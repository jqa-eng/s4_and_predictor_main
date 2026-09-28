# -*- coding: utf-8 -*-
"""Draw the molecular embedding with the actual virtual-screening score.

The original CLI is retained. Its three inputs are the complete predictions,
the output of mol_filter.py, and the experimentally validated structures.

For the 025 set, saved structure-embedding coordinates are reused so that the
same molecules keep the same positions as in the preceding figure. The cache
contains only canonical SMILES and x/y; SI, toxicity-derived IC50, and old
predictions are never used to color or rank points. For another molecule set,
the script computes a deterministic Morgan-fingerprint t-SNE embedding.

Outputs beside this script:
    molecules_2d_embedding.png
    molecules_2d_embedding.svg
    molecules_with_score.csv
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator


PROJECT_DIR = Path(__file__).resolve().parent
COORDINATE_CACHE = PROJECT_DIR / "graph" / "embedding_coordinates_025.csv"
OUT_PNG = PROJECT_DIR / "molecules_2d_embedding.png"
OUT_SVG = PROJECT_DIR / "molecules_2d_embedding.svg"
OUT_CSV = PROJECT_DIR / "molecules_with_score.csv"

SCORE_COLUMNS = [
    "sa_score", "similarity_335", "distance_335", "mic_percentile",
    "sa_percentile", "distance_335_percentile", "total_score", "rank",
]

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "svg.fonttype": "none",
    "axes.linewidth": 0.7,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
RDLogger.DisableLog("rdApp.*")


def read_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(name).strip() for name in df.columns]
    if "smiles" in df:
        df["smiles"] = df["smiles"].astype("string").str.strip().str.strip('"').str.strip("'")
    return df


def canonicalize_smiles(value):
    if pd.isna(value):
        return None
    mol = Chem.MolFromSmiles(str(value))
    return Chem.MolToSmiles(mol, canonical=True) if mol is not None else None


def require_columns(df: pd.DataFrame, columns, source: str):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{source} 缺少必要列: {', '.join(missing)}")


def load_inputs(processed_path: Path, filtered_path: Path, validated_path: Path):
    processed = read_clean(processed_path)
    filtered = read_clean(filtered_path)
    validated = read_clean(validated_path)
    require_columns(processed, ["smiles", "toxicity", "aureus_MIC", "ecoli_MIC"],
                    str(processed_path))
    require_columns(filtered, ["smiles", "aureus_MIC", *SCORE_COLUMNS],
                    str(filtered_path))
    require_columns(validated, ["label", "smiles"], str(validated_path))

    for df in (processed, filtered, validated):
        df["canonical_smiles"] = df["smiles"].map(canonicalize_smiles)
    if processed.canonical_smiles.isna().any():
        raise ValueError("完整分子列表含无效 SMILES，无法为全部结构生成嵌入坐标")
    if filtered.canonical_smiles.isna().any():
        raise ValueError("筛选结果含无效 SMILES")
    if not processed.canonical_smiles.is_unique or not filtered.canonical_smiles.is_unique:
        raise ValueError("输入存在重复的 canonical SMILES，无法逐一匹配分数与坐标")

    for column in ["aureus_MIC", "ecoli_MIC"]:
        processed[column] = pd.to_numeric(processed[column], errors="coerce")
    for column in ["aureus_MIC", *SCORE_COLUMNS]:
        filtered[column] = pd.to_numeric(filtered[column], errors="coerce")
    if filtered[["aureus_MIC", *SCORE_COLUMNS]].isna().any().any():
        raise ValueError("筛选结果存在缺失或非数值的综合排序字段")
    if not set(filtered.canonical_smiles).issubset(set(processed.canonical_smiles)):
        raise ValueError("筛选结果包含不在完整分子列表中的结构")

    expected_score = (filtered.mic_percentile + filtered.sa_percentile
                      + filtered.distance_335_percentile) / 3
    if not np.allclose(expected_score, filtered.total_score, atol=1e-8):
        raise ValueError("total_score 与三项百分位秩的等权平均不一致；请重跑 mol_filter.py")
    if not np.allclose(1 - filtered.similarity_335, filtered.distance_335,
                       atol=1e-8):
        raise ValueError("distance_335 与 1 - similarity_335 不一致")
    # The exported score is rounded; distinct full-precision scores may look tied.
    # Keep the rank written by mol_filter.py and check its ordering instead.
    if (not np.allclose(filtered["rank"], filtered["rank"].astype(int))
            or filtered["rank"].lt(1).any()
            or filtered["rank"].gt(len(filtered)).any()):
        raise ValueError("rank 必须是有效的正整数名次")
    ordered_by_rank = filtered.sort_values(["rank", "total_score"], kind="mergesort")
    if np.any(np.diff(ordered_by_rank.total_score.to_numpy()) < -1e-8):
        raise ValueError("rank 与 total_score 的升序顺序不一致")

    # mol_filter.py retains valid structures with 0 < predicted S. aureus MIC <= 20.
    expected_screen = processed.aureus_MIC.gt(0) & processed.aureus_MIC.le(20)
    if set(processed.loc[expected_screen, "canonical_smiles"]) != set(filtered.canonical_smiles):
        raise ValueError("filtered.csv 与当前预测文件的 MIC 初筛结果不一致；请重跑 mol_filter.py")

    matched_mic = processed[["canonical_smiles", "aureus_MIC"]].merge(
        filtered[["canonical_smiles", "aureus_MIC"]], on="canonical_smiles",
        how="inner", validate="one_to_one", suffixes=("_processed", "_filtered"))
    if not np.allclose(matched_mic.aureus_MIC_processed,
                       matched_mic.aureus_MIC_filtered, atol=1e-6):
        raise ValueError("filtered.csv 与当前预测文件中的金黄色葡萄球菌 MIC 不一致")

    if validated.canonical_smiles.isna().any():
        raise ValueError("实验验证分子文件含无效 SMILES")
    validated["label"] = validated["label"].astype(str).str.strip()
    validated = validated.drop_duplicates("canonical_smiles", keep="first")
    return processed, filtered, validated


def get_coordinates(processed: pd.DataFrame):
    """Preserve the prior layout when its coordinate cache matches this set."""
    if COORDINATE_CACHE.exists():
        cached = pd.read_csv(COORDINATE_CACHE, encoding="utf-8-sig")
        require_columns(cached, ["canonical_smiles", "x", "y"],
                        str(COORDINATE_CACHE))
        if (cached.canonical_smiles.is_unique
                and len(cached) == len(processed)
                and set(cached.canonical_smiles) == set(processed.canonical_smiles)
                and cached[["x", "y"]].notna().all().all()):
            xy = processed[["canonical_smiles"]].merge(
                cached[["canonical_smiles", "x", "y"]],
                on="canonical_smiles", how="left", validate="one_to_one")
            return xy[["x", "y"]].to_numpy(), "saved structural embedding"
        print("WARNING: 保存的坐标与当前分子集合不匹配，改为重新计算 Morgan/t-SNE")

    from sklearn.manifold import TSNE

    fingerprint = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    features = np.zeros((len(processed), 2048), dtype=np.uint8)
    for index, smiles in enumerate(processed.smiles):
        mol = Chem.MolFromSmiles(str(smiles))
        DataStructs.ConvertToNumpyArray(fingerprint.GetFingerprint(mol), features[index])
    coords = TSNE(n_components=2, perplexity=min(30, len(features) - 1),
                  learning_rate="auto", init="pca", random_state=42).fit_transform(features)
    print("WARNING: 已重新计算嵌入，整体坐标可能不同于以前保存的图")
    return coords, "new Morgan/t-SNE embedding"


def build_output(processed: pd.DataFrame, filtered: pd.DataFrame,
                 validated: pd.DataFrame, star_top_n: int):
    result = processed.merge(filtered[["canonical_smiles", *SCORE_COLUMNS]],
                             on="canonical_smiles", how="left", validate="one_to_one")
    result["rank"] = result["rank"].astype("Int64")
    result["is_candidate"] = result.total_score.notna()
    ordered = filtered.sort_values(["rank", "total_score"], kind="mergesort")
    stars = ordered if star_top_n <= 0 else ordered.head(star_top_n)
    result["is_star"] = result.canonical_smiles.isin(stars.canonical_smiles)
    validated_map = dict(zip(validated.canonical_smiles, validated.label))
    result["is_validated"] = result.canonical_smiles.isin(validated_map)
    result["validated_label"] = result.canonical_smiles.map(validated_map)
    coords, source = get_coordinates(processed)
    result[["x", "y"]] = coords
    columns = ["smiles", "canonical_smiles", "toxicity", "aureus_MIC", "ecoli_MIC",
               *SCORE_COLUMNS, "is_candidate", "is_star", "is_validated",
               "validated_label", "x", "y"]
    return result[columns], source


def draw(result: pd.DataFrame, star_top_n: int):
    norm = Normalize(vmin=0, vmax=1)
    cmap = mpl.colormaps["viridis"]
    fig, ax = plt.subplots(figsize=(7.20, 5.55), layout="constrained")
    outside = result.loc[~result.is_candidate]
    selected = result.loc[result.is_candidate]
    stars = result.loc[result.is_star]
    validated = result.loc[result.is_validated]

    ax.scatter(outside.x, outside.y, s=8, c="#BFC6CC", alpha=0.60,
               linewidths=0, rasterized=True, zorder=1)
    ax.scatter(selected.x, selected.y, s=12, c=selected.total_score,
               cmap=cmap, norm=norm, alpha=0.82, linewidths=0,
               rasterized=True, zorder=2)
    ax.scatter(stars.x, stars.y, s=47, marker="*", c=stars.total_score,
               cmap=cmap, norm=norm, edgecolors="black", linewidths=0.45,
               alpha=0.96, zorder=3)
    if not validated.empty:
        ax.scatter(validated.x, validated.y, s=125, marker="D",
                   facecolors="#E05A4F", edgecolors="black", linewidths=1.0,
                   zorder=5)
        for _, row in validated.iterrows():
            ax.annotate(row.validated_label, (row.x, row.y), xytext=(-8, 12),
                        textcoords="offset points", ha="right", fontsize=8.5,
                        fontweight="bold", color="#8A2522", zorder=6)

    ax.set_xlabel("Embedding 1")
    ax.set_ylabel("Embedding 2")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    n_stars = int(result.is_star.sum())
    diamond_label = "Validated compound"
    if len(validated) == 1:
        row = validated.iloc[0]
        diamond_label = f"Compound {row.validated_label}"
        if pd.notna(row["rank"]):
            diamond_label += f" (rank {int(row['rank'])}/{len(selected)})"
    handles = [
        Line2D([], [], linestyle="", marker="o", color="#BFC6CC", markersize=5,
               label=f"Outside MIC screen (n = {len(outside)})"),
        Line2D([], [], linestyle="", marker="o", color=cmap(0.45), markersize=5,
               label=f"Passed MIC screen (n = {len(selected)})"),
        Line2D([], [], linestyle="", marker="*", color=cmap(0.10),
               markeredgecolor="black", markersize=8,
               label=f"Top {n_stars} by composite rank"),
    ]
    if not validated.empty:
        handles.append(Line2D([], [], linestyle="", marker="D", color="#E05A4F",
                              markeredgecolor="black", markersize=6,
                              label=diamond_label))
    ax.legend(handles=handles, loc="upper left", fontsize=7.2,
              frameon=True, edgecolor="#D0D5D9", facecolor="white",
              framealpha=0.96)
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                            ax=ax, shrink=0.73, pad=0.02, aspect=30)
    colorbar.set_label("Composite score (lower is better)", fontsize=8)
    colorbar.ax.tick_params(labelsize=7)
    fig.savefig(OUT_PNG, dpi=400, bbox_inches="tight")
    fig.savefig(OUT_SVG, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="按综合得分绘制二维分子嵌入图")
    parser.add_argument("--mol-processed", type=str, default="molecules.csv",
                        help="完整分子预测结果 CSV")
    parser.add_argument("--mol-filtered", type=str, default="molecules_final.csv",
                        help="mol_filter.py 的综合排序结果 CSV")
    parser.add_argument("--star-top-n", type=int, default=100,
                        help="标星的综合排序前 N 个分子；<=0 表示全部通过初筛的分子")
    parser.add_argument("--mol-validated", type=str, default="validated_molecules.csv",
                        help="待突出显示的实验分子，含 label 与 smiles 两列")
    args = parser.parse_args()
    processed, filtered, validated = load_inputs(
        Path(args.mol_processed), Path(args.mol_filtered), Path(args.mol_validated))
    output, embedding_source = build_output(processed, filtered, validated,
                                            args.star_top_n)
    output.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    draw(output, args.star_top_n)
    print({
        "total_molecules": len(output),
        "screened_molecules": int(output.is_candidate.sum()),
        "starred_molecules": int(output.is_star.sum()),
        "validated_matches": int(output.is_validated.sum()),
        "embedding_source": embedding_source,
        "out_png": str(OUT_PNG),
        "out_svg": str(OUT_SVG),
        "out_csv": str(OUT_CSV),
    })


if __name__ == "__main__":
    main()
