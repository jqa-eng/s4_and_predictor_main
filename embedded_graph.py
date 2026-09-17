# -*- coding: utf-8 -*-
"""
embedded_graph.py
二维嵌入图（UMAP/t-SNE）+ SI 着色 + 候选星标 + 实验验证分子标注

输入：
- --mol-processed: 完整分子列表（用于计算嵌入与绘制全部散点）
- --mol-filtered: 筛选通过分子列表（仅用于决定哪些分子标星）
- --mol-validated: 实验验证分子列表（独立匹配、绘制并标注）

输出：
- molecules_2d_embedding.png
- molecules_2d_embedding.svg
- molecules_with_SI.csv
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from rdkit import Chem

# ---------------------- 可调参数 ----------------------
DATA_DIR = Path(".")

TOX_TO_IC50 = {
    "高毒": 5.0,
    "中毒": (10 + 75) / 2,
    "微毒": (75 + 200) / 2,
    "低毒": 250.0,
}

COLOR_MAP = {
    "both<5": "#4169e1",
    "one>5-other<=5": "#78c679",
    "both5-10": "#ffd000",
    "one>10-other<5": "#ff7f0e",
    "both>10": "#ff0000",
}

POINT_SIZE = 10
STAR_SIZE = 35
ALPHA_OTHER = 0.8
ALPHA_STAR = 0.75
VALIDATED_COLOR = "#8e44ad"
VALIDATED_SIZE = 90

FIGSIZE = (10, 8)
DPI = 180
OUT_PNG = "molecules_2d_embedding.png"
OUT_SVG = "molecules_2d_embedding.svg"
OUT_ENRICHED = "molecules_with_SI.csv"
# ----------------------------------------------------

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False


def read_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]
    if "smiles" in df.columns:
        df["smiles"] = df["smiles"].astype(str).str.strip().str.strip('"').str.strip("'")
    return df


def canonicalize_smiles(smiles):
    """返回 RDKit canonical SMILES；非法或空 SMILES 返回 None。"""
    try:
        mol = Chem.MolFromSmiles(str(smiles).strip())
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def build_validated_map(mol_validated: pd.DataFrame) -> dict:
    """建立 canonical SMILES -> label 映射，同结构多标签时保留第一条。"""
    validated_map = {}
    for _, row in mol_validated.iterrows():
        canonical = row["canonical_smiles"]
        if canonical is None or pd.isna(canonical):
            print(f"WARNING: validated molecule '{row['label']}' has an invalid SMILES")
            continue

        label = str(row["label"]).strip()
        if canonical in validated_map:
            if label != validated_map[canonical]:
                print(
                    "WARNING: multiple validated labels refer to the same structure; "
                    f"keeping '{validated_map[canonical]}' and ignoring '{label}'"
                )
            continue
        validated_map[canonical] = label
    return validated_map


def warn_unplotted_validated(
    validated_map: dict,
    mol_processed: pd.DataFrame,
    plot_df: pd.DataFrame,
    embedded_df: pd.DataFrame,
) -> None:
    """逐个报告未匹配或未进入最终二维图的验证分子。"""
    processed_structures = set(mol_processed["canonical_smiles"].dropna())
    plot_ready_structures = set(plot_df["canonical_smiles"].dropna())
    embedded_structures = set(embedded_df["canonical_smiles"].dropna())

    for canonical, label in validated_map.items():
        if canonical not in processed_structures:
            print(f"WARNING: validated molecule '{label}' was not found in mol_processed")
        elif canonical not in plot_ready_structures:
            print(
                f"WARNING: validated molecule '{label}' was matched but excluded "
                "because required MIC/toxicity data are missing"
            )
        elif canonical not in embedded_structures:
            print(
                f"WARNING: validated molecule '{label}' was matched but did not receive "
                "final embedding coordinates"
            )


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den = den.replace({0: np.nan})
    return num / den


def classify(si_sa: float, si_ec: float) -> str:
    if np.isnan(si_sa) or np.isnan(si_ec):
        return "both<5"
    if si_sa > 10 and si_ec > 10:
        return "both>10"
    if (si_sa > 10 and si_ec < 5) or (si_ec > 10 and si_sa < 5):
        return "one>10-other<5"
    if 5 <= si_sa <= 10 and 5 <= si_ec <= 10:
        return "both5-10"
    if (si_sa > 5 and si_ec <= 5) or (si_ec > 5 and si_sa <= 5):
        return "one>5-other<=5"
    return "both<5"


def compute_embedding(plot_df: pd.DataFrame):
    try:
        from rdkit import DataStructs
        from rdkit.Chem import AllChem

        def morgan_fp(smiles: str, n_bits=2048, radius=2):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            arr = np.zeros((n_bits,), dtype=np.int8)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr

        fps = []
        keep_idx = []
        for i, smi in enumerate(plot_df["smiles"].astype(str)):
            fp = morgan_fp(smi)
            if fp is not None:
                fps.append(fp)
                keep_idx.append(i)

        if not fps:
            raise RuntimeError("No valid Morgan fingerprints.")

        features = np.stack(fps, axis=0)
        emb_df = plot_df.iloc[keep_idx].copy()
    except Exception:
        features = plot_df[["aureus_MIC", "ecoli_MIC", "IC50_map", "SI_SA", "SI_EC"]].to_numpy(
            dtype=float
        )
        emb_df = plot_df.copy()

    try:
        import umap

        metric = "jaccard" if np.issubdtype(features.dtype, np.integer) else "euclidean"
        reducer = umap.UMAP(
            n_neighbors=30,
            min_dist=0.1,
            metric=metric,
            random_state=42,
        )
        coords = reducer.fit_transform(features)
    except Exception:
        from sklearn.manifold import TSNE

        n_samples = len(features)
        if n_samples < 2:
            raise ValueError("Not enough samples for dimensionality reduction.")
        perplexity = min(30, max(5, n_samples - 1))
        if perplexity >= n_samples:
            perplexity = max(1, n_samples // 3)
        coords = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate="auto",
            init="pca",
            random_state=42,
        ).fit_transform(features)

    emb_df["x"] = coords[:, 0]
    emb_df["y"] = coords[:, 1]
    return emb_df


def main():
    parser = argparse.ArgumentParser(description="绘制二维分子嵌入图")
    parser.add_argument(
        "--mol-processed",
        type=str,
        default="molecules.csv",
        help="完整分子列表 CSV 文件路径，用于计算嵌入和绘制全部散点",
    )
    parser.add_argument(
        "--mol-filtered",
        type=str,
        default="molecules_final.csv",
        help="筛选通过分子 CSV 文件路径，用于标星",
    )
    parser.add_argument(
        "--star-top-n",
        type=int,
        default=100,
        help="仅对 mol-filtered 中优先级最高的前 N 个分子标星；<=0 表示全部标星",
    )
    parser.add_argument(
        "--mol-validated",
        type=str,
        default="validated_molecules.csv",
        help="实验验证分子 CSV 文件路径，用于单独突出显示和标注",
    )
    args = parser.parse_args()

    processed_path = Path(args.mol_processed)
    filtered_path = Path(args.mol_filtered)
    validated_path = Path(args.mol_validated)

    mol_processed = read_clean(processed_path)
    mol_filtered = read_clean(filtered_path)
    mol_validated = read_clean(validated_path)

    required_processed_columns = ["smiles", "toxicity", "aureus_MIC", "ecoli_MIC"]
    missing_processed = [c for c in required_processed_columns if c not in mol_processed.columns]
    if missing_processed:
        raise ValueError(f"mol-processed 文件缺少必要列: {missing_processed}")

    if "smiles" not in mol_filtered.columns:
        raise ValueError("mol-filtered 文件缺少必要列: ['smiles']")

    missing_validated = [c for c in ["label", "smiles"] if c not in mol_validated.columns]
    if missing_validated:
        raise ValueError(f"mol-validated 文件缺少必要列: {missing_validated}")

    for df in [mol_processed, mol_filtered, mol_validated]:
        df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)

    filtered_smiles_series = mol_filtered["smiles"].astype(str).str.strip()
    filtered_df = mol_filtered.copy()
    filtered_df["smiles"] = filtered_smiles_series
    filtered_df = filtered_df[
        (filtered_df["smiles"] != "")
        & (filtered_df["smiles"].str.lower() != "nan")
    ].copy()

    if args.star_top_n > 0:
        if "aureus_MIC" in filtered_df.columns:
            filtered_df["aureus_MIC"] = pd.to_numeric(filtered_df["aureus_MIC"], errors="coerce")
            filtered_df = filtered_df.sort_values(by="aureus_MIC", ascending=True, na_position="last")
            star_canonical_smiles = filtered_df["canonical_smiles"].head(args.star_top_n)
        else:
            star_canonical_smiles = filtered_df["canonical_smiles"].head(args.star_top_n)
    else:
        star_canonical_smiles = filtered_df["canonical_smiles"]

    candidate_set = set(filtered_df["canonical_smiles"].dropna())
    star_set = set(star_canonical_smiles.dropna())
    mol_processed["is_candidate"] = mol_processed["canonical_smiles"].isin(candidate_set)
    mol_processed["is_star"] = mol_processed["canonical_smiles"].isin(star_set)

    validated_map = build_validated_map(mol_validated)
    mol_processed["is_validated"] = mol_processed["canonical_smiles"].isin(validated_map)
    mol_processed["validated_label"] = mol_processed["canonical_smiles"].map(validated_map)

    mol_processed["toxicity"] = mol_processed["toxicity"].astype(str).str.strip()
    for col in ["aureus_MIC", "ecoli_MIC"]:
        mol_processed[col] = pd.to_numeric(mol_processed[col], errors="coerce")
    mol_processed["IC50_map"] = mol_processed["toxicity"].map(TOX_TO_IC50).astype(float)

    plot_df = mol_processed.copy()
    plot_df["smiles"] = plot_df["smiles"].astype(str).str.strip()
    plot_df = plot_df[
        (plot_df["smiles"] != "")
        & (plot_df["smiles"].str.lower() != "nan")
    ].copy()
    plot_df = plot_df.dropna(subset=["aureus_MIC", "ecoli_MIC", "IC50_map"]).copy()

    plot_df["SI_SA"] = safe_div(plot_df["IC50_map"], plot_df["aureus_MIC"])
    plot_df["SI_EC"] = safe_div(plot_df["IC50_map"], plot_df["ecoli_MIC"])
    plot_df["cls"] = [classify(a, b) for a, b in zip(plot_df["SI_SA"], plot_df["SI_EC"])]

    embedded_df = compute_embedding(plot_df)
    warn_unplotted_validated(validated_map, mol_processed, plot_df, embedded_df)

    plt.close("all")
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    for key, color in COLOR_MAP.items():
        sub = embedded_df[
            (embedded_df["cls"] == key)
            & (~embedded_df["is_star"])
            & (~embedded_df["is_validated"])
        ]
        if len(sub) == 0:
            continue
        ax.scatter(
            sub["x"],
            sub["y"],
            s=POINT_SIZE,
            c=color,
            alpha=ALPHA_OTHER,
            edgecolors="none",
            label=None,
        )

    star = embedded_df[embedded_df["is_star"] & ~embedded_df["is_validated"]]
    if len(star) > 0:
        ax.scatter(
            star["x"],
            star["y"],
            s=STAR_SIZE,
            marker="*",
            c=COLOR_MAP["both>10"],
            edgecolors="k",
            linewidths=0.4,
            alpha=ALPHA_STAR,
            label=None,
        )

    validated = embedded_df[embedded_df["is_validated"]]
    if len(validated) > 0:
        ax.scatter(
            validated["x"],
            validated["y"],
            s=VALIDATED_SIZE,
            marker="D",
            c=VALIDATED_COLOR,
            edgecolors="black",
            linewidths=1.0,
            alpha=1.0,
            zorder=10,
            label=None,
        )
        for _, row in validated.iterrows():
            ax.annotate(
                str(row["validated_label"]),
                (row["x"], row["y"]),
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=9,
                color="black",
                zorder=11,
            )

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=COLOR_MAP["both<5"],
            markeredgecolor="none",
            markersize=8,
            label="SI$_{S.aureus}$<5, SI$_{E.coli}$<5",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=COLOR_MAP["one>5-other<=5"],
            markeredgecolor="none",
            markersize=8,
            label="SI$_{S.aureus}$>5 或 SI$_{E.coli}$>5",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=COLOR_MAP["both5-10"],
            markeredgecolor="none",
            markersize=8,
            label="5≤SI<10（两菌）",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=COLOR_MAP["one>10-other<5"],
            markeredgecolor="none",
            markersize=8,
            label="SI>10 且另一菌<5",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=COLOR_MAP["both>10"],
            markeredgecolor="none",
            markersize=8,
            label="SI>10（两菌）",
        ),
    ]
    shape_handles = [
        Line2D(
            [0],
            [0],
            marker="*",
            linestyle="",
            markerfacecolor=COLOR_MAP["both>10"],
            markeredgecolor="k",
            markersize=10,
            label="候选（星标）",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="",
            markerfacecolor=VALIDATED_COLOR,
            markeredgecolor="black",
            markersize=8,
            label="实验验证分子",
        ),
    ]
    handles = color_handles + shape_handles
    legend = ax.legend(
        handles=handles,
        title="图例：颜色=SI 等级；★=计算候选；◆=实验验证分子",
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=True,
        borderaxespad=0.4,
        borderpad=0.4,
        handletextpad=0.6,
        labelspacing=0.6,
    )
    legend._legend_box.align = "left"

    fig.tight_layout()
    fig.savefig(DATA_DIR / OUT_PNG, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(DATA_DIR / OUT_SVG, bbox_inches="tight", pad_inches=0.2)

    desired_columns = [
        "smiles",
        "canonical_smiles",
        "toxicity",
        "aureus_MIC",
        "ecoli_MIC",
        "IC50_map",
        "SI_SA",
        "SI_EC",
        "cls",
        "is_candidate",
        "is_star",
        "is_validated",
        "validated_label",
        "x",
        "y",
    ]
    existing_columns = [c for c in desired_columns if c in embedded_df.columns]
    extra_columns = [c for c in embedded_df.columns if c not in existing_columns]
    embedded_df = embedded_df[existing_columns + extra_columns]
    embedded_df.to_csv(DATA_DIR / OUT_ENRICHED, index=False, encoding="utf-8-sig")

    print(
        {
            "mol_processed": str(processed_path),
            "mol_filtered": str(filtered_path),
            "mol_validated": str(validated_path),
            "total_processed_rows": int(len(mol_processed)),
            "filtered_smiles": int(len(candidate_set)),
            "star_top_n": int(args.star_top_n),
            "star_smiles_used": int(len(star_set)),
            "matched_star_rows": int(mol_processed["is_star"].sum()),
            "validated_input_rows": int(len(mol_validated)),
            "validated_unique_structures": int(len(validated_map)),
            "matched_validated_rows": int(mol_processed["is_validated"].sum()),
            "validated_rows_in_plot": int(embedded_df["is_validated"].sum()),
            "embedded_rows": int(len(embedded_df)),
            "star_rows_in_plot": int(embedded_df["is_star"].sum()),
            "out_png": OUT_PNG,
            "out_svg": OUT_SVG,
            "out_csv": OUT_ENRICHED,
        }
    )


if __name__ == "__main__":
    main()
