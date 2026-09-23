# -*- coding: utf-8 -*-
"""
cubic_graph.py - 三维分子性质分布图绘制脚本

功能：
- 读取完整分子预测结果文件 mol-processed；
- 以 S.aureus_MIC 为 x 轴；
- 以 E.coli_MIC 为 y 轴；
- 将毒性标签 toxicity 映射为 IC50 代表值作为 z 轴；
- 使用 toxicity 控制散点颜色；
- 保留全部分子的背景散点，不再标记筛选候选；
- 从实验验证文件中只选取 336，使用 RDKit canonical SMILES 匹配；
- 以菱形单独标注 336。

输入：
1. --mol-processed：完整分子列表，必须包含 smiles、toxicity、aureus_MIC、ecoli_MIC；
2. --mol-filtered：兼容旧命令的参数，现已忽略。
3. --mol-validated：实验验证分子列表，至少包含 label、smiles，且包含 336。

输出：
- molecules_3d_scatter.png
- molecules_3d_scatter.svg
- molecules_3d_scatter.pdf
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from rdkit import Chem

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans", "Arial"]
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["axes.unicode_minus"] = False
mpl.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})


# =============== 参数区（按需修改）===============
DATA_DIR = Path(".")

TOX_TO_Z = {
    "高毒": 5.0,
    "中毒": (10 + 75) / 2,
    "微毒": (75 + 200) / 2,
    "低毒": 250.0,
}

TOX_TO_COLOR = {
    "低毒": "#2ca02c",
    "微毒": "#ffdd00",
    "中毒": "#ff7f0e",
    "高毒": "#d62728",
}

SIZE_CIRCLE = 12
ALPHA_CIRCLE = 0.6
VALIDATED_COLOR = "#8e44ad"
SIZE_VALIDATED = 100

ELEV = 22
AZIM = -65

Z_LIM = (0, 300)
Z_TICKS = [0, 10, 75, 200, 300]
X_LIM = (0, 100)
Y_LIM = (0, 100)

OUT_PNG = "molecules_3d_scatter.png"
OUT_SVG = "molecules_3d_scatter.svg"
OUT_PDF = "molecules_3d_scatter.pdf"
DPI = 300
FIGSIZE = (10, 8)
# ==============================================


def read_clean(path: Path) -> pd.DataFrame:
    """读取 CSV，并清理列名和 smiles 字段。"""
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
    coordinate_ready_df: pd.DataFrame,
    plot_df: pd.DataFrame,
) -> None:
    """逐个报告未匹配、缺少坐标数据或被坐标范围排除的验证分子。"""
    processed_structures = set(mol_processed["canonical_smiles"].dropna())
    coordinate_ready_structures = set(coordinate_ready_df["canonical_smiles"].dropna())
    plotted_structures = set(plot_df["canonical_smiles"].dropna())

    for canonical, label in validated_map.items():
        if canonical not in processed_structures:
            print(f"WARNING: validated molecule '{label}' was not found in mol_processed")
        elif canonical not in coordinate_ready_structures:
            print(
                f"WARNING: validated molecule '{label}' was matched but excluded "
                "because required MIC/toxicity coordinates are missing"
            )
        elif canonical not in plotted_structures:
            print(
                f"WARNING: validated molecule '{label}' was matched but excluded by plot range"
            )


def main():
    parser = argparse.ArgumentParser(description="绘制分子三维散点图")
    parser.add_argument(
        "--mol-processed",
        type=str,
        default="molecules.csv",
        help="完整分子列表 CSV 文件路径，用于绘制所有散点",
    )
    parser.add_argument(
        "--mol-filtered",
        type=str,
        default=None,
        help="兼容旧命令；不再读取或标记候选分子",
    )
    parser.add_argument(
        "--mol-validated",
        type=str,
        default="validated_molecules.csv",
        help="实验验证分子 CSV 文件路径，用于单独突出显示和标注",
    )
    args = parser.parse_args()

    base = DATA_DIR
    processed_path = Path(args.mol_processed)
    validated_path = Path(args.mol_validated)

    mol_processed = read_clean(processed_path)
    mol_validated = read_clean(validated_path)

    required_processed_columns = ["smiles", "toxicity", "aureus_MIC", "ecoli_MIC"]
    missing_processed = [c for c in required_processed_columns if c not in mol_processed.columns]
    if missing_processed:
        raise ValueError(f"mol-processed 文件缺少必要列: {missing_processed}")

    missing_validated = [c for c in ["label", "smiles"] if c not in mol_validated.columns]
    if missing_validated:
        raise ValueError(f"mol-validated 文件缺少必要列: {missing_validated}")

    mol_validated = mol_validated.loc[
        mol_validated["label"].astype(str).str.strip().eq("336")
    ].copy()
    if mol_validated.empty:
        raise ValueError("mol-validated 文件中未找到标签为 336 的分子")

    for df in [mol_processed, mol_validated]:
        df["canonical_smiles"] = df["smiles"].apply(canonicalize_smiles)

    validated_map = build_validated_map(mol_validated)
    mol_processed["is_validated"] = mol_processed["canonical_smiles"].isin(validated_map)
    mol_processed["validated_label"] = mol_processed["canonical_smiles"].map(validated_map)

    for col in ["aureus_MIC", "ecoli_MIC"]:
        mol_processed[col] = pd.to_numeric(mol_processed[col], errors="coerce")

    mol_processed["toxicity"] = mol_processed["toxicity"].astype(str).str.strip()
    mol_processed["IC50_z"] = mol_processed["toxicity"].map(TOX_TO_Z)

    must_cols = ["aureus_MIC", "ecoli_MIC", "IC50_z"]
    coordinate_ready_df = mol_processed.dropna(subset=must_cols).copy()

    x0, x1 = X_LIM
    y0, y1 = Y_LIM
    z0, z1 = Z_LIM
    in_range = (
        coordinate_ready_df["aureus_MIC"].between(x0, x1)
        & coordinate_ready_df["ecoli_MIC"].between(y0, y1)
        & coordinate_ready_df["IC50_z"].between(z0, z1)
    )
    plot_df = coordinate_ready_df.loc[in_range].copy()

    warn_unplotted_validated(validated_map, mol_processed, coordinate_ready_df, plot_df)

    plt.close("all")
    fig = plt.figure(figsize=FIGSIZE, dpi=DPI)
    # Disable mplot3d's collection depth sorting so the 336 marker can stay
    # above nearby background points regardless of the viewing angle.
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    ax.view_init(elev=ELEV, azim=AZIM)
    ax.set_xlabel(r"MIC$_{S.\,aureus}$ ($\mu$g/mL)")
    ax.set_ylabel(r"MIC$_{E.\,coli}$ ($\mu$g/mL)")
    ax.set_zlabel(r"IC$_{50}$ ($\mu$g/mL)")

    ax.set_zlim(*Z_LIM)
    ax.set_zticks(Z_TICKS)

    for tox, color in TOX_TO_COLOR.items():
        sub = plot_df[plot_df["toxicity"] == tox]
        if sub.empty:
            continue

        background = sub[~sub["is_validated"]]
        if not background.empty:
            ax.scatter(
                background["aureus_MIC"].values,
                background["ecoli_MIC"].values,
                background["IC50_z"].values,
                marker="o",
                s=SIZE_CIRCLE,
                edgecolors="none",
                c=color,
                alpha=ALPHA_CIRCLE,
                zorder=2,
            )

    validated = plot_df[plot_df["is_validated"]]
    if not validated.empty:
        ax.scatter(
            validated["aureus_MIC"].values,
            validated["ecoli_MIC"].values,
            validated["IC50_z"].values,
            marker="D",
            s=SIZE_VALIDATED,
            edgecolors="black",
            linewidths=1.0,
            c=VALIDATED_COLOR,
            alpha=1.0,
            depthshade=False,
            zorder=10,
        )
        for _, row in validated.iterrows():
            ax.text(
                row["aureus_MIC"] + 1.0,
                row["ecoli_MIC"] + 1.0,
                row["IC50_z"] + 3.0,
                str(row["validated_label"]),
                fontsize=9,
                color="black",
                zorder=11,
            )

    color_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=TOX_TO_COLOR[t],
            markeredgecolor="none",
            markersize=8,
            label=t,
        )
        for t in ["低毒", "微毒", "中毒", "高毒"]
        if (plot_df["toxicity"] == t).any()
    ]
    shape_handles = [
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="",
            markerfacecolor=VALIDATED_COLOR,
            markeredgecolor="black",
            markersize=8,
            label="336",
        ),
    ]
    handles = color_handles + shape_handles

    ax.legend(
        handles=handles,
        title="图例",
        loc="upper left",
        bbox_to_anchor=(1.00, 1.02),
        frameon=True,
        borderaxespad=0.6,
        borderpad=0.6,
        handletextpad=0.6,
        labelspacing=0.5,
        columnspacing=1.0,
        ncol=1,
    )

    ax.grid(True)
    ax.set_xlim(*X_LIM)
    ax.set_ylim(*Y_LIM)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    fig.tight_layout()
    fig.savefig(base / OUT_PNG, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(base / OUT_SVG, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(base / OUT_PDF, bbox_inches="tight", pad_inches=0.15)

    print(
        {
            "mol_processed": str(processed_path),
            "mol_validated": str(validated_path),
            "total_processed_rows": int(len(mol_processed)),
            "336_input_rows": int(len(mol_validated)),
            "336_matched_rows": int(mol_processed["is_validated"].sum()),
            "336_rows_in_plot": int(plot_df["is_validated"].sum()),
            "total_points_plotted": int(len(plot_df)),
            "background_points_plotted": int((~plot_df["is_validated"]).sum()),
            "dropped_missing_or_out_of_range": int(len(mol_processed) - len(plot_df)),
            "out_png": str(Path(OUT_PNG)),
            "out_svg": str(Path(OUT_SVG)),
            "out_pdf": str(Path(OUT_PDF)),
        }
    )


if __name__ == "__main__":
    main()
