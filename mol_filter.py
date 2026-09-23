#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Filter molecules by predicted S. aureus MIC and rank the survivors.

Only molecules with 0 < aureus_MIC <= 20 and valid SMILES enter the ranking.
The three equally weighted criteria are lower MIC, lower synthetic
accessibility (SA) score, and greater Morgan/Tanimoto similarity to 335.
Each criterion is converted to a percentile rank within the filtered set;
lower combined scores are better. QED, toxicity, and E. coli MIC are not
ranking criteria.
"""

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Contrib.SA_Score import sascorer


REFERENCE_335_SMILES = (
    "O=C(C1=C(N(CC2=CC=CC=C2)CC3=CC=CC=C3)NN=N1)C4=CC=CC=C4"
)
MAX_AUREUS_MIC = 20.0
MORGAN_RADIUS = 2
MORGAN_BITS = 2048


def _percentile_rank(values: pd.Series, *, ascending: bool) -> pd.Series:
    """Return 0-best to 1-worst midrank percentiles, including ties."""
    if len(values) == 1:
        return pd.Series(0.0, index=values.index)
    return (values.rank(method="average", ascending=ascending) - 1) / (len(values) - 1)


def filter_molecules(input_file: str, output_file: str) -> pd.DataFrame:
    """Write a MIC-filtered, three-criterion ranking and return it."""
    source = pd.read_csv(input_file, encoding="utf-8-sig")
    required = {"smiles", "aureus_MIC"}
    missing = required.difference(source.columns)
    if missing:
        raise ValueError(f"输入文件缺少必要列: {', '.join(sorted(missing))}")

    source["aureus_MIC"] = pd.to_numeric(source["aureus_MIC"], errors="coerce")
    # A missing, nonfinite, or nonpositive MIC cannot be meaningfully ranked.
    eligible = source.loc[
        source["aureus_MIC"].gt(0) & source["aureus_MIC"].le(MAX_AUREUS_MIC)
    ].copy()

    reference_mol = Chem.MolFromSmiles(REFERENCE_335_SMILES)
    if reference_mol is None:
        raise ValueError("脚本中的 335 SMILES 无法被 RDKit 解析")
    fingerprint_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
    )
    reference_fingerprint = fingerprint_generator.GetFingerprint(reference_mol)

    valid_indices = []
    sa_scores = []
    similarities = []
    for index, smiles in eligible["smiles"].items():
        if not isinstance(smiles, str) or not smiles.strip():
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        valid_indices.append(index)
        sa_scores.append(sascorer.calculateScore(mol))
        similarities.append(
            DataStructs.TanimotoSimilarity(
                fingerprint_generator.GetFingerprint(mol), reference_fingerprint
            )
        )

    ranked = eligible.loc[valid_indices].copy()
    ranked["sa_score"] = sa_scores
    ranked["similarity_335"] = similarities
    ranked["distance_335"] = 1 - ranked["similarity_335"]

    if not ranked.empty:
        ranked["mic_percentile"] = _percentile_rank(ranked["aureus_MIC"], ascending=True)
        ranked["sa_percentile"] = _percentile_rank(ranked["sa_score"], ascending=True)
        ranked["distance_335_percentile"] = _percentile_rank(
            ranked["distance_335"], ascending=True
        )
        ranked["total_score"] = (
            ranked["mic_percentile"]
            + ranked["sa_percentile"]
            + ranked["distance_335_percentile"]
        ) / 3
        ranked["rank"] = ranked["total_score"].rank(method="min").astype(int)
        ranked = ranked.sort_values("total_score", kind="mergesort").reset_index(drop=True)
    else:
        for column in (
            "mic_percentile", "sa_percentile", "distance_335_percentile", "total_score", "rank"
        ):
            ranked[column] = pd.Series(dtype="float64")

    ranked.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"输入: {input_file} ({len(source)} 个分子)")
    print(f"MIC 在 (0, {MAX_AUREUS_MIC:g}] 内: {len(eligible)} 个")
    print(f"有效 SMILES 并参与排序: {len(ranked)} 个")
    print(f"输出: {output_file}")
    return ranked


def main() -> None:
    parser = argparse.ArgumentParser(description="按 MIC、SA 和 335 相似度筛选并排序分子")
    parser.add_argument(
        "--input",
        default=str(Path("validation") / "mol_processed.csv"),
        help="含 smiles 和 aureus_MIC 列的输入 CSV",
    )
    parser.add_argument("--output", default="filter.csv", help="排序后的输出 CSV")
    args = parser.parse_args()
    filter_molecules(args.input, args.output)


if __name__ == "__main__":
    main()
