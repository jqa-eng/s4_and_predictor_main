#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot the complete prediction set against the three screening criteria.

The MIC-screened ranking in filtered.csv determines candidate membership,
composite scores and ranks. SA score and Tanimoto distance are computed for
all molecules using the same definitions as mol_filter.py. Compound 336 is
identified by canonical SMILES in the validated-molecule file.
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

from mol_filter import (MAX_AUREUS_MIC, MORGAN_BITS, MORGAN_RADIUS,
                        REFERENCE_335_SMILES, sascorer)


PROJECT_DIR = Path(__file__).resolve().parent
OUT_PNG = PROJECT_DIR / "molecules_3d_scatter.png"
OUT_SVG = PROJECT_DIR / "molecules_3d_scatter.svg"
MIC_TICK_STEP = 10
MIC_OVERVIEW_MAX = 100

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9,
    "mathtext.fontset": "dejavusans",
    "svg.fonttype": "none",
})
RDLogger.DisableLog("rdApp.*")


def require_matplotlib_panel_alignment(fig, overview_ax, detail_ax, colorbar_ax):
    """Check separation of intentionally unequal overview, detail, and legend axes."""
    fig.canvas.draw()
    overview = overview_ax.get_position()
    detail = detail_ax.get_position()
    colorbar = colorbar_ax.get_position()
    if not (overview.x1 + 0.015 < detail.x0
            and detail.x1 + 0.015 < colorbar.x0
            and overview.y0 < detail.y0 < detail.y1 < overview.y1):
        raise ValueError("The overview, MIC detail, and colorbar overlap")


def read_input(path: Path, required: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    frame.columns = [str(column).strip() for column in frame.columns]
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    frame["smiles"] = frame.smiles.astype("string").str.strip()
    frame["canonical_smiles"] = frame.smiles.map(
        lambda value: (Chem.MolToSmiles(molecule, canonical=True)
                       if (molecule := Chem.MolFromSmiles(str(value))) is not None
                       else None))
    if frame.canonical_smiles.isna().any():
        raise ValueError(f"{path} contains invalid SMILES")
    return frame


def load_inputs(processed_path: Path, filtered_path: Path,
                validated_path: Path):
    processed = read_input(processed_path, {"smiles", "aureus_MIC"})
    ranking_columns = {"smiles", "aureus_MIC", "sa_score", "distance_335",
                       "mic_percentile", "sa_percentile",
                       "distance_335_percentile", "total_score", "rank"}
    filtered = read_input(filtered_path, ranking_columns)
    validated = read_input(validated_path, {"label", "smiles"})
    if not processed.canonical_smiles.is_unique or not filtered.canonical_smiles.is_unique:
        raise ValueError("Processed or filtered structures are not unique")
    processed["aureus_MIC"] = pd.to_numeric(processed.aureus_MIC, errors="coerce")
    for column in ranking_columns - {"smiles"}:
        filtered[column] = pd.to_numeric(filtered[column], errors="coerce")
    if not np.isfinite(processed.aureus_MIC).all():
        raise ValueError("Processed MIC contains missing or nonfinite values")
    if not np.isfinite(filtered[list(ranking_columns - {"smiles"})]).all().all():
        raise ValueError("Filtered ranking contains missing or nonfinite values")
    expected = set(processed.loc[
        processed.aureus_MIC.gt(0) & processed.aureus_MIC.le(MAX_AUREUS_MIC),
        "canonical_smiles"])
    if set(filtered.canonical_smiles) != expected:
        raise ValueError("Filtered candidates do not match the processed MIC screen")
    matched = processed[["canonical_smiles", "aureus_MIC"]].merge(
        filtered[["canonical_smiles", "aureus_MIC"]], on="canonical_smiles",
        validate="one_to_one", suffixes=("_processed", "_filtered"))
    if not np.allclose(matched.aureus_MIC_processed,
                       matched.aureus_MIC_filtered, atol=1e-6):
        raise ValueError("Processed and filtered MIC predictions differ")
    expected_score = (filtered.mic_percentile + filtered.sa_percentile
                      + filtered.distance_335_percentile) / 3
    if not np.allclose(filtered.total_score, expected_score, atol=1e-8):
        raise ValueError("Composite scores do not match the screening rule")
    validated["label"] = validated.label.astype(str).str.strip()
    return processed, filtered, validated


def prepare_data(processed_path: Path, filtered_path: Path,
                 validated_path: Path) -> pd.DataFrame:
    processed, filtered, validated = load_inputs(
        processed_path, filtered_path, validated_path)
    match_336 = validated.loc[validated.label.eq("336")]
    if len(match_336) != 1:
        raise ValueError("The validated file must contain exactly one compound 336")
    smiles_336 = match_336.iloc[0].canonical_smiles
    if smiles_336 not in set(filtered.canonical_smiles):
        raise ValueError("Compound 336 is absent from the screened candidates")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_BITS)
    reference = Chem.MolFromSmiles(REFERENCE_335_SMILES)
    reference_fp = generator.GetFingerprint(reference)
    molecules = [Chem.MolFromSmiles(smiles) for smiles in processed.canonical_smiles]
    processed["sa_score"] = [sascorer.calculateScore(mol) for mol in molecules]
    processed["distance_335"] = [
        1 - DataStructs.TanimotoSimilarity(generator.GetFingerprint(mol), reference_fp)
        for mol in molecules
    ]
    joined = processed.merge(
        filtered[["canonical_smiles", "sa_score", "distance_335",
                  "total_score", "rank"]],
        on="canonical_smiles", how="left", validate="one_to_one",
        suffixes=("", "_filtered"), indicator=True)
    selected = joined._merge.eq("both")
    for column in ("sa_score", "distance_335"):
        discrepancy = np.abs(joined.loc[selected, column].to_numpy()
                             - joined.loc[selected, f"{column}_filtered"].to_numpy())
        if np.any(discrepancy > 1e-6):
            print(f"WARNING: {int(np.sum(discrepancy > 1e-6))} stored {column} "
                  "values differ from a fresh calculation; using filtered.csv "
                  "for all screened candidates")
        # The saved ranking file is authoritative for candidate coordinates.
        joined.loc[selected, column] = joined.loc[selected, f"{column}_filtered"].to_numpy()
    joined["is_candidate"] = selected
    joined["is_336"] = joined.canonical_smiles.eq(smiles_336)
    top_100 = filtered.sort_values(["rank", "total_score"], kind="mergesort").head(100)
    joined["is_top100"] = joined.canonical_smiles.isin(top_100.canonical_smiles)
    if not joined.aureus_MIC.ge(0).all():
        raise ValueError("Predicted MIC values must be nonnegative")
    if not np.isfinite(joined[["aureus_MIC", "sa_score", "distance_335"]]).all().all():
        raise ValueError("All three plot coordinates must be finite")
    return joined


def draw(data: pd.DataFrame):
    overview = data.loc[data.aureus_MIC.le(MIC_OVERVIEW_MAX)]
    outside = overview.loc[~overview.is_candidate]
    selected = overview.loc[overview.is_candidate]
    stars = overview.loc[overview.is_top100]
    compound_336 = data.loc[data.is_336].iloc[0]
    norm = Normalize(vmin=0, vmax=1)
    cmap = mpl.colormaps["viridis"]
    fig = plt.figure(figsize=(12.0, 7.2), dpi=300)
    # All screened candidates receive a score-colored circle, and the top 100
    # receive a star overlay. The larger 336 diamond is drawn last and above
    # its star, so it remains visible at the same coordinates.
    ax = fig.add_axes([0.02, 0.12, 0.60, 0.80], projection="3d",
                      computed_zorder=False)
    # The oblique overview preserves all three spatial dimensions within the
    # displayed 0–100 MIC range. Projection is not an MIC threshold test;
    # the companion detail view shows that test in linear MIC coordinates.
    ax.view_init(elev=22, azim=-58)
    ax.set_proj_type("ortho")
    ax.scatter(outside.aureus_MIC, outside.sa_score, outside.distance_335,
               marker="o", s=9, c="#BFC6CC", alpha=0.30,
               edgecolors="none", depthshade=False, zorder=1)
    ax.scatter(selected.aureus_MIC, selected.sa_score, selected.distance_335,
               marker="o", s=15, c=selected.total_score, cmap=cmap, norm=norm,
               alpha=0.72, edgecolors="none", depthshade=False, zorder=2)
    ax.scatter(stars.aureus_MIC, stars.sa_score, stars.distance_335,
               marker="*", s=55, c=stars.total_score, cmap=cmap, norm=norm,
               edgecolors="#202020", linewidths=0.35, alpha=0.96,
               depthshade=False, zorder=3)
    ax.scatter([compound_336.aureus_MIC], [compound_336.sa_score],
               [compound_336.distance_335], marker="D", s=145,
               c="#D94A38", edgecolors="black", linewidths=1.2,
               alpha=1, depthshade=False, zorder=20)
    ax.text(compound_336.aureus_MIC, compound_336.sa_score,
            compound_336.distance_335 + 0.04, "336", color="#A3261B",
            fontsize=10, fontweight="bold", zorder=21)

    ax.set_xlabel("Predicted S. aureus MIC (µg/mL)", labelpad=12)
    ax.set_ylabel("Synthetic accessibility score", labelpad=12)
    ax.set_zlabel("Tanimoto distance to compound 335", labelpad=11)
    mic_ticks = np.arange(0, MIC_OVERVIEW_MAX + 1, MIC_TICK_STEP)
    ax.set_xticks(mic_ticks)
    ax.set_xlim(0, MIC_OVERVIEW_MAX)
    sa_axis_max = max(7, int(np.ceil(data.sa_score.max())))
    ax.set_ylim(sa_axis_max, 0)
    ax.set_yticks(np.arange(0, sa_axis_max + 1, 1))
    ax.set_zlim(0, min(1, max(0.85, float(data.distance_335.max()) + 0.03)))
    ax.set_box_aspect((1.6, 1, 0.9))
    ax.grid(True, linewidth=0.5, alpha=0.4)

    # A companion projection makes the (0, 20] MIC screen unambiguous: every
    # gray molecule is to the right of the 20 µg/mL boundary. The overview
    # above still contains all 1514 molecules and all three ranking criteria.
    detail = fig.add_axes([0.72, 0.22, 0.18, 0.55])
    detail_max = 2 * MAX_AUREUS_MIC
    detail_outside = outside.loc[outside.aureus_MIC.le(detail_max)]
    detail.scatter(detail_outside.aureus_MIC, detail_outside.sa_score,
                   s=12, c="#BFC6CC", alpha=0.55, linewidths=0, zorder=1)
    # Opaque screening-region tint masks only the edge of a gray marker whose
    # center lies just above 20; it does not remove or relocate that molecule.
    detail.axvspan(0, MAX_AUREUS_MIC, color="#E8F2F0", alpha=1, zorder=1.5)
    detail.scatter(selected.aureus_MIC, selected.sa_score,
                   s=14, c=selected.total_score, cmap=cmap, norm=norm,
                   alpha=0.75, linewidths=0, zorder=2)
    detail.scatter(stars.aureus_MIC, stars.sa_score,
                   s=50, marker="*", c=stars.total_score, cmap=cmap, norm=norm,
                   edgecolors="#202020", linewidths=0.35, zorder=3)
    detail.scatter([compound_336.aureus_MIC], [compound_336.sa_score],
                   s=110, marker="D", c="#D94A38", edgecolors="black",
                   linewidths=1, zorder=4)
    detail.axvline(MAX_AUREUS_MIC, color="#555555", linestyle="--",
                   linewidth=1, zorder=5)
    detail.set(xlim=(0, detail_max), ylim=(sa_axis_max, 0),
               xticks=np.linspace(0, detail_max, 5),
               xlabel="Predicted MIC (µg/mL)", ylabel="SA score")
    detail.set_title(f"MIC screen detail (0–{detail_max:g} µg/mL)",
                     fontsize=9, pad=8)
    detail.tick_params(labelsize=8)
    detail.grid(alpha=0.2, linewidth=0.5)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", color="none",
               markerfacecolor="#BFC6CC", markersize=6,
               label=f"Outside MIC screen (n = {len(outside)} shown)"),
        Line2D([0], [0], marker="o", linestyle="", color="none",
               markerfacecolor=cmap(0.5), markersize=6,
               label=f"Passed MIC screen (n = {len(selected)})"),
        Line2D([0], [0], marker="*", linestyle="", color="none",
               markerfacecolor=cmap(0.25), markeredgecolor="black", markersize=9,
               label="Top 100 by composite rank"),
        Line2D([0], [0], marker="D", linestyle="", color="none",
               markerfacecolor="#D94A38", markeredgecolor="black", markersize=8,
               label=f"Compound 336 (rank {int(compound_336['rank'])}/{len(selected)})"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.04, 0.97),
               frameon=True, fontsize=8)
    colorbar_axis = fig.add_axes([0.94, 0.26, 0.018, 0.48])
    colorbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                            cax=colorbar_axis)
    colorbar.set_label("Composite score (lower is better)")
    require_matplotlib_panel_alignment(fig, ax, detail, colorbar_axis)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mol-processed",
                        default="molecules/molecules_025/molecules.csv")
    parser.add_argument("--mol-filtered",
                        default="molecules/molecules_025/filtered.csv")
    parser.add_argument("--mol-validated",
                        default="molecules/result/validated_molecules.csv")
    args = parser.parse_args()
    data = prepare_data(Path(args.mol_processed), Path(args.mol_filtered),
                        Path(args.mol_validated))
    fig = draw(data)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(OUT_SVG, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print({
        "total_molecules": len(data),
        "passed_mic_screen": int(data.is_candidate.sum()),
        "outside_mic_screen": int((~data.is_candidate).sum()),
        "overview_points": int(data.aureus_MIC.le(MIC_OVERVIEW_MAX).sum()),
        "overview_off_scale": int(data.aureus_MIC.gt(MIC_OVERVIEW_MAX).sum()),
        "top_100_candidates": int(data.is_top100.sum()),
        "candidate_stars": int(data.is_top100.sum()),
        "compound_336_diamonds": int(data.is_336.sum()),
        "compound_336_rank": int(data.loc[data.is_336, "rank"].iloc[0]),
        "out_png": str(OUT_PNG), "out_svg": str(OUT_SVG),
    })


if __name__ == "__main__":
    main()
