#!/usr/bin/env python3
"""Create the Delta F1 histogram used in the robustness analysis.

The input is the iteration-level output from the split-perturbation script.
For each split, the script calculates:

    Delta F1 = F1_hybrid - F1_baseline

Positive values therefore mean that the human-guided hybrid model performed
better than the rule-trained XGBoost baseline on the same held-out human cases.
"""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Inputs and outputs
ROOT = Path(__file__).resolve().parent
INPUT_CSV = ROOT / "Data" / "derived" / "ai_counterattack_detection" / "human_resampling_robustness" / "resampling_iteration_metrics.csv"
OUT_DIR = ROOT / "figures"
OUT_PNG = OUT_DIR / "hybrid_baseline_delta_f1_histogram.png"
OUT_PDF = OUT_DIR / "hybrid_baseline_delta_f1_histogram.pdf"


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_CSV}")

    # Convert the long iteration table into one row per split, then calculate
    # the paired F1 difference inside each split.
    df = pd.read_csv(INPUT_CSV)
    wide = df.pivot(index="iteration", columns="model", values="f1")
    delta_f1 = wide["hybrid_xgboost"] - wide["xgboost_baseline"]

    mean_delta = delta_f1.mean()
    median_delta = delta_f1.median()
    improved_share = (delta_f1 > 0).mean()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Keep the figure simple enough for the report: the dashed zero line shows
    # no model difference, and the solid line shows the observed mean gain.
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=180)
    ax.hist(
        delta_f1,
        bins=16,
        color="#2F6F73",
        edgecolor="white",
        linewidth=0.8,
        alpha=0.92,
    )
    ax.axvline(0, color="#9C3D3D", linestyle="--", linewidth=1.4, alpha=0.9)
    ax.axvline(mean_delta, color="#1A1A1A", linestyle="-", linewidth=1.5, alpha=0.95)

    ax.set_title("Paired F1 differences across 100 split perturbations", pad=12)
    ax.set_xlabel(r"$\Delta$F1 = F1$_{hybrid}$ - F1$_{baseline}$")
    ax.set_ylabel("Number of splits")
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)

    ylim_top = ax.get_ylim()[1]
    ax.text(
        0,
        ylim_top * 0.96,
        "No difference",
        ha="center",
        va="top",
        color="#9C3D3D",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.92},
    )
    ax.text(
        mean_delta,
        ylim_top * 0.86,
        f"Mean {mean_delta:+.3f}",
        ha="center",
        va="top",
        color="#1A1A1A",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5, "alpha": 0.92},
    )

    # Summary box used as an immediate visual interpretation of the histogram.
    annotation = (
        f"Median: {median_delta:+.3f}\n"
        f"Hybrid better: {improved_share:.0%}"
    )
    ax.text(
        0.98,
        0.82,
        annotation,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#D0D0D0"},
    )

    fig.tight_layout()
    # PNG is used in the report; PDF is kept for high-resolution export.
    fig.savefig(OUT_PNG, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PDF}")
    print(f"Mean delta F1: {mean_delta:+.3f}")
    print(f"Median delta F1: {median_delta:+.3f}")
    print(f"Hybrid better share: {improved_share:.0%}")


if __name__ == "__main__":
    main()
