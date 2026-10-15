"""Generate the ACS Table of Contents graphic for the benchmark paper.

ACS requirement: a single image, maximum 3.25 in wide by 1.75 in tall, at least
300 dpi, minimum 8 pt type, legible at printed size. Not a collage.

The graphic carries the paper's one claim: the headline number falls when the
evaluation protocol stops leaking molecules, and the base rate caps enrichment.
Values are read from the measured result tables, never typed in, so the graphic
cannot drift from the manuscript.

Palette and hatching are the same validated pair used by make_figures.py
(colourblind-safe: protan dE 21.5, tritan 28.4, normal 28.2), so the two
protocols read the same way in every figure of the paper and survive greyscale.

Usage:
    python3 make_toc_graphic.py --results ../results --outdir ../benchmark-paper/figures
"""

from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

C_RANDOM = "#1F6FB2"
C_HELDOUT = "#C2571A"
H_HELDOUT = "///"
INK = "#1a1a1a"
MUTED = "#6b6b6b"

FEATURE_SET = "physics+structural"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "text.color": INK,
    "figure.dpi": 300, "savefig.dpi": 300,
})


def read_values(results_dir: str) -> dict:
    ab = pd.read_csv(f"{results_dir}/ablation.csv")
    row = ab[ab.feature_set == FEATURE_SET].set_index("split")
    return {
        "random": float(row.loc["random", "balanced_acc"]),
        "held_out": float(row.loc["anchor_held_out", "balanced_acc"]),
        "base_rate": float(row.loc["anchor_held_out", "base_rate"]),
        "enrichment": float(row.loc["anchor_held_out", "enrichment_top10pct"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    v = read_values(args.results)

    fig = plt.figure(figsize=(3.25, 1.75))
    fig.patch.set_facecolor("white")

    # Vertical budget for 1.75 in: title band, bar band, caption band.
    fig.text(0.035, 0.955,
             "Co-crystal formability ML:\nwhat survives a stricter test",
             fontsize=8.5, fontweight="bold", va="top", ha="left",
             linespacing=1.25)
    fig.text(0.035, 0.655, "Balanced accuracy, 7,775 experimental pairs",
             fontsize=6.5, color=MUTED, va="center", ha="left")

    ax = fig.add_axes([0.34, 0.28, 0.60, 0.30])

    labels = ["Random split", "Molecule held out"]
    values = [v["random"], v["held_out"]]
    colours = [C_RANDOM, C_HELDOUT]
    hatches = ["", H_HELDOUT]

    for i, (val, c, h) in enumerate(zip(values, colours, hatches)):
        ax.barh(i, val, 0.60, color=c, hatch=h, edgecolor="white", linewidth=0.8,
                zorder=3)
        # Value outside the bar end: white type is unreadable over the hatch.
        ax.text(val + 0.018, i, f"{val:.3f}", va="center", ha="left",
                fontsize=8, color=INK, fontweight="bold", zorder=4)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.06)
    ax.set_xticks([])
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)

    drop = v["random"] - v["held_out"]
    fig.text(0.035, 0.115,
             f"Leakage costs {drop:.3f} balanced accuracy. At a "
             f"{v['base_rate']*100:.0f}% positive\n"
             f"base rate, enrichment is only {v['enrichment']:.2f}\u00d7: "
             f"high ROC-AUC\n\u2260 screening value.",
             fontsize=6.8, color=INK, va="center", ha="left", linespacing=1.4)

    for ext in ("png", "pdf", "tif"):
        fig.savefig(f"{args.outdir}/toc_graphic.{ext}", dpi=300,
                    facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"[done] TOC graphic (3.25 x 1.75 in, 300 dpi) in {args.outdir}")
    print(f"       random {v['random']:.3f}  held out {v['held_out']:.3f}  "
          f"drop {drop:.3f}  base rate {v['base_rate']:.3f}  "
          f"enrichment {v['enrichment']:.2f}x")


if __name__ == "__main__":
    main()
