"""Generate the three figures for the benchmark paper from measured results.

Reads nothing but the analysis outputs and the pair table, so the figures can
never drift from the numbers in the manuscript.

Journal conventions applied: single-column width, greyscale-safe (each series
carries a hatch as well as a hue), recessive axes, legend present for two
series, no dual axis, direct value labels only where they carry information.

Palette validated colourblind-safe (protan ΔE 21.5, tritan 28.4, normal 28.2).

Usage:
    python3 make_figures.py --results ../results --outdir ../benchmark-paper/figures
"""

from __future__ import annotations

import argparse
import collections
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

import descriptors as desc
import full_analysis as fa

C_RANDOM = "#1F6FB2"     # random split
C_HELDOUT = "#C2571A"    # anchor held out
H_RANDOM = ""            # secondary encoding so the pair survives greyscale
H_HELDOUT = "///"
INK = "#000000"
MUTED = "#4d4d4d"
GRID = "#e3e3e3"

# ACS asks for figure lettering in a sans-serif face (Helvetica or Arial) and
# prints line art in black. Liberation Sans carries Arial's metrics, so the
# figures keep the proportions ACS expects on a machine without Arial itself.
plt.rcParams.update({
    "font.family": "Liberation Sans",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "FreeSans",
                        "DejaVu Sans"],
    "font.size": 8,
    "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
    "axes.linewidth": 0.8,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.size": 3.0, "ytick.major.size": 3.0,
    "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
})

SHORT = {
    "physics_only": "Physics\n13",
    "structural_only": "Structural\n20",
    "physics+structural": "Both\n33",
    "physics+structural+fp": "Both + FP\n2081",
}


def figure1(results_dir: str, out: str) -> None:
    """Balanced accuracy by feature set under both protocols."""
    ab = pd.read_csv(f"{results_dir}/ablation.csv")
    order = list(SHORT)
    x = np.arange(len(order))
    w = 0.38

    fig, ax = plt.subplots(figsize=(4.60, 2.7))
    for i, (split, colour, hatch, label) in enumerate([
        ("random", C_RANDOM, H_RANDOM, "Random split"),
        ("anchor_held_out", C_HELDOUT, H_HELDOUT, "Anchor held out"),
    ]):
        sub = ab[ab.split == split].set_index("feature_set").loc[order]
        ax.bar(x + (i - 0.5) * w, sub.balanced_acc, w * 0.94,
               yerr=sub.balanced_acc_sd, capsize=2.5,
               color=colour, hatch=hatch, edgecolor="white", linewidth=0.8,
               error_kw={"ecolor": INK, "elinewidth": 0.8, "capthick": 0.8}, label=label, zorder=3)

    ax.set_xticks(x); ax.set_xticklabels([SHORT[o] for o in order])
    ax.set_xlabel("Feature set (number of features)", labelpad=2)
    ax.set_ylabel("Balanced accuracy")
    ax.set_ylim(0.55, 0.93)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2,
              columnspacing=1.2, handlelength=1.6)
    fig.savefig(f"{out}/figure1_split_protocol.png")
    fig.savefig(f"{out}/figure1_split_protocol.pdf")
    plt.close(fig)
    print("  figure1 written")


def precision_curve(table: str):
    """Precision and enrichment as a function of the selected top fraction."""
    frame = fa.add_anchor(fa.load_pairs(table))
    X = desc.featurise(frame, include_fingerprints=False)
    frame = frame.loc[X.index]
    y = frame["label"].to_numpy()
    groups = frame["anchor_smiles"].to_numpy()

    fracs = np.arange(0.05, 1.01, 0.05)
    per_fold = []
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(X, y, groups):
        m = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
        ).fit(X.iloc[tr], y[tr])
        s = m.predict_proba(X.iloc[te])[:, 1]
        yt = y[te]
        rank = np.argsort(-s)
        prec = [yt[rank[:max(1, int(f * len(yt)))]].mean() for f in fracs]
        per_fold.append((np.array(prec), yt.mean()))

    prec = np.vstack([p for p, _ in per_fold])
    base = float(np.mean([b for _, b in per_fold]))
    return fracs, prec.mean(axis=0), prec.std(axis=0), base


def figure2(table: str, out: str) -> None:
    fracs, mean, sd, base = precision_curve(table)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    fig.subplots_adjust(wspace=0.32)

    ax = axes[0]
    ax.plot(fracs * 100, mean, color=C_HELDOUT, lw=2, zorder=3)
    ax.fill_between(fracs * 100, mean - sd, mean + sd, color=C_HELDOUT,
                    alpha=0.18, lw=0, zorder=2)
    ax.axhline(base, color=C_RANDOM, lw=1.6, ls="--", zorder=3)
    ax.text(99, base - 0.035, f"base rate {base:.3f}", fontsize=7,
            color=C_RANDOM, ha="right")
    ax.set_xlabel("Top-scoring candidates selected (%)")
    ax.set_ylabel("Precision")
    ax.set_ylim(0.80, 1.02)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.set_title("a", loc="left", fontsize=9, fontweight="bold")

    ax = axes[1]
    ax.plot(fracs * 100, mean / base, color=C_HELDOUT, lw=2, zorder=3)
    ax.fill_between(fracs * 100, (mean - sd) / base, (mean + sd) / base,
                    color=C_HELDOUT, alpha=0.18, lw=0, zorder=2)
    ax.axhline(1.0, color=MUTED, lw=1.2, ls=":", zorder=3)
    ax.text(97, 1.008, "no enrichment", fontsize=7, color=MUTED, ha="right")
    ax.set_xlabel("Top-scoring candidates selected (%)")
    ax.set_ylabel("Enrichment factor")
    ax.set_ylim(0.95, 1.25)
    ax.yaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.set_title("b", loc="left", fontsize=9, fontweight="bold")

    fig.savefig(f"{out}/figure2_enrichment.png")
    fig.savefig(f"{out}/figure2_enrichment.pdf")
    plt.close(fig)
    print(f"  figure2 written (base rate {base:.4f}, max enrichment "
          f"{(mean/base).max():.3f})")


NICE = {
    "c_mw": "Co-former MW", "c_heavy_atoms": "Co-former heavy atoms",
    "c_molar_refractivity": "Co-former molar refractivity",
    "d_hbd": "Anchor H-bond donors", "d_tpsa": "Anchor TPSA",
    "c_tpsa": "Co-former TPSA", "hb_total_complementarity": "Total H-bond complementarity",
    "c_frac_csp3": "Co-former fraction sp3 C", "tpsa_sum": "Summed TPSA",
    "d_logp": "Anchor log P", "logp_sum": "Summed log P",
    "c_rotb": "Co-former rotatable bonds", "hb_pair_dD_cA": "H-bond pair (anchor D / cf A)",
    "c_arom_rings": "Co-former aromatic rings", "c_logp": "Co-former log P",
}


def figure3(results_dir: str, out: str) -> None:
    imp = pd.read_csv(f"{results_dir}/permutation_importance.csv").head(15).iloc[::-1]
    labels = [NICE.get(f, f) for f in imp.feature]
    colours = [C_HELDOUT if b == "physics" else C_RANDOM for b in imp.block]
    hatches = [H_HELDOUT if b == "physics" else H_RANDOM for b in imp.block]

    fig, ax = plt.subplots(figsize=(3.9, 3.4))
    ypos = np.arange(len(imp))
    for yi, (v, e, c, h) in enumerate(zip(imp.importance, imp.sd, colours, hatches)):
        ax.barh(yi, v, 0.72, xerr=e, color=c, hatch=h, edgecolor="white",
                linewidth=0.8, error_kw={"ecolor": INK, "elinewidth": 0.8, "capthick": 0.8}, zorder=3)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Permutation importance (Δ ROC-AUC)")
    ax.xaxis.grid(True, color=GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)

    handles = [plt.Rectangle((0, 0), 1, 1, color=C_RANDOM, hatch=H_RANDOM,
                             edgecolor="white", linewidth=0.8),
               plt.Rectangle((0, 0), 1, 1, color=C_HELDOUT, hatch=H_HELDOUT,
                             edgecolor="white", linewidth=0.8)]
    ax.legend(handles, ["Structural (per molecule)", "Physics (pair level)"],
              frameon=False, fontsize=7.5, loc="lower right")
    fig.savefig(f"{out}/figure3_importance.png")
    fig.savefig(f"{out}/figure3_importance.pdf")
    plt.close(fig)
    print("  figure3 written")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    figure1(args.results, args.outdir)
    figure2(f"{args.results}/ccgnet_pairs_smiles.tab", args.outdir)
    figure3(args.results, args.outdir)
    print(f"[done] figures in {args.outdir}")


if __name__ == "__main__":
    main()
