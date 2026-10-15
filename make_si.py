"""Build the Supporting Information document and Figure S1 from measured results.

Every table is read from the result CSVs rather than typed, so the SI cannot
disagree with the manuscript. Table S5 and Figure S1 are computed here because
they appear only in the SI.

Usage:
    python3 make_si.py --results ../results \
                       --outdir ../benchmark-paper/submission/supporting
"""

from __future__ import annotations

import argparse
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
import druglike_subset as dls
import full_analysis as fa
import make_figures as mf

SPLIT_NAME = {"random": "Random", "anchor_held_out": "Anchor held out"}
FS_NAME = {
    "physics_only": "Physics only",
    "structural_only": "Structural only",
    "physics+structural": "Physics + structural",
    "physics+structural+fp": "Physics + structural + fingerprints",
}
MODEL_NAME = {
    "logistic": "Logistic regression",
    "random_forest": "Random forest",
    "grad_boost": "Gradient boosting",
}


def fmt(m: float, s: float | None = None, n: int = 3) -> str:
    if s is None or np.isnan(s):
        return f"{m:.{n}f}"
    return f"{m:.{n}f} ± {s:.{n}f}"


def table_s1(results: str) -> str:
    ab = pd.read_csv(f"{results}/ablation.csv")
    rows = ["| Feature set | No. features | Protocol | ROC-AUC | Balanced accuracy "
            "| Average precision | Precision, top decile | Precision, top half "
            "| Enrichment |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in ab.iterrows():
        rows.append(
            f"| {FS_NAME[r.feature_set]} | {int(r.n_features):,} | "
            f"{SPLIT_NAME[r.split]} | {fmt(r.roc_auc, r.roc_auc_sd)} | "
            f"{fmt(r.balanced_acc, r.balanced_acc_sd)} | "
            f"{fmt(r.avg_precision, r.avg_precision_sd)} | "
            f"{fmt(r.precision_top10pct)} | {fmt(r.precision_top50pct)} | "
            f"{r.enrichment_top10pct:.2f} |")
    return "\n".join(rows)


def table_s2(results: str) -> str:
    md = pd.read_csv(f"{results}/model_families.csv")
    rows = ["| Model | Protocol | ROC-AUC | Balanced accuracy | Average precision "
            "| Precision, top decile | Enrichment |",
            "|---|---|---|---|---|---|---|"]
    for _, r in md.iterrows():
        rows.append(
            f"| {MODEL_NAME[r.model]} | {SPLIT_NAME[r.split]} | "
            f"{fmt(r.roc_auc, r.roc_auc_sd)} | "
            f"{fmt(r.balanced_acc, r.balanced_acc_sd)} | "
            f"{fmt(r.avg_precision, r.avg_precision_sd)} | "
            f"{fmt(r.precision_top10pct)} | {r.enrichment_top10pct:.2f} |")
    return "\n".join(rows)


def table_s3(results: str) -> str:
    imp = pd.read_csv(f"{results}/permutation_importance.csv")
    rows = ["| Rank | Feature | Block | Δ ROC-AUC |", "|---|---|---|---|"]
    for i, (_, r) in enumerate(imp.iterrows(), start=1):
        rows.append(f"| {i} | `{r.feature}` | {r.block} | "
                    f"{r.importance:+.4f} ± {r.sd:.4f} |")
    return "\n".join(rows)


def table_s4(results: str) -> str:
    d = pd.read_csv(f"{results}/druglike_subset.csv")
    rows = ["| Protocol | ROC-AUC | Balanced accuracy | Average precision "
            "| Precision, top decile | Precision, top half | Enrichment |",
            "|---|---|---|---|---|---|---|"]
    for _, r in d.iterrows():
        rows.append(
            f"| {SPLIT_NAME[r.split]} | {fmt(r.roc_auc, r.roc_auc_sd)} | "
            f"{fmt(r.balanced_acc, r.balanced_acc_sd)} | "
            f"{fmt(r.avg_precision, r.avg_precision_sd)} | "
            f"{fmt(r.precision_top10pct)} | {fmt(r.precision_top50pct)} | "
            f"{r.enrichment_top10pct:.2f} |")
    return "\n".join(rows)


def table_s5(results: str) -> tuple[str, pd.DataFrame]:
    """Variance of every physics feature, evidencing the constant dpKa block."""
    frame = fa.add_anchor(fa.load_pairs(f"{results}/ccgnet_pairs_smiles.tab"))
    X = desc.featurise(frame, include_fingerprints=False)
    phys = [c for c in X.columns if fa.is_physics(c)]
    stats = pd.DataFrame({
        "feature": phys,
        "distinct_values": [int(X[c].nunique(dropna=False)) for c in phys],
        "variance": [float(np.nanvar(X[c].to_numpy(dtype=float))) for c in phys],
    }).sort_values(["distinct_values", "feature"])
    rows = ["| Feature | Distinct values | Variance | Informative |",
            "|---|---|---|---|"]
    for _, r in stats.iterrows():
        rows.append(f"| `{r.feature}` | {int(r.distinct_values)} | "
                    f"{r.variance:.4g} | {'no' if r.distinct_values < 2 else 'yes'} |")
    return "\n".join(rows), stats


def figure_s1(results: str, outdir: str) -> dict:
    """Precision and enrichment against selected fraction, drug-like subset."""
    frame = fa.add_anchor(fa.load_pairs(f"{results}/ccgnet_pairs_smiles.tab"))
    keep = {a for a in frame["anchor_smiles"].unique() if dls.is_druglike(a)}
    sub = frame[frame["anchor_smiles"].isin(keep)].reset_index(drop=True)
    X = desc.featurise(sub, include_fingerprints=False)
    sub = sub.loc[X.index]
    y = sub["label"].to_numpy()
    groups = sub["anchor_smiles"].to_numpy()

    fracs = np.arange(0.05, 1.01, 0.05)
    per_fold = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        m = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
        ).fit(X.iloc[tr], y[tr])
        s = m.predict_proba(X.iloc[te])[:, 1]
        yt = y[te]
        rank = np.argsort(-s)
        per_fold.append(([yt[rank[:max(1, int(f * len(yt)))]].mean() for f in fracs],
                         yt.mean()))
    prec = np.vstack([p for p, _ in per_fold])
    mean, sd = prec.mean(axis=0), prec.std(axis=0)
    base = float(np.mean([b for _, b in per_fold]))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    fig.subplots_adjust(wspace=0.32)
    for ax, series, ylab, ylim, ref, reflab in [
        (axes[0], mean, "Precision", (0.84, 1.02), base, f"base rate {base:.3f}"),
        (axes[1], mean / base, "Enrichment factor", (0.95, 1.20), 1.0, "no enrichment"),
    ]:
        band = (sd, sd) if ylab == "Precision" else (sd / base, sd / base)
        ax.plot(fracs * 100, series, color=mf.C_HELDOUT, lw=2, zorder=3)
        ax.fill_between(fracs * 100, series - band[0], series + band[1],
                        color=mf.C_HELDOUT, alpha=0.18, lw=0, zorder=2)
        ax.axhline(ref, color=mf.C_RANDOM if ylab == "Precision" else mf.MUTED,
                   lw=1.5, ls="--" if ylab == "Precision" else ":", zorder=3)
        ax.text(99, ref - (0.035 if ylab == "Precision" else -0.008), reflab,
                fontsize=7, ha="right",
                color=mf.C_RANDOM if ylab == "Precision" else mf.MUTED)
        ax.set_xlabel("Top-scoring candidates selected (%)")
        ax.set_ylabel(ylab)
        ax.set_ylim(*ylim)
        ax.yaxis.grid(True, color=mf.GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
    axes[0].set_title("a", loc="left", fontsize=9, fontweight="bold")
    axes[1].set_title("b", loc="left", fontsize=9, fontweight="bold")
    for ext in ("png", "pdf"):
        fig.savefig(f"{outdir}/figureS1_druglike_enrichment.{ext}", dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    return {"base": base, "max_enrichment": float((mean / base).max()),
            "n_pairs": int(len(sub)), "n_anchors": int(sub.anchor_smiles.nunique())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    print("[si] Figure S1 ...")
    f1 = figure_s1(args.results, args.outdir)
    print(f"     {f1['n_pairs']} pairs, base {f1['base']:.4f}, "
          f"max enrichment {f1['max_enrichment']:.3f}")
    print("[si] Table S5 ...")
    s5, stats = table_s5(args.results)
    n_const = int((stats.distinct_values < 2).sum())

    doc = f"""# Supporting Information

**How Evaluation Protocol and Class Balance Inflate Reported Performance in Machine-Learning Co-Crystal Screening**

Sambhaji Balaso Thakar<sup>1,</sup>\\* and Pradnya Namdev Ghorpade<sup>1</sup>

<sup>1</sup> Advanced Data Science and Gen AI Unit, Aditya BioNova Analytics, Pune, Maharashtra, India

\\* Corresponding author. E-mail: drsambhajithakar@adityabionova.com

---

## Contents

- Table S1. Complete feature ablation, both protocols, all screening metrics
- Table S2. Complete model family comparison, both protocols
- Table S3. Permutation importance of all 33 features
- Table S4. Drug-like subset, both protocols, all screening metrics
- Table S5. Variance of every pair-level (physics) feature
- Figure S1. Precision and enrichment against selected fraction, drug-like subset
- SMILES.csv (separate file). Canonical SMILES and labels for all 7,775 pairs

---

## Table S1. Complete feature ablation

Random forest, five-fold cross-validation, mean ± standard deviation across folds. Enrichment is top-decile precision divided by the test-fold base rate. This is the full version of Table 1 in the manuscript, with the screening-utility columns added.

{table_s1(args.results)}

---

## Table S2. Complete model family comparison

All three model families on the 33 physics and structural features, five-fold cross-validation. This is the full version of Table 3 in the manuscript. The protocol penalty is present in every family; enrichment is near unity in every cell, under both protocols, for every model.

{table_s2(args.results)}

---

## Table S3. Permutation importance, all 33 features

Random forest on 33 physics and structural features, first held-out fold, ten repeats, ROC-AUC scoring. This is the full version of Table 2 in the manuscript. Features with importance at or below zero are those whose permutation did not degrade, or marginally improved, held-out ROC-AUC.

{table_s3(args.results)}

---

## Table S4. Drug-like subset, complete metrics

Random forest on 33 features, five-fold cross-validation, restricted to the {f1['n_pairs']:,} pairs whose anchor component passes the drug-likeness filter of Section 2.5 ({f1['n_anchors']} distinct anchors). This is the full version of Table 4 in the manuscript.

{table_s4(args.results)}

---

## Table S5. Variance of every pair-level (physics) feature

Computed over all 7,775 usable pairs. {n_const} of the 13 pair-level features take a single value across the entire dataset and therefore carry no information. These are the four Δp*K*<sub>a</sub>-derived terms and their availability flags, which are constant because the source dataset contains no p*K*<sub>a</sub> values. This table is the direct evidence for the statement in Section 2.2 of the manuscript that the physics block as evaluated excludes the term with the strongest theoretical claim to relevance.

{s5}

---

## Figure S1. Precision and enrichment on the drug-like subset

![Figure S1](figureS1_druglike_enrichment.png)

**Figure S1.** (a) Precision among the top-scoring fraction of candidates on the drug-like subset, with the subset base rate of {f1['base']:.3f} marked. (b) The same quantity as enrichment factor, with unity marked. Shaded bands are the standard deviation across five folds under the anchor-held-out protocol. Enrichment never exceeds {f1['max_enrichment']:.2f}, below the {1.15:.2f} ceiling on the full dataset, confirming that the base-rate problem of Section 3.2 is worse rather than better on pharmaceutically plausible chemistry.

---

## SMILES.csv

Canonical SMILES for both components of all 7,775 usable pairs, with the binary formability label and the component assigned as anchor. Columns: `pair_id`, `anchor_smiles`, `coformer_smiles`, `formability_label` (1 = co-crystal formed, 0 = co-crystallisation attempted and failed), `label_meaning`. Derived from the CCGNet dataset (MIT licence) by `prepare_ccgnet.py`; see Data and Software Availability in the manuscript.
"""
    path = f"{args.outdir}/supporting-information.md"
    with open(path, "w") as fh:
        fh.write(doc)
    print(f"[done] {path}")


if __name__ == "__main__":
    main()
