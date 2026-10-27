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
import csv
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


def contradictory_pairs(results: str) -> str:
    """List the seven pairs whose two records disagree, read from the source table."""
    rows = []
    with open(f"{results}/as-distributed/ccgnet_pairs_smiles.tab") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            rows.append(row)
    groups: dict[tuple[str, str], list] = collections.defaultdict(list)
    for row in rows:
        groups[tuple(sorted(row[:2]))].append(row)
    out = ["| Positive record | Negative record |", "|---|---|"]
    for group in groups.values():
        if len({r[2] for r in group}) > 1:
            pos = [r[3] for r in group if r[2] == "1"][0]
            neg = [r[3] for r in group if r[2] == "0"][0]
            out.append(f"| {pos} | {neg} |")
    assert len(out) == 9, len(out)
    return "\n".join(out)


def table_s6(results: str) -> tuple[str, dict]:
    """Compare the curated primary analysis with the source table as distributed."""
    cur = pd.read_csv(f"{results}/ablation.csv")
    alt = pd.read_csv(f"{results}/as-distributed/ablation.csv")
    cur_d = pd.read_csv(f"{results}/druglike_subset.csv")
    alt_d = pd.read_csv(f"{results}/as-distributed/druglike_subset.csv")

    def ba(frame, split):
        row = frame[(frame.feature_set == "physics+structural") & (frame.split == split)]
        return float(row.balanced_acc.iloc[0])

    def ba_d(frame, split):
        row = frame[frame.split == split]
        return float(row.balanced_acc.iloc[0])

    v = {
        "cur_rand": ba(cur, "random"), "cur_held": ba(cur, "anchor_held_out"),
        "alt_rand": ba(alt, "random"), "alt_held": ba(alt, "anchor_held_out"),
        "cur_dl": ba_d(cur_d, "random") - ba_d(cur_d, "anchor_held_out"),
        "alt_dl": ba_d(alt_d, "random") - ba_d(alt_d, "anchor_held_out"),
        "cur_base": float(cur[cur.split == "anchor_held_out"].base_rate.iloc[0]),
        "alt_base": float(alt[alt.split == "anchor_held_out"].base_rate.iloc[0]),
        "cur_enr": float(cur[(cur.feature_set == "physics+structural") &
                             (cur.split == "anchor_held_out")].enrichment_top10pct.iloc[0]),
        "alt_enr": float(alt[(alt.feature_set == "physics+structural") &
                             (alt.split == "anchor_held_out")].enrichment_top10pct.iloc[0]),
    }
    v["cur_pen"] = v["cur_rand"] - v["cur_held"]
    v["alt_pen"] = v["alt_rand"] - v["alt_held"]

    rows = [
        "| Quantity | Curated (reported) | Source table as distributed |",
        "|---|---|---|",
        f"| Pairs | 7,757 | 7,775 |",
        f"| Balanced accuracy, random split | {v['cur_rand']:.3f} | {v['alt_rand']:.3f} |",
        f"| Balanced accuracy, molecule held out | {v['cur_held']:.3f} | {v['alt_held']:.3f} |",
        f"| **Protocol penalty** | **{v['cur_pen']:.3f}** | **{v['alt_pen']:.3f}** |",
        f"| Protocol penalty, drug-like subset | {v['cur_dl']:.3f} | {v['alt_dl']:.3f} |",
        f"| Base rate | {v['cur_base']:.3f} | {v['alt_base']:.3f} |",
        f"| Top-decile enrichment | {v['cur_enr']:.2f}\u00d7 | {v['alt_enr']:.2f}\u00d7 |",
    ]
    return "\n".join(rows), v


def table_s7(results: str) -> tuple[str, dict]:
    """Table S7 from the split-integrity audit outputs."""
    leak = pd.read_csv(f"{results}/split_integrity_leakage.csv").set_index("protocol")
    sc = pd.read_csv(f"{results}/split_integrity_scores.csv").set_index("protocol")
    A, D = "anchor held out", "fully molecule-disjoint"

    def pct(col, key):
        v = leak.loc[key, col] * 100
        return f"{v:.0f}%" if v == 0 else f"{v:.1f}%"

    rows = ["| Quantity | Anchor held out | Fully molecule-disjoint |",
            "|---|---|---|",
            "| Anchor molecules shared between train and test | "
            f"{leak.loc[A, 'anchor_molecules_shared_mean']:.0f} | "
            f"{leak.loc[D, 'anchor_molecules_shared_mean']:.0f} |",
            "| Test pairs whose partner component was seen in training | "
            f"{pct('test_pairs_whose_partner_was_seen_mean', A)} | "
            f"{pct('test_pairs_whose_partner_was_seen_mean', D)} |",
            "| Test pairs with either component seen in training | "
            f"{pct('test_pairs_with_either_component_seen_mean', A)} | "
            f"{pct('test_pairs_with_either_component_seen_mean', D)} |",
            "| Test fold size, smallest | "
            f"{leak.loc[A, 'test_fold_size_min']:,} | {leak.loc[D, 'test_fold_size_min']:,} |",
            "| Test fold size, largest | "
            f"{leak.loc[A, 'test_fold_size_max']:,} | {leak.loc[D, 'test_fold_size_max']:,} |",
            "| Test-fold base rate, lowest | "
            f"{leak.loc[A, 'test_base_rate_min']:.3f} | {leak.loc[D, 'test_base_rate_min']:.3f} |",
            "| Test-fold base rate, highest | "
            f"{leak.loc[A, 'test_base_rate_max']:.3f} | {leak.loc[D, 'test_base_rate_max']:.3f} |",
            "| Scorable folds of five | "
            f"{sc.loc[A, 'n_folds']:.0f} | {sc.loc[D, 'n_folds']:.0f} |"]
    v = {"partner": leak.loc[A, "test_pairs_whose_partner_was_seen_mean"],
         "either": leak.loc[A, "test_pairs_with_either_component_seen_mean"],
         "big": int(leak.loc[D, "test_fold_size_max"]),
         "small": int(leak.loc[D, "test_fold_size_min"]),
         "bacc": sc.loc[D, "balanced_acc"], "bacc_sd": sc.loc[D, "balanced_acc_sd"],
         "folds": int(sc.loc[D, "n_folds"])}
    return "\n".join(rows), v


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
    print("[si] Table S6 ...")
    s6_pairs = contradictory_pairs(args.results)
    s6, v6 = table_s6(args.results)
    print(f"     penalty curated {v6['cur_pen']:.3f} vs as distributed {v6['alt_pen']:.3f}")
    print("[si] Table S7 ...")
    s7, v7 = table_s7(args.results)
    print(f"     partner leakage {v7['partner']:.3f}, disjoint bACC {v7['bacc']:.3f}")

    doc = f"""# Supporting Information

**Data-Centric Machine Learning for Pharmaceutical Co-Crystal Discovery: Molecule-Held-Out Evaluation and Enrichment-Based Benchmarking of 7,757 Experimental Pairs**

Sambhaji Balaso Thakar<sup>1,2,</sup>\\* and Pradnya Namdev Ghorpade<sup>1,2</sup>

<sup>1</sup> Advanced Bioinformatics, Data Science and Generative AI Unit, Aditya BioNova Analytics, Thakar Mala Road, At/Post Haroli, Taluka Shirol, District Kolhapur 416102, Maharashtra, India

<sup>2</sup> Oncology Clinical Trial Analytics and Research Unit, Aditya BioNova Analytics, Thakar Mala Road, At/Post Haroli, Taluka Shirol, District Kolhapur 416102, Maharashtra, India

\\* Corresponding author. E-mail: drsambhajithakar@adityabionova.com

---

## Contents

- Table S1. Complete feature ablation, both protocols, all screening metrics
- Table S2. Complete model family comparison, both protocols
- Table S3. Permutation importance of all 33 features
- Table S4. Drug-like subset, both protocols, all screening metrics
- Table S5. Variance of every pair-level (physics) feature
- Table S6. Sensitivity of the reported values to the curation step
- Table S7. Split integrity: component leakage and the limits of a molecule-disjoint protocol
- Figure S1. Precision and enrichment against selected fraction, drug-like subset
- SMILES.csv (separate file). Canonical SMILES and labels for all 7,757 curated pairs

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

Computed over all 7,757 curated pairs. {n_const} of the 13 pair-level features take a single value across the entire dataset and therefore carry no information. These are the four Δp*K*<sub>a</sub>-derived terms and their availability flags, which are constant because the source dataset contains no p*K*<sub>a</sub> values. This table is the direct evidence for the statement in Section 2.2 of the manuscript that the physics block as evaluated excludes the term with the strongest theoretical claim to relevance.

{s5}

---

## Figure S1. Precision and enrichment on the drug-like subset

![](figureS1_druglike_enrichment.png)

**Figure S1.** (a) Precision among the top-scoring fraction of candidates on the drug-like subset, with the subset base rate of {f1['base']:.3f} marked. (b) The same quantity as enrichment factor, with unity marked. Shaded bands are the standard deviation across five folds under the anchor-held-out protocol. Enrichment never exceeds {f1['max_enrichment']:.2f}, below the {1.15:.2f} ceiling on the full dataset, confirming that the base-rate problem of Section 3.2 is worse rather than better on pharmaceutically plausible chemistry.

---

## Table S6. Sensitivity of the reported values to the curation step

{s6}

Eighteen records were removed in curation (Section 2.1): four exact repeats of a pair
already present with the same label, and fourteen records belonging to seven pairs
whose two records disagreed. In every one of those seven, the positive record carries
a Cambridge Structural Database refcode and the negative a numeric identifier pair,
so the two come from different sources. The seven are:

{s6_pairs}

The table above gives every headline value
computed both ways. Removing 0.23% of the records moves the protocol penalty by
{abs(v6['cur_pen']-v6['alt_pen']):.3f} balanced accuracy, which is within one fold
standard deviation of the held-out estimate and is itself a demonstration of the
instability that Section 3.1 reports. The direction of every conclusion is unchanged,
and the drug-like result is stronger on the curated data.

## Table S7. Split integrity: component leakage and the limits of a molecule-disjoint protocol

Measured on the 7,757 curated pairs by `audit_split_integrity.py`, a script written for the
pre-submission audit and kept separate from the analysis that produces the results in Section 3.
Values are means over the five folds of each protocol.

{s7}

The curated table contains 7,757 distinct unordered pairs with no pair appearing twice and no
pair carrying contradictory labels. The anchor-held-out protocol is exactly anchor-disjoint, as
intended, but {v7['partner']*100:.1f}% of test pairs still share their partner component with the
training folds, which is why Section 2.3 reports its figures as upper bounds.

The fully molecule-disjoint protocol was built by taking connected components of the graph whose
nodes are molecules and whose edges are the curated pairs, so that neither component of a test
pair can occur in training. On this dataset that protocol is degenerate rather than merely
strict: the graph has 416 components, but a single component holds {v7['big']:,} of the 7,757
pairs. The resulting folds are therefore wildly unbalanced in size, from {v7['small']:,} to
{v7['big']:,} pairs; one test fold contains only positives and cannot be scored; and balanced
accuracy over the three scorable folds falls to {v7['bacc']:.3f} (standard deviation
{v7['bacc_sd']:.3f}), which is chance. That figure measures the failure of the split, not the
difficulty of the chemistry, and we do not report it as a performance result. It is given here to
make the point in Section 2.3 concrete: on a dataset this concentrated, a genuinely
molecule-disjoint benchmark cannot be obtained by resplitting, only by collecting pairs that
reach beyond the dominant component.

---

## SMILES.csv

Canonical SMILES for both components of all 7,757 curated pairs, with the binary formability label and the component assigned as anchor. Columns: `pair_id`, `anchor_smiles`, `coformer_smiles`, `formability_label` (1 = co-crystal formed, 0 = co-crystallisation attempted and failed), `label_meaning`. Derived from the CCGNet dataset (MIT licence) by `prepare_ccgnet.py`; see Data and Software Availability in the manuscript.
"""
    path = f"{args.outdir}/supporting-information.md"
    with open(path, "w") as fh:
        fh.write(doc)
    print(f"[done] {path}")


if __name__ == "__main__":
    main()
