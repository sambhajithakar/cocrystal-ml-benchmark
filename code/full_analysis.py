"""Complete analysis for the benchmark paper. Every number it prints is measured.

Runs on the SMILES pair table produced by prepare_ccgnet.py from the public
CCGNet dataset (Jiang et al., Nat. Commun. 2021, DOI 10.1038/s41467-021-26226-7,
MIT licence).

Analyses, in the order the paper reports them:
  A. Split protocol      random vs anchor-held-out, the core result
  B. Feature ablation    physics only / structural only / both / + fingerprints
  C. Model family        logistic regression, random forest, gradient boosting
  D. Screening utility   precision at top-k, enrichment factor, which is what a
                         coformer screen actually consumes
  E. Feature importance  permutation importance under the held-out split

Outputs CSVs into --outdir and prints everything.

Usage:
    python3 full_analysis.py --table pairs.tab --outdir ../results
"""

from __future__ import annotations

import argparse
import collections

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import descriptors as desc

SEED = 0
N_TREES = 300

# The physics block: pair-level terms with a thermodynamic or supramolecular
# reading, as opposed to plain per-molecule structural descriptors.
PHYSICS_PREFIXES = (
    "hb_pair", "hb_total", "tpsa_gap", "logp_gap", "mw_ratio",
    "heavy_atom_ratio", "tpsa_sum", "logp_sum", "delta_pka", "abs_delta_pka",
)


def load_pairs(path: str) -> pd.DataFrame:
    rows = []
    for line in open(path, encoding="utf-8", errors="replace"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 3 and p[2] in ("0", "1"):
            rows.append({"smiles_a": p[0], "smiles_b": p[1], "label": int(p[2])})
    return pd.DataFrame(rows)


def add_anchor(frame: pd.DataFrame) -> pd.DataFrame:
    counts = collections.Counter(frame["smiles_a"]) + collections.Counter(frame["smiles_b"])
    anchor, partner = [], []
    for a, b in zip(frame["smiles_a"], frame["smiles_b"]):
        if counts[a] >= counts[b]:
            anchor.append(a); partner.append(b)
        else:
            anchor.append(b); partner.append(a)
    out = frame.copy()
    out["anchor_smiles"] = anchor
    out["drug_smiles"] = anchor
    out["coformer_smiles"] = partner
    out["drug_pka"] = np.nan
    out["coformer_pka"] = np.nan
    return out


def is_physics(col: str) -> bool:
    return col.startswith(PHYSICS_PREFIXES)


def is_fingerprint(col: str) -> bool:
    return "_fp" in col


def make_model(kind: str):
    if kind == "logistic":
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=SEED))
    if kind == "random_forest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=N_TREES, random_state=SEED, n_jobs=-1))
    if kind == "grad_boost":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(random_state=SEED))
    raise ValueError(kind)


def precision_at_k(y_true: np.ndarray, score: np.ndarray, k: int) -> float:
    if k > len(score):
        return float("nan")
    top = np.argsort(-score)[:k]
    return float(y_true[top].mean())


def run_cv(X, y, groups, split: str, model_kind: str, folds: int = 5) -> dict:
    """One (split, model, feature-set) cell. Returns mean metrics across folds."""
    if split == "random":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
        it = splitter.split(X, y)
    else:
        n_groups = len(np.unique(groups))
        splitter = GroupKFold(n_splits=min(folds, n_groups))
        it = splitter.split(X, y, groups)

    aucs, baccs, aps, p10, p50, ef10 = [], [], [], [], [], []
    base_rate = y.mean()
    for tr, te in it:
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = make_model(model_kind).fit(X.iloc[tr], y[tr])
        s = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], s))
        baccs.append(balanced_accuracy_score(y[te], (s >= 0.5).astype(int)))
        aps.append(average_precision_score(y[te], s))
        k10 = max(1, len(te) // 10)
        p_at_10 = precision_at_k(y[te], s, k10)
        p10.append(p_at_10)
        p50.append(precision_at_k(y[te], s, max(1, len(te) // 2)))
        ef10.append(p_at_10 / y[te].mean() if y[te].mean() > 0 else np.nan)

    def ms(v):
        v = [x for x in v if not np.isnan(x)]
        return (float(np.mean(v)), float(np.std(v))) if v else (float("nan"),) * 2

    auc_m, auc_s = ms(aucs); bacc_m, bacc_s = ms(baccs); ap_m, ap_s = ms(aps)
    return {
        "split": split, "model": model_kind, "n_folds": len(aucs),
        "roc_auc": auc_m, "roc_auc_sd": auc_s,
        "balanced_acc": bacc_m, "balanced_acc_sd": bacc_s,
        "avg_precision": ap_m, "avg_precision_sd": ap_s,
        "precision_top10pct": ms(p10)[0], "precision_top50pct": ms(p50)[0],
        "enrichment_top10pct": ms(ef10)[0], "base_rate": float(base_rate),
    }


def section(title: str) -> None:
    print("\n" + "=" * 70); print(title); print("=" * 70)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    frame = add_anchor(load_pairs(args.table))
    print(f"[data] {len(frame)} pairs, "
          f"{int((frame.label==1).sum())} positive / {int((frame.label==0).sum())} negative")

    print("[features] computing (this includes fingerprints) ...")
    X_all = desc.featurise(frame, include_fingerprints=True)
    frame = frame.loc[X_all.index]
    y = frame["label"].to_numpy()
    groups = frame["anchor_smiles"].to_numpy()
    print(f"[features] {X_all.shape[1]} features, {X_all.shape[0]} pairs, "
          f"{len(np.unique(groups))} distinct anchors")

    phys_cols = [c for c in X_all.columns if is_physics(c)]
    fp_cols = [c for c in X_all.columns if is_fingerprint(c)]
    struct_cols = [c for c in X_all.columns if not is_physics(c) and not is_fingerprint(c)]
    print(f"[features] {len(phys_cols)} physics, {len(struct_cols)} structural, "
          f"{len(fp_cols)} fingerprint bits")

    feature_sets = {
        "physics_only": phys_cols,
        "structural_only": struct_cols,
        "physics+structural": phys_cols + struct_cols,
        "physics+structural+fp": phys_cols + struct_cols + fp_cols,
    }

    # --- A and B: split protocol x feature set, random forest -----------------
    section("A/B. Split protocol x feature set (random forest)")
    rows = []
    for fs_name, cols in feature_sets.items():
        for split in ("random", "anchor_held_out"):
            r = run_cv(X_all[cols], y, groups, split, "random_forest", args.folds)
            r["feature_set"] = fs_name
            r["n_features"] = len(cols)
            rows.append(r)
            print(f"  {fs_name:>24s} {split:>16s}  "
                  f"AUC {r['roc_auc']:.3f}+/-{r['roc_auc_sd']:.3f}  "
                  f"bACC {r['balanced_acc']:.3f}+/-{r['balanced_acc_sd']:.3f}  "
                  f"AP {r['avg_precision']:.3f}")
    ablation = pd.DataFrame(rows)

    # --- C: model family on the best compact feature set ----------------------
    section("C. Model family (physics+structural features)")
    cols = feature_sets["physics+structural"]
    rows_m = []
    for kind in ("logistic", "random_forest", "grad_boost"):
        for split in ("random", "anchor_held_out"):
            r = run_cv(X_all[cols], y, groups, split, kind, args.folds)
            r["feature_set"] = "physics+structural"
            r["n_features"] = len(cols)
            rows_m.append(r)
            print(f"  {kind:>14s} {split:>16s}  "
                  f"AUC {r['roc_auc']:.3f}+/-{r['roc_auc_sd']:.3f}  "
                  f"bACC {r['balanced_acc']:.3f}+/-{r['balanced_acc_sd']:.3f}")
    models = pd.DataFrame(rows_m)

    # --- D: screening utility -------------------------------------------------
    section("D. Screening utility under the held-out split (random forest)")
    for fs_name in ("physics+structural", "physics+structural+fp"):
        r = run_cv(X_all[feature_sets[fs_name]], y, groups,
                   "anchor_held_out", "random_forest", args.folds)
        print(f"  {fs_name:>24s}  base rate {r['base_rate']:.3f}  "
              f"precision@top10% {r['precision_top10pct']:.3f}  "
              f"enrichment {r['enrichment_top10pct']:.2f}x  "
              f"precision@top50% {r['precision_top50pct']:.3f}")

    # --- E: permutation importance -------------------------------------------
    section("E. Permutation importance, compact feature set, held-out split")
    gkf = GroupKFold(n_splits=min(args.folds, len(np.unique(groups))))
    tr, te = next(gkf.split(X_all[cols], y, groups))
    m = make_model("random_forest").fit(X_all[cols].iloc[tr], y[tr])
    pi = permutation_importance(m, X_all[cols].iloc[te], y[te],
                                n_repeats=10, random_state=SEED,
                                scoring="roc_auc", n_jobs=-1)
    imp = (pd.DataFrame({"feature": cols,
                         "importance": pi.importances_mean,
                         "sd": pi.importances_std})
           .sort_values("importance", ascending=False))
    imp["block"] = ["physics" if is_physics(f) else "structural" for f in imp["feature"]]
    for _, row in imp.head(15).iterrows():
        print(f"  {row['feature']:>26s} ({row['block']:>10s})  "
              f"{row['importance']:+.4f} +/- {row['sd']:.4f}")
    phys_share = imp.head(10)["block"].value_counts().get("physics", 0)
    print(f"  physics terms among the top 10: {phys_share}/10")

    out = args.outdir.rstrip("/")
    ablation.to_csv(f"{out}/ablation.csv", index=False)
    models.to_csv(f"{out}/model_families.csv", index=False)
    imp.to_csv(f"{out}/permutation_importance.csv", index=False)
    print(f"\n[done] CSVs written to {out}/")


if __name__ == "__main__":
    main()
