"""Independent audit checks on the evaluation splits (Project 8, 26 Oct 2026).

Written for the final audit, NOT part of the original analysis. It answers four
questions about the splits the paper reports, using the same curated pair table:

  1. Is the anchor-held-out split genuinely anchor-disjoint?
  2. How much does the PARTNER component leak across that split? Section 2.3 of
     the manuscript states that it may; this measures how often.
  3. What happens under a fully molecule-disjoint split, where neither component
     may cross folds? Built by taking connected components of the molecule graph
     as groups.
  4. Are there order-insensitive duplicate pairs or contradictory labels left in
     the curated table?

Usage:
    python3 audit_split_integrity.py --table ../results/ccgnet_pairs_smiles.tab \
                                     --out /path/to/audit_results.csv
"""

from __future__ import annotations

import argparse
import collections

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold

import descriptors as desc
import full_analysis as fa


def load(table: str) -> pd.DataFrame:
    return fa.add_anchor(fa.load_pairs(table))


def duplicate_report(frame: pd.DataFrame) -> dict:
    """Order-insensitive duplicates and label contradictions."""
    keyed = collections.defaultdict(list)
    for a, b, y in zip(frame["smiles_a"], frame["smiles_b"], frame["label"]):
        keyed[tuple(sorted((a, b)))].append(y)
    dup = sum(1 for v in keyed.values() if len(v) > 1)
    contra = sum(1 for v in keyed.values() if len(set(v)) > 1)
    return {"unique_unordered_pairs": len(keyed),
            "pairs_appearing_more_than_once": dup,
            "pairs_with_contradictory_labels": contra}


def component_groups(frame: pd.DataFrame) -> np.ndarray:
    """Connected components of the molecule graph; guarantees molecule-disjoint folds."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for a, b in zip(frame["smiles_a"], frame["smiles_b"]):
        union(a, b)
    return np.array([find(a) for a in frame["smiles_a"]])


def leak_report(frame: pd.DataFrame, groups: np.ndarray, label: str,
                folds: int = 5) -> dict:
    """Per fold: do anchors or partners cross the train/test boundary?"""
    gkf = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    y = frame["label"].to_numpy()
    anchors = frame["anchor_smiles"].to_numpy()
    partners = frame["coformer_smiles"].to_numpy()
    anchor_leak, partner_leak, any_leak, sizes, rates = [], [], [], [], []
    for tr, te in gkf.split(frame, y, groups):
        tr_anchor, te_anchor = set(anchors[tr]), set(anchors[te])
        tr_mol = tr_anchor | set(partners[tr])
        shared_anchor = tr_anchor & te_anchor
        shared_partner = {p for p in partners[te] if p in tr_mol}
        anchor_leak.append(len(shared_anchor))
        partner_leak.append(float(np.mean([p in tr_mol for p in partners[te]])))
        any_leak.append(float(np.mean([(a in tr_mol) or (p in tr_mol)
                                       for a, p in zip(anchors[te], partners[te])])))
        sizes.append(len(te))
        rates.append(float(y[te].mean()))
    return {"protocol": label,
            "folds": len(sizes),
            "anchor_molecules_shared_mean": float(np.mean(anchor_leak)),
            "test_pairs_whose_partner_was_seen_mean": float(np.mean(partner_leak)),
            "test_pairs_with_either_component_seen_mean": float(np.mean(any_leak)),
            "test_fold_size_min": int(min(sizes)),
            "test_fold_size_max": int(max(sizes)),
            "test_base_rate_min": float(min(rates)),
            "test_base_rate_max": float(max(rates))}


def score(frame: pd.DataFrame, X: pd.DataFrame, groups: np.ndarray,
          label: str, folds: int = 5) -> dict:
    """Random forest on the 33-feature set under an arbitrary grouping."""
    y = frame["label"].to_numpy()
    gkf = GroupKFold(n_splits=min(folds, len(np.unique(groups))))
    aucs, baccs, aps, ef10 = [], [], [], []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = fa.make_model("random_forest").fit(X.iloc[tr], y[tr])
        s = m.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], s))
        baccs.append(balanced_accuracy_score(y[te], (s >= 0.5).astype(int)))
        aps.append(average_precision_score(y[te], s))
        k10 = max(1, len(te) // 10)
        ef10.append(fa.precision_at_k(y[te], s, k10) / y[te].mean())
    return {"protocol": label, "n_folds": len(aucs),
            "roc_auc": float(np.mean(aucs)), "roc_auc_sd": float(np.std(aucs)),
            "balanced_acc": float(np.mean(baccs)),
            "balanced_acc_sd": float(np.std(baccs)),
            "avg_precision": float(np.mean(aps)),
            "enrichment_top10pct": float(np.mean(ef10))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    frame = load(args.table)
    print("[data]", len(frame), "pairs,", int(frame["label"].sum()), "positive")
    print("[duplicates]", duplicate_report(frame))

    anchors = frame["anchor_smiles"].to_numpy()
    comps = component_groups(frame)
    print("[graph] connected components:", len(np.unique(comps)))
    sizes = collections.Counter(comps)
    print("[graph] largest component holds",
          max(sizes.values()), "of", len(frame), "pairs")

    rows = [leak_report(frame, anchors, "anchor held out", args.folds),
            leak_report(frame, comps, "fully molecule-disjoint", args.folds)]
    for r in rows:
        print("[leakage]", r)

    X_all = desc.featurise(frame, include_fingerprints=False)
    frame = frame.loc[X_all.index]
    anchors = frame["anchor_smiles"].to_numpy()
    comps = component_groups(frame)
    X = X_all
    print("[features]", X.shape[1], "features (physics + structural)")
    scores = [score(frame, X, anchors, "anchor held out", args.folds),
              score(frame, X, comps, "fully molecule-disjoint", args.folds)]
    for s in scores:
        print("[score]", s)

    pd.DataFrame(rows).to_csv(args.out.replace(".csv", "_leakage.csv"), index=False)
    pd.DataFrame(scores).to_csv(args.out, index=False)
    print("[done] written", args.out)


if __name__ == "__main__":
    main()
