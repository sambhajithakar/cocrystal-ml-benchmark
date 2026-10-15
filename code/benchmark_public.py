"""Benchmark the formability head on a published public dataset.

Data source: the co-crystal table released with
  Jiang, Y. et al. "Coupling complementary strategy to flexible graph neural
  network for quick discovery of coformer in diverse co-crystal materials."
  Nat. Commun. 2021, 12, 5950. DOI: 10.1038/s41467-021-26226-7
  Code and data: https://github.com/Saoge123/ccgnet  (MIT licence)
  File: data/CC_Table/ECC&CC_Table.tab  -- columns are
  SMILES_1 <tab> SMILES_2 <tab> label <tab> CSD refcode

What this script does and does not establish
--------------------------------------------
DOES: it checks that the feature pipeline and the evaluation protocol in this
repository work on real published data, and it quantifies how much the score
drops when you stop splitting at random and start holding molecules out.

DOES NOT: it says nothing about BCS Class II/IV drugs specifically. This table
covers diverse organic co-crystals including energetic materials, and it has
no solid-form labels (co-crystal vs salt) and no eutectic constants. Heads B
and C of the project cannot be benchmarked here -- they need the curated
BCS II/IV dataset.

Two splits are reported because the contrast is the point:

  random      -- the same molecule can appear in train and test. This is what
                 much of the published literature does, and it reads high.
  anchor-wise -- GroupKFold on the more frequently occurring component of each
                 pair, so that component is never in both folds. Still not a
                 fully molecule-disjoint split (the partner molecule can leak),
                 so treat it as an upper bound on the honest number.

Usage:
    python3 benchmark_public.py --table path/to/ECC_CC_Table.tab
    python3 benchmark_public.py --table ... --max-rows 2000   # quick run
"""

from __future__ import annotations

import argparse
import collections

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline

import descriptors as desc

SEED = 0
N_TREES = 300


def read_table(path: str, max_rows: int | None) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            smi_a, smi_b, label = parts[0], parts[1], parts[2]
            if label not in ("0", "1"):
                continue
            rows.append({
                "smiles_a": smi_a,
                "smiles_b": smi_b,
                "label": int(label),
                "refcode": parts[3] if len(parts) > 3 else "",
            })
    frame = pd.DataFrame(rows)
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.sample(max_rows, random_state=SEED).reset_index(drop=True)
    return frame


def assign_anchor(frame: pd.DataFrame) -> pd.DataFrame:
    """Pick the more common component of each pair as the grouping key.

    In this table neither column is consistently 'the API', so the frequent
    component stands in for the anchor molecule a screen is built around.
    """
    counts = collections.Counter(frame["smiles_a"]) + collections.Counter(frame["smiles_b"])
    anchor, partner = [], []
    for a, b in zip(frame["smiles_a"], frame["smiles_b"]):
        if counts[a] >= counts[b]:
            anchor.append(a)
            partner.append(b)
        else:
            anchor.append(b)
            partner.append(a)
    out = frame.copy()
    out["anchor_smiles"] = anchor
    out["partner_smiles"] = partner
    # descriptors.featurise() expects these column names
    out["drug_smiles"] = anchor
    out["coformer_smiles"] = partner
    out["drug_pka"] = np.nan
    out["coformer_pka"] = np.nan
    return out


def evaluate(X: pd.DataFrame, y: np.ndarray, splitter, groups, name: str) -> dict:
    aucs, baccs = [], []
    it = (splitter.split(X, y, groups) if groups is not None
          else splitter.split(X, y))
    for fold, (tr, te) in enumerate(it, 1):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=N_TREES, random_state=SEED, n_jobs=-1),
        )
        model.fit(X.iloc[tr], y[tr])
        proba = model.predict_proba(X.iloc[te])[:, 1]
        aucs.append(roc_auc_score(y[te], proba))
        baccs.append(balanced_accuracy_score(y[te], (proba >= 0.5).astype(int)))
        print(f"    fold {fold}: ROC-AUC {aucs[-1]:.3f}  balanced acc {baccs[-1]:.3f}"
              f"  (n_test={len(te)})")

    res = {
        "split": name,
        "auc_mean": float(np.mean(aucs)), "auc_sd": float(np.std(aucs)),
        "bacc_mean": float(np.mean(baccs)), "bacc_sd": float(np.std(baccs)),
        "n_folds": len(aucs),
    }
    print(f"  {name}: ROC-AUC {res['auc_mean']:.3f} +/- {res['auc_sd']:.3f}   "
          f"balanced acc {res['bacc_mean']:.3f} +/- {res['bacc_sd']:.3f}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="ECC&CC_Table.tab from the CCGNet repo")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fingerprints", action="store_true",
                    help="add Morgan fingerprints (slower, 2000+ features)")
    ap.add_argument("--out", default=None, help="write results table to this CSV")
    args = ap.parse_args()

    frame = read_table(args.table, args.max_rows)
    print(f"[data] {len(frame)} pairs from {args.table}")
    print(f"[data] labels: {int((frame['label']==1).sum())} positive, "
          f"{int((frame['label']==0).sum())} negative "
          f"(ratio {(frame['label']==1).sum()/max((frame['label']==0).sum(),1):.1f}:1)")

    frame = assign_anchor(frame)
    print(f"[data] {frame['anchor_smiles'].nunique()} distinct anchor molecules")

    print(f"[features] computing{' with' if args.fingerprints else ' without'} fingerprints ...")
    X = desc.featurise(frame, include_fingerprints=args.fingerprints)
    frame = frame.loc[X.index]
    y = frame["label"].to_numpy()
    print(f"[features] {X.shape[1]} features for {X.shape[0]} pairs")

    # Known limitation of this particular file: its positive rows carry SMILES,
    # but its negative rows identify energetic materials by abbreviation
    # ("CL-20", "TNB", ...) rather than structure. Those rows cannot be
    # featurised from this file alone, so every negative is dropped above and
    # no classifier can be trained. Resolving them needs data/Mol_Blocks.dir
    # from the same repository, which maps each identifier to an sdf block.
    present = sorted(set(y.tolist()))
    if len(present) < 2:
        only = "positive" if present == [1] else "negative"
        print(f"\n[stop] after featurisation only {only} examples remain "
              f"({len(y)} rows), so there is nothing to classify.")
        print("[stop] The feature pipeline itself ran cleanly on real published "
              "data, which is what this script can confirm from this file.")
        print("[stop] To benchmark formability, obtain data/Mol_Blocks.dir from "
              "https://github.com/Saoge123/ccgnet (clone the repository), resolve "
              "the non-SMILES identifiers to structures, and re-run.")
        return

    results = []
    print("\n=== random split (comparable to much of the literature) ===")
    results.append(evaluate(
        X, y, StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=SEED),
        None, "random"))

    print("\n=== anchor-wise split (molecules held out) ===")
    groups = frame["anchor_smiles"].to_numpy()
    n_groups = len(np.unique(groups))
    results.append(evaluate(
        X, y, GroupKFold(n_splits=min(args.folds, n_groups)), groups, "anchor-wise"))

    drop = results[0]["auc_mean"] - results[1]["auc_mean"]
    print(f"\n[summary] holding molecules out costs {drop:.3f} ROC-AUC.")
    print("[summary] Report the anchor-wise number as the honest one, and show both "
          "so readers can compare with papers that split at random.")

    table = pd.DataFrame(results)
    if args.out:
        table.to_csv(args.out, index=False)
        print(f"[summary] written to {args.out}")
    print("\n" + table.to_string(index=False))


if __name__ == "__main__":
    main()
