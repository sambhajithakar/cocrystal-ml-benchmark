"""Do the protocol and base-rate findings hold on a drug-like subset?

The main analysis runs on a chemically diverse co-crystal dataset, which is
its most obvious limitation: the paper's claims are about evaluation protocol,
but a reader working on pharmaceutical co-crystals will reasonably ask whether
they survive on pharmaceutically relevant chemistry.

This script answers that with the same data. It restricts the dataset to pairs
whose anchor component satisfies conventional drug-likeness criteria, then
repeats the split-protocol comparison and the enrichment measurement.

Criteria applied to the anchor only (the co-former is unconstrained, as benign
co-formers are routinely small):
    150 <= MW <= 600, -1 <= logP <= 6, HBD <= 5, HBA <= 10
These are the Lipinski ranges with a lower molecular-weight bound added to
exclude solvents and small gases, which are common in crystallographic sources
but are not drug candidates.

This is a drug-LIKE filter on physicochemical properties. It does not identify
BCS Class II or IV drugs, which requires solubility and permeability data the
dataset does not carry.

Usage:
    python3 druglike_subset.py --table ../results/ccgnet_pairs_smiles.tab \
                               --outdir ../results
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors

import descriptors as desc
import full_analysis as fa

RDLogger.DisableLog("rdApp.*")

MW_MIN, MW_MAX = 150.0, 600.0
LOGP_MIN, LOGP_MAX = -1.0, 6.0
HBD_MAX, HBA_MAX = 5, 10


def is_druglike(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    return (MW_MIN <= Descriptors.MolWt(mol) <= MW_MAX
            and LOGP_MIN <= Crippen.MolLogP(mol) <= LOGP_MAX
            and Descriptors.NumHDonors(mol) <= HBD_MAX
            and Descriptors.NumHAcceptors(mol) <= HBA_MAX)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    frame = fa.add_anchor(fa.load_pairs(args.table))
    print(f"[data] full dataset: {len(frame)} pairs, "
          f"base rate {frame.label.mean():.4f}")

    anchors = frame["anchor_smiles"].unique()
    keep = {a for a in anchors if is_druglike(a)}
    print(f"[filter] {len(keep)}/{len(anchors)} anchor molecules are drug-like")

    sub = frame[frame["anchor_smiles"].isin(keep)].reset_index(drop=True)
    print(f"[data] drug-like subset: {len(sub)} pairs "
          f"({int((sub.label==1).sum())} positive / {int((sub.label==0).sum())} negative), "
          f"base rate {sub.label.mean():.4f}, "
          f"{sub.anchor_smiles.nunique()} distinct anchors")

    if sub.label.nunique() < 2 or sub.anchor_smiles.nunique() < args.folds:
        print("[stop] subset too small or single-class; nothing to report")
        return

    X = desc.featurise(sub, include_fingerprints=False)
    sub = sub.loc[X.index]
    y = sub["label"].to_numpy()
    groups = sub["anchor_smiles"].to_numpy()
    cols = [c for c in X.columns]
    print(f"[features] {X.shape[1]} features, {X.shape[0]} pairs")

    rows = []
    for split in ("random", "anchor_held_out"):
        r = fa.run_cv(X[cols], y, groups, split, "random_forest", args.folds)
        r["subset"] = "drug-like anchors"
        rows.append(r)
        print(f"  {split:>16s}  AUC {r['roc_auc']:.3f}+/-{r['roc_auc_sd']:.3f}  "
              f"bACC {r['balanced_acc']:.3f}+/-{r['balanced_acc_sd']:.3f}  "
              f"precision@top10% {r['precision_top10pct']:.3f}  "
              f"enrichment {r['enrichment_top10pct']:.2f}x")

    out = pd.DataFrame(rows)
    out.to_csv(f"{args.outdir.rstrip('/')}/druglike_subset.csv", index=False)

    d_auc = rows[0]["roc_auc"] - rows[1]["roc_auc"]
    d_bacc = rows[0]["balanced_acc"] - rows[1]["balanced_acc"]
    print(f"\n[summary] protocol penalty on the drug-like subset: "
          f"{d_auc:.3f} ROC-AUC, {d_bacc:.3f} balanced accuracy")
    print(f"[summary] base rate {sub.label.mean():.4f}, "
          f"held-out enrichment {rows[1]['enrichment_top10pct']:.2f}x")


if __name__ == "__main__":
    main()
