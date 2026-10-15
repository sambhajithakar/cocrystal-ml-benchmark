"""Write the machine-readable SMILES table that accompanies the manuscript.

Uses the same anchor assignment as full_analysis.py, so the file a reader
imports is the file the analysis saw.

Usage:
    python3 make_smiles_csv.py --table ../results/ccgnet_pairs_smiles.tab \
                               --out ../benchmark-paper/submission/supporting/SMILES.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

import full_analysis as fa

MEANING = {1: "co-crystal formed",
           0: "co-crystallisation attempted, failed"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frame = fa.add_anchor(fa.load_pairs(args.table))
    out = pd.DataFrame({
        "pair_id": [f"P{i:05d}" for i in range(1, len(frame) + 1)],
        "anchor_smiles": frame["anchor_smiles"],
        "coformer_smiles": frame["coformer_smiles"],
        "formability_label": frame["label"].astype(int),
    })
    out["label_meaning"] = out["formability_label"].map(MEANING)
    out.to_csv(args.out, index=False)
    print(f"[done] {len(out)} pairs written to {args.out}")
    print(f"       {int(out.formability_label.sum())} positive, "
          f"{int((1 - out.formability_label).sum())} negative")


if __name__ == "__main__":
    main()
