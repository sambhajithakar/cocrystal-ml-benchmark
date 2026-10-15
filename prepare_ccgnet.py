"""Convert the published CCGNet dataset into the SMILES table this repo reads.

Source (MIT licence):
  Jiang, Y. et al. Nat. Commun. 2021, 12, 5950. DOI: 10.1038/s41467-021-26226-7
  https://github.com/Saoge123/ccgnet
  Files used: data/CC_Table/CC_Table.tab  (pairs: id_a, id_b, label, pair_id)
              data/Mol_Blocks.zip -> Mol_Blocks.dir  (identifier -> sdf block)

CC_Table.tab identifies each component by an opaque identifier, and
Mol_Blocks.dir maps every identifier to a 3D sdf block. This script resolves
each pair through that mapping and writes a four-column tab file
(smiles_a, smiles_b, label, pair_id) that benchmark_public.py reads directly.

Mol_Blocks.dir is a Python dict literal. It is parsed with ast.literal_eval,
never eval, so no code in the file can execute.

Usage:
    python3 prepare_ccgnet.py \
        --table  /path/to/CC_Table.tab \
        --blocks /path/to/Mol_Blocks.dir \
        --out    cc_pairs_smiles.tab
"""

from __future__ import annotations

import argparse
import ast
import collections

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def load_blocks(path: str) -> dict:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return ast.literal_eval(fh.read())


def block_to_smiles(block: str) -> str | None:
    """sdf block -> canonical SMILES, or None if RDKit cannot make sense of it."""
    mol = Chem.MolFromMolBlock(block, sanitize=True, removeHs=True)
    if mol is None:
        # Some deposited blocks have valence or aromaticity records RDKit
        # rejects on a strict pass; retry without sanitisation and sanitise
        # what can be sanitised.
        mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=True)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True)
    ap.add_argument("--blocks", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[blocks] parsing {args.blocks} ...")
    blocks = load_blocks(args.blocks)
    print(f"[blocks] {len(blocks)} identifiers")

    rows = []
    for line in open(args.table, encoding="utf-8", errors="replace"):
        parts = line.rstrip("\n").split("\t")
        if len(parts) >= 3 and parts[2] in ("0", "1"):
            rows.append((parts[0], parts[1], parts[2],
                         parts[3] if len(parts) > 3 else f"{parts[0]}&{parts[1]}"))
    print(f"[table] {len(rows)} labelled pairs")

    # Resolve each identifier once; many recur across pairs.
    needed = {i for r in rows for i in r[:2]}
    smiles: dict[str, str] = {}
    unresolved_ids, unparsed_ids = [], []
    for ident in needed:
        block = blocks.get(ident, blocks.get(str(ident)))
        if block is None:
            unresolved_ids.append(ident)
            continue
        smi = block_to_smiles(block)
        if smi is None:
            unparsed_ids.append(ident)
        else:
            smiles[ident] = smi
    print(f"[resolve] {len(smiles)}/{len(needed)} identifiers resolved to SMILES "
          f"({len(unresolved_ids)} missing from the block file, "
          f"{len(unparsed_ids)} unreadable by RDKit)")

    kept, dropped = [], 0
    for a, b, label, pid in rows:
        if a in smiles and b in smiles:
            kept.append((smiles[a], smiles[b], label, pid))
        else:
            dropped += 1

    counts = collections.Counter(r[2] for r in kept)
    print(f"[out] {len(kept)} pairs usable, {dropped} dropped for an "
          f"unresolvable component")
    print(f"[out] labels: {counts.get('1', 0)} positive, {counts.get('0', 0)} negative")

    with open(args.out, "w", encoding="utf-8") as fh:
        for row in kept:
            fh.write("\t".join(row) + "\n")
    print(f"[out] written to {args.out}")


if __name__ == "__main__":
    main()
