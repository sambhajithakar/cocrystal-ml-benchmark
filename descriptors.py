"""Build model inputs for an API-coformer pair.

Two blocks of features:
  structural  -- RDKit 2D descriptors + Morgan fingerprints for each molecule
  physics     -- pair-level terms that carry the thermodynamics: delta pKa,
                 hydrogen-bond donor/acceptor complementarity, polarity gap

The physics block is what separates this work from formability-only models,
so keep it explicit and auditable rather than hidden inside a fingerprint.

Not included here: COSMO-RS excess enthalpy and molecular electrostatic
potential extremes. Both need external software (COSMOtherm, a QM package).
Compute them separately and merge on (drug_name, coformer_name) -- see
merge_external_descriptors().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdFingerprintGenerator

RDLogger.DisableLog("rdApp.*")

FP_BITS = 1024
FP_RADIUS = 2  # radius 2 == ECFP4, the fingerprint that worked best in the
               # deep-forest study (Crystals 2024, 14, 313)

_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

# Per-molecule descriptors. Chosen to cover size, polarity, hydrogen bonding
# and flexibility -- the four things that drive co-crystal formation.
_MOL_DESCRIPTORS = {
    "mw": Descriptors.MolWt,
    "logp": Crippen.MolLogP,
    "tpsa": Descriptors.TPSA,
    "hbd": Descriptors.NumHDonors,
    "hba": Descriptors.NumHAcceptors,
    "rotb": Descriptors.NumRotatableBonds,
    "arom_rings": Descriptors.NumAromaticRings,
    "heavy_atoms": Descriptors.HeavyAtomCount,
    "frac_csp3": Descriptors.FractionCSP3,
    "molar_refractivity": Crippen.MolMR,
}


def parse_smiles(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol


def molecule_descriptors(mol: Chem.Mol, prefix: str) -> dict[str, float]:
    return {f"{prefix}_{name}": float(fn(mol)) for name, fn in _MOL_DESCRIPTORS.items()}


def morgan_bits(mol: Chem.Mol, prefix: str) -> dict[str, int]:
    fp = _MORGAN.GetFingerprintAsNumPy(mol)
    return {f"{prefix}_fp{i}": int(bit) for i, bit in enumerate(fp)}


def physics_features(
    drug: Chem.Mol,
    coformer: Chem.Mol,
    drug_pka: float | None = None,
    coformer_pka: float | None = None,
) -> dict[str, float]:
    """Pair-level terms with a physical reading.

    delta_pka is the one that matters most for telling a salt from a
    co-crystal: a large positive difference favours proton transfer, so a
    salt; a small one favours a neutral co-crystal. The usual rule of thumb
    puts the changeover around 2-3 units, with a grey zone between. Leave
    the pKa columns blank and the model falls back on structure alone --
    delta_pka_known tells it which case it is looking at.
    """
    d = molecule_descriptors(drug, "d")
    c = molecule_descriptors(coformer, "c")

    feats: dict[str, float] = {
        # Hydrogen-bond complementarity: can the coformer's donors meet the
        # drug's acceptors, and the other way round.
        "hb_pair_dD_cA": min(d["d_hbd"], c["c_hba"]),
        "hb_pair_cD_dA": min(c["c_hbd"], d["d_hba"]),
        "hb_total_complementarity": min(d["d_hbd"], c["c_hba"]) + min(c["c_hbd"], d["d_hba"]),
        # Gaps. Similar polarity and size tend to help packing.
        "tpsa_gap": abs(d["d_tpsa"] - c["c_tpsa"]),
        "logp_gap": abs(d["d_logp"] - c["c_logp"]),
        "mw_ratio": c["c_mw"] / d["d_mw"] if d["d_mw"] else 0.0,
        "heavy_atom_ratio": c["c_heavy_atoms"] / d["d_heavy_atoms"] if d["d_heavy_atoms"] else 0.0,
        # Combined polarity, a crude stand-in for Hansen matching until you
        # compute real Hansen parameters.
        "tpsa_sum": d["d_tpsa"] + c["c_tpsa"],
        "logp_sum": d["d_logp"] + c["c_logp"],
    }

    if drug_pka is not None and coformer_pka is not None and not (
        pd.isna(drug_pka) or pd.isna(coformer_pka)
    ):
        delta = float(drug_pka) - float(coformer_pka)
        feats["delta_pka"] = delta
        feats["abs_delta_pka"] = abs(delta)
        feats["delta_pka_known"] = 1.0
        # Flag the grey zone where the salt/co-crystal call is genuinely uncertain.
        feats["delta_pka_grey_zone"] = 1.0 if 0.0 <= abs(delta) <= 3.0 else 0.0
    else:
        feats["delta_pka"] = 0.0
        feats["abs_delta_pka"] = 0.0
        feats["delta_pka_known"] = 0.0
        feats["delta_pka_grey_zone"] = 0.0

    return feats


def featurise_row(row: pd.Series, include_fingerprints: bool = True) -> dict[str, float]:
    drug = parse_smiles(row["drug_smiles"])
    coformer = parse_smiles(row["coformer_smiles"])

    feats: dict[str, float] = {}
    feats.update(molecule_descriptors(drug, "d"))
    feats.update(molecule_descriptors(coformer, "c"))
    feats.update(
        physics_features(
            drug,
            coformer,
            row.get("drug_pka"),
            row.get("coformer_pka"),
        )
    )
    if include_fingerprints:
        feats.update(morgan_bits(drug, "d"))
        feats.update(morgan_bits(coformer, "c"))
    return feats


def featurise(frame: pd.DataFrame, include_fingerprints: bool = True) -> pd.DataFrame:
    """Featurise every row. Rows with unparseable SMILES are dropped and reported."""
    rows, kept, dropped = [], [], []
    for idx, row in frame.iterrows():
        try:
            rows.append(featurise_row(row, include_fingerprints))
            kept.append(idx)
        except ValueError as exc:
            dropped.append((idx, str(exc)))

    if dropped:
        print(f"[descriptors] dropped {len(dropped)} row(s) with bad SMILES:")
        for idx, msg in dropped[:10]:
            print(f"  row {idx}: {msg}")

    return pd.DataFrame(rows, index=kept)


def merge_external_descriptors(
    features: pd.DataFrame,
    frame: pd.DataFrame,
    external: pd.DataFrame,
    on: tuple[str, str] = ("drug_name", "coformer_name"),
) -> pd.DataFrame:
    """Attach descriptors computed outside this script.

    `external` must have the two key columns plus one column per descriptor,
    e.g. cosmo_dhex, mep_min_drug, mep_max_coformer, hansen_dd, hansen_dp,
    hansen_dh. Missing pairs come back as NaN, which the training script
    imputes -- so a partial COSMO-RS run is still usable.
    """
    keys = frame.loc[features.index, list(on)]
    merged = keys.merge(external, on=list(on), how="left")
    merged.index = features.index
    added = [c for c in merged.columns if c not in on]
    print(f"[descriptors] merged {len(added)} external descriptor(s): {added}")
    return pd.concat([features, merged[added]], axis=1)
