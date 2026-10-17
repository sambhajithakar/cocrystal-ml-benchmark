# Co-crystal ML benchmark — code and derived data

Everything needed to reproduce every number in:

> Thakar, S. B.; Ghorpade, P. N. *Data-Centric Machine Learning for
> Pharmaceutical Co-Crystal Discovery: Molecule-Held-Out Evaluation and
> Enrichment-Based Benchmarking of 7,757 Experimental Pairs.*
> Manuscript submitted for publication, 2026.

No proprietary data or software is used. Licence: MIT.

## What this measures

Two features of current evaluation practice inflate reported co-crystal
formability performance:

1. **Split protocol.** A random split over molecule *pairs* lets the same
   molecule appear in training and test. Holding out the recurring molecule
   costs 0.090–0.109 balanced accuracy, in every feature set tested.
2. **Class balance.** The data are 86.6% positive, so a model at 0.997
   top-decile precision enriches by only 1.15×.

Both effects are *larger* on a drug-like subset (0.151 penalty, 90.7% base
rate, 1.10× enrichment).

The analysed set is **curated**: of the 7,775 pairs that resolve, four are
exact duplicates and seven appear twice with opposite labels, so 18 records
are removed and 7,757 remain. `results/as-distributed/` holds the same
analysis run on the uncurated table, and Table S6 of the paper reports every
headline value both ways.

## Setup

```
python3 -m pip install -r requirements.txt
```

Pinned to the exact versions that produced the published numbers (Python 3.13.16).

## Reproduce from scratch

```
git clone https://github.com/Saoge123/ccgnet.git
unzip ccgnet/data/Mol_Blocks.zip -d blocks/

python3 code/prepare_ccgnet.py --table ccgnet/data/CC_Table/CC_Table.tab \
                               --blocks blocks/Mol_Blocks.dir \
                               --out results/ccgnet_pairs_smiles.tab

python3 code/full_analysis.py   --table results/ccgnet_pairs_smiles.tab --outdir results
python3 code/druglike_subset.py --table results/ccgnet_pairs_smiles.tab --outdir results
python3 code/make_figures.py    --results results --outdir figures
```

`results/` already contains the outputs, so the analysis scripts can be run
against the shipped pair table without re-deriving it.

## What each file does

| File | Purpose |
|---|---|
| `code/prepare_ccgnet.py` | Resolves CCGNet numeric identifiers to canonical SMILES via `Mol_Blocks.dir`. 7,775 of 7,884 pairs resolve; curation then removes 18, leaving 7,757. |
| `code/descriptors.py` | The three descriptor blocks: 20 structural, 13 pair-level physics, 2,048 Morgan fingerprint bits. |
| `code/full_analysis.py` | Split protocol × feature set, model families, screening utility, permutation importance. Produces Tables 1–3. |
| `code/druglike_subset.py` | Repeats the protocol comparison on drug-like anchors. Produces Table 4. |
| `code/make_figures.py` | Figures 1–3, regenerated from the result tables. |
| `code/make_toc_graphic.py` | The journal Table of Contents graphic. |
| `code/make_si.py` | Supporting Information tables S1–S6 and Figure S1. |
| `code/make_smiles_csv.py` | The published `SMILES.csv`, built through the same loader the analysis uses. |
| `code/benchmark_public.py` | Standalone formability benchmark on any SMILES pair table. |
| `code/build_docx.py` | Manuscript formatting (not part of the analysis). |

Figures and SI tables are generated *from* the result CSVs, so they cannot
disagree with the numbers in the paper.

## Derived data

`results/ccgnet_pairs_smiles.tab` — the curated table, 7,757 pairs (6,714
positive, 1,043 negative), both components as canonical SMILES.
`results/as-distributed/ccgnet_pairs_smiles.tab` is the same table before
curation, 7,775 pairs. Derived from the CCGNet
dataset (Jiang et al., *Nat. Commun.* 2021, 12, 5950, MIT licence),
https://github.com/Saoge123/ccgnet.

The remaining CSVs are the measured results: `ablation.csv`,
`model_families.csv`, `permutation_importance.csv`, `druglike_subset.csv`,
each with its uncurated counterpart under `results/as-distributed/`.
`results/SMILES.csv` is the table published as Supporting Information.

## A note on scope

This is a benchmark of evaluation methodology. It does not propose a new model,
and the dataset carries no therapeutic-class or indication annotation, so no
claim about any disease area is made or supported.
