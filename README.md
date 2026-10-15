# Co-crystal ML benchmark — code and derived data

Everything needed to reproduce every number in:

> Thakar, S. B.; Ghorpade, P. N. *Data-Centric Machine Learning for
> Pharmaceutical Co-Crystal Discovery: Molecule-Held-Out Evaluation and
> Enrichment-Based Benchmarking of 7,775 Experimental Pairs.*
> Submitted to *Journal of Chemical Information and Modeling*, 2026.

No proprietary data or software is used. Licence: MIT.

## What this measures

Two features of current evaluation practice inflate reported co-crystal
formability performance:

1. **Split protocol.** A random split over molecule *pairs* lets the same
   molecule appear in training and test. Holding out the recurring molecule
   costs 0.112–0.125 balanced accuracy, in every feature set and model family.
2. **Class balance.** The data are 86.4% positive, so a model at 0.994
   top-decile precision enriches by only 1.15×.

Both effects are *larger* on a drug-like subset (0.137 penalty, 90.6% base
rate, 1.10× enrichment).

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
| `code/prepare_ccgnet.py` | Resolves CCGNet numeric identifiers to canonical SMILES via `Mol_Blocks.dir`. 7,775 of 7,884 pairs resolve. |
| `code/descriptors.py` | The three descriptor blocks: 20 structural, 13 pair-level physics, 2,048 Morgan fingerprint bits. |
| `code/full_analysis.py` | Split protocol × feature set, model families, screening utility, permutation importance. Produces Tables 1–3. |
| `code/druglike_subset.py` | Repeats the protocol comparison on drug-like anchors. Produces Table 4. |
| `code/make_figures.py` | Figures 1–3, regenerated from the result tables. |
| `code/make_toc_graphic.py` | The journal Table of Contents graphic. |
| `code/make_si.py` | Supporting Information tables S1–S5 and Figure S1. |
| `code/benchmark_public.py` | Standalone formability benchmark on any SMILES pair table. |
| `code/build_docx.py` | Manuscript formatting (not part of the analysis). |

Figures and SI tables are generated *from* the result CSVs, so they cannot
disagree with the numbers in the paper.

## Derived data

`results/ccgnet_pairs_smiles.tab` — 7,775 pairs (6,721 positive, 1,054
negative), both components as canonical SMILES. Derived from the CCGNet
dataset (Jiang et al., *Nat. Commun.* 2021, 12, 5950, MIT licence),
https://github.com/Saoge123/ccgnet.

The remaining CSVs are the measured results: `ablation.csv`,
`model_families.csv`, `permutation_importance.csv`, `druglike_subset.csv`.

## A note on scope

This is a benchmark of evaluation methodology. It does not propose a new model,
and the dataset carries no therapeutic-class or indication annotation, so no
claim about any disease area is made or supported.
