# PredictSNPprot

PredictSNPprot is a gradient-boosting consensus predictor of the effect of single amino acid substitutions on protein function. It combines the scores of nine sequence-based variant effect predictors (VEPs) into one pathogenicity/deleteriousness score.

This repository contains the code to train the model and generate predictions from precomputed tool scores. It accompanies the manuscript *Leakage-Controlled Benchmarking of Variant Effect Predictors and Development of Gradient-Boosting Consensus Model*.

## Contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Installation](#installation)
- [Input data](#input-data)
- [Usage](#usage)
- [Data and related resources](#data-and-related-resources)
- [Citation](#citation)

## Overview

PredictSNPprot takes the predictions of nine VEPs for each variant and outputs a consensus score:

| Constituent tool | Type |
|---|---|
| 3Cnet | Supervised, multi-task RNN |
| ESM1b | Protein language model, zero-shot |
| VESPAl | pLM embeddings + supervised model |
| VespaG | pLM embeddings (ESM-2) + supervised model |
| PROVEAN | Alignment-based |
| VariPred | pLM embeddings + supervised model |
| PHACT | Phylogeny-aware, MSA-based |
| PHACTboost | PHACT + gradient boosting |
| GEMME | Evolutionary, MSA-based |

PredictSNPprot does not run the constituent tools. You must provide their scores (see [Input data](#input-data)).

## Repository structure

```
PredictSNPprot/
├── predictsnp_prot_predictor/
│   ├── predict_scores.py              # generate PredictSNPprot scores from tool scores (`predict-score`)
│   ├── reconstructors.py              # rebuild the fitted pipeline from the JSON artifacts
│   ├── custom_transformers.py         # FeatureSelector and ColumnFlipper pipeline steps
│   ├── pipeline/
│   │   └── pipeline_artifacts_full/   # trained model used for prediction
│   │       ├── pipeline_config.json   # features, sign flips, scaler and imputer parameters
│   │       ├── classifier.json        # XGBoost model
│   │       └── imputer_estimators/    # BayesianRidge estimators of the iterative imputer
│   └── training/
│       ├── data_preparation.py        # stratified 80/20 train/test split (`data-preparation`)
│       ├── train_scaler.py            # fit feature selection, sign flips and min–max scaler (`train-scaler`)
│       ├── train_imputer.py           # select and fit the imputation strategy (`train-imputer`)
│       ├── train_classifier.py        # train and select the consensus classifier (`train-classifier`)
│       ├── convert_artifact.py        # convert the training output into pipeline artifacts (`convert-artifact`)
│       └── features.py                # names of the nine constituent tools
├── setup.py                           # package metadata and console scripts
├── requirements.txt                   # dependencies
├── MANIFEST.in
└── README.md
```

The pipeline applied at prediction time is: FeatureSelector → ColumnFlipper → MinMaxScaler → IterativeImputer → XGBClassifier.

## Installation

Requires Python 3.12 or newer.

```bash
git clone https://github.com/loschmidt/PredictSNPprot.git
cd PredictSNPprot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies (`requirements.txt`):

| Package | Version |
|---|---|
| NumPy | ≥ 2.4.5 |
| pandas | ≥ 3.0.3 |
| scikit-learn | ≥ 1.8.0 |
| XGBoost | ≥ 3.2.0 |

## Input data

PredictSNPprot expects a CSV with one row per variant and one score column per constituent tool.

| Column | Description |
|---|---|
| `ID` | Protein identifier |
| `Mutation` | Amino acid substitution |
| [tool columns] | One raw score column for each of the nine tools |

How to generate the nine scores:

- Tool versions, parameters and modified source code are listed in Supplementary Table S3 of the manuscript. The modified forks used for benchmarking are in separate repositories: [VariPred](https://github.com/loschmidt/VariPred), [PHACT](https://github.com/loschmidt/PHACT), [PHACTboost](https://github.com/loschmidt/PHACTboost) and [3Cnet](https://github.com/loschmidt/3Cnet).
- PHACT, PHACTboost and GEMME require multiple sequence alignments built with the shared MSA pipeline described in Supplementary Note 1 and Table S3.
- Missing scores are filled in by the bundled imputer. If many are missing, we recommend training your own imputer with `train_imputer.py`.

Preprocessing applied automatically by the code:

1. Sign alignment: scores for which lower values mean more damaging variants are negated so that higher always means more deleterious.
2. Min–max rescaling to the unit range using the 1st and 99th percentiles of the training scores. Values outside the range are clamped.

## Usage

Predict:

```bash
python -m predictsnp_prot_predictor.predict_scores --input scores.csv --output predictions.csv
```

The output contains the input columns plus `PredictSNPProt`, the predicted probability that the variant is deleterious. Use `--cols` (repeatable) to keep only selected columns and `--artifacts-dir` to use a different trained model.

Train (run from `predictsnp_prot_predictor/training/`):

```bash
python data_preparation.py --input full_dataset.csv --outdir splits
python train_scaler.py     --trainset splits/trainset.csv --out models/scaler.json
python train_imputer.py    --trainset splits/trainset.csv --scaler models/scaler.json --out models/imputer.json
python train_classifier.py --trainset splits/trainset.csv --scaler models/scaler.json --imputer models/imputer.json --out models/model
python convert_artifact.py --input models/model_classifier.json --outdir artifacts --variant full
```

Then predict with `--artifacts-dir artifacts`.

## Data and related resources

- Datasets (compiled training sets, raw, filtered and benchmark datasets, PredictSNPprot train and test sets): [Zenodo DOI: https://doi.org/10.5281/zenodo.23597130].

## Citation

If you use PredictSNPprot or the benchmark data, please cite:

> Havlasek M., Kabourek P., Musil M., Damborsky J., Mazurenko S., Bednar D. *Leakage-Controlled Benchmarking of Variant Effect Predictors and Development of Gradient-Boosting Consensus Model* [journal, year, DOI pending]

Please also cite the original publications of the constituent tools (see the manuscript reference list).

