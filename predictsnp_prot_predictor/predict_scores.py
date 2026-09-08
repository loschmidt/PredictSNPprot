#!/usr/bin/env python3
"""
apply_pipeline.py
=================
Reconstructs the fitted sklearn Pipeline from JSON artifacts (no pkl needed)
and applies it to new input data.

Pipeline: FeatureSelector -> ColumnFlipper -> MinMaxScaler -> IterativeImputer -> XGBClassifier

CLI usage:
    python predict_scores.py --input data.csv --output predictions.csv

Module usage:
    from apply_pipeline import load_pipeline, run_pipeline
    pipeline = load_pipeline(Path("pipeline_artifacts/"))
    df_out   = run_pipeline(df, pipeline)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
import shlex

import logging
from predictsnp_prot_predictor.predictor_merger import merge_predictors
from predictsnp_prot_predictor.reconstructors import reconstruct_feature_selector, reconstruct_column_flipper, \
    reconstruct_scaler, reconstruct_imputer, reconstruct_xgb

logger = logging.getLogger(__name__)

AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

# ---------------------------------------------------------------------------
# Default artifact directories (relative to this script)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent / "pipeline"

PIPELINE_VARIANTS: dict[str, Path] = {"full": _HERE / "pipeline_artifacts_full"}

# ===========================================================================
# Pipeline loader
# ===========================================================================

def load_pipeline(artifacts_dir: Path) -> dict[str, Any]:
    """
    Loads and reconstructs all pipeline steps from JSON artifacts.

    Parameters
    ----------
    artifacts_dir : Path to directory containing pipeline_config.json,
                    classifier.json, and imputer_estimators/

    Returns
    -------
    dict with keys:
        "selector"   : FeatureSelector (fitted)
        "flipper"    : ColumnFlipper
        "scaler"     : MinMaxScaler (fitted)
        "imputer"    : IterativeImputer (fitted)
        "classifier" : xgb.Booster
        "config"     : raw pipeline_config.json dict
    """
    artifacts_dir = Path(artifacts_dir)
    config_path   = artifacts_dir / "pipeline_config.json"

    if not config_path.exists():
        raise FileNotFoundError(
            f"pipeline_config.json not found in {artifacts_dir}.\n"
            "Run the extraction script first."
        )

    with open(config_path) as f:
        cfg = json.load(f)

    n_feat = len(cfg["feature_cols"])
    logger.info(f"Loading pipeline from : {artifacts_dir}")
    logger.info(f"  Version  : {cfg.get('version', 'unknown')}")
    logger.info(f"  Created  : {cfg.get('created', 'unknown')}")
    logger.info(f"  Features : {n_feat} -> {cfg['feature_cols']}")

    selector = reconstruct_feature_selector(cfg)
    logger.info("  OK  FeatureSelector")

    flipper = reconstruct_column_flipper(cfg)
    logger.info(f"  OK  ColumnFlipper   (cols_to_flip: {cfg['cols_to_flip']})")

    scaler = reconstruct_scaler(cfg["scaler"])
    logger.info("  OK  MinMaxScaler")

    imputer = reconstruct_imputer(cfg["imputer"], artifacts_dir)
    logger.info(
        f"  OK  IterativeImputer  "
        f"({cfg['imputer']['n_rounds']} round(s) x "
        f"{cfg['imputer']['entries_per_round']} features)"
    )

    clf_path   = artifacts_dir / cfg["classifier"]["model_file"]
    classifier = reconstruct_xgb(clf_path, cfg["classifier"])
    logger.info(f"  OK  XGBClassifier   ({clf_path.stat().st_size / 1024:.1f} KB)")

    return {
        "selector":   selector,
        "flipper":    flipper,
        "scaler":     scaler,
        "imputer":    imputer,
        "classifier": classifier,
        "config":     cfg,
    }


# ===========================================================================
# Inference
# ===========================================================================

def run_pipeline(df: pd.DataFrame, pipeline: dict[str, Any]) -> pd.DataFrame:
    selector   = pipeline["selector"]
    flipper    = pipeline["flipper"]
    scaler     = pipeline["scaler"]
    imputer    = pipeline["imputer"]
    classifier = pipeline["classifier"]

    X = selector.transform(df)
    X = flipper.transform(X)
    # MinMaxScaler returns a numpy array — restore DataFrame with column names
    # so IterativeImputer receives the feature names it was fitted with
    feature_cols = pipeline["config"]["feature_cols"]
    X = pd.DataFrame(
        scaler.transform(X),
        columns=feature_cols,
        index=df.index,
    )
    X[:] = imputer.transform(X)
    feature_names = classifier._inference_feature_names
    X_arr = X.astype(np.float32) if isinstance(X, np.ndarray) else X.values.astype(np.float32)
    dmat  = xgb.DMatrix(X_arr, feature_names=feature_names)
    proba = classifier.predict(dmat)

    cols = X.columns.intersection(df.columns)
    df["PredictSNPProt"] = proba

    logger.info(f"  Processed {len(df)} rows.")
    return df


# ===========================================================================
# Matrix output
# ===========================================================================
def save_as_matrix(df: pd.DataFrame, output_path: str) -> None:
    matrix = (
        df[["WT residue", "Position", "MT", "proba"]]
        .drop_duplicates(subset=["Position", "MT"])
        .pivot(index=["Position", "WT residue"], columns="MT", values="proba")
        .reset_index()
    )

    for aa in AA_ORDER:
        if aa not in matrix.columns:
            matrix[aa] = np.nan

    matrix = matrix[["WT residue", "Position"] + AA_ORDER].sort_values("Position")
    matrix[AA_ORDER] = matrix[AA_ORDER].apply(pd.to_numeric, errors="coerce")

    matrix.to_csv(output_path, index=False, na_rep="NaN")

# ===========================================================================
# CLI
# ===========================================================================
def parse_args():

    parser = argparse.ArgumentParser(
        description="Apply reconstructed JSON pipeline to new input data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch-input", required=False,
                        help="Input CSV with raw feature scores.")
    parser.add_argument("--output", required=False,
                        help="Output CSV path for predictions.")
    parser.add_argument("--cols", action="append", default=[],
                        help="Columns to be stored in output csv")
    parser.add_argument("--matrix", required=False,
                        help="Path for matrix format output")
    parser.add_argument("--artifacts-dir", default=None,
                        help="Override default artifact directory,")
    return parser.parse_args()

def main():
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    # Resolve artifact directory
    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir)
    else:
        artifacts_dir = PIPELINE_VARIANTS["full"]

    # Load pipeline
    pipeline = load_pipeline(artifacts_dir)

    if args.batch_input:
        df = pd.read_csv(args.batch_input)
        logger.info(f"Loaded existing input CSV: {args.batch_input} ({len(df)} rows)")
    else:
        logger.error(f"No input method selected! Exiting now!")
        sys.exit(-1)

    out = run_pipeline(df, pipeline)

    if args.output:
        if args.cols:
            out = out[args.cols]
        out.to_csv(args.output, index=False)
        logger.info(f"Output: {args.output} ({len(out)} rows)")
    if args.matrix:
        save_as_matrix(out, args.matrix)

if __name__ == "__main__":
    main()
