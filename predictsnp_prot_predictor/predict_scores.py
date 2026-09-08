#!/usr/bin/env python3
"""
apply_pipeline.py
=================
Reconstructs the fitted sklearn Pipeline from JSON artifacts (no pkl needed)
and applies it to new input data.

Pipeline: FeatureSelector -> ColumnFlipper -> MinMaxScaler -> IterativeImputer -> XGBClassifier

CLI usage:
    python predict_scores.py --input data.csv --variant full --output predictions.csv
    python predict_scores.py --input data.csv --variant fast --output predictions.csv

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

PIPELINE_VARIANTS: dict[str, Path] = {
    "full": _HERE / "pipeline_artifacts_full",
    "fast": _HERE / "pipeline_artifacts_fast",
}

REQUIRED_COLS = {"Position", "WT residue", "MT"}

ALLOWED_PREDICTORS = {
    "3cnet",
    "esmVariants",
    "provean",
    "vespal",
    "varipred",
    "vespag",
    "phact",
    "phactBoost",
    "gemme",
    "fullMetaModel",
    "fastMetaModel"
}


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
        "variant"    : "full" | "fast"
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
    logger.info(f"  Variant  : {cfg.get('pipeline_variant', 'unknown')}")
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
        "variant":    cfg['pipeline_variant'],
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
    out = df.drop(columns=cols).merge(X, left_index=True, right_index=True, how="left")
    out["proba"]           = proba
    out["predicted_label"] = (proba >= 0.5).astype(int)

    n_path   = int((out["predicted_label"] == 1).sum())
    n_benign = int((out["predicted_label"] == 0).sum())
    logger.info(
        f"  Processed {len(out)} rows -> "
        f"{n_path} pathogenic ({100 * n_path / len(out):.1f}%), "
        f"{n_benign} benign ({100 * n_benign / len(out):.1f}%)"
    )
    return out


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
# vvvv this is needed, as core passes predictors as a single str, not individual tokens
# we split it and let argparse process normally
PREDICTORS_PATTERN = "|".join(re.escape(p) for p in sorted(ALLOWED_PREDICTORS))
PREDICTOR_BLOB_RE = re.compile(rf"^(?:--({PREDICTORS_PATTERN})\b\s\S+\s?)+$")

def normalize_argv(argv: list[str]) -> list[str]:
    out = []
    for arg in argv:
        if isinstance(arg, str) and PREDICTOR_BLOB_RE.match(arg):
            out.extend(shlex.split(arg))
        else:
            out.append(arg)
    return out


class PredictorAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        name = self.dest  # e.g. "phact", "vespag"
        if not hasattr(namespace, 'predictor_paths') or namespace.predictor_paths is None:
            namespace.predictor_paths = {}
        namespace.predictor_paths[name] = Path(values)


def parse_args():
    argv = normalize_argv(sys.argv[1:])

    parser = argparse.ArgumentParser(
        description="Apply reconstructed JSON pipeline to new input data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch-input", required=False,
                        help="Input CSV with raw feature scores.")
    parser.add_argument("--output", required=False,
                        help="Output CSV path for predictions.")
    parser.add_argument("--variant", choices=["full", "fast"], default="full",
                        help="Pipeline variant: full (9 features) or fast (5 features).")
    parser.add_argument("--cols", action="append", default=[],
                        help="Columns to be stored in output csv")
    parser.add_argument("--matrix", required=False,
                        help="Path for matrix format output")
    parser.add_argument("--artifacts-dir", default=None,
                        help="Override default artifact directory for the chosen variant.")

    for name in ALLOWED_PREDICTORS:
        parser.add_argument(f"--{name}", dest=name, metavar="PATH", action=PredictorAction, default=argparse.SUPPRESS, help=f"Path to {name} predictor CSV")

    return parser.parse_args(argv)


def main():
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    # Resolve artifact directory
    if args.artifacts_dir:
        artifacts_dir = Path(args.artifacts_dir)
    else:
        artifacts_dir = PIPELINE_VARIANTS[args.variant]

    # Load pipeline
    pipeline = load_pipeline(artifacts_dir)

    if args.batch_input:
        df = pd.read_csv(args.batch_input)
        logger.info(f"Loaded existing input CSV: {args.batch_input} ({len(df)} rows)")
    elif args.predictor_paths:
        df = merge_predictors(args, args.predictor_paths, ALLOWED_PREDICTORS)
    else:
        logger.error(f"No input method selected! Exiting now!")
        sys.exit(-1)

    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        logger.error(f"Columns {missing_cols} not present in predictor CSV!")
        sys.exit(-1)


    out = run_pipeline(df, pipeline)

    if args.output:
        if args.cols:
            out = out[args.cols]
        out.to_csv(args.output, index=False)
        logger.info(f"Output: {args.output} ({len(out)} rows, columns: proba, predicted_label)")
    if args.matrix:
        save_as_matrix(out, args.matrix)

if __name__ == "__main__":
    main()
