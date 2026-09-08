#!/usr/bin/env python3
"""
convert_artifact.py
-------------------
Converts the new unified classifier.json (produced by train_classifier.py)
into the three-file/folder structure expected by apply_pipeline.py:

    <outdir>/
        pipeline_config.json           ← top-level config with all pipeline metadata
        classifier.json                ← XGBoost native model (booster format)
        imputer_estimators/
            round_0_entry_0.json       ← BayesianRidge for feature 0
            round_0_entry_1.json       ← BayesianRidge for feature 1
            ...

The XGBoost model is written as a proper native JSON booster file so that
apply_pipeline.py can load it with xgb.Booster().load_model(path).

Usage
-----
    python convert_artifact.py \
        --input  fast_pipeline_artifacts_classifier.json \
        --outdir pipeline_artifacts_fast \
        [--variant fast]
        [--version 2.0]
"""

from __future__ import annotations

import argparse
import json
import tempfile
import os
from pathlib import Path

import numpy as np

try:
    import xgboost as xgb
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _save(obj: dict, path: Path, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)
    print(f"  wrote  {path}  ({path.stat().st_size / 1024:.1f} KB)")


# ═══════════════════════════════════════════════════════════════════════════
# Imputer conversion
# ═══════════════════════════════════════════════════════════════════════════

def _convert_imputer(
    imp: dict,
    outdir: Path,
) -> dict:
    """
    Explode the inline imputation_sequence_ into per-feature files under
    outdir/imputer_estimators/round_0_entry_N.json and return the
    pipeline_config.json-compatible imputer dict.

    The old format groups entries by round. Our new format stores the
    full flattened sequence (one entry per feature per iteration).
    Since n_iter_ == 1 in the fast pipeline, all entries belong to round 0.
    """
    est_dir = outdir / "imputer_estimators"
    est_dir.mkdir(parents=True, exist_ok=True)

    sequence   = imp["imputation_sequence_"]
    n_features = imp["n_features_in_"]
    n_iter     = imp.get("n_iter_", 1)
    entries_per_round = len(sequence) // n_iter

    rounds = []
    for round_idx in range(n_iter):
        round_entries = []
        for entry_idx in range(entries_per_round):
            seq_idx   = round_idx * entries_per_round + entry_idx
            entry     = sequence[seq_idx]
            est       = entry["estimator"]
            filename  = f"round_{round_idx}_entry_{entry_idx}.json"
            est_path  = est_dir / filename

            br_payload = {
                "coef_":      est["coef_"],
                "intercept_": est["intercept_"],
                "alpha_":     est["alpha_"],
                "lambda_":    est["lambda_"],
                "sigma_":     est["sigma_"],
            }
            _save(br_payload, est_path)

            round_entries.append({
                "feat_idx":            entry["feat_idx"],
                "neighbor_feat_indices": entry["neighbor_feat_idx"],
                "estimator_type":      "BayesianRidge",
                "estimator_file":      f"imputer_estimators/{filename}",
            })
        rounds.append(round_entries)

    # Reconstruct initial_imputer config
    init_imp = {
        "strategy":         imp.get("initial_strategy", "mean"),
        "statistics_":      imp.get("initial_statistics_", []),
        "n_features_in_":   n_features,
        "feature_names_in_": imp.get("feature_names", []),
    }

    return {
        "max_iter":          imp.get("max_iter", 10),
        "tol":               1e-3,                        # sklearn default
        "n_nearest_features": None,                        # use all
        "initial_strategy":  imp.get("initial_strategy", "mean"),
        "imputation_order":  "ascending",                  # sklearn default
        "n_iter_":           n_iter,
        "n_features_in_":    n_features,
        "feature_names_in_": imp.get("feature_names", []),
        "n_rounds":          n_iter,
        "entries_per_round": entries_per_round,
        "initial_imputer":   init_imp,
        "rounds":            rounds,
    }


# ═══════════════════════════════════════════════════════════════════════════
# XGBoost conversion
# ═══════════════════════════════════════════════════════════════════════════

def _convert_xgb(clf: dict, outdir: Path) -> dict:
    """
    Write the XGBoost native model to outdir/classifier.json
    and return the classifier sub-dict for pipeline_config.json.

    The new format stores the full native JSON tree inline under
    xgb_native_model. We round-trip it through xgb.Booster to get a
    proper loadable file.
    """
    if not _XGB_AVAILABLE:
        raise ImportError("xgboost is required to convert the classifier.  pip install xgboost")

    native_model = clf["xgb_native_model"]
    model_bytes  = json.dumps(native_model).encode("utf-8")

    booster = xgb.Booster()
    # load_model accepts a file path or a bytearray; write to a temp file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp.write(model_bytes)
        tmp_path = tmp.name

    try:
        booster.load_model(tmp_path)
    finally:
        os.unlink(tmp_path)

    model_out = outdir / "classifier.json"
    booster.save_model(str(model_out))
    print(f"  wrote  {model_out}  ({model_out.stat().st_size / 1024:.1f} KB)")

    return {
        "model_file":    "classifier.json",
        "classes":       clf.get("classes_", [0, 1]),
        "n_features_in_": clf.get("n_features_in_", len(clf.get("feature_names_in_", []))),
        "feature_names": clf.get("feature_names_in_", []),
        "best_params":   clf.get("best_params", {}),
        "best_cv_score": clf.get("best_cv_score", None),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Main converter
# ═══════════════════════════════════════════════════════════════════════════

def convert(
    input_path: str | Path,
    outdir:     str | Path,
    variant:    str = "fast",
    version:    str = "2.0",
) -> None:
    """
    Convert a unified classifier.json to the three-file apply_pipeline.py format.

    Parameters
    ----------
    input_path : path to the new-format classifier.json
    outdir     : output directory (will be created); maps to a PIPELINE_VARIANTS entry
    variant    : "fast" or "full" — stored in pipeline_config.json for informational use
    version    : version string stored in pipeline_config.json
    """
    input_path = Path(input_path)
    outdir     = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"Converting  {input_path}")
    print(f"      →     {outdir}/")

    art   = _load(input_path)
    steps = art["pipeline_steps"]

    sel     = steps["feature_selector"]
    flipper = steps["column_flipper"]
    scaler  = steps["scaler"]
    imp     = steps["imputer"]
    clf     = steps["classifier"]

    # ── imputer → files ───────────────────────────────────────────────────
    print("[1/3] Exploding imputer BayesianRidge estimators …")
    imputer_cfg = _convert_imputer(imp, outdir)

    # ── XGBoost model → file ──────────────────────────────────────────────
    print("[2/3] Writing XGBoost booster model …")
    classifier_cfg = _convert_xgb(clf, outdir)

    # ── pipeline_config.json ──────────────────────────────────────────────
    print("[3/3] Writing pipeline_config.json …")

    # Scaler: copy fields, rename n_samples_seen_ if needed
    scaler_cfg = {
        "feature_range":   scaler.get("feature_range", [0.0, 1.0]),
        "clip":            scaler.get("clip", True),
        "scale":           scaler["scale"],
        "min":             scaler["min"],
        "data_min":        scaler["data_min"],
        "data_max":        scaler["data_max"],
        "data_range":      scaler["data_range"],
        "n_samples_seen":  scaler.get("n_samples_seen", scaler.get("n_samples_seen_", 2)),
        "feature_names":   scaler["feature_names"],
    }

    from datetime import datetime, timezone
    pipeline_config = {
        "version":          version,
        "created":          datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pipeline_variant": variant,
        # ── step configs (apply_pipeline.py reads these directly) ──────────
        "feature_cols":     sel["feature_cols"],
        "cols_to_flip":     flipper["cols_to_flip"],
        "scaler":           scaler_cfg,
        "imputer":          imputer_cfg,
        "classifier":       classifier_cfg,
        # ── training provenance (informational) ───────────────────────────
        "training_meta": {
            "best_model":         art.get("best_model"),
            "best_model_type":    art.get("best_model_type"),
            "cv_folds":           art.get("cv_folds"),
            "random_state":       art.get("random_state"),
            "n_train_samples":    art.get("n_train_samples"),
            "class_distribution": art.get("class_distribution"),
            "scores":             art.get("scores"),
            "best_params":        art.get("best_params"),
        },
    }

    _save(pipeline_config, outdir / "pipeline_config.json")

    # ── summary ───────────────────────────────────────────────────────────
    n_est_files = sum(1 for _ in (outdir / "imputer_estimators").glob("*.json"))
    print(f"✔  Conversion complete")
    print(f"   {outdir}/")
    print(f"   ├── pipeline_config.json")
    print(f"   ├── classifier.json")
    print(f"   └── imputer_estimators/  ({n_est_files} BayesianRidge files)")
    print(f"Features  : {sel['feature_cols']}")
    print(f"   Flip cols : {flipper['cols_to_flip']}")
    print(f"   Imputer   : {imp.get('imputer_type')}  "
          f"(n_iter={imp.get('n_iter_')}, n_features={imp.get('n_features_in_')})")
    print(f"   Classifier: {clf.get('classifier_class')}  "
          f"(CV AUC={clf.get('best_cv_score', art.get('scores', {}).get('cv_auc', 'n/a'))})")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Convert a unified train_classifier.py artifact (classifier.json) "
            "into the apply_pipeline.py three-file directory structure."
        )
    )
    p.add_argument("--input",   required=True,
                   help="Path to new-format classifier.json")
    p.add_argument("--outdir",  required=True,
                   help="Output directory for pipeline_config.json, "
                        "classifier.json, imputer_estimators/")
    p.add_argument("--variant", default="fast", choices=["fast", "full"],
                   help="Pipeline variant label stored in pipeline_config.json")
    p.add_argument("--version", default="2.0",
                   help="Version string stored in pipeline_config.json")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    convert(
        input_path = args.input,
        outdir     = args.outdir,
        variant    = args.variant,
        version    = args.version,
    )


if __name__ == "__main__":
    main()
