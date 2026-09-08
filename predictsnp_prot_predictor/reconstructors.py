from __future__ import annotations

import json
from collections import namedtuple
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import MinMaxScaler

from predictsnp_prot_predictor.custom_transformers import FeatureSelector, ColumnFlipper

# ---------------------------------------------------------------------------
# _ImputerTriplet
# Strategy A: import sklearn's internal namedtuple so imputation_sequence_
# has the exact type sklearn expects. Fallback for version safety.
# ---------------------------------------------------------------------------

_ImputerTriplet = namedtuple(
    "_ImputerTriplet", ["feat_idx", "neighbor_feat_idx", "estimator"]
)

# ===========================================================================
# Step reconstructors
# ===========================================================================

def reconstruct_feature_selector(cfg: dict[str, Any]) -> FeatureSelector:
    sel = FeatureSelector(feature_cols=cfg["feature_cols"])
    sel.feature_names_in_ = np.array(cfg["feature_cols"])
    return sel


def reconstruct_column_flipper(cfg: dict[str, Any]) -> ColumnFlipper:
    return ColumnFlipper(cols_to_flip=cfg["cols_to_flip"])


def reconstruct_scaler(cfg: dict[str, Any]) -> MinMaxScaler:
    scaler = MinMaxScaler(
        feature_range=tuple(cfg["feature_range"]),
        clip=cfg["clip"],
    )
    scaler.scale_            = np.array(cfg["scale"])
    scaler.min_              = np.array(cfg["min"])
    scaler.data_min_         = np.array(cfg["data_min"])
    scaler.data_max_         = np.array(cfg["data_max"])
    scaler.data_range_       = np.array(cfg["data_range"])
    scaler.n_samples_seen_   = cfg["n_samples_seen"]
    scaler.n_features_in_    = len(cfg["feature_names"])
    scaler.feature_names_in_ = np.array(cfg["feature_names"])
    return scaler


def _reconstruct_bayesian_ridge(path: Path) -> BayesianRidge:
    with open(path) as f:
        data = json.load(f)
    br                = BayesianRidge()
    br.coef_          = np.array(data["coef_"])
    br.intercept_     = float(data["intercept_"])
    br.alpha_         = float(data["alpha_"])
    br.lambda_        = float(data["lambda_"])
    br.n_features_in_ = len(data["coef_"])
    if "sigma_" in data:
        br.sigma_  = np.array(data["sigma_"])
    if "n_iter_" in data:
        br.n_iter_ = data["n_iter_"]
    if "scores_" in data:
        br.scores_ = np.array(data["scores_"])
    return br


def reconstruct_imputer(
    cfg: dict[str, Any], artifacts_dir: Path
) -> IterativeImputer:
    imputer = IterativeImputer(
        max_iter           = cfg["max_iter"],
        tol                = cfg["tol"],
        n_nearest_features = cfg["n_nearest_features"],
        initial_strategy   = cfg["initial_strategy"],
        imputation_order   = cfg["imputation_order"],
    )

    # --- Scalar fitted attributes ---
    imputer.n_iter_           = cfg["n_iter_"]
    imputer.n_features_in_    = cfg["n_features_in_"]
    imputer.feature_names_in_ = np.array(cfg["feature_names_in_"])

    # --- Attributes set by fit_transform, needed by transform() ---
    n_feat = cfg["n_features_in_"]

    # True for features where ALL rows were missing during fit — none in our case
    imputer._is_empty_feature = np.zeros(n_feat, dtype=bool)

    # Per-feature clipping bounds — config shows min=-inf, max=+inf (no clipping)
    imputer._min_value = np.full(n_feat, -np.inf)
    imputer._max_value = np.full(n_feat,  np.inf)

    # --- Reconstruct initial_imputer_ (SimpleImputer) ---
    init_cfg                   = cfg["initial_imputer"]
    init_imp                   = SimpleImputer(strategy=init_cfg["strategy"])
    init_imp.statistics_       = np.array(init_cfg["statistics_"])
    init_imp.n_features_in_    = init_cfg["n_features_in_"]
    init_imp._fit_dtype        = np.dtype("float64")
    init_imp._fill_dtype       = np.dtype("float64")
    if init_cfg.get("feature_names_in_") is not None:
        init_imp.feature_names_in_ = np.array(init_cfg["feature_names_in_"])
    imputer.initial_imputer_ = init_imp

    # --- Reconstruct imputation_sequence_ ---
    sequence: list = []
    for round_entries in cfg["rounds"]:
        for entry in round_entries:
            est_path = artifacts_dir / entry["estimator_file"]
            if entry["estimator_type"] == "BayesianRidge":
                estimator = _reconstruct_bayesian_ridge(est_path)
            else:
                raise NotImplementedError(
                    f"Unsupported estimator type: {entry['estimator_type']}. "
                    "Only BayesianRidge is currently supported."
                )
            triplet = _ImputerTriplet(
                feat_idx          = entry["feat_idx"],
                neighbor_feat_idx = np.array(entry["neighbor_feat_indices"]),
                estimator         = estimator,
            )
            sequence.append(triplet)

    imputer.imputation_sequence_ = sequence
    return imputer


def reconstruct_xgb(path: Path, cfg: dict[str, Any]) -> xgb.Booster:
    booster = xgb.Booster()
    booster.load_model(str(path))
    # Attach metadata used at predict time
    booster._inference_classes       = np.array(cfg["classes"])
    booster._inference_n_features    = cfg["n_features_in_"]
    booster._inference_feature_names = [str(f) for f in cfg["feature_names"]]
    return booster