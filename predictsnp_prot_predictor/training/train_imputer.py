"""
train_imputer.py
----------------
Selects the best imputation strategy (Mean / Median / IterativeImputer) via
CV reconstruction RMSE and serialises the fitted imputer to JSON.

Pipeline contract
-----------------

The scaler artefact from train_scaler.py is required as input so the imputer
is always trained on the same scaled space it will see at inference time.

Every step accepts and returns a pd.DataFrame with named columns.

Usage
-----
python train_imputer.py \
    --trainset data/splits/trainset.csv \
    --scaler models/pipeline_A_scaler.json \
    --out models/pipeline_A_imputer.json

Optional overrides:
    --cv-folds 10
    --val-masks 5
    --max-iter 20
    --random-state 0

missing_proba (per-feature probability used for CV masking) can be added to
the pipeline config JSON under the key "missing_proba":
    { ..., "missing_proba": {"feat1": 0.1, "feat2": 0.3} }
If absent, a uniform 0.1 is used for all features.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.impute._iterative import _ImputerTriplet
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold

from train_scaler import load_preprocessing_steps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# DataFrame-preserving imputer wrapper
# ===========================================================================

class DataFrameImputer(BaseEstimator, TransformerMixin):
    """
    DataFrame-preserving wrapper around sklearn SimpleImputer / IterativeImputer.

    Guarantees that transform() always returns a pd.DataFrame with the
    original column names and index intact sklearn's imputers return
    numpy arrays by default, silently dropping column names.

    Input : pd.DataFrame (may contain NaN)
    Output : pd.DataFrame (same columns and index, NaN filled)
    """

    def __init__(self, imputer):
        self.imputer = imputer

    def fit(self, X: pd.DataFrame, y=None) -> "DataFrameImputer":
        self._validate(X)
        self.feature_names_in_ = np.array(X.columns.tolist())
        self.imputer.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate(X)
        cols  = X.columns.tolist()
        index = X.index
        arr   = self.imputer.transform(X)  # np.ndarray from sklearn
        return pd.DataFrame(arr, columns=cols, index=index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is not None:
            return np.array(input_features)
        return self.feature_names_in_

    def _validate(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"DataFrameImputer expects a DataFrame, got {type(X).__name__}.")


# ===========================================================================
# Missing-value injection (mirrors old pipeline script)
# ===========================================================================

def introduce_missing_values(
    X: pd.DataFrame,
    missing_proba: dict[str, float],
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Introduce NaN into X according to per-feature probabilities.
    Features absent from X are silently skipped.
    """
    rng   = np.random.RandomState(random_state)
    X_aug = X.copy()
    for feat, prob in missing_proba.items():
        if feat not in X_aug.columns or prob <= 0:
            continue
        mask = rng.rand(len(X_aug)) < prob
        X_aug.loc[mask, feat] = np.nan
    return X_aug


# ===========================================================================
# CV reconstruction RMSE
# ===========================================================================

def evaluate_imputers_cv(
    X_scaled: pd.DataFrame,
    missing_proba: dict[str, float],
    imputer_configs: list[dict] | None = None,
    n_cv_folds: int = 5,
    n_val_masks: int = 3,
    random_state: int = 42,
    max_iter: int = 10,
) -> pd.DataFrame:
    """
    Compare imputation strategies via CV reconstruction RMSE.

    For each KFold split:
    - Fit each candidate imputer on the clean training fold.
    - Apply n_val_masks independent random masks to the validation fold.
    - Impute and measure RMSE / MAE against the known true values.

    The same masks are applied to all candidates so comparisons are fair.
    """
    features = list(missing_proba.keys())

    if imputer_configs is None:
        imputer_configs = [
            {"label": "Mean",
             "imputer": SimpleImputer(strategy="mean")},
            {"label": "Median",
             "imputer": SimpleImputer(strategy="median")},
            {"label": "IterativeImputer",
             "imputer": IterativeImputer(max_iter=max_iter,
                                         random_state=random_state,
                                         verbose=0)},
        ]

    kf = KFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
    scores = {
        cfg["label"]: {"rmse": [], "mae": [], "feat_rmse": {f: [] for f in features}}
        for cfg in imputer_configs
    }

    print(f"\n{'='*70}")
    print("IMPUTER SELECTION CV RECONSTRUCTION EVALUATION")
    print(f"  Folds: {n_cv_folds}  Val masks/fold: {n_val_masks}  "
          f"Features evaluated: {len(features)}")
    print(f"{'='*70}\n")

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_scaled)):
        X_fold_tr  = X_scaled.iloc[tr_idx].reset_index(drop=True)
        X_fold_val = X_scaled.iloc[val_idx].reset_index(drop=True)

        val_masks = []
        for j in range(n_val_masks):
            seed     = random_state + 99999 + fold_idx * 1000 + j
            X_masked = introduce_missing_values(X_fold_val, missing_proba, seed)
            nan_flags = X_masked[features].isna()
            val_masks.append((X_masked, nan_flags))

        for cfg in imputer_configs:
            label   = cfg["label"]
            imputer = clone(cfg["imputer"])
            imputer.fit(X_fold_tr)

            for X_val_masked, nan_flags in val_masks:
                if nan_flags.values.sum() == 0:
                    continue
                X_imp = pd.DataFrame(
                    imputer.transform(X_val_masked),
                    columns=X_val_masked.columns,
                )
                all_sq, all_ae = [], []
                for feat in features:
                    feat_mask = nan_flags[feat]
                    if feat_mask.sum() == 0:
                        continue
                    y_true = X_fold_val.loc[feat_mask, feat].values
                    y_pred = X_imp.loc[feat_mask, feat].values
                    sq = (y_true - y_pred) ** 2
                    ae = np.abs(y_true - y_pred)
                    scores[label]["feat_rmse"][feat].append(np.sqrt(np.mean(sq)))
                    all_sq.extend(sq)
                    all_ae.extend(ae)
                if all_sq:
                    scores[label]["rmse"].append(np.sqrt(np.mean(all_sq)))
                    scores[label]["mae"].append(np.mean(all_ae))

    rows = []
    for cfg in imputer_configs:
        lbl = cfg["label"]
        s   = scores[lbl]
        rows.append({
            "label":     lbl,
            "mean_rmse": float(np.mean(s["rmse"])),
            "std_rmse":  float(np.std(s["rmse"])),
            "mean_mae":  float(np.mean(s["mae"])),
            "std_mae":   float(np.std(s["mae"])),
            "per_feature_rmse": {
                f: float(np.mean(v)) if v else float("nan")
                for f, v in s["feat_rmse"].items()
            },
        })

    results = (
        pd.DataFrame(rows)
        .sort_values("mean_rmse")
        .reset_index(drop=True)
    )

    print(f"  {'Rank':<5} {'Imputer':<35} {'RMSE (mean±std)':>20} {'MAE':>12}")
    print(f"  {'-'*76}")
    for rank, row in results.iterrows():
        print(f"  {rank+1:<5} {row['label']:<35} "
              f"{row['mean_rmse']:>8.4f} ± {row['std_rmse']:<8.4f} "
              f"{row['mean_mae']:>8.4f} ± {row['std_mae']:<8.4f}")

    print(f"\n  Per-feature RMSE:")
    hdr = f"  {'Feature':<25}" + "".join(
        f"  {r['label'][:13]:>15}" for _, r in results.iterrows()
    )
    print(hdr)
    print(f"  {'-' * (len(hdr) - 2)}")
    for feat in features:
        line = f"  {feat:<25}"
        for _, row in results.iterrows():
            val = row["per_feature_rmse"].get(feat, float("nan"))
            line += f"  {val:>15.4f}"
        print(line)

    print(f"\nBest imputer by RMSE: {results.iloc[0]['label']}\n")
    return results


# ===========================================================================
# Serialisation captures every private attr needed by transform()
# ===========================================================================

def _serialise_simple_imputer(inner: SimpleImputer, feature_names: list[str]) -> dict:
    """
    Serialise a fitted SimpleImputer.
    Captures _fit_dtype (sklearn < 1.4) / _fill_dtype (sklearn >= 1.4).
    Both names are stored under fit_dtype_kind so reconstruction can set both.
    """
    # Support both sklearn naming conventions
    dtype_attr = "_fit_dtype" if hasattr(inner, "_fit_dtype") else "_fill_dtype"
    return {
        "imputer_type":   "SimpleImputer",
        "strategy":       inner.strategy,
        "statistics_":    inner.statistics_.tolist(),
        "fit_dtype_kind": getattr(inner, dtype_attr).kind,
        "feature_names":  feature_names,
    }


def _serialise_bayesian_ridge(est: BayesianRidge) -> dict:
    return {
        "coef_":      est.coef_.tolist(),
        "intercept_": float(est.intercept_),
        "alpha_":     float(est.alpha_),
        "lambda_":    float(est.lambda_),
        "sigma_":     np.array(est.sigma_).tolist(),
    }


def _serialise_iterative_imputer(
    inner: IterativeImputer, feature_names: list[str]
) -> dict:
    """
    Serialise a fitted IterativeImputer.
    Captures every private attribute that transform() consults.
    """
    seq = []
    for feat_idx, predictor_cols, est in inner.imputation_sequence_:
        seq.append({
            "feat_idx":          int(feat_idx),
            "neighbor_feat_idx": predictor_cols.tolist(),
            "estimator":         _serialise_bayesian_ridge(est),
        })

    # Support both sklearn naming conventions for the initial imputer dtype
    init = inner.initial_imputer_
    dtype_attr = "_fit_dtype" if hasattr(init, "_fit_dtype") else "_fill_dtype"

    return {
        "imputer_type":             "IterativeImputer",
        "max_iter":                 inner.max_iter,
        "n_iter_":                  int(inner.n_iter_),
        "n_features_in_":           int(inner.n_features_in_),
        "n_features_with_missing_": int(inner.n_features_with_missing_),
        "initial_strategy":         init.strategy,
        "initial_statistics_":      init.statistics_.tolist(),
        "initial_fit_dtype_kind":   getattr(init, dtype_attr).kind,
        "imputation_sequence_":     seq,
        "_min_value":               inner._min_value.tolist(),
        "_max_value":               inner._max_value.tolist(),
        "_is_empty_feature":        inner._is_empty_feature.tolist(),
        "feature_names":            feature_names,
    }


def serialise_imputer(imputer: DataFrameImputer, feature_names: list[str]) -> dict:
    inner = imputer.imputer
    if isinstance(inner, SimpleImputer):
        body = _serialise_simple_imputer(inner, feature_names)
    elif isinstance(inner, IterativeImputer):
        body = _serialise_iterative_imputer(inner, feature_names)
    else:
        raise TypeError(f"Unsupported inner imputer: {type(inner).__name__}")
    body["step"] = "DataFrameImputer"
    return body


# ===========================================================================
# Reconstruction mirrors serialisation exactly
# ===========================================================================

def _set_dtype_attrs(obj, dtype: np.dtype) -> None:
    """
    Set both _fit_dtype and _fill_dtype on a SimpleImputer instance.

    sklearn renamed _fit_dtype _fill_dtype between versions. Setting both
    ensures the reconstructed imputer works regardless of installed version.
    """
    obj._fit_dtype  = dtype  # sklearn < 1.4
    obj._fill_dtype = dtype  # sklearn >= 1.4


def _reconstruct_simple_imputer(cfg: dict) -> SimpleImputer:
    feature_names = cfg["feature_names"]
    inner = SimpleImputer(strategy=cfg["strategy"])
    inner.statistics_        = np.array(cfg["statistics_"])
    inner.n_features_in_     = len(feature_names)
    inner.feature_names_in_  = np.array(feature_names)
    inner.indicator_         = None
    _set_dtype_attrs(inner, np.dtype(cfg["fit_dtype_kind"]))
    return inner


def _reconstruct_bayesian_ridge(cfg: dict) -> BayesianRidge:
    est            = BayesianRidge()
    est.coef_      = np.array(cfg["coef_"])
    est.intercept_ = float(cfg["intercept_"])
    est.alpha_     = float(cfg["alpha_"])
    est.lambda_    = float(cfg["lambda_"])
    est.sigma_     = np.array(cfg["sigma_"])
    return est


def _reconstruct_iterative_imputer(cfg: dict) -> IterativeImputer:
    feature_names = cfg["feature_names"]
    inner = IterativeImputer(max_iter=cfg["max_iter"])

    # Public fitted attributes
    inner.n_features_in_          = cfg["n_features_in_"]
    inner.n_features_with_missing_ = cfg["n_features_with_missing_"]
    inner.n_iter_                 = cfg["n_iter_"]
    inner.feature_names_in_       = np.array(feature_names)
    inner.indicator_              = None

    # Private attributes consulted during transform()
    inner._min_value        = np.array(cfg["_min_value"])
    inner._max_value        = np.array(cfg["_max_value"])
    inner._is_empty_feature = np.array(cfg["_is_empty_feature"])
    inner._estimator        = BayesianRidge()  # prototype only

    # Reconstruct initial_imputer_ do NOT set feature_names_in_ here;
    # IterativeImputer calls it internally with a numpy array and sklearn
    # warns when feature names are present but input is an array.
    init = SimpleImputer(strategy=cfg["initial_strategy"])
    init.statistics_    = np.array(cfg["initial_statistics_"])
    init.n_features_in_ = len(feature_names)
    init.indicator_     = None
    _set_dtype_attrs(init, np.dtype(cfg["initial_fit_dtype_kind"]))
    inner.initial_imputer_ = init

    # Reconstruct imputation_sequence_
    seq = []
    for entry in cfg["imputation_sequence_"]:
        seq.append(_ImputerTriplet(
            feat_idx          = np.intp(entry["feat_idx"]),
            neighbor_feat_idx = np.array(entry["neighbor_feat_idx"]),
            estimator         = _reconstruct_bayesian_ridge(entry["estimator"]),
        ))
    inner.imputation_sequence_ = seq

    return inner


def _reconstruct_imputer(cfg: dict) -> DataFrameImputer:
    """Reconstruct a DataFrameImputer from its serialised JSON dict."""
    feature_names = cfg["feature_names"]

    if cfg["imputer_type"] == "SimpleImputer":
        inner = _reconstruct_simple_imputer(cfg)
    elif cfg["imputer_type"] == "IterativeImputer":
        inner = _reconstruct_iterative_imputer(cfg)
    else:
        raise ValueError(f"Unknown imputer_type: {cfg['imputer_type']}")

    wrapper = DataFrameImputer(inner)
    wrapper.feature_names_in_ = np.array(feature_names)
    return wrapper


def load_imputer_steps(
    scaler_path: str | Path,
    imputer_path: str | Path,
) -> tuple:
    """
    Load all four preprocessing steps from their JSON artefacts.

    Returns
    -------
    (selector, flipper, scaler, imputer)
    All four steps accept and return pd.DataFrame with named columns.

    Example
    -------
    selector, flipper, scaler, imputer = load_imputer_steps(
        "models/pipeline_A_scaler.json",
        "models/pipeline_A_imputer.json",
    )
    X_ready = imputer.transform(
                  scaler.transform(
                      flipper.transform(
                          selector.transform(df))))
    """
    selector, flipper, scaler = load_preprocessing_steps(scaler_path)
    with open(imputer_path) as f:
        art = json.load(f)
    imputer = _reconstruct_imputer(art["imputer"])
    log.info("Loaded all four preprocessing steps from artefacts")
    return selector, flipper, scaler, imputer


# ===========================================================================
# Main training function (importable + CLI)
# ===========================================================================

def train_imputer(
    trainset_path: str | Path,
    scaler_artefact_path: str | Path,
    out_path: str | Path,
    missing_proba: dict[str, float] | None = None,
    n_cv_folds: int = 5,
    n_val_masks: int = 3,
    random_state: int = 42,
    max_iter: int = 10,
) -> DataFrameImputer:
    """
    Full imputer training flow.

    1. Load preprocessing steps from the scaler artefact.
    2. Apply selecto flipper scaler to the raw trainset.
    3. Evaluate Mean / Median / IterativeImputer via CV reconstruction RMSE.
    4. Fit the winning strategy on the full scaled trainset.
    5. Serialise all fitted attributes to JSON.
    """
    selector, flipper, scaler = load_preprocessing_steps(scaler_artefact_path)
    feature_cols = selector.feature_cols
    log.info("Preprocessing steps loaded from %s", scaler_artefact_path)

    train_df = pd.read_csv(trainset_path)
    log.info("Raw trainset: %d rows %d cols", *train_df.shape)

    X_scaled = scaler.transform(flipper.transform(selector.transform(train_df)))
    assert isinstance(X_scaled, pd.DataFrame), "Preprocessing chain must return DataFrame"
    assert list(X_scaled.columns) == feature_cols, "Column order must match feature_cols"
    log.info(
        "Preprocessed trainset: %d rows %d features (selector flipper scaler)",
        *X_scaled.shape,
    )

    if missing_proba is None:
        missing_proba = {f: 0.1 for f in feature_cols}
        log.info("No missing_proba provided — using uniform 0.1 for all features")
    else:
        missing_proba = {f: p for f, p in missing_proba.items() if f in feature_cols}

    imputer_configs = [
        {"label": "Mean",
         "imputer": SimpleImputer(strategy="mean")},
        {"label": "Median",
         "imputer": SimpleImputer(strategy="median")},
        {"label": "IterativeImputer",
         "imputer": IterativeImputer(max_iter=max_iter,
                                     random_state=random_state,
                                     verbose=0)},
    ]

    results = evaluate_imputers_cv(
        X_scaled        = X_scaled,
        missing_proba   = missing_proba,
        imputer_configs = imputer_configs,
        n_cv_folds      = n_cv_folds,
        n_val_masks     = n_val_masks,
        random_state    = random_state,
        max_iter        = max_iter,
    )

    best_label  = results.iloc[0]["label"]
    best_config = next(c for c in imputer_configs if c["label"] == best_label)
    final_inner = clone(best_config["imputer"])

    log.info("Fitting final imputer [%s] on full scaled trainset …", best_label)
    final_imputer = DataFrameImputer(final_inner)
    final_imputer.fit(X_scaled)
    log.info("Final imputer fitted successfully.")

    # Sanity check
    X_check = introduce_missing_values(X_scaled, missing_proba, random_state=0)
    X_out   = final_imputer.transform(X_check)
    assert isinstance(X_out, pd.DataFrame), "transform() must return DataFrame"
    assert list(X_out.columns) == feature_cols, "Column names must be preserved"
    assert X_out.isna().sum().sum() == 0, "transform() must leave no NaN"
    log.info("Post-fit DataFrame contract verified.")

    artefact = {
        "best_label":   best_label,
        "missing_proba": missing_proba,
        "cv_results":   results[
            ["label", "mean_rmse", "std_rmse", "mean_mae", "std_mae"]
        ].to_dict(orient="records"),
        "imputer":      serialise_imputer(final_imputer, feature_cols),
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(artefact, f, indent=2)
    log.info("Imputer artefact saved %s", out_path)

    return final_imputer


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Select best imputer (Mean/Median/IterativeImputer) by CV RMSE "
            "and save the fitted imputer to JSON."
        )
    )
    p.add_argument("--trainset",     required=True, help="Path to raw trainset.csv")
    p.add_argument("--scaler",       required=True, help="Path to scaler artefact JSON")
    p.add_argument("--out",          required=True, help="Output path for imputer JSON")
    p.add_argument("--cv-folds",     type=int, default=5)
    p.add_argument("--val-masks",    type=int, default=3)
    p.add_argument("--max-iter",     type=int, default=10)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    train_imputer(
        trainset_path        = args.trainset,
        scaler_artefact_path = args.scaler,
        out_path             = args.out,
        n_cv_folds           = args.cv_folds,
        n_val_masks          = args.val_masks,
        max_iter             = args.max_iter,
        random_state         = args.random_state,
    )


if __name__ == "__main__":
    main()
