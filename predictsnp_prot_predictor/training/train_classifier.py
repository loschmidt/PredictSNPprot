"""
train_classifier.py
-------------------
Trains all classifier variants on the fully preprocessed trainset,
selects the best by mean CV AUC, and serialises:
  - <out_stem>_classifier.joblib   — fitted best estimator (or GridSearchCV wrapper)
  - <out_stem>_classifier.json     — full pipeline artifact: feature_names, cv scores,
                                     train scores, best params, all model comparison
                                     results, + fitted-state dump of every pipeline step
                                     (selector → flipper → scaler → imputer → classifier)


Pipeline applied before training
---------------------------------
selector → flipper → scaler → imputer
(imputer is a pass-through on feature-complete rows; included so the JSON
artifact describes the full inference pipeline end-to-end)


Usage
-----
    python train_classifier.py \
        --trainset   data/splits/trainset.csv \
        --scaler     models/pipeline_A_scaler.json \
        --imputer    models/pipeline_A_imputer.json \
        --label-col  Label \
        --out        models/pipeline_A \
        [--models logreg_l1 logreg_en rf_2 rf_3 xgb_2 xgb_3 svm_2 mlp_2]
        [--cv-folds 5]
        [--random-state 42]
        [--n-jobs -1]


Importable API
--------------
    from train_classifier import train_classifier, load_classifier_steps

    best_model, meta = train_classifier(
        trainset_path        = "data/splits/trainset.csv",
        scaler_artefact_path = "models/pipeline_A_scaler.json",
        imputer_artefact_path= "models/pipeline_A_imputer.json",
        label_col            = "Label",
        out_stem             = "models/pipeline_A",
    )

    selector, flipper, scaler, imputer, model = load_classifier_steps(
        scaler_path   = "models/pipeline_A_scaler.json",
        imputer_path  = "models/pipeline_A_imputer.json",
        classifier_path = "models/pipeline_A_classifier.joblib",
    )
    X_prep = imputer.transform(scaler.transform(flipper.transform(selector.transform(df))))
    proba  = model.predict_proba(X_prep)[:, 1]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
    _XGB_AVAILABLE = True
except ImportError:
    _XGB_AVAILABLE = False

# Reuse serialisation helpers from sibling scripts — single source of truth
from .train_scaler import (
    load_preprocessing_steps,
    serialise_feature_selector,
    serialise_column_flipper,
    serialise_scaler,
)
from .train_imputer import load_imputer_steps, serialise_imputer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# JSON helpers
# ═══════════════════════════════════════════════════════════════════════════

def _json_safe(obj: Any) -> Any:
    """Recursively coerce numpy / non-serialisable types to plain Python."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(i) for i in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# Classifier fitted-state extraction
# ═══════════════════════════════════════════════════════════════════════════

def _serialise_classifier(fitted_model) -> dict:
    """
    Extract the fitted state of the best classifier.

    Handles all three wrappers that get_model() may return:
      - GridSearchCV        → unwrap to .best_estimator_
      - LogisticRegressionCV→ direct fitted model
      - Pipeline (logreg_rfe) → RFECV selector + LogisticRegressionCV

    For XGBoost, the full native learner JSON is embedded via
    booster.save_raw("json"), producing the identical structure to
    booster.save_model() / the reference JSON format.
    """
    info = {"wrapper_class": type(fitted_model).__name__}

    # ── unwrap GridSearchCV ───────────────────────────────────────────────
    if isinstance(fitted_model, GridSearchCV):
        clf = fitted_model.best_estimator_
        info["best_params"]   = _json_safe(fitted_model.best_params_)
        info["best_cv_score"] = float(fitted_model.best_score_)
    elif isinstance(fitted_model, Pipeline):
        # logreg_rfe: inner Pipeline([("selector", RFECV), ("model", LogReg)])
        if "selector" in fitted_model.named_steps:
            rfe = fitted_model.named_steps["selector"]
            info["rfe_selector"] = {
                "class":                  type(rfe).__name__,
                "n_features_selected_":   int(rfe.n_features_),
                "selected_features_mask": rfe.support_.tolist(),
                "ranking_":               rfe.ranking_.tolist(),
            }
        clf = fitted_model.named_steps.get("model", fitted_model)
    else:
        clf = fitted_model

    info["classifier_class"]  = type(clf).__name__
    info["classifier_module"] = type(clf).__module__

    if hasattr(clf, "classes_"):
        info["classes_"] = clf.classes_.tolist()
    if hasattr(clf, "feature_names_in_"):
        info["feature_names_in_"] = clf.feature_names_in_.tolist()
    if hasattr(clf, "n_features_in_"):
        info["n_features_in_"] = int(clf.n_features_in_)

    # ── XGBoost — embed full native learner JSON ──────────────────────────
    if hasattr(clf, "get_booster"):
        booster = clf.get_booster()
        native_json_str = booster.save_raw(raw_format="json").decode("utf-8")
        info["xgb_native_model"] = json.loads(native_json_str)
        if hasattr(clf, "feature_importances_"):
            info["feature_importances_"] = clf.feature_importances_.tolist()

    # ── Random Forest / Extra Trees ───────────────────────────────────────
    elif hasattr(clf, "estimators_") and hasattr(clf, "feature_importances_"):
        info["n_estimators_fit"]     = len(clf.estimators_)
        info["feature_importances_"] = clf.feature_importances_.tolist()
        info["max_depth"]            = clf.max_depth
        info["n_features_in_"]       = int(clf.n_features_in_)

    # ── Logistic Regression (plain + CV) ─────────────────────────────────
    elif hasattr(clf, "coef_"):
        info["coef_"]      = clf.coef_.tolist()
        info["intercept_"] = clf.intercept_.tolist()
        if hasattr(clf, "C_"):
            info["best_C_"] = (
                float(clf.C_) if np.ndim(clf.C_) == 0 else clf.C_.tolist()
            )
        if hasattr(clf, "l1_ratio_"):
            info["best_l1_ratio_"] = (
                float(clf.l1_ratio_)
                if np.ndim(clf.l1_ratio_) == 0
                else clf.l1_ratio_.tolist()
            )

    # ── SVM ───────────────────────────────────────────────────────────────
    elif hasattr(clf, "support_vectors_"):
        info["n_support_"]       = clf.n_support_.tolist()
        info["support_vectors_"] = clf.support_vectors_.tolist()
        info["dual_coef_"]       = clf.dual_coef_.tolist()
        info["intercept_"]       = clf.intercept_.tolist()
        if hasattr(clf, "gamma"):
            info["gamma"] = clf.gamma

    # ── MLP ───────────────────────────────────────────────────────────────
    elif hasattr(clf, "coefs_"):
        info["coefs_"]             = [c.tolist() for c in clf.coefs_]
        info["intercepts_"]        = [b.tolist() for b in clf.intercepts_]
        info["n_iter_"]            = int(clf.n_iter_)
        info["loss_"]              = float(clf.loss_)
        info["hidden_layer_sizes"] = list(clf.hidden_layer_sizes)

    return info


# ═══════════════════════════════════════════════════════════════════════════
# Model registry
# ═══════════════════════════════════════════════════════════════════════════

def get_model(
    model_name: str,
    cv_folds: int = 5,
    n_jobs: int = -1,
    random_state: int = 42,
):
    """
    Return a configured (unfitted) estimator for the given model name.

    All sklearn classifiers natively accept pd.DataFrame and set
    feature_names_in_ when fit on a DataFrame — no custom wrapper needed.

    GridSearchCV wrappers expose:
      .best_estimator_   — the winning estimator
      .best_params_      — its hyperparameters
      .best_score_       — mean CV AUC (test fold)
      .cv_results_       — full CV result table incl. mean_train_score

    LogisticRegressionCV exposes:
      .C_                — chosen regularisation strength
      .scores_           — per-fold AUC array
    """

    # ── Logistic Regression ───────────────────────────────────────────────
    if model_name == "logreg_l1":
        return LogisticRegressionCV(
            cv=cv_folds, l1_ratios=(1,), solver="saga",
            Cs=20, max_iter=10000, random_state=random_state,
            scoring="roc_auc", n_jobs=n_jobs,
            use_legacy_attributes=False,
        )

    if model_name == "logreg_en":
        return LogisticRegressionCV(
            cv=cv_folds, l1_ratios=[.01, .05, .1, .3, .5, .7, .9, .95, .99],
            solver="saga", Cs=20, max_iter=10000, random_state=random_state,
            scoring="roc_auc", n_jobs=n_jobs,
            use_legacy_attributes=False,
        )

    if model_name == "logreg_rfe":
        selector = RFECV(
            estimator=LogisticRegression(
                penalty="l2", max_iter=1000, random_state=random_state
            ),
            step=1, cv=cv_folds, scoring="roc_auc", n_jobs=n_jobs,
        )
        model = LogisticRegressionCV(
            cv=cv_folds, penalty="l2", solver="lbfgs",
            max_iter=1000, random_state=random_state, scoring="roc_auc",
        )
        return Pipeline([("selector", selector), ("model", model)])

    # ── Random Forest ─────────────────────────────────────────────────────
    if model_name == "rf_1":
        param_grid = {
            "n_estimators":      [50, 100, 150],
            "max_depth":         [3, 5, 7],
            "min_samples_split": [50, 100, 150],
            "min_samples_leaf":  [20, 50, 100],
            "max_features":      [0.3, 0.5],
        }
        return GridSearchCV(
            RandomForestClassifier(random_state=random_state, n_jobs=1),
            param_grid=param_grid, cv=cv_folds,
            scoring="roc_auc", n_jobs=n_jobs, return_train_score=True,
        )

    if model_name == "rf_2":
        param_grid = {
            "n_estimators":     [100, 200, 300],
            "max_depth":        [3, 5, 7],
            "min_samples_leaf": [20, 50, 100],
            "max_features":     [0.2, 0.3, 0.5],
            "max_samples":      [0.6, 0.7, 0.8],
            "class_weight":     ["balanced", "balanced_subsample", None],
            "ccp_alpha":        [0.0, 0.001, 0.005],
        }
        return GridSearchCV(
            RandomForestClassifier(random_state=random_state, n_jobs=1),
            param_grid=param_grid, cv=cv_folds,
            scoring="roc_auc", n_jobs=n_jobs, return_train_score=True,
        )

    # ── XGBoost ───────────────────────────────────────────────────────────
    if model_name == "xgb_1":
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is required. pip install xgboost")
        param_grid = {
            "n_estimators":     [50, 100, 150],
            "max_depth":        [2, 3, 4],
            "learning_rate":    [0.01, 0.05],
            "reg_alpha":        [1.0, 5.0],
            "reg_lambda":       [2.0, 5.0, 10.0],
            "subsample":        [0.6, 0.8],
            "colsample_bytree": [0.6, 0.8],
        }
        return GridSearchCV(
            XGBClassifier(
                random_state=random_state, eval_metric="logloss",
                n_jobs=1, verbosity=0,
            ),
            param_grid=param_grid, cv=cv_folds,
            scoring="roc_auc", n_jobs=n_jobs, return_train_score=True,
        )

    if model_name == "xgb_2":
        if not _XGB_AVAILABLE:
            raise ImportError("xgboost is required. pip install xgboost")
        param_grid = {
             "n_estimators":     [100, 200, 300],
             "max_depth":        [2, 3, 4],
             "learning_rate":    [0.005, 0.01, 0.05],
             "reg_alpha":        [1.0, 5.0, 10.0],
             "reg_lambda":       [2.0, 5.0, 10.0],
             "subsample":        [0.5, 0.6, 0.8],
             "colsample_bytree": [0.5, 0.6, 0.8],
             "min_child_weight": [5, 10, 20],
             "gamma":            [0.1, 0.5, 1.0],
         }
        #param_grid = {
        #    "n_estimators":     [300, 350, 400],
        #    "max_depth":        [3, 4, 5],
        #    "learning_rate":    [0.025, 0.05, 0.075, 0.1],
        #    "reg_alpha":        [0.5, 1.0, 1.5],
        #    "reg_lambda":       [2.0, 5.0, 7.5, 10.0, 12.5],
        #    "subsample":        [0.6, 0.8, 0.9],
        #    "colsample_bytree": [0.8, 0.9, 1.0],
        #    "min_child_weight": [3, 4, 5],
        #    "gamma":            [0.05, 0.1, 0.5],
        #}
        return GridSearchCV(
            XGBClassifier(
                random_state=random_state, eval_metric="logloss",
                n_jobs=1, verbosity=0,
            ),
            param_grid=param_grid, cv=cv_folds,
            scoring="roc_auc", n_jobs=n_jobs, return_train_score=True,
        )

    # ── SVM ───────────────────────────────────────────────────────────────
    if model_name == "svm_1":
        param_grid = {
            "C":      [0.1, 1, 10, 100],
            "gamma":  ["scale", "auto", 0.001, 0.01, 0.1],
            "kernel": ["rbf", "linear"],
        }
        return GridSearchCV(
            SVC(probability=True, random_state=random_state),
            param_grid=param_grid, cv=cv_folds,
            scoring="roc_auc", n_jobs=n_jobs, return_train_score=True,
        )

    if model_name == "svm_2":
        param_grid = {
            "C":            [0.01, 0.1, 1, 10],
            "gamma":        ["scale", 0.001, 0.01],
            "kernel":       ["rbf"],
            "class_weight": ["balanced", None],
        }
        return GridSearchCV(
            SVC(probability=True, random_state=random_state),
            param_grid=param_grid, cv=cv_folds,
            scoring="roc_auc", n_jobs=n_jobs, return_train_score=True,
        )

    # ── MLP ───────────────────────────────────────────────────────────────
    if model_name == "mlp_1":
        param_grid = {
            "hidden_layer_sizes": [(50,), (100,), (50, 25), (100, 50)],
            "alpha":              [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01],
            "max_iter":           [500, 1000],
        }
        return GridSearchCV(
            MLPClassifier(
                solver="adam", activation="relu",
                early_stopping=True, random_state=random_state,
            ),
            param_grid=param_grid, cv=cv_folds,
            scoring="roc_auc", n_jobs=n_jobs, return_train_score=True,
        )

    if model_name == "mlp_2":
        param_grid = {
            "hidden_layer_sizes": [
                (64,), (128,), (256,),
                (128, 64), (256, 128),
                (128, 64, 32), (256, 128, 64),
            ],
            "alpha":              [0.001, 0.01, 0.1, 1.0],
            "batch_size":         [32, 64, 128, "auto"],
            "learning_rate":      ["constant", "adaptive"],
            "learning_rate_init": [0.0001, 0.001, 0.01],
            "beta_1":             [0.85, 0.9, 0.95],
            "max_iter":           [500, 1000],
        }
        return GridSearchCV(
            MLPClassifier(
                solver="adam", activation="relu",
                early_stopping=True, random_state=random_state,
            ),
            param_grid=param_grid, cv=cv_folds,
            scoring="roc_auc", n_jobs=n_jobs, return_train_score=True,
        )

    raise ValueError(
        f"Unknown model name: {model_name!r}. "
        "Valid names: logreg_l1, logreg_en, logreg_rfe, "
        "rf_1, rf_2, xgb_1, xgb_2, svm_1, svm_2, mlp_1, mlp_2"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Score extraction helpers
# ═══════════════════════════════════════════════════════════════════════════

def _extract_scores(
    fitted_model,
    X: pd.DataFrame,
    y: np.ndarray,
    cv_folds: int,
    random_state: int,
    model_name: str,
) -> dict[str, Any]:
    """
    Unified score extraction for all model types.

    GridSearchCV         → .best_score_ for CV AUC; mean_train_score from cv_results_
    LogisticRegressionCV → extract from .scores_; compute train AUC directly
    Pipeline             → rerun cross_val_score for CV AUC; predict_proba for train
    """
    if isinstance(fitted_model, GridSearchCV):
        cv_auc    = float(fitted_model.best_score_)
        cv_std    = float(
            fitted_model.cv_results_["std_test_score"][fitted_model.best_index_]
        )
        train_auc = float(
            fitted_model.cv_results_["mean_train_score"][fitted_model.best_index_]
        )
        train_std = float(
            fitted_model.cv_results_["std_train_score"][fitted_model.best_index_]
        )
        best_params = fitted_model.best_params_
        all_cv = {
            "mean_test_score":  fitted_model.cv_results_["mean_test_score"].tolist(),
            "mean_train_score": fitted_model.cv_results_["mean_train_score"].tolist(),
            "params":           [str(p) for p in fitted_model.cv_results_["params"]],
        }

    elif isinstance(fitted_model, LogisticRegressionCV):
        scores_arr  = fitted_model.scores_
        mean_scores = scores_arr.mean(axis=0)
        best_flat   = np.argmax(mean_scores)
        best_r, best_c = np.unravel_index(best_flat, mean_scores.shape)
        cv_auc    = float(mean_scores[best_r, best_c])
        cv_std    = float(scores_arr[:, best_r, best_c].std())
        train_auc = float(roc_auc_score(y, fitted_model.predict_proba(X)[:, 1]))
        train_std = 0.0
        best_params = {
            "C":        float(fitted_model.C_),
            "l1_ratio": float(fitted_model.l1_ratio_),
        }
        all_cv = {}

    elif isinstance(fitted_model, Pipeline):
        cv_scores = cross_val_score(
            fitted_model, X, y,
            cv=KFold(n_splits=cv_folds, shuffle=True, random_state=random_state),
            scoring="roc_auc", n_jobs=-1,
        )
        cv_auc    = float(cv_scores.mean())
        cv_std    = float(cv_scores.std())
        train_auc = float(roc_auc_score(y, fitted_model.predict_proba(X)[:, 1]))
        train_std = 0.0
        best_params = {}
        all_cv = {"cv_scores": cv_scores.tolist()}

    else:
        raise TypeError(f"Unsupported model type: {type(fitted_model).__name__}")

    return {
        "model":       model_name,
        "cv_auc":      round(cv_auc, 6),
        "cv_std":      round(cv_std, 6),
        "train_auc":   round(train_auc, 6),
        "train_std":    round(train_std, 6),
        "overfit_gap": round(train_auc - cv_auc, 6),
        "best_params": best_params,
        "all_cv":      all_cv,
    }


def _print_results_table(results: list[dict]) -> None:
    w = 76
    print(f"\n{'='*w}")
    print("CLASSIFIER SELECTION — CV RESULTS (sorted by CV AUC ↓)")
    print(f"{'='*w}")
    print(f"  {'Rank':<5} {'Model':<20} {'CV AUC':>10}  {'STD':>8}  "
          f"{'Train AUC':>10}  {'Train_std':>8} {'Gap':>8}")
    print(f"  {'-'*w}")
    for rank, r in enumerate(results, 1):
        flag = " ⚠" if r["overfit_gap"] > 0.05 else "  "
        print(
            f"  {rank:<5} {r['model']:<20} {r['cv_auc']:>10.4f}  "
            f"{r['cv_std']:>8.4f}  {r['train_auc']:>10.4f} {r['train_std']:>8.4f} "
            f"{r['overfit_gap']:>+8.4f}{flag}"
        )
    print(f"{'='*w}")
    print("  ⚠  Gap > 0.05 may indicate overfitting")
    print(f"  ✔  Best: {results[0]['model']}  "
          f"(CV AUC = {results[0]['cv_auc']:.4f})\n")


# ═══════════════════════════════════════════════════════════════════════════
# Load artefacts
# ═══════════════════════════════════════════════════════════════════════════

def load_classifier_steps(
    scaler_path: str | Path,
    imputer_path: str | Path,
    classifier_path: str | Path,
) -> tuple:
    """
    Load all five pipeline steps from their artefacts.

    Returns
    -------
    (selector, flipper, scaler, imputer, model)

    Example
    -------
    selector, flipper, scaler, imputer, model = load_classifier_steps(
        scaler_path     = "models/pipeline_A_scaler.json",
        imputer_path    = "models/pipeline_A_imputer.json",
        classifier_path = "models/pipeline_A_classifier.joblib",
    )
    X_prep = imputer.transform(
                 scaler.transform(
                     flipper.transform(
                         selector.transform(df))))
    proba = model.predict_proba(X_prep)[:, 1]
    """
    selector, flipper, scaler, imputer = load_imputer_steps(scaler_path, imputer_path)
    model = joblib.load(classifier_path)
    log.info(
        "Classifier loaded from %s  [%s]",
        classifier_path, type(model).__name__,
    )
    return selector, flipper, scaler, imputer, model


# ═══════════════════════════════════════════════════════════════════════════
# Main training function
# ═══════════════════════════════════════════════════════════════════════════

def train_classifier(
    trainset_path: str | Path,
    scaler_artefact_path: str | Path,
    imputer_artefact_path: str | Path,
    label_col: str,
    out_stem: str | Path,
    model_names: list[str] | None = None,
    cv_folds: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> tuple:
    """
    Full classifier training flow.

    1. Load all preprocessing steps (selector → flipper → scaler → imputer).
    2. Apply the full preprocessing chain to the trainset.
       The imputer is a pass-through on feature-complete rows (no NaN introduced),
       but including it here means the JSON artifact describes the complete
       inference pipeline end-to-end.
    3. Train every model in model_names.
    4. Rank by CV AUC, print comparison table.
    5. Serialise the best estimator:
         <out_stem>_classifier.joblib  — fitted model binary
         <out_stem>_classifier.json    — full pipeline artifact

    Parameters
    ----------
    trainset_path         : raw trainset CSV (feature-complete rows only)
    scaler_artefact_path  : scaler JSON from train_scaler.py
    imputer_artefact_path : imputer JSON from train_imputer.py
    label_col             : binary label column name (values must be 0/1)
    out_stem              : path prefix for output files
    model_names           : which models to train (default: all available)
    cv_folds              : KFold splits
    random_state          : reproducibility seed
    n_jobs                : parallelism for GridSearchCV

    Returns
    -------
    (best_model, metadata_dict)
    """
    # ── 1. Load all preprocessing steps ──────────────────────────────────
    selector, flipper, scaler, imputer = load_imputer_steps(
        scaler_artefact_path, imputer_artefact_path
    )
    feature_cols = selector.feature_cols
    log.info(
        "All preprocessing steps loaded  "
        "(scaler: %s  imputer: %s)",
        scaler_artefact_path, imputer_artefact_path,
    )

    # ── 2. Load and preprocess trainset ───────────────────────────────────
    train_df = pd.read_csv(trainset_path)
    log.info("Raw trainset: %d rows × %d cols", *train_df.shape)

    if label_col not in train_df.columns:
        raise ValueError(
            f"Label column {label_col!r} not found. "
            f"Available columns: {list(train_df.columns)}"
        )
    y = train_df[label_col].values
    if not np.isin(y, [0, 1]).all():
        raise ValueError(
            f"Label column {label_col!r} must be binary (0/1). "
            f"Found unique values: {np.unique(y)}"
        )

    # Full chain: selector → flipper → scaler → imputer
    # Imputer is pass-through on complete rows; included for pipeline consistency
    X = imputer.transform(
            scaler.transform(
                flipper.transform(
                    selector.transform(train_df))))

    assert isinstance(X, pd.DataFrame), "Preprocessing chain must return DataFrame"
    assert list(X.columns) == feature_cols, "Column order must match feature_cols"

    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        raise ValueError(
            f"Trainset still contains {nan_count} NaN values after full preprocessing. "
            "Check that trainset.csv contains only feature-complete rows "
            "(output of data_preparation.py) and that the imputer artefact is valid."
        )

    log.info(
        "Preprocessed trainset: %d rows × %d features  "
        "(selector→flipper→scaler→imputer, 0 NaN)",
        *X.shape,
    )
    log.info(
        "Class distribution: %s",
        dict(zip(*np.unique(y, return_counts=True))),
    )

    # ── 3. Resolve model list ──────────────────────────────────────────────
    _defaults = ["logreg_l1", "logreg_en", "rf_1", "rf_2", "mlp_1", "mlp_2", "svm_1", "svm_2"]
    if _XGB_AVAILABLE:
        _defaults += ["xgb_1", "xgb_2"]
    if model_names is None:
        model_names = _defaults
    log.info("Models to train (%d): %s", len(model_names), model_names)

    # ── 4. Train each model ───────────────────────────────────────────────
    results: list[dict] = []
    fitted_models: dict[str, Any] = {}

    for name in model_names:
        log.info("─" * 60)
        log.info("Training: %s", name)
        try:
            model = get_model(
                name, cv_folds=cv_folds, n_jobs=n_jobs, random_state=random_state
            )
            model.fit(X, y)
            scores = _extract_scores(model, X, y, cv_folds, random_state, name)
            results.append(scores)
            fitted_models[name] = model
            log.info(
                "  CV AUC = %.4f ± %.4f   Train AUC = %.4f ± %.4f   Gap = %+.4f",
                scores["cv_auc"], scores["cv_std"],
                scores["train_auc"], scores["train_std"], scores["overfit_gap"],
            )
        except Exception as exc:
            log.warning("  SKIPPED %s — %s", name, exc)

    if not results:
        raise RuntimeError("All models failed to train.")

    # ── 5. Rank and print ─────────────────────────────────────────────────
    results.sort(key=lambda r: r["cv_auc"], reverse=True)
    _print_results_table(results)

    best_name  = results[0]["model"]
    best_model = fitted_models[best_name]

    # ── 6. Serialise ──────────────────────────────────────────────────────
    out_stem = Path(out_stem)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    joblib_path = Path(str(out_stem) + "_classifier.joblib")
    json_path   = Path(str(out_stem) + "_classifier.json")

    # Save full GridSearchCV wrapper — preserves best_params_, cv_results_, etc.
    joblib.dump(best_model, joblib_path)
    log.info("Model binary saved → %s", joblib_path)

    best_inner_type = (
        type(best_model.best_estimator_).__name__
        if isinstance(best_model, GridSearchCV)
        else type(best_model).__name__
    )
    feature_names = list(
        best_model.feature_names_in_
        if hasattr(best_model, "feature_names_in_")
        else feature_cols
    )

    # ── build JSON artifact ───────────────────────────────────────────────
    # pipeline_steps reuses the canonical serialise_* functions from
    # train_scaler.py and train_imputer.py — single source of truth
    metadata = {
        "best_model":       best_name,
        "best_model_type":  best_inner_type,
        "feature_names_in": feature_names,
        "label_col":        label_col,
        "cv_folds":         cv_folds,
        "random_state":     random_state,
        "n_train_samples":  int(len(y)),
        "class_distribution": {
            str(int(k)): int(v)
            for k, v in zip(*np.unique(y, return_counts=True))
        },
        "best_params":  results[0]["best_params"],
        "scores": {
            "cv_auc":      results[0]["cv_auc"],
            "cv_std":      results[0]["cv_std"],
            "train_auc":   results[0]["train_auc"],
            "overfit_gap": results[0]["overfit_gap"],
        },
        "all_models": [
            {k: v for k, v in r.items() if k != "all_cv"}
            for r in results
        ],
        # Full fitted-state of every pipeline step — uses the same
        # serialise_* functions as train_scaler.py and train_imputer.py
        "pipeline_steps": {
            "feature_selector": serialise_feature_selector(selector),
            "column_flipper":   serialise_column_flipper(flipper),
            "scaler":           serialise_scaler(scaler, feature_cols),
            "imputer":          serialise_imputer(imputer, feature_cols),
            "classifier":       _json_safe(_serialise_classifier(best_model)),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    log.info("Full pipeline artifact saved → %s", json_path)

    return best_model, metadata


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

_ALL_MODELS = [
    "logreg_l1", "logreg_en",
    "rf_1", "rf_2",
    "xgb_1", "xgb_2",
    "mlp_1", "mlp_2",
    "svm_1", "svm_2",
]


def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=(
            "Train all classifier variants, report CV/train AUC, "
            "serialise the best model with full pipeline artifact."
        )
    )
    p.add_argument("--trainset",  required=True, help="Path to trainset.csv")
    p.add_argument("--scaler",    required=True, help="Path to scaler artefact JSON")
    p.add_argument("--imputer",   required=True, help="Path to imputer artefact JSON")
    p.add_argument("--label-col", required=True, help="Binary label column name")
    p.add_argument("--out",       required=True,
                   help="Output path stem (e.g. models/pipeline_A)")
    p.add_argument("--models",    nargs="+", default=None,
                   choices=_ALL_MODELS, metavar="MODEL",
                   help=f"Models to train. Choices: {_ALL_MODELS}")
    p.add_argument("--cv-folds",     type=int, default=5)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--n-jobs",       type=int, default=-1)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    train_classifier(
        trainset_path         = args.trainset,
        scaler_artefact_path  = args.scaler,
        imputer_artefact_path = args.imputer,
        label_col             = args.label_col,
        out_stem              = args.out,
        model_names           = args.models,
        cv_folds              = args.cv_folds,
        random_state          = args.random_state,
        n_jobs                = args.n_jobs,
    )


if __name__ == "__main__":
    main()
