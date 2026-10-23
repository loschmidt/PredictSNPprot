"""
train_scaler.py
---------------
Trains and serialises the first three steps of the prediction pipeline:

    FeatureSelector ColumnFlipper DataFrameMinMaxScaler (clip-trained)

The scaler is fit on clipped scores to prevent outliers from dominating the
[0, 1] range: percentile-based bounds are computed on the trainset and used
as the effective data_min / data_max seen by MinMaxScaler.

All three steps are serialised to a single JSON artefact so the production
pipeline can reconstruct them without pickling and without sklearn version
pinning.

DataFrame contract
------------------
Every step in the pipeline receives a pd.DataFrame with named columns and
returns a pd.DataFrame with the same named columns.  Column names are never
dropped.  This is enforced by DataFrameMinMaxScaler, which wraps sklearn's
MinMaxScaler and re-attaches column names after transform.

Usage
-----
    python train_scaler.py \
        --trainset        data/splits/trainset.csv \
        --config          configs/pipeline_A.json \
        --out             models/pipeline_A_scaler.json

Config schema (pipeline_*.json)
--------------------------------
{
    "feature_cols":  ["feat1", "feat2", ...],   // ordered feature list for FeatureSelector
    "cols_to_flip":  ["feat2", ...],             // columns to negate (ColumnFlipper)
    "clip_lower_pct": 1.0,                       // lower percentile for score clipping (default 1)
    "clip_upper_pct": 99.0,                      // upper percentile for score clipping (default 99)
    "feature_range": [0, 1]                      // MinMaxScaler output range (default [0,1])
}
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from features import *
# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ===========================================================================
# Pipeline step definitions
# ===========================================================================

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Selects and reorders feature columns; drops everything else.

    Input  : pd.DataFrame  (any columns)
    Output : pd.DataFrame  (only feature_cols, in declared order)
    """

    def __init__(self, feature_cols: list[str]):
        self.feature_cols = feature_cols

    def fit(self, X: pd.DataFrame, y=None):
        self._validate(X)
        self.feature_names_in_ = np.array(self.feature_cols)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate(X)
        return X[self.feature_cols].copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.feature_cols)

    def _validate(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                f"FeatureSelector expects a DataFrame, got {type(X).__name__}. "
                "Load your CSV with pd.read_csv() first."
            )
        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise ValueError(
                f"FeatureSelector: {len(missing)} required feature(s) missing:\n"
                + "\n".join(f"  - {c}" for c in missing)
                + f"\n\nAvailable columns: {list(X.columns)}"
            )


class ColumnFlipper(BaseEstimator, TransformerMixin):
    """
    Negates columns that are negatively correlated with the target.
    Columns absent from the input are silently skipped (pipeline-safe).

    Input  : pd.DataFrame
    Output : pd.DataFrame  (same columns, sign-flipped where specified)
    """

    def __init__(self, cols_to_flip: list[str]):
        self.cols_to_flip = cols_to_flip

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"ColumnFlipper expects a DataFrame, got {type(X).__name__}.")
        # Record which flips are actually active for this feature set
        self.active_flips_      = [c for c in self.cols_to_flip if c in X.columns]
        self.feature_names_in_  = np.array(X.columns.tolist())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Recompute active flips dynamically — robust to column reordering
        active = [c for c in self.cols_to_flip if c in X.columns]
        X = X.copy()
        X[active] = -X[active]
        return X                               # always returns pd.DataFrame

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        # Column names are unchanged by negation
        if input_features is not None:
            return np.array(input_features)
        return self.feature_names_in_          # set during fit()


class DataFrameMinMaxScaler(BaseEstimator, TransformerMixin):
    """
    DataFrame-preserving wrapper around sklearn's MinMaxScaler.

    Guarantees that transform() always returns a pd.DataFrame with the
    original column names intact, regardless of what sklearn's internal
    MinMaxScaler returns (which is a numpy array).

    Input  : pd.DataFrame
    Output : pd.DataFrame  (same columns, values scaled to feature_range)
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0), clip: bool = True):
        self.feature_range = feature_range
        self.clip          = clip
        self._scaler       = MinMaxScaler(feature_range=feature_range, clip=clip)

    def fit(self, X: pd.DataFrame, y=None) -> "DataFrameMinMaxScaler":
        self._validate(X)
        self._scaler.fit(X)
        self.feature_names_in_ = np.array(X.columns.tolist())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate(X)
        cols   = X.columns.tolist()
        index  = X.index
        scaled = self._scaler.transform(X)     # returns np.ndarray
        return pd.DataFrame(scaled, columns=cols, index=index)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is not None:
            return np.array(input_features)
        return self.feature_names_in_

    # ------------------------------------------------------------------
    # Delegate sklearn scaler attributes for external inspection
    # ------------------------------------------------------------------
    @property
    def scale_(self):       return self._scaler.scale_
    @property
    def min_(self):         return self._scaler.min_
    @property
    def data_min_(self):    return self._scaler.data_min_
    @property
    def data_max_(self):    return self._scaler.data_max_
    @property
    def data_range_(self):  return self._scaler.data_range_
    @property
    def n_samples_seen_(self): return self._scaler.n_samples_seen_

    def _validate(self, X: pd.DataFrame):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"DataFrameMinMaxScaler expects a DataFrame, got {type(X).__name__}.")


# ===========================================================================
# Clip-aware scaler training
# ===========================================================================

def fit_clipped_scaler(
    X_flipped: pd.DataFrame,
    feature_cols: list[str],
    clip_lower_pct: float = 1.0,
    clip_upper_pct: float = 99.0,
    feature_range: tuple[float, float] = (0.0, 1.0),
) -> DataFrameMinMaxScaler:
    """
    Fit a DataFrameMinMaxScaler whose effective data range is the
    [clip_lower_pct, clip_upper_pct] percentile interval of each feature.

    Values outside that interval are hard-clamped to feature_range at
    transform time (clip=True).  This prevents extreme outliers from
    compressing the useful score distribution into a narrow band.

    The scaler is fit on a synthetic 2-row DataFrame [lo, hi] the
    percentile bounds rather than on the raw trainset.  This is exact
    because MinMaxScaler only uses min/max from the training data.

    Parameters
    ----------
    X_flipped       : DataFrame output of ColumnFlipper
    feature_cols    : ordered column list (subset of X_flipped.columns)
    clip_lower_pct  : lower percentile (default 1.0)
    clip_upper_pct  : upper percentile (default 99.0)
    feature_range   : output range of the scaler (default (0.0, 1.0))

    Returns
    -------
    Fitted DataFrameMinMaxScaler instance
    """
    arr = X_flipped[feature_cols].values.astype(float)

    lo = np.nanpercentile(arr, clip_lower_pct, axis=0)
    hi = np.nanpercentile(arr, clip_upper_pct, axis=0)

    # Guard: zero-range feature would produce NaN scale_
    zero_range_mask = hi == lo
    if zero_range_mask.any():
        bad_cols = [feature_cols[i] for i in np.where(zero_range_mask)[0]]
        log.warning("Zero-range features after clipping (bumping hi by 1): %s", bad_cols)
        hi[zero_range_mask] = lo[zero_range_mask] + 1.0

    # Build synthetic 2-row DataFrame so DataFrameMinMaxScaler.fit()
    # receives a properly named DataFrame (not a bare numpy array)
    synthetic_df = pd.DataFrame(
        np.vstack([lo, hi]),
        columns=feature_cols,
    )

    scaler = DataFrameMinMaxScaler(feature_range=feature_range, clip=True)
    scaler.fit(synthetic_df)

    log.info(
        "Scaler fit on percentile bounds [p%.0f, p%.0f] across %d features",
        clip_lower_pct, clip_upper_pct, len(feature_cols),
    )
    _log_scaler_stats(scaler, feature_cols)
    return scaler


def _log_scaler_stats(scaler: DataFrameMinMaxScaler, feature_cols: list[str]) -> None:
    rows = [
        {
            "feature":    col,
            "data_min":   round(float(scaler.data_min_[i]), 4),
            "data_max":   round(float(scaler.data_max_[i]), 4),
            "data_range": round(float(scaler.data_range_[i]), 4),
        }
        for i, col in enumerate(feature_cols)
    ]
    df = pd.DataFrame(rows).set_index("feature")
    log.info("Per-feature scaler bounds:\n%s", df.to_string())


# ===========================================================================
# Serialisation helpers
# ===========================================================================

def serialise_feature_selector(sel: FeatureSelector) -> dict[str, Any]:
    return {
        "step":         "FeatureSelector",
        "feature_cols": list(sel.feature_cols),
    }


def serialise_column_flipper(flipper: ColumnFlipper) -> dict[str, Any]:
    return {
        "step":         "ColumnFlipper",
        "cols_to_flip": list(flipper.cols_to_flip),
    }


def serialise_scaler(scaler: DataFrameMinMaxScaler, feature_cols: list[str]) -> dict[str, Any]:
    return {
        "step":           "DataFrameMinMaxScaler",
        "feature_range":  list(scaler.feature_range),
        "clip":           bool(scaler.clip),
        "scale":          scaler.scale_.tolist(),
        "min":            scaler.min_.tolist(),
        "data_min":       scaler.data_min_.tolist(),
        "data_max":       scaler.data_max_.tolist(),
        "data_range":     scaler.data_range_.tolist(),
        "n_samples_seen": int(scaler.n_samples_seen_),
        "feature_names":  feature_cols,
    }


def save_artefact(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Artefact saved → %s", path)


# ===========================================================================
# Reconstruction helpers  (used by the production inference script)
# ===========================================================================

def _reconstruct_feature_selector(cfg: dict[str, Any]) -> FeatureSelector:
    sel = FeatureSelector(feature_cols=cfg["feature_cols"])
    sel.feature_names_in_ = np.array(cfg["feature_cols"])
    return sel


def _reconstruct_column_flipper(cfg: dict[str, Any]) -> ColumnFlipper:
    flipper = ColumnFlipper(cols_to_flip=cfg["cols_to_flip"])
    # Provide a minimal fit state so active_flips_ and feature_names_in_
    # are always present, even before the first transform call
    flipper.active_flips_     = list(cfg["cols_to_flip"])
    flipper.feature_names_in_ = np.array(cfg["cols_to_flip"])
    return flipper


def _reconstruct_scaler(cfg: dict[str, Any]) -> DataFrameMinMaxScaler:
    scaler = DataFrameMinMaxScaler(
        feature_range=tuple(cfg["feature_range"]),
        clip=cfg["clip"],
    )
    # Inject pre-trained parameters directly into the inner sklearn scaler
    inner = scaler._scaler
    inner.scale_            = np.array(cfg["scale"])
    inner.min_              = np.array(cfg["min"])
    inner.data_min_         = np.array(cfg["data_min"])
    inner.data_max_         = np.array(cfg["data_max"])
    inner.data_range_       = np.array(cfg["data_range"])
    inner.n_samples_seen_   = np.intp(cfg["n_samples_seen"])
    inner.n_features_in_    = len(cfg["feature_names"])
    inner.feature_names_in_ = np.array(cfg["feature_names"])
    # Mirror on the wrapper so get_feature_names_out() works
    scaler.feature_names_in_ = np.array(cfg["feature_names"])
    return scaler


def load_preprocessing_steps(
    artefact_path: str | Path,
) -> tuple[FeatureSelector, ColumnFlipper, DataFrameMinMaxScaler]:
    """
    Reconstruct all three preprocessing steps from a saved JSON artefact.

    Every step accepts and returns a pd.DataFrame, so they can be chained
    directly or placed inside an sklearn Pipeline.

    Returns
    -------
    (selector, flipper, scaler)

    Example
    -------
    selector, flipper, scaler = load_preprocessing_steps("models/pipeline_A_scaler.json")
    X_out = scaler.transform(flipper.transform(selector.transform(df)))
    # X_out is a pd.DataFrame with the same named columns as the selector output
    """
    with open(artefact_path) as f:
        art = json.load(f)

    selector = _reconstruct_feature_selector(art["feature_selector"])
    flipper  = _reconstruct_column_flipper(art["column_flipper"])
    scaler   = _reconstruct_scaler(art["scaler"])
    log.info("Loaded preprocessing steps from %s", artefact_path)
    return selector, flipper, scaler


def create_pipeline_config(**kwargs):
    return kwargs


# ===========================================================================
# Main training function  (importable + CLI)
# ===========================================================================

def train_scaler(
    trainset_path: str | Path,
    feature_cols: list[str],
    out_path: str | Path,
) -> tuple[FeatureSelector, ColumnFlipper, DataFrameMinMaxScaler]:
    """
    Full training flow for the three preprocessing steps.

    Parameters
    ----------
    trainset_path   : path to trainset.csv (output of data_preparation.py)
    pipeline_config : dict with keys: feature_cols, cols_to_flip,
                      clip_lower_pct, clip_upper_pct, feature_range
    out_path        : where to write the JSON artefact

    Returns
    -------
    (selector, flipper, scaler) all three fitted, all DataFrame-compatible
    """
    train_df = pd.read_csv(trainset_path)
    log.info("Trainset loaded: %d rows %d cols", *train_df.shape)
    
    cols_to_flip   = get_cols_to_flip(train_df, feature_cols)
    clip_lower_pct = 1.0
    clip_upper_pct = 99.0
    feature_range  = tuple([0.0, 1.0])

    pipeline_config = create_pipeline_config(cols_to_flip=cols_to_flip, clip_lower_pct=clip_lower_pct, clip_upper_pct=clip_upper_pct, feature_range=feature_range)

    # Step 1: FeatureSelector df (feature_cols only)
    selector = FeatureSelector(feature_cols=feature_cols)
    selector.fit(train_df)
    X_selected = selector.transform(train_df)
    assert isinstance(X_selected, pd.DataFrame), "FeatureSelector must return DataFrame"
    log.info("FeatureSelector: %d features selected", len(feature_cols))

    # Step 2: ColumnFlipper df (same cols, sign-flipped where specified)
    flipper = ColumnFlipper(cols_to_flip=cols_to_flip)
    flipper.fit(X_selected)
    X_flipped = flipper.transform(X_selected)
    assert isinstance(X_flipped, pd.DataFrame), "ColumnFlipper must return DataFrame"
    if cols_to_flip:
        log.info(
            "ColumnFlipper: %d/%d cols flipped: %s",
            len(flipper.active_flips_), len(cols_to_flip), flipper.active_flips_,
        )
    else:
        log.info("ColumnFlipper: no columns to flip")

    # Step 3: DataFrameMinMaxScaler df (same cols, values in feature_range)
    scaler = fit_clipped_scaler(
        X_flipped      = X_flipped,
        feature_cols   = feature_cols,
        clip_lower_pct = clip_lower_pct,
        clip_upper_pct = clip_upper_pct,
        feature_range  = feature_range,
    )
    X_scaled = scaler.transform(X_flipped)
    assert isinstance(X_scaled, pd.DataFrame), "DataFrameMinMaxScaler must return DataFrame"
    assert list(X_scaled.columns) == feature_cols, "Column names must be preserved after scaling"

    artefact = {
        "pipeline_config":  pipeline_config,
        "feature_selector": serialise_feature_selector(selector),
        "column_flipper":   serialise_column_flipper(flipper),
        "scaler":           serialise_scaler(scaler, feature_cols),
    }
    save_artefact(Path(out_path), artefact)

    return selector, flipper, scaler


def get_cols_to_flip(df: pd.DataFrame, cols: list[str]) -> list[str]:
    """Return the columns in `cols` that are negatively correlated with `df[label_col]`."""
    labels = df["Label"].to_numpy()
    cols_to_flip = []

    for col in cols:
        scores = df[col].to_numpy()
        flipped = bool(np.corrcoef(scores, labels)[0, 1] < 0)
        if flipped:
            cols_to_flip.append(col)

    return cols_to_flip


# ===========================================================================
# CLI
# ===========================================================================

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train FeatureSelector + ColumnFlipper + DataFrameMinMaxScaler and save to JSON."
    )
    p.add_argument("--trainset", required=True, help="Path to trainset.csv")
    p.add_argument("--out",      required=True, help="Output path for the scaler artefact JSON")
    p.add_argument("--cols",      default=FEATURES)
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    train_scaler(
        trainset_path   = args.trainset,
        feature_cols    = args.cols,
        out_path        = args.out,
    )


if __name__ == "__main__":
    main()
