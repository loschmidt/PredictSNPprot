import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ===========================================================================
# Custom transformers — must match training definitions exactly
# ===========================================================================

class FeatureSelector(BaseEstimator, TransformerMixin):
    """Selects and reorders feature columns; drops everything else."""

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
    """Negates columns that are negatively correlated with the target.
    Columns absent from the input are silently skipped (fast pipeline safe)."""

    def __init__(self, cols_to_flip: list[str]):
        self.cols_to_flip = cols_to_flip

    def fit(self, X: pd.DataFrame, y=None):
        self.active_flips_ = [c for c in self.cols_to_flip if c in X.columns]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Recompute dynamically — never relies on active_flips_ from fit
        active = [c for c in self.cols_to_flip if c in X.columns]
        X = X.copy()
        X[active] = -X[active]
        return X

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is not None:
            return np.array(input_features)
        return np.array(self.feature_names_in_)