"""
data_preparation.py
-------------------
Splits a full dataset into:
  - trainset            : 80% stratified split (Criteria × Source_type)
  - testset             : 20% stratified split
  - testset_stringent   : testset rows whose Sequence is NOT present in trainset
  - rows_with_missing   : rows that had NaN in any feature column (excluded from train/test)

Usage:
    python data_preparation.py \
        --input  data/full_dataset.csv \
        --outdir data/splits \
        --feature-cols feature1 feature2 feature3
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def load_and_annotate(path: Path) -> pd.DataFrame:
    """Load CSV and add Source_type + stratify_key columns."""
    df = pd.read_csv(path)
    log.info("Loaded %d rows from %s", len(df), path)

    human_sources = {"ClinVar", "Humsavar"}
    df["Source_type"] = df["Source"].apply(
        lambda x: "Human" if x in human_sources else "Mixed"
    )
    df["stratify_key"] = df["Criteria"].astype(str) + "_" + df["Source_type"]
    return df


def separate_missing(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into (complete_rows, rows_with_missing).
    Rows that have NaN in ANY feature column go to rows_with_missing.
    """
    nan_mask = df[feature_cols].isna().any(axis=1)
    complete = df[~nan_mask].copy()
    missing  = df[nan_mask].copy()
    log.info(
        "Feature-complete rows: %d  |  rows with missing features: %d",
        len(complete), len(missing),
    )
    return complete, missing


def stratified_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """80/20 stratified split on stratify_key."""
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["stratify_key"],
        random_state=random_state,
    )
    log.info("Train: %d rows  |  Test: %d rows", len(train_df), len(test_df))
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_stringent_testset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sequence_col: str = "Sequence",
) -> pd.DataFrame:
    """
    Stringent test subset: test rows whose Sequence is NOT present in trainset.
    Requires the Sequence column to exist in both DataFrames.
    """
    if sequence_col not in test_df.columns:
        log.warning(
            "Column '%s' not found — stringent testset will be empty.", sequence_col
        )
        return test_df.iloc[0:0].copy()

    train_sequences = set(train_df[sequence_col].dropna())
    mask = ~test_df[sequence_col].isin(train_sequences)
    stringent = test_df[mask].copy()
    log.info(
        "Stringent testset (novel sequences only): %d / %d rows (%.1f%%)",
        len(stringent), len(test_df), 100 * len(stringent) / max(len(test_df), 1),
    )
    return stringent.reset_index(drop=True)


def log_distributions(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Print stratification diagnostics."""
    for name, df in [("Train", train_df), ("Test", test_df)]:
        log.info("--- %s distributions ---", name)
        log.info("stratify_key:\n%s", df["stratify_key"].value_counts(normalize=True).to_string())
        log.info("Criteria:\n%s",     df["Criteria"].value_counts(normalize=True).to_string())
        log.info("Source_type:\n%s",  df["Source_type"].value_counts(normalize=True).to_string())


def save_splits(
    outdir: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    stringent_df: pd.DataFrame,
    missing_df: pd.DataFrame,
) -> None:
    """Write all four splits to CSV."""
    outdir.mkdir(parents=True, exist_ok=True)

    paths = {
        "trainset.csv":          train_df,
        "testset.csv":           test_df,
        "testset_stringent.csv": stringent_df,
        "rows_with_missing.csv": missing_df,
    }
    for filename, df in paths.items():
        dest = outdir / filename
        df.to_csv(dest, index=False)
        log.info("Saved %d rows → %s", len(df), dest)


# ---------------------------------------------------------------------------
# Public API (importable)
# ---------------------------------------------------------------------------

def prepare_data(
    input_path: str | Path,
    feature_cols: list[str],
    outdir: str | Path = "splits",
    test_size: float = 0.2,
    random_state: int = 42,
    sequence_col: str = "Sequence",
    save: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Full preparation pipeline.

    Parameters
    ----------
    input_path   : path to full_dataset.csv
    feature_cols : list of feature column names used to detect missing rows
    outdir       : directory to write CSV outputs
    test_size    : fraction reserved for testing (default 0.2)
    random_state : reproducibility seed
    sequence_col : column whose values identify novel sequences for the stringent set
    save         : whether to write CSVs to disk

    Returns
    -------
    dict with keys: train, test, test_stringent, missing
    """
    df = load_and_annotate(Path(input_path))
    complete_df, missing_df = separate_missing(df, feature_cols)
    train_df, test_df = stratified_split(complete_df, test_size, random_state)
    stringent_df = build_stringent_testset(train_df, test_df, sequence_col)
    log_distributions(train_df, test_df)

    if save:
        save_splits(Path(outdir), train_df, test_df, stringent_df, missing_df)

    return {
        "train":          train_df,
        "test":           test_df,
        "test_stringent": stringent_df,
        "missing":        missing_df,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Split full_dataset.csv into train / test / stringent / missing splits."
    )
    parser.add_argument("--input",        required=True, help="Path to full_dataset.csv")
    parser.add_argument("--outdir",       default="splits", help="Output directory (default: splits/)")
    parser.add_argument("--feature-cols", nargs="+", required=True, metavar="COL",
                        help="Feature columns used to detect missing-value rows")
    parser.add_argument("--test-size",    type=float, default=0.2,
                        help="Fraction for test set (default: 0.2)")
    parser.add_argument("--random-state", type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--sequence-col", default="Sequence",
                        help="Column for novel-sequence filtering (default: Sequence)")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    prepare_data(
        input_path   = args.input,
        feature_cols = args.feature_cols,
        outdir       = args.outdir,
        test_size    = args.test_size,
        random_state = args.random_state,
        sequence_col = args.sequence_col,
        save         = True,
    )


if __name__ == "__main__":
    main()