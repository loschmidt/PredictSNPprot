#!/usr/bin/env python3

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def load_predictor_wide(path: Path) -> pd.DataFrame:
    """
    Load a per-position wide predictor CSV and return a long-form DataFrame:

        Position, WT, MT, score

    where `score` is the predictor value for WT -> MT at that position.
    """
    df = pd.read_csv(path)

    missing_cols = {"Position", "WT residue"} - set(df.columns)
    if missing_cols:
        raise ValueError(f"{path}: missing required columns: {missing_cols}")

    # Infer AA columns from header: take all single-letter AA columns,
    # excluding Position and WT. This respects the order present in the file.
    AA_SET = set("ACDEFGHIKLMNPQRSTVWY")
    aa_cols = [c for c in df.columns if c in AA_SET]
    if not aa_cols:
        raise ValueError(
            f"{path}: could not infer any amino acid columns; "
            f"expected single-letter AA columns like A,C,D,..."
        )

    long = df.melt(
        id_vars=["Position", "WT residue"],
        value_vars=aa_cols,
        var_name="MT",
        value_name="score",
    )

    long["Position"] = long["Position"].astype(int)
    long["WT residue"] = long["WT residue"].astype(str)
    long["MT"] = long["MT"].astype(str)

    # Drop WT->WT if only mutants are desired
    mask_mut = long["WT residue"] != long["MT"]
    long = long[mask_mut].reset_index(drop=True)

    return long


def build_mutation_keys(predictor_longs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build the union of all (Position, WT, MT) triplets across predictors.
    """
    key_dfs = []
    for df in predictor_longs.values():
        keys = df[["Position", "WT residue", "MT"]].drop_duplicates()
        key_dfs.append(keys)

    master_keys = pd.concat(key_dfs, ignore_index=True)
    master_keys = master_keys.drop_duplicates().reset_index(drop=True)
    return master_keys


def merge_all_predictors(predictor_longs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all predictors into one table keyed by (Position, WT, MT).

    predictor_longs: mapping predictor_name -> long df [Position, WT, MT, score]
    """
    master = build_mutation_keys(predictor_longs)

    for pred_name, long_df in predictor_longs.items():
        master = master.merge(
            long_df.rename(columns={"score": pred_name}),
            on=["Position", "WT residue", "MT"],
            how="left",
            validate="one_to_one",
        )

    return master


def ensure_model_features(df: pd.DataFrame, predictors: List[str]) -> pd.DataFrame:
    """
    Make sure all model features exist as columns.

    For predictors that were not provided, corresponding columns are added
    and filled with NaN so the fitted IterativeImputer in apply_pipeline.py
    can impute them.
    """
    df = df.copy()
    for col in predictors:
        if col not in df.columns:
            df[col] = np.nan
    return df


def add_mutation_code(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct a simple Mutation string like 'V506A' from WT, Position, MT.
    """
    df = df.copy()
    df["Mutation"] = df["WT"] + df["Position"].astype(str) + df["MT"]
    return df


def merge_predictors(args, predictor_paths: Dict[str, Path], predictors: List[str]) -> pd.DataFrame:
    """
    High-level entry point used by the main script when --input is NOT given.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed arguments; only args.add_mutation_code is used here.
    predictor_paths : dict[str, Path]
        Mapping from predictor name (e.g. 'provean') to CSV path.
    predictors: list[str]
        List of supported predictors
    Returns
    -------
    pd.DataFrame
        DataFrame ready to be passed to apply_pipeline.run_pipeline,
        containing at least the 9 model feature columns, plus:
            - Position, WT, MT
            - Mutation (if args.add_mutation_code is True)
    """
    if not predictor_paths:
        raise ValueError("merge_predictors called with empty predictor_paths")

    # 1) Load all predictors (wide -> long)
    predictor_longs: Dict[str, pd.DataFrame] = {}
    for name, path in predictor_paths.items():
        logger.info(f"Loading predictor '{name}' from {path}")
        predictor_longs[name] = load_predictor_wide(path)

    # 2) Merge all predictors into a union table
    df = merge_all_predictors(predictor_longs)

    # 3) Ensure all model features exist (missing predictors -> NaN columns)
    df = ensure_model_features(df, predictors)

    # 4) Optional: add 'Mutation' string
    if getattr(args, "add_mutation_code", False):
        df = add_mutation_code(df)

    return df