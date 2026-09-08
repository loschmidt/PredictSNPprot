"""
evaluate.py
------------------
Calculates AUC, Accuracy, Precision, Recall, MCC, and F1 for every score
column in a CSV, with optional bootstrap confidence intervals.

Threshold selection
-------------------
All threshold-dependent metrics (Accuracy, Precision, Recall, MCC, F1)
use the **Youden J** optimal threshold derived from --train-csv.
The threshold is then applied unchanged to --eval-csv.

This separation ensures no information leakage from the evaluation set:
  - Youden thresholds (individual tools + ensembles) come from trainset
  - Ensemble majority-vote per-tool thresholds come from trainset
  - All metrics are reported on evalset

Bootstrap
---------
When --bootstrap N is provided (recommended N ≥ 1000), metrics are estimated
on N stratified resamples of the --eval-csv (sampling with replacement,
preserving class ratio). The threshold is fixed to the trainset-derived
value in every resample — double leakage prevention.

Baseline ensembles
------------------
When --ensemble is set, four baselines are computed:
  simple_average_{fast|full}  — row-wise mean of _prep columns
  majority_vote_{fast|full}   — fraction of positive binary votes,
                                 where each tool's vote threshold is
                                 derived from the trainset

Usage
-----
    python evaluate.py \
        --train-csv  trainset.csv \
        --eval-csv   testset.csv \
        --output-csv metrics.csv

    python evaluate.py \
        --train-csv trainset.csv --eval-csv testset.csv \
        --output-csv metrics.csv --bootstrap 1000 --ensemble

    # Same split (legacy / exploration mode — thresholds and eval on same data)
    python evaluate.py \
        --train-csv results.csv --eval-csv results.csv \
        --output-csv metrics.csv
"""

from __future__ import annotations

import argparse
import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_DEFAULT_FLIP_COLS = frozenset(
    ["esmVariants", "provean", "vespag", "phact", "gemme", "VespaG"]
)

_DEFAULT_COLS_IN_FAST = [
    "3cnet_prep", "esmVariants_prep", "provean_prep",
    "vespag_prep", "vespal_prep",
]
_DEFAULT_COLS_IN_FULL = [
    "3cnet_prep", "esmVariants_prep", "provean_prep",
    "vespag_prep",  "vespal_prep",
    "varipred_prep", "phact_prep", "phactboost_prep", "gemme_prep",
]


# ═══════════════════════════════════════════════════════════════════════════
# Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_csv(path: str, label_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"Label column {label_col!r} not found in {path!r}.")
    if df[label_col].dtype == object:
        unique = df[label_col].unique()
        print(f"  Label is categorical {unique} in {path} — encoding via factorize.")
        df[label_col] = pd.factorize(df[label_col])[0]
    return df


def _clean_pair(
    train_df:  pd.DataFrame,
    eval_df:   pd.DataFrame,
    label_col: str,
    col:       str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Return (y_train, s_train, y_eval, s_eval) for one score column,
    dropping NaNs independently in each split.
    Returns None when either split has no valid rows or only one class.
    """
    def _extract(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
        sub = df[[label_col, col]].copy()
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub = sub.dropna()
        if len(sub) == 0 or sub[label_col].nunique() < 2:
            return None
        return sub[label_col].values.astype(int), sub[col].values.astype(float)

    train_pair = _extract(train_df)
    eval_pair  = _extract(eval_df)

    if train_pair is None or eval_pair is None:
        return None

    return train_pair[0], train_pair[1], eval_pair[0], eval_pair[1]


# ═══════════════════════════════════════════════════════════════════════════
# Metric computation
# ═══════════════════════════════════════════════════════════════════════════

class MetricBundle(NamedTuple):
    auc:       float
    accuracy:  float
    precision: float
    recall:    float
    mcc:       float
    f1:        float
    threshold: float


def _youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Threshold that maximises Youden J = Sensitivity + Specificity - 1."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    best_idx = int(np.argmax(tpr - fpr))
    return float(thresholds[best_idx])


def _compute_metrics(
    y_true:    np.ndarray,
    y_score:   np.ndarray,
    threshold: float,
) -> MetricBundle:
    """Compute all six metrics given a pre-determined threshold."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = (y_score >= threshold).astype(int)
        return MetricBundle(
            auc       = float(roc_auc_score(y_true, y_score)),
            accuracy  = float(accuracy_score(y_true, y_pred)),
            precision = float(precision_score(y_true, y_pred, zero_division=0)),
            recall    = float(recall_score(y_true, y_pred, zero_division=0)),
            mcc       = float(matthews_corrcoef(y_true, y_pred)),
            f1        = float(f1_score(y_true, y_pred, zero_division=0)),
            threshold = threshold,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Bootstrap
# ═══════════════════════════════════════════════════════════════════════════

def _bootstrap_metrics(
    y_true:       np.ndarray,
    y_score:      np.ndarray,
    threshold:    float,
    n_iterations: int,
    random_state: int = 42,
) -> dict[str, float]:
    """
    Stratified bootstrap on the eval set with a fixed (train-derived) threshold.

    The threshold is never re-estimated from bootstrap resamples, preventing
    both optimistic bias and eval-set leakage.
    """
    rng     = np.random.default_rng(random_state)
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    boot: dict[str, list[float]] = {
        "auc": [], "accuracy": [], "precision": [],
        "recall": [], "mcc": [], "f1": [],
    }

    for _ in range(n_iterations):
        s_pos = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        s_neg = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx   = np.concatenate([s_pos, s_neg])
        yt    = y_true[idx]
        ys    = y_score[idx]

        if len(np.unique(yt)) < 2:
            continue
        try:
            m = _compute_metrics(yt, ys, threshold=threshold)
            for key in boot:
                boot[key].append(getattr(m, key))
        except Exception:
            continue

    return {k: float(np.std(v)) if v else float("nan") for k, v in boot.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Per-column evaluation
# ═══════════════════════════════════════════════════════════════════════════

def _evaluate_column(
    y_train:      np.ndarray,
    s_train:      np.ndarray,
    y_eval:       np.ndarray,
    s_eval:       np.ndarray,
    col:          str,
    n_bootstrap:  int,
    random_state: int,
) -> dict:
    """
    Derive threshold from trainset, compute all metrics on evalset.
    """
    threshold = _youden_threshold(y_train, s_train)
    point     = _compute_metrics(y_eval, s_eval, threshold)

    row: dict = {
        "Score_Column":      col,
        "N_Train_Samples":   len(y_train),
        "N_Eval_Samples":    len(y_eval),
        "Threshold":         round(threshold, 6),
        "AUC":               round(point.auc,       4),
        "Accuracy":          round(point.accuracy,  4),
        "Precision":         round(point.precision, 4),
        "Recall":            round(point.recall,    4),
        "MCC":               round(point.mcc,       4),
        "F1":                round(point.f1,        4),
    }

    if n_bootstrap > 0:
        std = _bootstrap_metrics(
            y_eval, s_eval,
            threshold    = threshold,
            n_iterations = n_bootstrap,
            random_state = random_state,
        )
        for metric in ("AUC", "Accuracy", "Precision", "Recall", "MCC", "F1"):
            row[f"{metric}_std"] = round(std.get(metric.lower(), float("nan")), 4)

    return row


# ═══════════════════════════════════════════════════════════════════════════
# Baseline ensembles
# ═══════════════════════════════════════════════════════════════════════════

def _complete_case_mask(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].notna().all(axis=1)


def _per_tool_thresholds_from_train(
    train_df:  pd.DataFrame,
    label_col: str,
    tool_cols: list[str],
) -> dict[str, float]:
    """
    Derive each tool's Youden threshold from the trainset complete-case rows.
    Falls back to the score midpoint if threshold derivation fails.
    """
    mask  = _complete_case_mask(train_df, tool_cols + [label_col])
    sub   = train_df.loc[mask]
    ytrue = sub[label_col].values.astype(int)

    thresholds: dict[str, float] = {}
    for col in tool_cols:
        ys = sub[col].values.astype(float)
        try:
            thresholds[col] = _youden_threshold(ytrue, ys)
        except Exception:
            thresholds[col] = float(np.nanmean(ys))

    return thresholds


def _build_ensemble_arrays(
    df:              pd.DataFrame,
    label_col:       str,
    tool_cols:       list[str],
    tool_thresholds: dict[str, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """
    Build (y_true, avg_scores, vote_scores) for complete-case rows of df.
    tool_thresholds must already be derived from the trainset.
    """
    mask = _complete_case_mask(df, tool_cols + [label_col])
    sub  = df.loc[mask]
    if len(sub) == 0 or sub[label_col].nunique() < 2:
        return None

    y_true     = sub[label_col].values.astype(int)
    avg_scores = sub[tool_cols].mean(axis=1).values.astype(float)

    votes = np.stack(
        [(sub[col].values >= tool_thresholds[col]).astype(float)
         for col in tool_cols],
        axis=1,
    )
    vote_scores = votes.mean(axis=1)  # fraction of positive votes ∈ [0,1]

    return y_true, avg_scores, vote_scores

def _cross_validate_ensemble(
    train_df:        pd.DataFrame,
    label_col:       str,
    tool_cols:       list[str],
    tool_thresholds: dict[str, float],  # used only for "average" type
    ensemble_type:   str,               # "average" | "vote"
    n_splits:        int = 5,
    random_state:    int = 42,
) -> None:
    from sklearn.model_selection import StratifiedKFold

    mask = _complete_case_mask(train_df, tool_cols + [label_col])
    sub  = train_df.loc[mask].reset_index(drop=True)
    y    = sub[label_col].values.astype(int)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    train_aucs, val_aucs = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(sub, y), 1):
        tr, val   = sub.iloc[tr_idx], sub.iloc[val_idx]
        yt_tr, yt_val = y[tr_idx], y[val_idx]

        if ensemble_type == "average":
            s_tr  = tr[tool_cols].mean(axis=1).values
            s_val = val[tool_cols].mean(axis=1).values

        else:  # "vote" — derive per-tool thresholds from THIS fold's train rows only
            fold_thresholds = {}
            for c in tool_cols:
                try:
                    fold_thresholds[c] = _youden_threshold(yt_tr, tr[c].values.astype(float))
                except Exception:
                    fold_thresholds[c] = float(np.nanmean(tr[c].values))

            s_tr  = np.stack(
                [(tr[c].values  >= fold_thresholds[c]).astype(float) for c in tool_cols], axis=1
            ).mean(axis=1)
            s_val = np.stack(
                [(val[c].values >= fold_thresholds[c]).astype(float) for c in tool_cols], axis=1
            ).mean(axis=1)

        auc_tr  = float(roc_auc_score(yt_tr,  s_tr))
        auc_val = float(roc_auc_score(yt_val, s_val))
        train_aucs.append(auc_tr)
        val_aucs.append(auc_val)
        print(f"      fold {fold}  train_AUC={auc_tr:.4f}  val_AUC={auc_val:.4f}")

    print(f"      CV summary  train={np.mean(train_aucs):.4f}±{np.std(train_aucs):.4f}"
          f"  val={np.mean(val_aucs):.4f}±{np.std(val_aucs):.4f}")
    

def build_ensemble_results(
    train_df:     pd.DataFrame,
    eval_df:      pd.DataFrame,
    label_col:    str,
    n_bootstrap:  int,
    random_state: int,
) -> list[dict]:
    """
    Compute and evaluate all four baseline ensembles.
    Thresholds (per-tool and ensemble-level) are always derived from train_df.
    Metrics are always reported on eval_df.
    """
    configs = {
        "fast": _DEFAULT_COLS_IN_FAST,
        "full": _DEFAULT_COLS_IN_FULL,
    }
    results: list[dict] = []

    for config_name, tool_cols in configs.items():
        missing_train = [c for c in tool_cols if c not in train_df.columns]
        missing_eval  = [c for c in tool_cols if c not in eval_df.columns]

        if missing_train or missing_eval:
            missing = set(missing_train) | set(missing_eval)
            print(f"  SKIP ensemble '{config_name}' — missing columns: {missing}")
            continue

        # ── derive all thresholds from train ──────────────────────────────
        tool_thresholds = _per_tool_thresholds_from_train(
            train_df, label_col, tool_cols
        )

        # ── build arrays for train and eval ───────────────────────────────
        train_arrays = _build_ensemble_arrays(
            train_df, label_col, tool_cols, tool_thresholds
        )
        eval_arrays = _build_ensemble_arrays(
            eval_df, label_col, tool_cols, tool_thresholds
        )

        if train_arrays is None:
            print(f"  SKIP ensemble '{config_name}' — no complete rows in trainset")
            continue
        if eval_arrays is None:
            print(f"  SKIP ensemble '{config_name}' — no complete rows in evalset")
            continue

        yt_train, avg_train, vote_train = train_arrays
        yt_eval,  avg_eval,  vote_eval  = eval_arrays

        n_train = len(yt_train)
        n_eval  = len(yt_eval)
        print(f"  Ensemble '{config_name}': "
              f"train={n_train} rows, eval={n_eval} rows, {len(tool_cols)} tools")

        for ensemble_name, s_train, s_eval in [
            (f"simple_average_{config_name}", avg_train,  avg_eval),
            (f"majority_vote_{config_name}",  vote_train, vote_eval),
        ]:
            
            print(f"    {ensemble_name} — 5-fold CV on trainset:")
            _cross_validate_ensemble(
                train_df, label_col, tool_cols, tool_thresholds,
                ensemble_type = "average" if "average" in ensemble_name else "vote",
                random_state  = random_state,
            )
            try:
                # Ensemble-level Youden threshold derived from train
                ens_threshold = _youden_threshold(yt_train, s_train)
                point         = _compute_metrics(yt_eval, s_eval, ens_threshold)

                row: dict = {
                    "Score_Column":    ensemble_name,
                    "N_Train_Samples": n_train,
                    "N_Eval_Samples":  n_eval,
                    "Threshold":       round(ens_threshold, 6),
                    "AUC":             round(point.auc,       4),
                    "Accuracy":        round(point.accuracy,  4),
                    "Precision":       round(point.precision, 4),
                    "Recall":          round(point.recall,    4),
                    "MCC":             round(point.mcc,       4),
                    "F1":              round(point.f1,        4),
                }

                if n_bootstrap > 0:
                    std = _bootstrap_metrics(
                        yt_eval, s_eval,
                        threshold    = ens_threshold,
                        n_iterations = n_bootstrap,
                        random_state = random_state,
                    )
                    for metric in ("AUC","Accuracy","Precision","Recall","MCC","F1"):
                        row[f"{metric}_std"] = round(
                            std.get(metric.lower(), float("nan")), 4
                        )

                results.append(row)
                print(f"    {ensemble_name:<38}  AUC={row['AUC']:.4f}")
            except Exception as exc:
                print(f"  ERROR '{ensemble_name}': {exc}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every score column: AUC, Accuracy, Precision, Recall, "
            "MCC, F1.  Youden threshold derived from --train-csv, applied to "
            "--eval-csv.  Optional bootstrap ±std and baseline ensembles."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-csv",   required=True,
                        help="Trainset CSV — used ONLY for threshold derivation.")
    parser.add_argument("--eval-csv",    required=True,
                        help="Evaluation CSV — all metrics are reported on this set.")
    parser.add_argument("--output-csv",  default="evaluation_results.csv",
                        help="Path to save the results CSV.")
    parser.add_argument("--label-col",   default="Label",
                        help="Name of the binary label column (0/1).")
    parser.add_argument("--start-col",   default="3cnet",
                        help="First score column in the eval CSV; all columns "
                             "from here onward are treated as scores.")
    parser.add_argument("--flip-cols",   nargs="*", default=None,
                        help="Columns negatively correlated with pathogenicity "
                             "(negated before evaluation). Defaults to: "
                             "esmVariants provean vespag phact gemme VespaG")
    parser.add_argument("--bootstrap",   type=int, default=0, metavar="N",
                        help="Bootstrap iterations for ±std on eval set. "
                             "0 = point estimates only.  Recommended: 1000.")
    parser.add_argument("--ensemble",    action="store_true",
                        help="Add simple_average and majority_vote baselines "
                             "for fast (5-tool) and full (9-tool) configs.")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    flip_cols = []
    #    set(args.flip_cols) if args.flip_cols is not None else _DEFAULT_FLIP_COLS
    #)

    # ── Load both splits ───────────────────────────────────────────────────
    print(f"Loading train : {args.train_csv} …")
    train_df = _load_csv(args.train_csv, args.label_col)
    print(f"  {len(train_df):,} rows, {train_df.shape[1]} columns")

    print(f"Loading eval  : {args.eval_csv} …")
    eval_df = _load_csv(args.eval_csv, args.label_col)
    print(f"  {len(eval_df):,} rows, {eval_df.shape[1]} columns")

    if args.start_col not in eval_df.columns:
        raise ValueError(
            f"Start column {args.start_col!r} not found in eval CSV."
        )

    start_idx  = eval_df.columns.get_loc(args.start_col)
    score_cols = [c for c in eval_df.columns[start_idx:] if c != args.label_col]
    print(f"Evaluating {len(score_cols)} score columns "
          f"(threshold from train, bootstrap N={args.bootstrap}) …")

    # ── Evaluate individual score columns ──────────────────────────────────
    results: list[dict] = []

    for col in score_cols:
        if col not in train_df.columns:
            print(f"  SKIP  {col!r} — not found in trainset")
            continue

        packed = _clean_pair(train_df, eval_df, args.label_col, col)
        if packed is None:
            n_eval = int(eval_df[col].notna().sum()) if col in eval_df.columns else 0
            results.append({
                "Score_Column":    col,
                "N_Train_Samples": 0,
                "N_Eval_Samples":  n_eval,
                **{k: np.nan for k in
                   ["Threshold","AUC","Accuracy","Precision","Recall","MCC","F1"]},
            })
            print(f"  SKIP  {col!r} — insufficient data or single class in a split")
            continue

        y_train, s_train, y_eval, s_eval = packed

        if col in flip_cols:
            s_train = -s_train
            s_eval  = -s_eval

        try:
            row = _evaluate_column(
                y_train, s_train, y_eval, s_eval, col,
                n_bootstrap  = args.bootstrap,
                random_state = args.random_state,
            )
            if col in flip_cols:
                row["Threshold"] = -row["Threshold"]
            results.append(row)
        except Exception as exc:
            print(f"  ERROR {col!r}: {exc}")
            results.append({
                "Score_Column":    col,
                "N_Train_Samples": len(y_train),
                "N_Eval_Samples":  len(y_eval),
                **{k: np.nan for k in
                   ["Threshold","AUC","Accuracy","Precision","Recall","MCC","F1"]},
            })

    # ── Baseline ensembles ─────────────────────────────────────────────────
    if args.ensemble:
        print(f"Building baseline ensembles …")
        ensemble_rows = build_ensemble_results(
            train_df     = train_df,
            eval_df      = eval_df,
            label_col    = args.label_col,
            n_bootstrap  = args.bootstrap,
            random_state = args.random_state,
        )
        results.extend(ensemble_rows)
        print(f"  Added {len(ensemble_rows)} ensemble entries.")

    # ── Compile and save ───────────────────────────────────────────────────
    results_df = pd.DataFrame(results).sort_values("AUC", ascending=False)
    results_df.to_csv(args.output_csv, index=False)
    print(f"Results saved → {args.output_csv}")

    # ── Console summary ────────────────────────────────────────────────────
    display_cols = ["Score_Column", "N_Train_Samples", "N_Eval_Samples",
                    "Threshold", "AUC", "Accuracy", "Precision",
                    "Recall", "MCC", "F1"]
    if args.bootstrap > 0:
        display_cols += [f"{m}_std" for m in
                         ("AUC","Accuracy","Precision","Recall","MCC","F1")]

    print("--- Evaluation Summary (sorted by AUC ↓) ---")
    print(results_df[[c for c in display_cols if c in results_df.columns]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
