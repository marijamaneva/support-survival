"""Held-out validation, calibration, uncertainty, and fairness checks.

Phase 3/4 used cross-validation on the full cohort to *compare* candidate
models (logistic vs gradient boosting; Cox covariate/PH choices) — CV is the
right tool for that kind of comparison. Phase 5's job is different: report
final validation numbers for the *already-chosen* model on data that played
no role whatsoever in choosing it. That needs a genuinely held-out split,
fixed once and never revisited — not just another CV fold.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


def split_train_val_test(
    feat: pd.DataFrame,
    y_col: str = "event",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three-way, stratified-by-event split: train / validation / test.

    `test` is touched exactly once, at the end of Phase 5, for final
    reporting. `val` exists so a recalibrator can be fit on data the model
    never trained on, without spending the test set to do it. Returned frames
    keep their original index (not reset) so callers can align predictions.
    """
    train_val, test = train_test_split(
        feat, test_size=test_size, stratify=feat[y_col], random_state=random_state
    )
    val_fraction_of_remainder = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=val_fraction_of_remainder,
        stratify=train_val[y_col], random_state=random_state,
    )
    return train, val, test


def brier(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Brier score: mean squared error between predicted probability and outcome.

    Lower is better; 0 is a perfect forecast, 0.25 is what a constant 50%
    guess scores against a balanced outcome.
    """
    return float(brier_score_loss(y_true, proba))


def fit_isotonic_recalibrator(proba_val: np.ndarray, y_val: np.ndarray) -> IsotonicRegression:
    """Fit an isotonic-regression recalibrator on validation predictions only.

    Never fit this on the test set — that would let the test set influence
    the very predictions later evaluated on it, which is exactly the kind of
    validation leakage Phase 5 exists to avoid.
    """
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(proba_val, y_val)
    return ir


def bootstrap_ci(
    y_true: np.ndarray,
    proba: np.ndarray,
    metric_fn,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Percentile bootstrap CI for any `metric_fn(y_true, proba) -> float`,
    resampling patients (with replacement) from the held-out data provided.
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    point = metric_fn(y_true, proba)
    boot_scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        boot_scores[i] = metric_fn(y_true[idx], proba[idx])
    alpha = (1 - ci) / 2
    lower, upper = np.quantile(boot_scores, [alpha, 1 - alpha])
    return {"point": float(point), "lower": float(lower), "upper": float(upper)}


def bootstrap_cox_concordance_ci(
    cph,
    test_df: pd.DataFrame,
    covariates: list[str],
    strata: list[str] | None = None,
    n_boot: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> dict:
    """Percentile bootstrap CI for a fitted Cox model's concordance, resampling
    patients (with replacement) from the held-out test set only.
    """
    rng = np.random.RandomState(random_state)
    cols = covariates + ["time", "event"] + (strata or [])
    n = len(test_df)
    point = cph.score(test_df[cols], scoring_method="concordance_index")
    boot_scores = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, n)
        sample = test_df.iloc[idx].reset_index(drop=True)
        boot_scores[i] = cph.score(sample[cols], scoring_method="concordance_index")
    alpha = (1 - ci) / 2
    lower, upper = np.quantile(boot_scores, [alpha, 1 - alpha])
    return {"point": float(point), "lower": float(lower), "upper": float(upper)}


def subgroup_report(
    df: pd.DataFrame, y_col: str, proba: np.ndarray, group_col: str
) -> pd.DataFrame:
    """AUROC and Brier score, computed separately per level of `group_col`.

    Expects `df` to have a clean 0..n-1 index whose row order matches `proba`
    (i.e. call `.reset_index(drop=True)` on the frame the predictions came from).
    """
    rows = []
    for level in sorted(df[group_col].dropna().unique()):
        mask = (df[group_col] == level).to_numpy()
        y_sub = df.loc[mask, y_col].to_numpy()
        p_sub = proba[mask]
        auroc = roc_auc_score(y_sub, p_sub) if len(np.unique(y_sub)) > 1 else float("nan")
        rows.append({
            "group": level,
            "n": int(mask.sum()),
            "event_rate": float(y_sub.mean()),
            "auroc": auroc,
            "brier": brier(y_sub, p_sub),
        })
    return pd.DataFrame(rows)


def plot_calibration_before_after(
    y_test: np.ndarray,
    proba_raw: np.ndarray,
    proba_recalibrated: np.ndarray,
    n_bins: int = 10,
    save_path: Path | None = None,
) -> plt.Figure:
    """Reliability diagram comparing raw vs recalibrated predictions on the
    same held-out test set, so any improvement from recalibration is visible.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    for label, proba in [("Raw", proba_raw), ("Recalibrated (isotonic)", proba_recalibrated)]:
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event frequency")
    ax.set_title("Calibration on held-out test set: raw vs recalibrated")
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def worst_predictions(
    df: pd.DataFrame, y_col: str, proba: np.ndarray, cols: list[str], n: int = 15
) -> pd.DataFrame:
    """The `n` patients whose predicted probability was furthest from the truth.

    Expects `df`'s row order to match `proba` (see `subgroup_report`).
    """
    out = df[cols + [y_col]].copy()
    out["predicted_proba"] = proba
    out["abs_error"] = (out[y_col] - out["predicted_proba"]).abs()
    return out.sort_values("abs_error", ascending=False).head(n)
