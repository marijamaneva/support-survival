"""Binary-mortality model comparison and evaluation for the SUPPORT cohort.

Cross-validated comparison of logistic regression vs gradient boosting, plus
the plots used to report it honestly (ROC/PR, calibration, SHAP). Kept here
(not in `notebooks/03_baselines.ipynb`) so it is testable and reusable, per the
project's notebook-is-narrative-only convention.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from support_survival import models
from support_survival.data import ROOT

REPORTS_DIR = ROOT / "reports"

MODEL_LABELS = {"logistic": "Logistic regression", "gradient_boosting": "Gradient boosting"}


def feature_columns(feat: pd.DataFrame) -> list[str]:
    """All engineered columns except the survival targets (`time`, `event`)."""
    return [c for c in feat.columns if c not in ("time", "event")]


def compare_models(
    feat: pd.DataFrame, y_col: str = "event", cv: int = 5, random_state: int = 42
) -> dict:
    """Out-of-fold predicted probabilities for logistic regression and gradient
    boosting, using identical stratified folds for both so the comparison is fair.

    `feat` is expected to already be the output of `features.build_features`
    (raw covariates + missingness indicators + derived clinical flags).
    """
    numeric_cols = feature_columns(feat)
    X = feat[numeric_cols]
    y = feat[y_col].to_numpy()
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)

    logistic_proba = cross_val_predict(
        models.logistic_baseline(numeric_cols), X, y, cv=skf, method="predict_proba"
    )[:, 1]
    gb_proba = cross_val_predict(
        models.gradient_boosting(), X, y, cv=skf, method="predict_proba"
    )[:, 1]

    return {
        "y_true": y,
        "logistic": logistic_proba,
        "gradient_boosting": gb_proba,
        "auroc": {
            "logistic": roc_auc_score(y, logistic_proba),
            "gradient_boosting": roc_auc_score(y, gb_proba),
        },
        "average_precision": {
            "logistic": average_precision_score(y, logistic_proba),
            "gradient_boosting": average_precision_score(y, gb_proba),
        },
    }


def plot_roc_pr(result: dict, save_path: Path | None = None) -> plt.Figure:
    """Side-by-side ROC and precision-recall curves for each model in `result`."""
    y_true = result["y_true"]
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(11, 5))
    for key, label in MODEL_LABELS.items():
        proba = result[key]
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax_roc.plot(fpr, tpr, label=f"{label} (AUROC={roc_auc_score(y_true, proba):.3f})")
        precision, recall, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax_pr.plot(recall, precision, label=f"{label} (AP={ap:.3f})")
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("ROC")
    ax_roc.legend()
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall")
    ax_pr.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_calibration(result: dict, n_bins: int = 10, save_path: Path | None = None) -> plt.Figure:
    """Reliability diagram: mean predicted probability vs observed event frequency,
    per model, bucketed into `n_bins` equal-frequency bins.
    """
    y_true = result["y_true"]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfect calibration")
    for key, label in MODEL_LABELS.items():
        frac_pos, mean_pred = calibration_curve(
            y_true, result[key], n_bins=n_bins, strategy="quantile"
        )
        ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event frequency")
    ax.set_title("Calibration")
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def shap_summary(feat: pd.DataFrame, y_col: str = "event", save_path: Path | None = None):
    """Fit gradient boosting on the full cohort and return its SHAP values.

    For interpretation, not evaluation: unlike `compare_models`, this model is
    fit on the full data with no held-out split, so the SHAP values describe
    what the model learned, not out-of-sample performance.
    """
    import shap

    numeric_cols = feature_columns(feat)
    X = feat[numeric_cols]
    y = feat[y_col].to_numpy()

    gb = models.gradient_boosting()
    gb.fit(X, y)

    explainer = shap.TreeExplainer(gb)
    shap_values = explainer(X)

    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    if save_path is not None:
        plt.gcf().savefig(save_path, dpi=150, bbox_inches="tight")
    return shap_values
