"""Exploratory data analysis helpers for the SUPPORT cohort.

Summary statistics and plots used by `notebooks/01_eda.ipynb`. Kept here (not in
the notebook) so they are testable and reusable, per the project's
notebook-is-narrative-only convention.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

from support_survival.data import ROOT

REPORTS_DIR = ROOT / "reports"


def outcome_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One-row summary of events vs censoring and follow-up time."""
    n = len(df)
    n_events = int(df["event"].sum())
    return pd.DataFrame({
        "n_patients": [n],
        "n_deaths": [n_events],
        "n_censored": [n - n_events],
        "event_rate": [n_events / n],
        "median_followup_days": [df["time"].median()],
        "max_followup_days": [df["time"].max()],
    })


def scan_implausible(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Count zero values for each column, to spot physiologically impossible zeros.

    A zero is only a data-quality problem for measurements that cannot truly be
    zero in a live patient (e.g. blood pressure, heart rate); the caller decides
    which columns to flag based on clinical plausibility.
    """
    rows = [
        {"column": col, "n_zero": int((df[col] == 0).sum()), "pct_zero": float((df[col] == 0).mean())}
        for col in cols
    ]
    return pd.DataFrame(rows).sort_values("n_zero", ascending=False).reset_index(drop=True)


def plot_km_overall(df: pd.DataFrame, save_path: Path | None = None) -> KaplanMeierFitter:
    """Fit and plot the overall Kaplan-Meier survival curve."""
    kmf = KaplanMeierFitter()
    kmf.fit(df["time"], df["event"], label="Overall")
    ax = kmf.plot_survival_function()
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival probability")
    ax.set_title("Kaplan-Meier — overall survival")
    if save_path is not None:
        ax.figure.savefig(save_path, dpi=150, bbox_inches="tight")
    return kmf


def plot_feature_distributions(
    df: pd.DataFrame, cols: list[str], save_path: Path | None = None, ncols: int = 4
) -> plt.Figure:
    """Grid of histograms, one per column, to eyeball distributions and outliers."""
    nrows = -(-len(cols) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 2.8 * nrows))
    axes = axes.flatten()
    for ax, col in zip(axes, cols):
        ax.hist(df[col].dropna(), bins=30)
        ax.set_title(col, fontsize=10)
    for ax in axes[len(cols):]:
        ax.axis("off")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_km_by_group(df: pd.DataFrame, group_col: str, save_path: Path | None = None) -> dict:
    """Stratified KM curves for each level of `group_col`, plus a log-rank test.

    Uses the multivariate log-rank test so it works for two or more groups (e.g.
    `cancer` has three levels in SUPPORT: none / yes / metastatic).
    """
    sub = df.dropna(subset=[group_col, "time", "event"])
    groups = sorted(sub[group_col].unique())
    if len(groups) < 2:
        raise ValueError(f"need at least 2 groups in {group_col!r}, got {groups}")

    fig, ax = plt.subplots(figsize=(7, 5))
    fitters = {}
    for g in groups:
        mask = sub[group_col] == g
        kmf = KaplanMeierFitter()
        kmf.fit(sub.loc[mask, "time"], sub.loc[mask, "event"], label=f"{group_col}={g}")
        kmf.plot_survival_function(ax=ax)
        fitters[g] = kmf

    result = multivariate_logrank_test(sub["time"], sub[group_col], sub["event"])
    ax.set_xlabel("Days")
    ax.set_ylabel("Survival probability")
    ax.set_title(f"Kaplan-Meier by {group_col} (log-rank p={result.p_value:.4g})")
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return {"fitters": fitters, "logrank": result}
