"""Tests for EDA helpers."""
import matplotlib
matplotlib.use("Agg")  # headless, no display needed for tests

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from support_survival import eda


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _toy() -> pd.DataFrame:
    return pd.DataFrame({
        "time": [5, 10, 15, 20, 25, 30, 35, 40],
        "event": [1, 0, 1, 1, 0, 1, 0, 1],
        "cancer": [0, 0, 1, 1, 1, 2, 2, 2],
        "mean_bp": [80, 0, 90, 70, 0, 85, 95, 100],
    })


def test_outcome_summary_counts():
    summary = eda.outcome_summary(_toy())
    assert summary.loc[0, "n_patients"] == 8
    assert summary.loc[0, "n_deaths"] == 5
    assert summary.loc[0, "n_censored"] == 3
    assert summary.loc[0, "event_rate"] == pytest.approx(5 / 8)


def test_scan_implausible_counts_zeros_and_sorts_desc():
    out = eda.scan_implausible(_toy(), ["mean_bp", "cancer"])
    row = out[out["column"] == "mean_bp"].iloc[0]
    assert row["n_zero"] == 2
    assert row["pct_zero"] == pytest.approx(2 / 8)
    # cancer has 2 zeros too, but mean_bp comes first by construction; check ordering holds
    assert out["n_zero"].is_monotonic_decreasing


def test_plot_km_overall_returns_fitted_kmf():
    kmf = eda.plot_km_overall(_toy())
    assert kmf.event_observed.sum() == 5


def test_plot_feature_distributions_returns_figure_with_one_axes_per_col():
    fig = eda.plot_feature_distributions(_toy(), ["time", "mean_bp"])
    assert len(fig.axes) >= 2


def test_plot_km_by_group_requires_two_groups():
    df = _toy()
    df["constant"] = 1
    with pytest.raises(ValueError):
        eda.plot_km_by_group(df, "constant")


def test_plot_km_by_group_handles_three_levels_and_returns_logrank():
    result = eda.plot_km_by_group(_toy(), "cancer")
    assert set(result["fitters"].keys()) == {0, 1, 2}
    assert 0 <= result["logrank"].p_value <= 1
