"""Tests for model comparison and evaluation."""
import matplotlib
matplotlib.use("Agg")  # headless, no display needed for tests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from support_survival import evaluate


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _toy_features() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    n = 80
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    logit = 1.5 * x1 - 0.5 * x2
    p = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({
        "age": x1 * 10 + 60,
        "mean_bp": x2 * 10 + 80,
        "time": rng.uniform(1, 500, n),
        "event": y,
    })


def test_feature_columns_excludes_targets():
    cols = evaluate.feature_columns(_toy_features())
    assert set(cols) == {"age", "mean_bp"}


def test_feature_columns_excludes_race():
    feat = _toy_features()
    feat["race"] = 1
    cols = evaluate.feature_columns(feat)
    assert "race" not in cols
    assert set(cols) == {"age", "mean_bp"}


def test_compare_models_returns_valid_probabilities():
    result = evaluate.compare_models(_toy_features(), cv=3)
    n = len(result["y_true"])
    for key in ("logistic", "gradient_boosting"):
        proba = result[key]
        assert len(proba) == n
        assert ((proba >= 0) & (proba <= 1)).all()
        assert 0 <= result["auroc"][key] <= 1
        assert 0 <= result["average_precision"][key] <= 1


def test_compare_models_is_reproducible():
    feat = _toy_features()
    r1 = evaluate.compare_models(feat, cv=3, random_state=1)
    r2 = evaluate.compare_models(feat, cv=3, random_state=1)
    np.testing.assert_array_equal(r1["logistic"], r2["logistic"])


def test_plot_roc_pr_returns_figure_with_two_axes():
    result = evaluate.compare_models(_toy_features(), cv=3)
    fig = evaluate.plot_roc_pr(result)
    assert len(fig.axes) == 2


def test_plot_calibration_returns_figure():
    result = evaluate.compare_models(_toy_features(), cv=3)
    fig = evaluate.plot_calibration(result, n_bins=4)
    assert len(fig.axes) == 1


def test_shap_summary_returns_one_row_per_patient():
    feat = _toy_features()
    shap_values = evaluate.shap_summary(feat)
    assert shap_values.values.shape[0] == len(feat)
