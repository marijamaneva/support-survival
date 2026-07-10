"""Tests for held-out validation, calibration, and fairness utilities."""
import matplotlib
matplotlib.use("Agg")  # headless, no display needed for tests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from support_survival import validate


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _toy_feat(n=200, seed=0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    x1 = rng.normal(0, 1, n)
    logit = 1.2 * x1
    p = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(size=n) < p).astype(int)
    return pd.DataFrame({"x1": x1, "event": y})


def test_split_train_val_test_sizes_and_no_overlap():
    feat = _toy_feat(n=500)
    train, val, test = validate.split_train_val_test(feat, test_size=0.2, val_size=0.2)
    assert len(train) + len(val) + len(test) == len(feat)
    # No row appears in more than one split.
    all_idx = pd.concat([train, val, test]).index
    assert len(all_idx) == len(set(all_idx))
    # Roughly the requested proportions.
    assert abs(len(test) / len(feat) - 0.2) < 0.02
    assert abs(len(val) / len(feat) - 0.2) < 0.02


def test_split_train_val_test_is_reproducible():
    feat = _toy_feat(n=300)
    t1, v1, s1 = validate.split_train_val_test(feat, random_state=7)
    t2, v2, s2 = validate.split_train_val_test(feat, random_state=7)
    pd.testing.assert_frame_equal(t1, t2)
    pd.testing.assert_frame_equal(s1, s2)


def test_brier_perfect_forecast_is_zero():
    y = np.array([1, 0, 1, 0])
    proba = np.array([1.0, 0.0, 1.0, 0.0])
    assert validate.brier(y, proba) == pytest.approx(0.0)


def test_brier_constant_half_guess_on_balanced_outcome():
    y = np.array([1, 0, 1, 0])
    proba = np.array([0.5, 0.5, 0.5, 0.5])
    assert validate.brier(y, proba) == pytest.approx(0.25)


def test_fit_isotonic_recalibrator_improves_a_miscalibrated_forecast():
    rng = np.random.RandomState(0)
    n = 500
    true_p = rng.uniform(0, 1, n)
    y = (rng.uniform(size=n) < true_p).astype(int)
    # Miscalibrated: systematically overconfident (pushed toward the extremes).
    miscalibrated = np.clip(true_p * 1.8 - 0.4, 0.01, 0.99)

    ir = validate.fit_isotonic_recalibrator(miscalibrated, y)
    recalibrated = ir.predict(miscalibrated)

    assert validate.brier(y, recalibrated) < validate.brier(y, miscalibrated)


def test_bootstrap_ci_contains_point_estimate_and_has_lower_le_upper():
    rng = np.random.RandomState(0)
    n = 300
    y = rng.randint(0, 2, n)
    proba = rng.uniform(0, 1, n)
    result = validate.bootstrap_ci(y, proba, validate.brier, n_boot=200, random_state=1)
    assert result["lower"] <= result["point"] <= result["upper"]


def test_subgroup_report_has_one_row_per_group_with_expected_columns():
    df = pd.DataFrame({
        "event": [1, 0, 1, 0, 1, 0, 1, 0],
        "sex": [0, 0, 0, 0, 1, 1, 1, 1],
    })
    proba = np.array([0.9, 0.1, 0.8, 0.2, 0.6, 0.4, 0.7, 0.3])
    out = validate.subgroup_report(df, "event", proba, "sex")
    assert list(out["group"]) == [0, 1]
    assert set(out.columns) == {"group", "n", "event_rate", "auroc", "brier"}
    assert (out["n"] == 4).all()


def test_worst_predictions_returns_largest_absolute_errors_first():
    df = pd.DataFrame({"age": [50, 60, 70, 80], "event": [1, 0, 1, 0]})
    proba = np.array([0.1, 0.9, 0.5, 0.5])  # errors: 0.9, 0.9, 0.5, 0.5
    out = validate.worst_predictions(df, "event", proba, ["age"], n=2)
    assert len(out) == 2
    assert out["abs_error"].is_monotonic_decreasing
    assert set(out["age"]) == {50, 60}


def test_plot_calibration_before_after_returns_figure():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 100)
    raw = rng.uniform(0, 1, 100)
    recal = rng.uniform(0, 1, 100)
    fig = validate.plot_calibration_before_after(y, raw, recal)
    assert len(fig.axes) == 1


def test_bootstrap_cox_concordance_ci_contains_point_estimate():
    from support_survival import models

    rng = np.random.RandomState(0)
    n = 150
    age = rng.normal(60, 10, n)
    time_ = rng.exponential(200, n)
    event = rng.randint(0, 2, n)
    df = pd.DataFrame({"age": age, "time": time_, "event": event})
    train, test = df.iloc[:100].reset_index(drop=True), df.iloc[100:].reset_index(drop=True)

    cph = models.fit_cox(train, ["age"])
    result = validate.bootstrap_cox_concordance_ci(cph, test, ["age"], n_boot=100, random_state=1)
    assert result["lower"] <= result["point"] <= result["upper"]
    assert 0 <= result["point"] <= 1
