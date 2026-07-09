"""Tests for model construction."""
import pandas as pd

from support_survival import models


def _toy():
    return pd.DataFrame({
        "age": [50, 60, 70, 80, 55, 65, 75, 85],
        "mean_bp": [80, 70, 60, 90, 85, 75, 65, 95],
    }), pd.Series([0, 0, 1, 1, 0, 1, 1, 0])


def test_logistic_baseline_has_no_class_weight_balancing():
    pipe = models.logistic_baseline(["age", "mean_bp"])
    assert pipe.named_steps["clf"].class_weight is None


def test_logistic_baseline_fits_and_predicts_probabilities():
    X, y = _toy()
    pipe = models.logistic_baseline(["age", "mean_bp"])
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert len(proba) == len(X)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_gradient_boosting_fits_and_predicts_probabilities():
    X, y = _toy()
    gb = models.gradient_boosting()
    gb.fit(X, y)
    proba = gb.predict_proba(X)[:, 1]
    assert len(proba) == len(X)
    assert ((proba >= 0) & (proba <= 1)).all()
