"""Tests for model construction."""
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

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


def _toy_survival(n=100, seed=0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    age = rng.normal(60, 10, n)
    mean_bp = rng.normal(80, 15, n)
    group = rng.randint(0, 2, n)
    risk = 0.03 * age - 0.01 * mean_bp
    baseline = rng.exponential(scale=200, size=n)
    true_time = baseline / np.exp(risk - risk.mean())
    censor_time = rng.uniform(50, 400, n)
    observed_time = np.minimum(true_time, censor_time)
    event = (true_time <= censor_time).astype(int)
    return pd.DataFrame({
        "age": age, "mean_bp": mean_bp, "group": group,
        "time": observed_time, "event": event,
    })


def test_fit_cox_returns_model_with_expected_covariates():
    df = _toy_survival()
    cph = models.fit_cox(df, ["age", "mean_bp"])
    assert set(cph.summary.index) == {"age", "mean_bp"}
    assert 0 <= cph.concordance_index_ <= 1


def test_fit_cox_with_strata_excludes_strata_from_summary():
    df = _toy_survival()
    cph = models.fit_cox(df, ["age"], strata=["group"])
    assert set(cph.summary.index) == {"age"}


def test_encode_cancer_stage_creates_expected_dummies():
    df = pd.DataFrame({"cancer": [0, 1, 2, 1]})
    out = models.encode_cancer_stage(df)
    assert list(out["cancer_present"]) == [0, 1, 0, 1]
    assert list(out["cancer_metastatic"]) == [0, 0, 1, 0]


def test_encode_cancer_stage_does_not_mutate_input():
    df = pd.DataFrame({"cancer": [0, 1, 2]})
    _ = models.encode_cancer_stage(df)
    assert "cancer_present" not in df.columns


def test_check_ph_assumption_returns_dataframe_with_p_values():
    df = _toy_survival()
    cph = models.fit_cox(df, ["age", "mean_bp"])
    result = models.check_ph_assumption(cph, df, ["age", "mean_bp"])
    assert set(result.index) == {"age", "mean_bp"}
    assert "p" in result.columns
    assert ((result["p"] >= 0) & (result["p"] <= 1)).all()


def test_check_ph_assumption_works_on_a_stratified_model():
    df = _toy_survival()
    cph = models.fit_cox(df, ["age"], strata=["group"])
    result = models.check_ph_assumption(cph, df, ["age"])
    assert set(result.index) == {"age"}
    assert "p" in result.columns


def test_cross_val_concordance_returns_one_score_per_fold_in_range():
    df = _toy_survival(n=120)
    scores = models.cross_val_concordance(df, ["age", "mean_bp"], cv=4)
    assert len(scores) == 4
    assert all(0 <= s <= 1 for s in scores)


def test_cross_val_concordance_supports_strata():
    df = _toy_survival(n=120)
    scores = models.cross_val_concordance(df, ["age"], strata=["group"], cv=4)
    assert len(scores) == 4
    assert all(0 <= s <= 1 for s in scores)


def test_cross_val_concordance_handles_missing_values_per_fold():
    df = _toy_survival(n=120)
    df.loc[df.index[:10], "mean_bp"] = float("nan")
    scores = models.cross_val_concordance(df, ["age", "mean_bp"], cv=4)
    assert len(scores) == 4
    assert all(0 <= s <= 1 for s in scores)


def test_cross_val_concordance_imputes_with_train_fold_median_only():
    """Regression test for the leakage question: the value used to fill a
    missing covariate must come from that fold's training data only -- never
    the whole dataset, and never the held-out fold itself.

    Row 0's covariate is missing. With cv=2, it lands in the training set of
    one fold and the test set of the other. Both fills should equal that
    fold's train-only median. The two folds' train medians are deliberately
    different (asserted below), so the test can actually distinguish "used
    the right median" from "used the wrong one."
    """
    n, cv, seed = 20, 2, 0
    df = pd.DataFrame({
        "x": [float(i) for i in range(1, n + 1)],
        "time": np.linspace(10, 200, n),
        "event": [1, 0] * (n // 2),
    })
    df.loc[0, "x"] = np.nan

    splits = list(KFold(n_splits=cv, shuffle=True, random_state=seed).split(df))
    train_medians = [df.iloc[tr]["x"].median() for tr, _ in splits]
    assert train_medians[0] != train_medians[1], (
        "test setup is degenerate: both folds' train medians match, so this "
        "test cannot tell a fold-safe fill from a leaky one"
    )

    captured_train, captured_test = [], []

    class _FakeCox:
        def fit(self, training_df, duration_col=None, event_col=None, strata=None):
            captured_train.append(training_df["x"].reset_index(drop=True))
            return self

        def score(self, df, scoring_method=None):
            captured_test.append(df["x"].reset_index(drop=True))
            return 0.5

    with patch("lifelines.CoxPHFitter", _FakeCox):
        models.cross_val_concordance(df, ["x"], cv=cv, random_state=seed)

    assert len(captured_train) == cv
    for fold, (train_idx, test_idx) in enumerate(splits):
        if 0 in train_idx:
            position = list(train_idx).index(0)
            assert captured_train[fold].iloc[position] == train_medians[fold]
        else:
            position = list(test_idx).index(0)
            assert captured_test[fold].iloc[position] == train_medians[fold]


def test_hazard_ratio_table_has_expected_columns_and_sorting():
    df = _toy_survival()
    cph = models.fit_cox(df, ["age", "mean_bp"])
    table = models.hazard_ratio_table(cph)
    assert list(table.columns) == ["hazard_ratio", "hr_lower_95", "hr_upper_95", "p_value"]
    assert table["p_value"].is_monotonic_increasing
    assert (table["hazard_ratio"] > 0).all()


def test_predict_risk_at_horizons_returns_one_row_per_patient_in_range():
    df = _toy_survival(n=100)
    cph = models.fit_cox(df, ["age", "mean_bp"])
    risk = models.predict_risk_at_horizons(cph, df, ["age", "mean_bp"], horizons=[30, 90])
    assert list(risk.columns) == ["risk_at_30d", "risk_at_90d"]
    assert len(risk) == len(df)
    assert ((risk >= 0) & (risk <= 1)).all().all()


def test_predict_risk_at_horizons_is_non_decreasing_in_time():
    df = _toy_survival(n=100)
    cph = models.fit_cox(df, ["age", "mean_bp"])
    risk = models.predict_risk_at_horizons(cph, df, ["age", "mean_bp"], horizons=[30, 90])
    # Probability of death by day 90 can't be less than by day 30.
    assert (risk["risk_at_90d"] >= risk["risk_at_30d"] - 1e-9).all()
