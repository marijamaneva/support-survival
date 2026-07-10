"""Smoke test for the headless CLI pipeline."""
from support_survival import pipeline


def test_run_quick_returns_expected_keys_and_plausible_values():
    results = pipeline.run(quick=True)
    assert set(results) == {"n_patients", "event_rate", "logistic_auroc_cv", "cox_c_index"}
    assert results["n_patients"] == 8873
    assert 0 <= results["event_rate"] <= 1
    assert 0.5 <= results["logistic_auroc_cv"] <= 0.9
    assert 0.5 <= results["cox_c_index"] <= 0.9
