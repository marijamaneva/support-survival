"""Permanent guard-rails for the SUPPORT survival project.

Encodes, as executable code, the checks a careful reviewer would run by hand
before trusting any number in this repo: a Kaplan-Meier curve that isn't
monotonic, a leaked target column, or a suspiciously perfect AUROC/C-index
are all symptoms of the same family of bugs (data leakage, broken
preprocessing, a misconfigured baseline).

Run standalone:

    python scripts/sanity_checks.py

Exits 0 if every check passes, 1 otherwise. Also run by
`tests/test_sanity_checks.py` (as a subprocess), so `pytest -q` re-verifies
these on every change.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from support_survival import data, eda, evaluate, features, models

COX_COVARIATES = ["age", "n_comorbidities", "mean_bp", "heart_rate", "serum_creatinine", "wbc"]
COX_CANCER_DUMMIES = ["cancer_present", "cancer_metastatic"]
AUROC_CEILING = 0.90
COX_CONCORDANCE_RANGE = (0.5, 0.85)


@dataclass
class CheckResult:
    name: str
    passed: bool
    message: str


def check_km_monotonic() -> CheckResult:
    """(a) The overall Kaplan-Meier survival function must never increase."""
    df = data.load()
    kmf = eda.plot_km_overall(df)
    plt.close("all")

    sf = kmf.survival_function_.iloc[:, 0].to_numpy()
    diffs = np.diff(sf)
    is_monotonic = bool((diffs <= 1e-12).all())
    starts_at_one = bool(abs(sf[0] - 1.0) < 1e-9)
    passed = is_monotonic and starts_at_one

    if passed:
        message = f"KM survival function is non-increasing and starts at {sf[0]:.6f} ({len(sf)} steps)."
    else:
        bad = int((diffs > 1e-12).sum())
        message = (
            f"KM survival function violates monotonicity at {bad} step(s), or does not start "
            f"at 1.0 (starts at {sf[0]:.6f}). This should be mathematically impossible for a "
            f"Kaplan-Meier estimator -- check eda.plot_km_overall / the lifelines version."
        )
    return CheckResult("km_monotonic", passed, message)


def check_targets_excluded_from_features() -> CheckResult:
    """(b) `time`/`event` must never appear as predictive covariates, binary or Cox."""
    df = data.load()
    feat = features.build_features(df)
    binary_cols = evaluate.feature_columns(feat)
    leaked_binary = [c for c in ("time", "event") if c in binary_cols]

    df2 = models.encode_cancer_stage(df)
    cph = models.fit_cox(df2, COX_COVARIATES + COX_CANCER_DUMMIES)
    leaked_cox = [c for c in ("time", "event") if c in cph.summary.index]

    passed = not leaked_binary and not leaked_cox
    if passed:
        message = "`time`/`event` absent from both the binary feature list and the fitted Cox summary."
    else:
        message = (
            f"Target leakage into predictive features! binary_leaked={leaked_binary}, "
            f"cox_leaked={leaked_cox}."
        )
    return CheckResult("targets_excluded_from_features", passed, message)


def check_cox_concordance_range() -> CheckResult:
    """(c) Cross-validated Cox concordance must land in a plausible, non-leaked range."""
    df = data.load()
    df2 = models.encode_cancer_stage(df)
    scores = models.cross_val_concordance(df2, COX_COVARIATES + COX_CANCER_DUMMIES, cv=5)
    mean_score = float(np.mean(scores))
    low, high = COX_CONCORDANCE_RANGE
    passed = low <= mean_score <= high

    if passed:
        message = f"CV concordance = {mean_score:.3f}, within the plausible range [{low}, {high}]."
    elif mean_score > high:
        message = (
            f"CV concordance = {mean_score:.3f} is ABOVE {high} -- suspiciously high for this "
            f"dataset and covariate set. Check for leakage (e.g. a covariate derived from the "
            f"outcome, or imputation/scaling fit on the full dataset) before trusting this number."
        )
    else:
        message = (
            f"CV concordance = {mean_score:.3f} is BELOW {low} -- the model is barely better "
            f"than random ranking. Check the covariate list and data loading."
        )
    return CheckResult("cox_concordance_range", passed, message)


def check_classifier_auroc_below_ceiling() -> CheckResult:
    """(d) Binary-mortality AUROC above 0.90 on this cohort is a near-certain leakage signal."""
    df = data.load()
    feat = features.build_features(df)
    result = evaluate.compare_models(feat, cv=5)
    aurocs = result["auroc"]
    offenders = {k: v for k, v in aurocs.items() if v >= AUROC_CEILING}
    passed = not offenders

    if passed:
        message = f"AUROC below {AUROC_CEILING} for both models: {aurocs}."
    else:
        message = (
            f"AUROC >= {AUROC_CEILING} for {list(offenders)}: {offenders}. This dataset's binary "
            f"mortality signal does not support AUROC this high -- check for leakage (e.g. a "
            f"feature derived from `event`, or CV folds that leak across patients) before "
            f"treating this as a good result."
        )
    return CheckResult("classifier_auroc_below_ceiling", passed, message)


CHECKS = [
    check_km_monotonic,
    check_targets_excluded_from_features,
    check_cox_concordance_range,
    check_classifier_auroc_below_ceiling,
]


def run_all() -> list[CheckResult]:
    return [check() for check in CHECKS]


def main() -> int:
    results = run_all()
    print("Sanity checks")
    print("-" * 60)
    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.name}")
        print(f"       {r.message}")
        all_passed = all_passed and r.passed
    print("-" * 60)
    print("ALL CHECKS PASSED" if all_passed else "SOME CHECKS FAILED")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
