"""Model construction for the SUPPORT project.

Two modeling views of the same cohort:

1. Binary mortality classification (did the patient die during follow-up?) — for
   familiar ML baselines with calibration.
2. Survival analysis (time-to-death with censoring) — Kaplan-Meier and Cox PH.

Keeping model construction here (rather than in notebooks) means the same objects
are used for experiments, tests, and serving.
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def logistic_baseline(numeric_cols: list[str]) -> Pipeline:
    """Interpretable baseline: median-imputed, scaled logistic regression.

    Logistic regression is the baseline regulated-healthcare reviewers respect,
    and a strong one is often close to gradient boosting — reporting that honestly
    is part of the methodology.

    No `class_weight="balanced"`: measured on this cohort (68%/32% event split),
    it left AUROC unchanged (0.646 vs 0.647) but badly miscalibrated the model,
    inflating predicted risk by 0.13-0.23 across probability bins (see Phase 3
    model card entry). Not imbalanced enough to justify that cost.
    """
    pre = ColumnTransformer(
        [("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols)],
        remainder="drop",
    )
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000)),
    ])


def gradient_boosting():
    """XGBoost mortality classifier. Imported lazily to keep import cost low."""
    from xgboost import XGBClassifier

    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        n_jobs=-1,
    )


def fit_cox(df: pd.DataFrame, covariates: list[str], strata: list[str] | None = None):
    """Fit a Cox proportional hazards model. Returns the fitted CoxPHFitter.

    Expects `df` to contain `covariates` (+ `strata`, if given) plus `time` and
    `event`. Passing `strata` gives each stratum its own baseline hazard instead
    of assuming proportional hazards for that variable — the standard fix when
    `check_ph_assumption` finds a covariate that badly violates the assumption.
    Stratified covariates get no hazard ratio in the resulting summary.
    """
    from lifelines import CoxPHFitter

    cols = covariates + ["time", "event"] + (strata or [])
    cph = CoxPHFitter()
    cph.fit(df[cols], duration_col="time", event_col="event", strata=strata)
    return cph


def encode_cancer_stage(df: pd.DataFrame) -> pd.DataFrame:
    """Dummy-code `cancer` (0=none/other diagnosis, 1=present, 2=metastatic)
    against a `cancer == 0` reference, instead of feeding it to Cox as a single
    linear ordinal term.

    `cancer` is not monotonic in hazard in this cohort: patients with no cancer
    have *worse* observed survival than those with non-metastatic cancer
    (SUPPORT's other admission diagnoses — e.g. multi-organ failure, sepsis,
    coma — carry high short-term mortality). A linear coefficient averages
    this into a misleading, sign-flipped effect; dummy-coding fixes it and
    measurably improves concordance (see the Phase 4 model card entry).

    Returns a copy; does not mutate the input.
    """
    out = df.copy()
    out["cancer_present"] = (out["cancer"] == 1).astype(int)
    out["cancer_metastatic"] = (out["cancer"] == 2).astype(int)
    return out


def check_ph_assumption(cph, df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """Schoenfeld-residual test for the Cox proportional-hazards assumption.

    Returns a DataFrame indexed by covariate with a test statistic and p-value;
    p < 0.05 means that covariate's effect on the hazard likely changes over
    time, violating the assumption Cox regression relies on. Uses the same
    statistical test as lifelines' `CoxPHFitter.check_assumptions`, but returns
    a DataFrame instead of printing to stdout, so it's usable in code and tests.

    With a large cohort (as here), this test is very sensitive — near-zero p
    values are common even for modest, practically tolerable deviations, so
    weigh the test statistic magnitude alongside the p-value, not the p-value
    alone.
    """
    from lifelines.statistics import proportional_hazard_test

    strata = cph.strata or []
    strata_cols = [strata] if isinstance(strata, str) else list(strata)
    results = proportional_hazard_test(
        cph, df[covariates + ["time", "event"] + strata_cols], time_transform="rank"
    )
    return results.summary


def cross_val_concordance(
    df: pd.DataFrame,
    covariates: list[str],
    strata: list[str] | None = None,
    cv: int = 5,
    random_state: int = 42,
) -> list[float]:
    """K-fold cross-validated concordance index for a Cox model.

    Mirrors the out-of-fold rigor used for the binary-mortality comparison in
    Phase 3 (`evaluate.compare_models`): fits on each training fold and scores
    concordance on the held-out fold only, so discrimination isn't inflated by
    evaluating on the same data used to fit the model — unlike a plain
    `fit_cox(...).concordance_index_`, which is in-sample. Missing values in
    `covariates` are median-imputed per fold (train-fold median only, applied
    to both train and test) to avoid leaking test-fold statistics into training.
    """
    from lifelines import CoxPHFitter
    from sklearn.model_selection import KFold

    all_cols = covariates + (strata or [])
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    scores = []
    for train_idx, test_idx in kf.split(df):
        train_df = df.iloc[train_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        medians = train_df[covariates].median()
        train_df[covariates] = train_df[covariates].fillna(medians)
        test_df[covariates] = test_df[covariates].fillna(medians)

        cph = CoxPHFitter()
        cph.fit(
            train_df[all_cols + ["time", "event"]],
            duration_col="time", event_col="event", strata=strata,
        )
        scores.append(
            cph.score(test_df[all_cols + ["time", "event"]], scoring_method="concordance_index")
        )
    return scores


def hazard_ratio_table(cph) -> pd.DataFrame:
    """Clean hazard-ratio table for reporting, sorted by significance.

    `hazard_ratio` (`exp(coef)`) > 1 means higher hazard (worse prognosis) per
    unit increase in the covariate (or, for a 0/1 dummy, relative to the
    reference level); < 1 means protective. A 95% CI that excludes 1 indicates
    a statistically significant effect at the 0.05 level.
    """
    table = cph.summary[
        ["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
    ].copy()
    table.columns = ["hazard_ratio", "hr_lower_95", "hr_upper_95", "p_value"]
    return table.sort_values("p_value")
