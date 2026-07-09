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


def fit_cox(df: pd.DataFrame, covariates: list[str]):
    """Fit a Cox proportional hazards model. Returns the fitted CoxPHFitter.

    Expects `df` to contain `covariates` plus `time` and `event`.
    """
    from lifelines import CoxPHFitter

    cph = CoxPHFitter()
    cph.fit(df[covariates + ["time", "event"]], duration_col="time", event_col="event")
    return cph
