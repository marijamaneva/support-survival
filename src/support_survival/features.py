"""Feature engineering and data-quality handling for the SUPPORT cohort.

All transformations live here (not in notebooks) so they are testable and reusable
by both the training pipeline and any serving code. Each cleaning decision is
explicit and documented, because in regulated settings *why* a value was changed
matters as much as the change itself.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Physiologically implausible values that are really missing-data sentinels or
# measurement artifacts, found via the Phase 1/2 EDA zero-scan (see model card).
# Each is a vital sign that cannot truly be zero in a live patient.
IMPLAUSIBLE_ZERO_COLS = [
    "mean_bp",      # a live patient cannot have MAP == 0
    "heart_rate",   # a live patient cannot have a heart rate of 0
    "resp_rate",    # a live patient cannot have a respiration rate of 0
    "wbc",          # a live patient cannot have a white blood cell count of 0
]


def flag_implausible(df: pd.DataFrame) -> pd.DataFrame:
    """Replace physiologically impossible zeros with NaN so they can be imputed.

    Returns a copy; does not mutate the input.
    """
    out = df.copy()
    for col in IMPLAUSIBLE_ZERO_COLS:
        if col in out.columns:
            out[col] = out[col].replace(0, np.nan)
    return out


def add_missingness_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a binary "was this implausible-zero column missing" flag per column.

    Missingness in these vitals may not be random (e.g. a vital sign that wasn't
    recorded can itself reflect how unstable a patient was, or differences in
    monitoring protocol across the 5 SUPPORT hospitals). Recording *that* a value
    was missing, before it gets imputed, keeps that signal available to the model
    instead of erasing it.
    """
    out = df.copy()
    for col in IMPLAUSIBLE_ZERO_COLS:
        if col in out.columns:
            out[f"{col}_missing"] = out[col].isna().astype(int)
    return out


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple, clinically motivated derived features.

    Each threshold is a standard clinical cutoff, not a fitted statistic — kept
    intentionally small and interpretable.
    """
    out = df.copy()
    if "age" in out:
        # Frailty flag: mortality risk rises markedly past 70 due to reduced
        # physiological reserve, independent of any single diagnosis.
        out["age_over_70"] = (out["age"] >= 70).astype(int)
    if "serum_creatinine" in out:
        # Renal-impairment flag: > 1.5 mg/dL is a commonly used threshold for
        # reduced kidney function.
        out["creatinine_high"] = (out["serum_creatinine"] > 1.5).astype(int)
    if "heart_rate" in out:
        # Tachycardia: resting heart rate above 100 bpm.
        out["tachycardic"] = (out["heart_rate"] > 100).astype(int)
    if "mean_bp" in out:
        # Hypotension: MAP < 65 mmHg is the threshold used in sepsis guidelines
        # (Surviving Sepsis Campaign) below which organ perfusion is at risk.
        out["hypotensive"] = (out["mean_bp"] < 65).astype(int)
    if "wbc" in out:
        # Leukocytosis: white blood cell count above the normal upper limit
        # (~11 x1000/mm3), suggestive of infection/inflammation.
        out["leukocytosis"] = (out["wbc"] > 11).astype(int)
    if "serum_sodium" in out:
        # Dysnatremia: sodium outside the normal reference range (135-145 mEq/L);
        # both directions are clinically significant, so this flags either.
        out["sodium_abnormal"] = (
            (out["serum_sodium"] < 135) | (out["serum_sodium"] > 145)
        ).astype(int)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline: quality flags, missingness indicators, then derived
    clinical features.

    Note: imputation of the NaNs introduced here is done inside the model pipeline
    (Phase 3+) so that imputation statistics are learned on training folds only,
    avoiding leakage.
    """
    flagged = flag_implausible(df)
    with_missing_flags = add_missingness_indicators(flagged)
    return add_clinical_features(with_missing_flags)
