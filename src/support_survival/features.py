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
# measurement artifacts. Extend this as EDA surfaces more (document each one).
IMPLAUSIBLE_ZERO_COLS = ["mean_bp"]  # a live patient cannot have MAP == 0


def flag_implausible(df: pd.DataFrame) -> pd.DataFrame:
    """Replace physiologically impossible zeros with NaN so they can be imputed.

    Returns a copy; does not mutate the input.
    """
    out = df.copy()
    for col in IMPLAUSIBLE_ZERO_COLS:
        if col in out.columns:
            out[col] = out[col].replace(0, np.nan)
    return out


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple, clinically motivated derived features.

    Kept intentionally small and interpretable. Extend in Phase 2.
    """
    out = df.copy()
    # Example derived features — refine with domain reasoning in Phase 2.
    if "age" in out:
        out["age_over_70"] = (out["age"] >= 70).astype(int)
    if "serum_creatinine" in out:
        # Rough renal-impairment flag; a real project would justify the threshold.
        out["creatinine_high"] = (out["serum_creatinine"] > 1.5).astype(int)
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature pipeline: quality flags then derived features.

    Note: imputation of the NaNs introduced here is done inside the model pipeline
    (Phase 3+) so that imputation statistics are learned on training folds only,
    avoiding leakage.
    """
    return add_clinical_features(flag_implausible(df))
