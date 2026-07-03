"""Tests for feature engineering."""
import numpy as np
import pandas as pd

from support_survival import features


def _toy() -> pd.DataFrame:
    return pd.DataFrame({
        "age": [55, 75, 40],
        "mean_bp": [80, 0, 95],  # the 0 is physiologically impossible
        "serum_creatinine": [0.9, 2.0, 1.1],
    })


def test_implausible_zero_becomes_nan():
    out = features.flag_implausible(_toy())
    assert np.isnan(out.loc[1, "mean_bp"])
    # Legitimate values are untouched.
    assert out.loc[0, "mean_bp"] == 80


def test_flag_does_not_mutate_input():
    df = _toy()
    _ = features.flag_implausible(df)
    assert df.loc[1, "mean_bp"] == 0  # original unchanged


def test_derived_features_added():
    out = features.build_features(_toy())
    assert "age_over_70" in out.columns
    assert "creatinine_high" in out.columns
    assert out.loc[1, "age_over_70"] == 1   # age 75
    assert out.loc[1, "creatinine_high"] == 1  # creatinine 2.0 > 1.5
