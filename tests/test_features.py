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


def _toy_vitals() -> pd.DataFrame:
    return pd.DataFrame({
        "heart_rate": [70, 0, 110, 75],        # 0 is implausible; 110 is tachycardic
        "resp_rate": [16, 0, 20, 18],          # 0 is implausible
        "mean_bp": [80, 95, 50, 0],            # 50 is hypotensive (< 65); 0 is implausible
        "wbc": [8, 0, 15, 10],                 # 0 is implausible; 15 is leukocytosis (> 11)
        "serum_sodium": [140, 130, 150, 138],  # 130 and 150 are both abnormal (outside 135-145)
    })


def test_implausible_zero_extends_to_heart_rate_resp_rate_wbc():
    out = features.flag_implausible(_toy_vitals())
    assert np.isnan(out.loc[1, "heart_rate"])
    assert np.isnan(out.loc[1, "resp_rate"])
    assert np.isnan(out.loc[1, "wbc"])
    assert out.loc[0, "heart_rate"] == 70  # legitimate value untouched


def test_missingness_indicators_flag_implausible_rows():
    flagged = features.flag_implausible(_toy_vitals())
    out = features.add_missingness_indicators(flagged)

    # heart_rate: implausible (0) at row 1, present elsewhere
    assert out.loc[1, "heart_rate_missing"] == 1
    assert out.loc[0, "heart_rate_missing"] == 0

    # resp_rate: implausible (0) at row 1, present elsewhere
    assert out.loc[1, "resp_rate_missing"] == 1
    assert out.loc[0, "resp_rate_missing"] == 0

    # mean_bp: implausible (0) at row 3, present elsewhere
    assert out.loc[3, "mean_bp_missing"] == 1
    assert out.loc[1, "mean_bp_missing"] == 0

    # wbc: implausible (0) at row 1, present elsewhere
    assert out.loc[1, "wbc_missing"] == 1
    assert out.loc[0, "wbc_missing"] == 0


def test_tachycardic_flag():
    out = features.add_clinical_features(_toy_vitals())
    assert out.loc[2, "tachycardic"] == 1  # heart_rate 110
    assert out.loc[0, "tachycardic"] == 0  # heart_rate 70


def test_hypotensive_flag():
    out = features.add_clinical_features(_toy_vitals())
    assert out.loc[2, "hypotensive"] == 1  # mean_bp 50 < 65
    assert out.loc[0, "hypotensive"] == 0  # mean_bp 80


def test_leukocytosis_flag():
    out = features.add_clinical_features(_toy_vitals())
    assert out.loc[2, "leukocytosis"] == 1  # wbc 15 > 11
    assert out.loc[0, "leukocytosis"] == 0  # wbc 8


def test_sodium_abnormal_flag():
    out = features.add_clinical_features(_toy_vitals())
    assert out.loc[1, "sodium_abnormal"] == 1  # 130 < 135
    assert out.loc[2, "sodium_abnormal"] == 1  # 150 > 145
    assert out.loc[0, "sodium_abnormal"] == 0  # 140 within range


def test_build_features_includes_missingness_and_clinical_flags():
    out = features.build_features(_toy_vitals())
    for col in ["heart_rate_missing", "resp_rate_missing", "wbc_missing",
                "tachycardic", "hypotensive", "leukocytosis", "sodium_abnormal"]:
        assert col in out.columns
