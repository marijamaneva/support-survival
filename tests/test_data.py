"""Tests for data loading."""
import importlib
import os
from pathlib import Path

from support_survival import data


def test_load_shape():
    df = data.load()
    # Known size of the SUPPORT cohort in this preprocessing.
    assert len(df) == 8873
    assert "time" in df.columns
    assert "event" in df.columns


def test_event_is_binary():
    df = data.load()
    assert set(df["event"].unique()) <= {0, 1}


def test_all_features_present():
    df = data.load()
    for col in data.FEATURES:
        assert col in df.columns, f"missing feature {col}"


def test_time_positive():
    df = data.load()
    assert (df["time"] > 0).all()


def test_wbc_and_serum_sodium_are_in_clinically_plausible_ranges():
    """Regression test for the wbc/serum_sodium label swap fixed in Phase 2.

    The raw DeepSurv h5 has these two columns' values swapped relative to their
    documented labels (see the NOTE in data.py). Real serum sodium clusters
    tightly around 135-145 mEq/L; real white blood cell count is right-skewed
    with a median around 9-11 (x1000/mm3). If FEATURES is ever "corrected" back
    to the documented-but-wrong order, this test catches it.
    """
    df = data.load()
    assert 130 <= df["serum_sodium"].median() <= 145
    assert 5 <= df["wbc"].median() <= 15


def test_root_is_cwd_relative_not_file_relative():
    """Regression test for the path-resolution bug once found (and fixed) in
    `api.py`'s `MODEL_PATH`: `ROOT` must never be computed from `__file__`,
    which breaks under a non-editable install (the file lands in
    site-packages). It should resolve relative to the working directory,
    honoring `SUPPORT_SURVIVAL_ROOT` when set.
    """
    assert data.ROOT == Path(".")

    os.environ["SUPPORT_SURVIVAL_ROOT"] = "/tmp/somewhere-else"
    try:
        reloaded = importlib.reload(data)
        assert reloaded.ROOT == Path("/tmp/somewhere-else")
        assert reloaded.DATA_DIR == Path("/tmp/somewhere-else/data")
    finally:
        del os.environ["SUPPORT_SURVIVAL_ROOT"]
        importlib.reload(data)
