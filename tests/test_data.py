"""Tests for data loading."""
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
