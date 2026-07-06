"""Data acquisition and loading for the SUPPORT cohort.

The SUPPORT study (~8,900 seriously ill patients from 5 US medical centers) is the
canonical open dataset for survival analysis. We use the DeepSurv-preprocessed
version, which ships time-to-event, an event indicator, and 14 clinical covariates.
No credentialing required.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Repo root = three parents up from this file (src/support_survival/data.py).
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_H5 = DATA_DIR / "support_raw.h5"
CSV = DATA_DIR / "support.csv"

SOURCE_URL = (
    "https://raw.githubusercontent.com/jaredleekatzman/DeepSurv/"
    "master/experiments/data/support/support_train_test.h5"
)

# The 14 covariates, in the column order used by the DeepSurv preprocessing.
#
# NOTE: positions 11 and 12 are swapped relative to the DeepSurv/pycox documented
# order ("...temperature, wbc, serum_sodium, creatinine"). The raw values at those
# two positions do not match their documented labels: the documented "wbc" slot
# holds a tight 110-181 range (textbook serum sodium, mEq/L, for critically ill
# patients), and the documented "serum_sodium" slot holds a right-skewed 0-200
# range with median ~10.6 (textbook white blood cell count, x1000/mm3; matches
# the SUPPORT codebook's default imputation value of 9 for wblc). Verified via
# Phase 2 EDA against the original SUPPORT/DeepSurv literature; see model card.
FEATURES: list[str] = [
    "age",
    "sex",
    "race",
    "n_comorbidities",
    "diabetes",
    "dementia",
    "cancer",
    "mean_bp",
    "heart_rate",
    "resp_rate",
    "temperature",
    "serum_sodium",
    "wbc",
    "serum_creatinine",
]

DESCRIPTIONS: dict[str, str] = {
    "age": "Age in years",
    "sex": "Sex (encoded)",
    "race": "Race (encoded)",
    "n_comorbidities": "Number of comorbidities",
    "diabetes": "Diabetes (0/1)",
    "dementia": "Dementia (0/1)",
    "cancer": "Cancer present (0/1)",
    "mean_bp": "Mean arterial blood pressure (mmHg)",
    "heart_rate": "Heart rate (bpm)",
    "resp_rate": "Respiration rate (breaths/min)",
    "temperature": "Body temperature",
    "wbc": "White blood cell count",
    "serum_sodium": "Serum sodium",
    "serum_creatinine": "Serum creatinine",
    "time": "Follow-up time in days",
    "event": "1 = death observed, 0 = right-censored",
}


def download(force: bool = False) -> Path:
    """Download the raw SUPPORT HDF5 file into data/. Idempotent."""
    DATA_DIR.mkdir(exist_ok=True)
    if RAW_H5.exists() and not force:
        return RAW_H5
    req = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "Mozilla/5.0"})
    RAW_H5.write_bytes(urllib.request.urlopen(req, timeout=30).read())
    return RAW_H5


def _h5_to_frame(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as f:
        x = np.vstack([f["train"]["x"][:], f["test"]["x"][:]])
        t = np.concatenate([f["train"]["t"][:], f["test"]["t"][:]])
        e = np.concatenate([f["train"]["e"][:], f["test"]["e"][:]])
    df = pd.DataFrame(x, columns=FEATURES)
    df["time"] = t
    df["event"] = e.astype(int)
    return df


def build_csv(force: bool = False) -> Path:
    """Download and materialize the dataset as data/support.csv."""
    download(force=force)
    df = _h5_to_frame(RAW_H5)
    df.to_csv(CSV, index=False)
    return CSV


def load() -> pd.DataFrame:
    """Load the SUPPORT cohort as a DataFrame, building the CSV if needed."""
    if not CSV.exists():
        build_csv()
    return pd.read_csv(CSV)
