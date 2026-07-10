"""FastAPI serving layer for the mortality-risk model.

Train/serve consistency is the whole point of this module: a prediction
request is converted to a one-row DataFrame and run through the exact same
`features.build_features` used in training (imported, not reimplemented) --
see `predict` below. If that function ever changes, training and serving
pick up the change together; there is no second copy of the preprocessing
logic to forget to update.
"""
from __future__ import annotations

import os
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from support_survival import features

# Deliberately NOT `Path(__file__).resolve().parents[N]`: once this package is
# installed normally (as in the Dockerfile's `pip install .`, rather than the
# `pip install -e .` used for local dev), `__file__` resolves inside
# site-packages and counting parent directories no longer lands on the repo
# root. `train_and_save.py` always runs as a plain script from the project
# root (locally, and via `WORKDIR /app` in the container), so it saves the
# model to a `models/` directory relative to the current working directory --
# resolve it the same way here, with an env var escape hatch for anything
# unusual.
MODEL_PATH = Path(os.environ.get("SUPPORT_SURVIVAL_MODEL_PATH", "models/gradient_boosting.joblib"))

app = FastAPI(
    title="SUPPORT Mortality Risk API",
    description="Educational / portfolio demonstration of survival modeling. Not for clinical use.",
)

_artifact: dict | None = None


def _load_artifact() -> dict:
    global _artifact
    if _artifact is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"No trained model found at {MODEL_PATH}. Run "
                f"`python scripts/train_and_save.py` first."
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


class PatientFeatures(BaseModel):
    """The 14 raw SUPPORT covariates (see `data.DESCRIPTIONS`). Ranges are
    physiologically generous bounds, not clinical thresholds -- genuinely
    impossible values (negative age, negative heart rate, ...) are rejected
    here; physiologically-implausible-but-representable sentinels (e.g.
    mean_bp == 0) are intentionally let through, because `features.build_features`
    is responsible for handling those exactly as it does in training.
    """

    age: float = Field(ge=0, le=120, description="Age in years")
    sex: int = Field(ge=0, le=1, description="Sex (encoded)")
    race: int = Field(ge=0, le=9, description="Race (encoded)")
    n_comorbidities: int = Field(ge=0, le=20, description="Number of comorbidities")
    diabetes: int = Field(ge=0, le=1, description="Diabetes (0/1)")
    dementia: int = Field(ge=0, le=1, description="Dementia (0/1)")
    cancer: int = Field(ge=0, le=2, description="Cancer: 0=none, 1=present, 2=metastatic")
    mean_bp: float = Field(ge=0, le=300, description="Mean arterial blood pressure (mmHg)")
    heart_rate: float = Field(ge=0, le=350, description="Heart rate (bpm)")
    resp_rate: float = Field(ge=0, le=100, description="Respiration rate (breaths/min)")
    temperature: float = Field(ge=25, le=45, description="Body temperature")
    serum_sodium: float = Field(ge=0, le=250, description="Serum sodium (mEq/L)")
    wbc: float = Field(ge=0, le=500, description="White blood cell count (x1000/mm3)")
    serum_creatinine: float = Field(ge=0, le=30, description="Serum creatinine (mg/dL)")


class PredictionResponse(BaseModel):
    risk_probability: float
    model_version: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientFeatures) -> PredictionResponse:
    artifact = _load_artifact()

    raw = pd.DataFrame([patient.model_dump()])
    feat = features.build_features(raw)  # exact same function training used

    missing = [c for c in artifact["feature_columns"] if c not in feat.columns]
    if missing:
        raise HTTPException(status_code=500, detail=f"Feature mismatch: missing {missing}")

    x = feat[artifact["feature_columns"]]
    proba = float(artifact["model"].predict_proba(x)[:, 1][0])
    return PredictionResponse(risk_probability=proba, model_version=artifact["version"])
