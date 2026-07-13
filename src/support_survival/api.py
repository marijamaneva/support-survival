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
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from support_survival import data, features, models, validate

# Frontend assets (HTML/CSS/JS) ship as package data alongside this module --
# unlike MODEL_PATH/DATA_DIR below, this is genuinely safe to resolve from
# `__file__`: these files travel with the installed package itself (editable
# or not), they aren't written at runtime by a separate process at a
# different location, so there's no site-packages relocation risk here.
STATIC_DIR = Path(__file__).parent / "static"

# Triage-panel tuning. `risk_at_30d`/`risk_at_90d` come from the Cox model's
# individual survival curve (models.predict_risk_at_horizons) -- that's the
# whole point of using Cox here rather than the binary classifier alone: a
# single risk score can't distinguish "high risk very soon" from "high risk
# eventually", which is exactly what triage/urgency needs to distinguish.
TRIAGE_SAMPLE_SIZE = 15
TRIAGE_URGENT_30D_RISK = 0.40   # ~90th percentile of 30-day risk cohort-wide
TRIAGE_MONITOR_OVERALL_RISK = 0.65  # matches the single-patient "Elevated risk" band

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
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

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
    """13 of the 14 raw SUPPORT covariates (see `data.DESCRIPTIONS`); `race` is
    deliberately excluded (never used as a model input -- see
    `evaluate.EXCLUDED_FROM_FEATURES` and the model card). Ranges are
    physiologically generous bounds, not clinical thresholds -- genuinely
    impossible values (negative age, negative heart rate, ...) are rejected
    here; physiologically-implausible-but-representable sentinels (e.g.
    mean_bp == 0) are intentionally let through, because `features.build_features`
    is responsible for handling those exactly as it does in training.
    """

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "age": 65, "sex": 1, "n_comorbidities": 2, "diabetes": 0,
                "dementia": 0, "cancer": 1, "mean_bp": 80, "heart_rate": 90,
                "resp_rate": 20, "temperature": 37.0, "serum_sodium": 138,
                "wbc": 9.5, "serum_creatinine": 1.1,
            }]
        }
    }

    age: float = Field(ge=0, le=120, description="Age in years")
    sex: int = Field(ge=0, le=1, description="Sex (encoded; mapping to male/female not independently confirmed)")
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


class TriagePatient(BaseModel):
    patient_id: int
    age: float
    cancer: str
    overall_risk: float
    risk_30d: float
    risk_90d: float
    tier: str


class TriageResponse(BaseModel):
    patients: list[TriagePatient]
    model_version: str


def _triage_tier(risk_30d: float, overall_risk: float) -> str:
    if risk_30d >= TRIAGE_URGENT_30D_RISK:
        return "Urgent"
    if overall_risk >= TRIAGE_MONITOR_OVERALL_RISK:
        return "Monitor"
    return "Routine"


@app.get("/triage", response_model=TriageResponse)
def triage(n: int = TRIAGE_SAMPLE_SIZE) -> TriageResponse:
    """A risk-stratified patient panel, for decision-support triage -- not
    automated resource allocation. Patients are a fixed random sample from
    the Phase 5 held-out test set (this project has no live hospital feed),
    so this reflects genuinely unseen data, sampled deterministically
    (`random_state=7`) so the demo panel is stable across requests.
    """
    artifact = _load_artifact()

    # Split on the raw df, not the featurized one: `models.fit_cox` in
    # train_and_save.py was trained on raw covariates (Phase 4 never applied
    # `features.flag_implausible` to the Cox inputs), so predicting from
    # NaN-flagged featurized data here would silently skew Cox's inputs away
    # from what it was trained on (implausible zeros predict as NaN survival
    # curves -- exactly the train/serve mismatch this project has been careful
    # about elsewhere). `df` and `feat` share the same row index, so the split
    # itself (which patients land in `test`) is identical either way.
    df = data.load()
    feat = features.build_features(df)
    _, _, test_raw = validate.split_train_val_test(df, test_size=0.2, val_size=0.2, random_state=42)
    sample_raw = test_raw.sample(n=min(n, len(test_raw)), random_state=7)
    sample_feat = feat.loc[sample_raw.index]

    overall_risk = artifact["model"].predict_proba(sample_feat[artifact["feature_columns"]])[:, 1]

    cox_input = models.encode_cancer_stage(sample_raw)
    horizon_risk = models.predict_risk_at_horizons(
        artifact["cox_model"], cox_input, artifact["cox_covariates"], horizons=[30, 90]
    )

    cancer_labels = {0: "None", 1: "Present", 2: "Metastatic"}
    patients = []
    for i, (idx, patient) in enumerate(sample_raw.iterrows()):
        r30 = float(horizon_risk["risk_at_30d"].iloc[i])
        r90 = float(horizon_risk["risk_at_90d"].iloc[i])
        overall = float(overall_risk[i])
        patients.append(TriagePatient(
            patient_id=int(idx),
            age=round(float(patient["age"]), 1),
            cancer=cancer_labels.get(int(patient["cancer"]), "?"),
            overall_risk=round(overall, 3),
            risk_30d=round(r30, 3),
            risk_90d=round(r90, 3),
            tier=_triage_tier(r30, overall),
        ))

    patients.sort(key=lambda p: p.risk_30d, reverse=True)
    return TriageResponse(patients=patients, model_version=artifact["version"])


@app.get("/triage-view", response_class=HTMLResponse)
def triage_page() -> str:
    return (STATIC_DIR / "triage.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def demo_page() -> str:
    return (STATIC_DIR / "predict.html").read_text(encoding="utf-8")


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
