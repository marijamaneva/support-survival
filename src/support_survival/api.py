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
from pydantic import BaseModel, Field

from support_survival import data, features, models, validate

# Observed event rate in the SUPPORT cohort (see reports/model_card.md) -- the
# reference point the demo page compares a prediction against.
COHORT_BASELINE_EVENT_RATE = 0.68

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


_SHARED_STYLE = """
  :root {
    --surface-1:      #fcfcfb;
    --surface-2:      #f3f3f1;
    --page:           #eef1f5;
    --text-primary:   #0b0b0b;
    --text-secondary: #52514e;
    --text-muted:     #898781;
    --border:         rgba(11,11,11,0.09);
    --track:          #cde2fb;
    --fill:           #2a78d6;
    --fill-2:         #1c5cab;
    --baseline:       #52514e;
    --status-good:      #0ca30c;
    --status-good-bg:   #e7f6e7;
    --status-warning:   #c98500;
    --status-warning-bg: #fff3d9;
    --status-serious:   #ec835a;
    --status-serious-bg: #fdece5;
    --status-critical:  #d03b3b;
    --status-critical-bg: #fbe6e6;
    --error-bg:       #fbeaea;
    --error-text:     #8a1f1f;
    --shadow:         0 1px 2px rgba(20,20,20,0.04), 0 12px 28px rgba(20,20,20,0.07);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --surface-1:      #1c1c1b;
      --surface-2:      #232322;
      --page:           #0c0d10;
      --text-primary:   #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted:     #8b8a84;
      --border:         rgba(255,255,255,0.09);
      --track:          #1e355a;
      --fill:           #3987e5;
      --fill-2:         #6da7ec;
      --baseline:       #c3c2b7;
      --status-good:      #3fce3f;
      --status-good-bg:   #12291a;
      --status-warning:   #fab219;
      --status-warning-bg: #2e2410;
      --status-serious:   #ec835a;
      --status-serious-bg: #2e1c14;
      --status-critical:  #e66767;
      --status-critical-bg: #301616;
      --error-bg:       #2a1414;
      --error-text:     #e88a8a;
      --shadow:         0 1px 2px rgba(0,0,0,0.3), 0 16px 32px rgba(0,0,0,0.35);
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 0 16px 48px;
    background:
      radial-gradient(1200px 420px at 50% -120px, color-mix(in srgb, var(--fill) 10%, transparent), transparent),
      var(--page);
    color: var(--text-primary);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  .icon { width: 18px; height: 18px; display: inline-block; vertical-align: -4px; }

  .topnav { max-width: 960px; margin: 0 auto; padding: 22px 4px 20px; display: flex;
    align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px; }
  .brand { display: flex; align-items: center; gap: 8px; font-weight: 700; font-size: 15px; color: var(--text-primary); }
  .brand .icon { color: var(--fill); width: 22px; height: 22px; vertical-align: -5px; }
  .navlinks { display: flex; gap: 4px; background: var(--surface-1); border: 1px solid var(--border);
    border-radius: 999px; padding: 3px; box-shadow: var(--shadow); }
  .navlinks a { text-decoration: none; color: var(--text-secondary); font-size: 13px; font-weight: 600;
    padding: 7px 14px; border-radius: 999px; }
  .navlinks a.active { background: var(--fill); color: #ffffff; }

  .card {
    max-width: 640px; margin: 0 auto;
    background: var(--surface-1);
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    border-radius: 18px;
    padding: 30px 32px 34px;
  }
  h1 { font-size: 21px; margin: 0 0 4px; letter-spacing: -0.01em; }
  .subtitle { color: var(--text-secondary); font-size: 13.5px; margin: 0 0 26px; line-height: 1.5; }
  .subtitle a { color: var(--fill); font-weight: 600; text-decoration: none; }
  .subtitle a:hover { text-decoration: underline; }

  fieldset.section { border: none; padding: 0; margin: 0 0 20px; }
  fieldset.section legend {
    font-size: 11px; font-weight: 700; color: var(--fill); text-transform: uppercase;
    letter-spacing: 0.06em; padding: 0 0 8px; width: 100%; border-bottom: 1px solid var(--border);
    margin-bottom: 12px;
  }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px 16px; }
  @media (max-width: 480px) { .grid { grid-template-columns: 1fr; } }
  label { display: block; font-size: 12px; color: var(--text-secondary); margin-bottom: 4px; }
  input, select {
    width: 100%; padding: 8px 10px; font-size: 14px;
    border: 1px solid var(--border); border-radius: 8px;
    background: var(--surface-2); color: var(--text-primary);
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
  }
  input:hover, select:hover { border-color: color-mix(in srgb, var(--fill) 40%, var(--border)); }
  input:focus, select:focus {
    outline: none; border-color: var(--fill);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--fill) 20%, transparent);
  }
  .field { margin-bottom: 4px; }
  button {
    margin-top: 6px; width: 100%; padding: 12px;
    background: linear-gradient(180deg, var(--fill), var(--fill-2));
    color: #ffffff; font-size: 14.5px; font-weight: 700;
    border: none; border-radius: 10px; cursor: pointer;
    box-shadow: 0 6px 16px color-mix(in srgb, var(--fill) 35%, transparent);
    transition: transform 0.1s ease, box-shadow 0.15s ease;
  }
  button:hover { transform: translateY(-1px); box-shadow: 0 8px 20px color-mix(in srgb, var(--fill) 45%, transparent); }
  button:active { transform: translateY(0); }

  #error {
    margin-top: 16px; padding: 11px 13px; border-radius: 10px;
    background: var(--error-bg); color: var(--error-text); font-size: 13px;
  }

  #result { margin-top: 26px; padding-top: 24px; border-top: 1px solid var(--border);
    animation: rise 0.35s ease; }
  @keyframes rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
  .result-label { font-size: 12.5px; color: var(--text-secondary); margin-bottom: 4px; text-transform: uppercase;
    letter-spacing: 0.04em; font-weight: 600; }
  .hero-row { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
  .hero-value { font-size: 50px; font-weight: 700; line-height: 1; letter-spacing: -0.02em; }
  .status { display: inline-flex; align-items: center; gap: 6px; font-size: 14px; font-weight: 600; color: var(--text-secondary); }
  .status-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

  .meter-wrap { margin-top: 22px; position: relative; padding-bottom: 22px; }
  .meter-track {
    height: 14px; border-radius: 999px; background: var(--track); overflow: visible; position: relative;
  }
  .meter-fill {
    height: 14px; border-radius: 999px; width: 0%;
    background: linear-gradient(90deg, var(--fill-2), var(--fill));
    transition: width 0.6s cubic-bezier(.2,.8,.2,1);
  }
  .meter-baseline {
    position: absolute; top: -4px; bottom: -4px; width: 2px; background: var(--baseline); opacity: 0.6;
  }
  .meter-baseline::before {
    content: ""; position: absolute; top: -7px; left: 50%; transform: translateX(-50%);
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid var(--baseline); opacity: 0.8;
  }
  .meter-baseline-label {
    position: absolute; top: 20px; font-size: 11px; color: var(--text-muted);
    transform: translateX(-50%); white-space: nowrap;
  }

  .meta { margin-top: 20px; font-size: 12px; color: var(--text-muted); line-height: 1.6; }
  [hidden] { display: none !important; }
"""

_NAV_PULSE_ICON = (
    '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
    'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M2 12h4l2-8 4 16 3-11 2 3h5"/></svg>'
)

_DEMO_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SUPPORT Mortality Risk -- demo</title>
<style>""" + _SHARED_STYLE + """</style>
</head>
<body>
  <div class="topnav">
    <div class="brand">""" + _NAV_PULSE_ICON + """ SUPPORT Risk Suite</div>
    <div class="navlinks">
      <a href="/" class="active">Predictor</a>
      <a href="/triage-view">Triage Panel</a>
    </div>
  </div>

  <div class="card">
    <h1>Mortality risk predictor</h1>
    <p class="subtitle">Educational / portfolio demonstration of survival modeling. Not for clinical use.</p>

    <form id="predict-form">
      <fieldset class="section">
        <legend>Demographics</legend>
        <div class="grid">
          <div class="field"><label for="age">Age (years)</label>
            <input id="age" name="age" type="number" min="0" max="120" step="1" value="65" required></div>
          <div class="field"><label for="sex">Sex (code; mapping unconfirmed)</label>
            <select id="sex" name="sex"><option value="0">0</option><option value="1" selected>1</option></select></div>
        </div>
      </fieldset>

      <fieldset class="section">
        <legend>Comorbidities</legend>
        <div class="grid">
          <div class="field"><label for="n_comorbidities">Number of comorbidities</label>
            <input id="n_comorbidities" name="n_comorbidities" type="number" min="0" max="20" step="1" value="2" required></div>
          <div class="field"><label for="cancer">Cancer</label>
            <select id="cancer" name="cancer">
              <option value="0">None</option>
              <option value="1" selected>Present</option>
              <option value="2">Metastatic</option>
            </select></div>
          <div class="field"><label for="diabetes">Diabetes</label>
            <select id="diabetes" name="diabetes"><option value="0" selected>No</option><option value="1">Yes</option></select></div>
          <div class="field"><label for="dementia">Dementia</label>
            <select id="dementia" name="dementia"><option value="0" selected>No</option><option value="1">Yes</option></select></div>
        </div>
      </fieldset>

      <fieldset class="section">
        <legend>Vital signs</legend>
        <div class="grid">
          <div class="field"><label for="mean_bp">Mean arterial BP (mmHg)</label>
            <input id="mean_bp" name="mean_bp" type="number" min="0" max="300" step="0.1" value="80" required></div>
          <div class="field"><label for="heart_rate">Heart rate (bpm)</label>
            <input id="heart_rate" name="heart_rate" type="number" min="0" max="350" step="0.1" value="90" required></div>
          <div class="field"><label for="resp_rate">Respiration rate (breaths/min)</label>
            <input id="resp_rate" name="resp_rate" type="number" min="0" max="100" step="0.1" value="20" required></div>
          <div class="field"><label for="temperature">Temperature</label>
            <input id="temperature" name="temperature" type="number" min="25" max="45" step="0.1" value="37.0" required></div>
        </div>
      </fieldset>

      <fieldset class="section">
        <legend>Labs</legend>
        <div class="grid">
          <div class="field"><label for="serum_sodium">Serum sodium (mEq/L)</label>
            <input id="serum_sodium" name="serum_sodium" type="number" min="0" max="250" step="0.1" value="138" required></div>
          <div class="field"><label for="wbc">White blood cell count (x1000/mm3)</label>
            <input id="wbc" name="wbc" type="number" min="0" max="500" step="0.1" value="9.5" required></div>
          <div class="field"><label for="serum_creatinine">Serum creatinine (mg/dL)</label>
            <input id="serum_creatinine" name="serum_creatinine" type="number" min="0" max="30" step="0.01" value="1.1" required></div>
        </div>
      </fieldset>

      <button type="submit">Predict risk</button>
    </form>

    <div id="error" hidden></div>

    <div id="result" hidden>
      <div class="result-label">Predicted probability of death during follow-up</div>
      <div class="hero-row">
        <div class="hero-value" id="hero-value">--</div>
        <span class="status"><span class="status-dot" id="status-dot"></span><span id="status-label"></span></span>
      </div>

      <div class="meter-wrap">
        <div class="meter-track">
          <div class="meter-fill" id="meter-fill"></div>
          <div class="meter-baseline" style="left: 68%;"></div>
        </div>
        <div class="meter-baseline-label" style="left: 68%;">Cohort average 68%</div>
      </div>

      <div class="meta">
        Model version: <span id="model-version"></span><br>
        Illustrative risk band, not a validated clinical risk score. See the
        model card for calibration, uncertainty, and subgroup performance.
      </div>
    </div>
  </div>

<script>
function statusForRisk(p) {
  if (p < 0.50) return { label: "Low risk", color: "var(--status-good)" };
  if (p < 0.65) return { label: "Moderate risk", color: "var(--status-warning)" };
  if (p < 0.80) return { label: "Elevated risk", color: "var(--status-serious)" };
  return { label: "High risk", color: "var(--status-critical)" };
}

document.getElementById("predict-form").addEventListener("submit", async function (e) {
  e.preventDefault();
  const form = e.target;
  const errorBox = document.getElementById("error");
  const resultBox = document.getElementById("result");
  errorBox.hidden = true;
  resultBox.hidden = true;

  const payload = {
    age: parseFloat(form.age.value),
    sex: parseInt(form.sex.value, 10),
    n_comorbidities: parseInt(form.n_comorbidities.value, 10),
    diabetes: parseInt(form.diabetes.value, 10),
    dementia: parseInt(form.dementia.value, 10),
    cancer: parseInt(form.cancer.value, 10),
    mean_bp: parseFloat(form.mean_bp.value),
    heart_rate: parseFloat(form.heart_rate.value),
    resp_rate: parseFloat(form.resp_rate.value),
    temperature: parseFloat(form.temperature.value),
    serum_sodium: parseFloat(form.serum_sodium.value),
    wbc: parseFloat(form.wbc.value),
    serum_creatinine: parseFloat(form.serum_creatinine.value),
  };

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const body = await res.json();

    if (!res.ok) {
      const detail = body.detail;
      const messages = Array.isArray(detail)
        ? detail.map((d) => (d.loc || []).slice(-1)[0] + ": " + d.msg)
        : [String(detail)];
      errorBox.textContent = messages.join(" \\u00b7 ");
      errorBox.hidden = false;
      return;
    }

    const pct = body.risk_probability * 100;
    const status = statusForRisk(body.risk_probability);
    document.getElementById("hero-value").textContent = pct.toFixed(1) + "%";
    document.getElementById("status-dot").style.background = status.color;
    document.getElementById("status-label").textContent = status.label;
    document.getElementById("meter-fill").style.width = pct + "%";
    document.getElementById("model-version").textContent = body.model_version;
    resultBox.hidden = false;
  } catch (err) {
    errorBox.textContent = "Could not reach the API: " + err;
    errorBox.hidden = false;
  }
});
</script>
</body>
</html>
"""

_TRIAGE_EXTRA_STYLE = """
  .banner {
    display: flex; gap: 10px; align-items: flex-start;
    background: var(--status-warning-bg); border: 1px solid color-mix(in srgb, var(--status-warning) 55%, transparent);
    border-radius: 12px; padding: 13px 15px; font-size: 13px; color: var(--text-primary);
    margin-bottom: 22px; line-height: 1.55;
  }
  .banner .icon { flex: none; margin-top: 1px; color: var(--status-warning); }
  .banner b { display: block; margin-bottom: 3px; }

  .stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 22px; }
  .stat {
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 12px;
    padding: 14px 16px; border-top: 3px solid var(--stat-color, var(--fill));
  }
  .stat .stat-label { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
    color: var(--text-secondary); font-weight: 700; margin-bottom: 4px; }
  .stat .stat-value { font-size: 26px; font-weight: 700; letter-spacing: -0.02em; }
  .stat-urgent { --stat-color: var(--status-critical); }
  .stat-monitor { --stat-color: var(--status-warning); }
  .stat-routine { --stat-color: var(--status-good); }

  .table-wrap { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
  table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
  thead th {
    position: sticky; top: 0; background: var(--surface-2);
    text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border);
    color: var(--text-secondary); font-weight: 700; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.04em;
  }
  tbody td { text-align: left; padding: 10px 12px; border-bottom: 1px solid var(--border); }
  tbody tr:last-child td { border-bottom: none; }
  tbody tr:nth-child(even) { background: color-mix(in srgb, var(--surface-2) 55%, transparent); }
  tbody tr:hover { background: color-mix(in srgb, var(--fill) 7%, transparent); }
  td.num { font-variant-numeric: tabular-nums; }

  .tier {
    display: inline-flex; align-items: center; gap: 6px; font-weight: 700; font-size: 12.5px;
    padding: 4px 10px; border-radius: 999px;
  }
  .tier-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
  .tier-urgent { background: var(--status-critical-bg); color: var(--status-critical); }
  .tier-urgent .tier-dot { background: var(--status-critical); }
  .tier-monitor { background: var(--status-warning-bg); color: var(--status-warning); }
  .tier-monitor .tier-dot { background: var(--status-warning); }
  .tier-routine { background: var(--status-good-bg); color: var(--status-good); }
  .tier-routine .tier-dot { background: var(--status-good); }

  .meta { margin-top: 16px; font-size: 12px; color: var(--text-muted); }
"""

_TRIAGE_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SUPPORT Triage Panel -- demo</title>
<style>""" + _SHARED_STYLE + _TRIAGE_EXTRA_STYLE + """</style>
</head>
<body>
  <div class="topnav">
    <div class="brand">""" + _NAV_PULSE_ICON + """ SUPPORT Risk Suite</div>
    <div class="navlinks">
      <a href="/">Predictor</a>
      <a href="/triage-view" class="active">Triage Panel</a>
    </div>
  </div>

  <div class="card" style="max-width: 920px;">
    <h1>Triage panel</h1>
    <p class="subtitle">Decision-support risk stratification, not automated resource allocation.</p>

    <div class="banner">
      <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"
        stroke-linecap="round" stroke-linejoin="round">
        <path d="M12 9v4M12 17h.01M10.3 3.9 2 18a2 2 0 0 0 1.7 3h16.6a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>
      </svg>
      <div>
        <b>Read before using this panel</b>
        This ranks a fixed sample of patients from a historical 1990s research cohort, not a live
        hospital feed -- a demonstration, not a deployed clinical tool. Discrimination is markedly
        worse for patients 85+ (AUROC ~0.60 vs ~0.74 for under-50s) -- treat risk estimates for the
        oldest patients as lower-confidence. Tiers are meant to prompt a care-team conversation
        (e.g. a goals-of-care discussion), never to trigger an automatic decision.
      </div>
    </div>

    <div class="stats" id="stats" hidden>
      <div class="stat stat-urgent"><div class="stat-label">Urgent</div><div class="stat-value" id="count-urgent">--</div></div>
      <div class="stat stat-monitor"><div class="stat-label">Monitor</div><div class="stat-value" id="count-monitor">--</div></div>
      <div class="stat stat-routine"><div class="stat-label">Routine</div><div class="stat-value" id="count-routine">--</div></div>
    </div>

    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Patient</th><th>Age</th><th>Cancer</th>
            <th>Overall risk</th><th>30-day risk</th><th>90-day risk</th><th>Tier</th>
          </tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </div>

    <div class="meta" id="meta">Loading...</div>
  </div>

<script>
const tierClass = { "Urgent": "tier-urgent", "Monitor": "tier-monitor", "Routine": "tier-routine" };

fetch("/triage")
  .then((res) => res.json())
  .then((data) => {
    const rows = document.getElementById("rows");
    rows.innerHTML = data.patients.map((p) => `
      <tr>
        <td>#${p.patient_id}</td>
        <td class="num">${p.age}</td>
        <td>${p.cancer}</td>
        <td class="num">${(p.overall_risk * 100).toFixed(1)}%</td>
        <td class="num">${(p.risk_30d * 100).toFixed(1)}%</td>
        <td class="num">${(p.risk_90d * 100).toFixed(1)}%</td>
        <td><span class="tier ${tierClass[p.tier] || ''}"><span class="tier-dot"></span>${p.tier}</span></td>
      </tr>
    `).join("");

    const counts = { Urgent: 0, Monitor: 0, Routine: 0 };
    data.patients.forEach((p) => { counts[p.tier] = (counts[p.tier] || 0) + 1; });
    document.getElementById("count-urgent").textContent = counts.Urgent;
    document.getElementById("count-monitor").textContent = counts.Monitor;
    document.getElementById("count-routine").textContent = counts.Routine;
    document.getElementById("stats").hidden = false;

    document.getElementById("meta").textContent =
      `${data.patients.length} patients, sampled from the held-out test set. Model version: ${data.model_version}`;
  })
  .catch((err) => {
    document.getElementById("meta").textContent = "Could not load the triage panel: " + err;
  });
</script>
</body>
</html>
"""


@app.get("/triage-view", response_class=HTMLResponse)
def triage_page() -> str:
    return _TRIAGE_PAGE


@app.get("/", response_class=HTMLResponse)
def demo_page() -> str:
    return _DEMO_PAGE


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
