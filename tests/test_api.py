"""End-to-end smoke test for the FastAPI serving layer.

Trains the model artifact if it doesn't exist yet (e.g. a fresh checkout, or
CI), so this test is self-sufficient and doesn't depend on a manual step
having been run first.
"""
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "gradient_boosting.joblib"

VALID_PATIENT = {
    "age": 65, "sex": 1, "n_comorbidities": 2, "diabetes": 0,
    "dementia": 0, "cancer": 1, "mean_bp": 80, "heart_rate": 90,
    "resp_rate": 20, "temperature": 37.0, "serum_sodium": 138,
    "wbc": 9.5, "serum_creatinine": 1.1,
}


@pytest.fixture(scope="module", autouse=True)
def _ensure_model_trained():
    if not MODEL_PATH.exists():
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train_and_save.py")],
            cwd=ROOT, check=True, timeout=300,
        )


@pytest.fixture
def client():
    from support_survival.api import app
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_end_to_end_returns_risk_and_version(client):
    response = client.post("/predict", json=VALID_PATIENT)
    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["risk_probability"] <= 1.0
    assert isinstance(body["model_version"], str) and body["model_version"]


def test_predict_rejects_out_of_range_value(client):
    bad = {**VALID_PATIENT, "age": -5}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_rejects_missing_field(client):
    incomplete = dict(VALID_PATIENT)
    del incomplete["age"]
    response = client.post("/predict", json=incomplete)
    assert response.status_code == 422


def test_predict_rejects_wrong_type(client):
    bad = {**VALID_PATIENT, "age": "not-a-number"}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_demo_page_serves_html_with_the_predict_form(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert 'id="predict-form"' in response.text
    assert "/predict" in response.text


def test_triage_returns_ranked_patients_with_both_risk_horizons(client):
    response = client.get("/triage")
    assert response.status_code == 200
    body = response.json()
    patients = body["patients"]
    assert len(patients) == 15
    for p in patients:
        assert 0.0 <= p["overall_risk"] <= 1.0
        assert 0.0 <= p["risk_30d"] <= 1.0
        assert 0.0 <= p["risk_90d"] <= 1.0
        assert p["risk_90d"] >= p["risk_30d"] - 1e-9
        assert p["tier"] in {"Urgent", "Monitor", "Routine"}
    # sorted by 30-day risk, descending
    risks_30d = [p["risk_30d"] for p in patients]
    assert risks_30d == sorted(risks_30d, reverse=True)


def test_triage_is_stable_across_requests():
    from support_survival.api import app
    client_a, client_b = TestClient(app), TestClient(app)
    ids_a = [p["patient_id"] for p in client_a.get("/triage").json()["patients"]]
    ids_b = [p["patient_id"] for p in client_b.get("/triage").json()["patients"]]
    assert ids_a == ids_b


def test_triage_view_serves_html_pointing_at_triage_endpoint(client):
    response = client.get("/triage-view")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "/triage" in response.text


@pytest.mark.parametrize("path,content_type", [
    ("/static/css/monitor.css", "text/css"),
    ("/static/js/monitor.js", "text/javascript"),
    ("/static/js/predict.js", "text/javascript"),
    ("/static/js/triage.js", "text/javascript"),
])
def test_static_frontend_assets_are_served(client, path, content_type):
    response = client.get(path)
    assert response.status_code == 200
    assert content_type in response.headers["content-type"]
    assert len(response.text) > 0
