import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

SAMPLE_PATIENT = {
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1,
}


def test_health_endpoint():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_endpoint_returns_valid_shape():
    r = client.post("/predict", json=SAMPLE_PATIENT)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert body["label"] in ("disease_risk", "low_risk")
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_endpoint_rejects_bad_input():
    bad_payload = {"age": "not_a_number"}
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_metrics_endpoint_exposes_prometheus_format():
    r = client.get("/metrics")
    assert r.status_code == 200
    assert b"predict_requests_total" in r.content
