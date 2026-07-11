"""
Model Serving API + Logging/Monitoring
----------------------------------------------------
FastAPI app that loads the trained pipeline (preprocessing + model bundled
together via joblib) and exposes:
  GET  /health   -> liveness check
  POST /predict  -> heart disease risk prediction
  GET  /metrics  -> Prometheus-scrapeable metrics

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import time

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("api_requests.log")],
)
logger = logging.getLogger("heart-disease-api")

MODEL_PATH = os.getenv("MODEL_PATH", "models/heart_disease_model.joblib")

app = FastAPI(title="Heart Disease Risk Prediction API", version="1.0.0")

REQUEST_COUNT = Counter("predict_requests_total", "Total number of /predict requests")
REQUEST_LATENCY = Histogram("predict_latency_seconds", "Latency of /predict requests")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
        logger.info(f"Loaded model from {MODEL_PATH}")
    return _model


class PatientFeatures(BaseModel):
    age: float = Field(..., json_schema_extra={"example": 63})
    sex: int = Field(..., description="1 = male, 0 = female", json_schema_extra={"example": 1})
    cp: int = Field(..., description="chest pain type (0-3)", json_schema_extra={"example": 3})
    trestbps: float = Field(..., description="resting blood pressure", json_schema_extra={"example": 145})
    chol: float = Field(..., description="serum cholesterol mg/dl", json_schema_extra={"example": 233})
    fbs: int = Field(..., description="fasting blood sugar > 120 mg/dl (1/0)", json_schema_extra={"example": 1})
    restecg: int = Field(..., description="resting ECG results (0-2)", json_schema_extra={"example": 0})
    thalach: float = Field(..., description="max heart rate achieved", json_schema_extra={"example": 150})
    exang: int = Field(..., description="exercise induced angina (1/0)", json_schema_extra={"example": 0})
    oldpeak: float = Field(..., json_schema_extra={"example": 2.3})
    slope: int = Field(..., json_schema_extra={"example": 0})
    ca: float = Field(..., description="number of major vessels (0-3)", json_schema_extra={"example": 0})
    thal: int = Field(..., json_schema_extra={"example": 1})


class PredictionResponse(BaseModel):
    prediction: int
    label: str
    confidence: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    start = time.time()
    REQUEST_COUNT.inc()
    try:
        model = get_model()
        row = pd.DataFrame([features.model_dump()])
        pred = int(model.predict(row)[0])
        proba = float(model.predict_proba(row)[0][pred])

        logger.info(f"request={features.model_dump()} prediction={pred} confidence={proba:.3f}")

        return PredictionResponse(
            prediction=pred,
            label="disease_risk" if pred == 1 else "low_risk",
            confidence=round(proba, 4),
        )
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        REQUEST_LATENCY.observe(time.time() - start)
