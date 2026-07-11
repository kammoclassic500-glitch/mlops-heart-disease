"""
FastAPI serving app for the Heart Disease risk model.

Endpoints:
    GET  /health   -> {"status": "ok"}
    POST /predict  -> {"prediction": 0/1, "confidence": float, "risk_label": str}

Run locally (outside Docker) with:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""

import logging
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Logging setup (Task 8 will build on this — keep the format simple/parseable)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("api_requests.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("heart-disease-api")

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "models/heart_disease_model.joblib")

app = FastAPI(
    title="Heart Disease Risk Prediction API",
    description="Predicts heart disease risk from patient clinical features.",
    version="1.0.0",
)

model = None


@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model file not found at {MODEL_PATH}")
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    logger.info(f"Model loaded from {MODEL_PATH} — type: {type(model)}")


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class PatientFeatures(BaseModel):
    age: int = Field(..., example=63)
    sex: int = Field(..., ge=0, le=1, example=1)
    cp: int = Field(..., example=3)
    trestbps: float = Field(..., example=145)
    chol: float = Field(..., example=233)
    fbs: int = Field(..., ge=0, le=1, example=1)
    restecg: int = Field(..., example=0)
    thalach: float = Field(..., example=150)
    exang: int = Field(..., ge=0, le=1, example=0)
    oldpeak: float = Field(..., example=2.3)
    slope: int = Field(..., example=0)
    ca: int = Field(..., example=0)
    thal: int = Field(..., example=1)


class PredictionResponse(BaseModel):
    prediction: int
    risk_label: str
    confidence: float


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "Heart Disease Risk Prediction API. See /docs for usage."}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: PatientFeatures):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        input_df = pd.DataFrame([features.dict()])
        pred = int(model.predict(input_df)[0])

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(input_df)[0]
            confidence = float(np.max(proba))
        else:
            confidence = 1.0

        risk_label = "High Risk" if pred == 1 else "Low Risk"

        logger.info(
            f"Prediction served | input={features.dict()} | "
            f"prediction={pred} | confidence={confidence:.4f}"
        )

        return PredictionResponse(
            prediction=pred, risk_label=risk_label, confidence=round(confidence, 4)
        )

    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")
