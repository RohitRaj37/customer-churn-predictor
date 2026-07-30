import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import uvicorn

from src.preprocess import encode_features, scale_features
from src.interpret import interpret_prediction

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

app = FastAPI(
    title="Customer Churn Predictor API",
    description="Predict customer churn with SHAP explanations and retention suggestions",
    version="1.0.0",
)


class CustomerFeatures(BaseModel):
    gender: str = "Male"
    SeniorCitizen: str = "No"
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = 1
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "Fiber optic"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "Yes"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 70.0
    TotalCharges: float = 100.0


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: str
    top_factors: list
    retention_suggestion: str


class HealthResponse(BaseModel):
    status: str
    model: str
    features: int


def _load_artifacts():
    scores = joblib.load(MODELS_DIR / "scores.pkl")
    best_name = max(scores, key=scores.get)
    model = joblib.load(MODELS_DIR / f"{best_name}.pkl")
    encoders = joblib.load(MODELS_DIR / "encoders.pkl")
    scaler = joblib.load(MODELS_DIR / "scaler.pkl")
    feature_names = joblib.load(PROCESSED_DIR / "X_train.pkl").columns.tolist()
    return model, best_name, encoders, scaler, feature_names


model, model_name, encoders, scaler, feature_names = _load_artifacts()


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="ok",
        model=model_name,
        features=len(feature_names),
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    try:
        row = pd.DataFrame([customer.model_dump()])
        row, _ = encode_features(row, fit=False, encoders=encoders)
        row, _ = scale_features(row, fit=False, scaler=scaler)

        if set(row.columns) != set(feature_names):
            row = row[feature_names]

        result = interpret_prediction(model, row, feature_names, raw_features=customer.model_dump())
        return PredictionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
