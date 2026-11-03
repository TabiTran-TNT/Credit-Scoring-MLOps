from contextlib import asynccontextmanager
from datetime import datetime, timezone
import json
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd

from app.model_loader import get_feature_columns, get_model, load_model
from app.schemas import HealthResponse, LoanApplication, PredictionResponse
from credit_scoring.features import prepare_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("credit_scoring_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # If loading fails, startup fails instead of serving a broken model.
    load_model()
    logger.info("Model loaded; API ready.")
    yield


app = FastAPI(
    title="Home Credit Scoring API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    try:
        get_model()
        return HealthResponse(status="healthy", model_loaded=True)
    except RuntimeError:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "model_loaded": False},
        )


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication):
    # A synchronous handler lets FastAPI run blocking model work in a thread.
    started = perf_counter()
    request_id = str(uuid4())

    try:
        raw_data = pd.DataFrame([application.model_dump()])
        features = prepare_features(raw_data)
        features = features.loc[:, get_feature_columns()]

        model = get_model()
        probability = float(model.predict_proba(features)[0, 1])
        # Same default binary threshold as this trained classifier.
        prediction = int(probability >= 0.5)

        # Demo categories from the blog; not validated lending policy.
        risk_category = (
            "Low" if probability < 0.3
            else "Medium" if probability < 0.6
            else "High"
        )
        response = PredictionResponse(
            prediction=prediction,
            probability_of_default=probability,
            risk_category=risk_category,
        )
        logger.info(json.dumps({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": "prediction",
            "request_id": request_id,
            "outputs": response.model_dump(),
            "inference_time_ms": round((perf_counter() - started) * 1000, 2),
        }))
        return response
    except Exception:
        logger.exception("Prediction failed; request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed. Request ID: {request_id}",
        ) from None
