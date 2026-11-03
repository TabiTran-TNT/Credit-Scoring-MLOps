from contextlib import asynccontextmanager
import logging
from time import perf_counter
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
import pandas as pd

from app.model_loader import (
    get_feature_columns, get_model, get_model_id, get_model_version, load_model,
)
from app.schemas import HealthResponse, LoanApplication, PredictionResponse
from credit_scoring.features import prepare_features
from monitoring.logger import get_logger, log_event
from monitoring.metrics import REQUESTS, LATENCY, CONTENT_TYPE_LATEST, generate_latest

FEATURE_REPRESENTATION = "engineered_features_before_imputation"


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_logger()
    try:
        load_model()
    except Exception:
        log_event("startup_error", level=logging.ERROR, exc_info=True)
        raise
    log_event("startup", model_id=get_model_id())
    try:
        yield
    finally:
        log_event("shutdown", model_id=get_model_id())


app = FastAPI(
    title="Home Credit Scoring API",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware('http')
async def prediction_metrics(request, call_next):
    if request.url.path != '/predict':
        return await call_next(request)
    started = perf_counter()
    try:
        result = await call_next(request)
        REQUESTS.labels(status=str(result.status_code)).inc()
        return result
    except Exception:
        REQUESTS.labels(status='500').inc()
        raise
    finally:
        LATENCY.observe(perf_counter() - started)


@app.get('/metrics', include_in_schema=False)
def metrics():
    return Response(content=generate_latest(), headers={'Content-Type': CONTENT_TYPE_LATEST})


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


def _feature_record(features):
    """Convert one feature row into JSON-compatible values, including nulls."""
    result = {}
    for name, value in features.iloc[0].items():
        if pd.isna(value):
            result[name] = None
        elif hasattr(value, "item"):
            result[name] = value.item()
        else:
            result[name] = value
    return result


@app.post("/predict", response_model=PredictionResponse)
def predict(application: LoanApplication, response: Response):
    started = perf_counter()
    request_id = str(uuid4())
    model_id = None

    try:
        model = get_model()
        model_id = get_model_id()
        raw_data = pd.DataFrame([application.model_dump()])
        features = prepare_features(raw_data)
        features = features.loc[:, get_feature_columns()]

        probability = float(model.predict_proba(features)[0, 1])
        prediction = int(probability >= 0.5)
        # Demo thresholds from the blog, not a validated lending policy.
        risk_category = (
            "Low" if probability < 0.3
            else "Medium" if probability < 0.6
            else "High"
        )
        result = PredictionResponse(
            prediction=prediction,
            probability_of_default=probability,
            risk_category=risk_category,
        )
        log_event(
            "prediction",
            request_id=request_id,
            model_id=model_id,
            model_version=get_model_version(),
            feature_representation=FEATURE_REPRESENTATION,
            features=_feature_record(features),
            outputs=result.model_dump(),
            inference_time_ms=round((perf_counter() - started) * 1000, 2),
        )
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Model-ID"] = model_id
        if get_model_version() is not None:
            response.headers["X-Model-Version"] = get_model_version()
        return result
    except Exception:
        log_event(
            "prediction_error",
            level=logging.ERROR,
            exc_info=True,
            request_id=request_id,
            model_id=model_id,
            inference_time_ms=round((perf_counter() - started) * 1000, 2),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed. Request ID: {request_id}",
            headers={"X-Request-ID": request_id},
        ) from None
