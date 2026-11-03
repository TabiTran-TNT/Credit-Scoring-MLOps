"""Training settings loaded from environment, with optional project .env."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrackingConfig:
    enabled: bool
    tracking_uri: str
    experiment_name: str
    registered_model_name: str


def get_tracking_config():
    # Explicit environment variables take precedence over .env.
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    enabled = os.environ.get("MLFLOW_ENABLED", "true").strip().lower()
    if enabled not in {"true", "false", "1", "0"}:
        raise ValueError("MLFLOW_ENABLED must be true/false or 1/0")
    return TrackingConfig(
        enabled=enabled in {"true", "1"},
        tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
        experiment_name=os.environ.get("MLFLOW_EXPERIMENT_NAME", "credit-scoring"),
        registered_model_name=os.environ.get("MLFLOW_REGISTERED_MODEL_NAME", "credit-scoring"),
    )
