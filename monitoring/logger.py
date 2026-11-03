"""JSON logging for the local, single-worker API."""

from datetime import datetime, timezone
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER_NAME = "credit_scoring_api"


class JsonFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            **getattr(record, "event_data", {}),
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "event": record.getMessage(),
        }
        if record.exc_info:
            # JSON escapes newlines, keeping the entire event on one line.
            entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False, allow_nan=False)


def get_logger():
    logger = logging.getLogger(LOGGER_NAME)
    if getattr(logger, "_credit_scoring_configured", False):
        return logger

    log_dir = Path(os.environ.get("LOG_DIR", "logs"))
    if not log_dir.is_absolute():
        log_dir = PROJECT_ROOT / log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = JsonFormatter()
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        log_dir / "predictions.jsonl",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    # Configure only this application's logger; leave Uvicorn logging alone.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.addHandler(console)
    logger.addHandler(file_handler)
    logger._credit_scoring_configured = True
    return logger


def log_event(event, *, level=logging.INFO, exc_info=False, **fields):
    get_logger().log(
        level,
        event,
        extra={"event_data": fields},
        exc_info=exc_info,
    )
