from __future__ import annotations
import json
import os
import threading
from pathlib import Path
from typing import Any
import boto3
import joblib

_LOCK = threading.RLock()
_MODEL: Any = None
_METADATA: dict = {}


def _paths() -> tuple[Path, Path]:
    model_path = Path(os.getenv("MODEL_LOCAL_PATH", "/app/model_cache/model.pkl"))
    return model_path, model_path.with_suffix(".metadata.json")


def download_production_model() -> dict:
    bucket = os.environ["MODEL_S3_BUCKET"]
    model_key = os.getenv("MODEL_S3_KEY", "nyc-resilience/models/production/model.pkl")
    metadata_key = os.getenv("MODEL_METADATA_S3_KEY", "nyc-resilience/models/production/metadata.json")
    model_path, metadata_path = _paths()
    model_path.parent.mkdir(parents=True, exist_ok=True)
    s3 = boto3.client("s3")
    s3.download_file(bucket, model_key, str(model_path))
    try:
        s3.download_file(bucket, metadata_key, str(metadata_path))
        metadata = json.loads(metadata_path.read_text())
    except Exception:
        metadata = {"model_key": model_key}
    with _LOCK:
        global _MODEL, _METADATA
        _MODEL = joblib.load(model_path)
        _METADATA = metadata
    return metadata


def get_model() -> Any:
    with _LOCK:
        return _MODEL


def get_metadata() -> dict:
    with _LOCK:
        return dict(_METADATA)
