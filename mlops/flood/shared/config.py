from __future__ import annotations

import os
from pathlib import Path


# ============================================================
# PROJECT / EXPERIMENT
# ============================================================

MLFLOW_EXPERIMENT_NAME = os.getenv(
    "FLOOD_MLFLOW_EXPERIMENT",
    "nyc-flood-adaptive-spatial-mlops",
)


# ============================================================
# MLFLOW MODEL REGISTRY / SERVING
# ============================================================

MLFLOW_TRACKING_URI = (
    os.getenv("MLFLOW_TRACKING_URI")
    or "http://localhost:5000"
)

FLOOD_GCN_REGISTERED_MODEL_NAME = os.getenv(
    "FLOOD_GCN_REGISTERED_MODEL_NAME",
    "nyc-resilience-flood-gcn",
)

FLOOD_GCN_MODEL_ALIAS = os.getenv(
    "FLOOD_GCN_MODEL_ALIAS",
    "champion",
)


# ============================================================
# MODEL AVAILABILITY
# ============================================================

ENABLE_GCN = os.getenv(
    "ENABLE_GCN",
    "true",
).lower() == "true"

# Logistic is intentionally supported in the architecture
# even though the production implementation is not ready yet.
ENABLE_LOGISTIC = os.getenv(
    "ENABLE_LOGISTIC",
    "false",
).lower() == "true"


# ============================================================
# SENSOR ELIGIBILITY THRESHOLDS
# ============================================================

MIN_SENSOR_RECALL = float(
    os.getenv(
        "MIN_SENSOR_RECALL",
        "0.30",
    )
)

MIN_SENSOR_PRECISION = float(
    os.getenv(
        "MIN_SENSOR_PRECISION",
        "0.25",
    )
)

MODEL_SELECTION_METRIC = os.getenv(
    "MODEL_SELECTION_METRIC",
    "f1",
)


# ============================================================
# FLOOD MODEL SETTINGS
# ============================================================

# Active GCN response threshold:
# 1 inch = 25.4 mm
FLOOD_DEPTH_THRESHOLD_MM = float(
    os.getenv(
        "FLOOD_DEPTH_THRESHOLD_MM",
        "25.4",
    )
)

EVENT_MATCH_WINDOW_HOURS = int(
    os.getenv(
        "EVENT_MATCH_WINDOW_HOURS",
        "24",
    )
)

GRID_SIZE_KM = float(
    os.getenv(
        "FLOOD_GRID_SIZE_KM",
        "1.0",
    )
)


# ============================================================
# GCN FEATURE / OUTPUT CONTRACT
# ============================================================

GCN_FEATURE_COLUMNS = [
    "precip_current_hour_mm",
    "precip_previous_6h_mm",
    "daily_total_precip_mm",
]

GCN_SENSOR_ID_COLUMN = "deployment_id"

GCN_PREDICTION_COLUMN = (
    "predicted_minutes_above_1inch"
)

GCN_PREDICTED_EVENT_COLUMN = (
    "predicted_event"
)

# Prediction duration threshold used by the event evaluator.
GCN_EVENT_DURATION_THRESHOLD_MINUTES = float(
    os.getenv(
        "GCN_EVENT_DURATION_THRESHOLD_MINUTES",
        "1.0",
    )
)


# ============================================================
# RETRAINING / PIPELINE
# ============================================================

RETRAIN_FREQUENCY = os.getenv(
    "FLOOD_RETRAIN_FREQUENCY",
    "weekly",
)


# ============================================================
# LOCAL RUNTIME PATHS
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[3]
)

MODEL_CACHE_DIR = Path(
    os.getenv(
        "MODEL_CACHE_DIR",
        str(
            PROJECT_ROOT
            / "model_cache"
        ),
    )
)

FLOOD_ARTIFACT_DIR = Path(
    os.getenv(
        "FLOOD_ARTIFACT_DIR",
        str(
            PROJECT_ROOT
            / "artifacts"
            / "flood"
        ),
    )
)

FLOOD_ARTIFACT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MODEL ARTIFACT NAMES
# ============================================================

# Legacy/local artifact names are retained for compatibility
# with older utilities and for possible offline fallback.
#
# Production GCN inference should use the MLflow registered
# model alias instead of loading this file directly.
GCN_MODEL_FILENAME = (
    "gcn_model.pt"
)

LOGISTIC_MODEL_FILENAME = (
    "logistic_model.pkl"
)

SENSOR_MODEL_REGISTRY_FILENAME = (
    "sensor_model_registry.parquet"
)

GRID_SENSOR_REGISTRY_FILENAME = (
    "grid_sensor_registry.parquet"
)


# ============================================================
# MLFLOW MODEL URI
# ============================================================

def get_gcn_model_uri(
    alias: str | None = None,
) -> str:
    """
    Return the production-facing MLflow URI for the GCN.

    Example:

        models:/nyc-resilience-flood-gcn@champion

    Passing an alias allows testing another promoted alias
    without changing global configuration.
    """

    resolved_alias = (
        alias
        or FLOOD_GCN_MODEL_ALIAS
    )

    return (
        f"models:/"
        f"{FLOOD_GCN_REGISTERED_MODEL_NAME}"
        f"@{resolved_alias}"
    )


GCN_MODEL_URI = (
    get_gcn_model_uri()
)


# ============================================================
# S3 MODEL LOCATIONS
# ============================================================

MODEL_S3_BUCKET = os.getenv(
    "MODEL_S3_BUCKET",
    "",
)

FLOOD_MODEL_S3_PREFIX = os.getenv(
    "FLOOD_MODEL_S3_PREFIX",
    "model-artifacts/flood",
)