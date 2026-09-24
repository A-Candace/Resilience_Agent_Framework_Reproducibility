from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd

from mlops.flood.shared.config import (
    ENABLE_GCN,
    ENABLE_LOGISTIC,
    GCN_FEATURE_COLUMNS,
    GCN_MODEL_FILENAME,
    GCN_MODEL_URI,
    GCN_PREDICTED_EVENT_COLUMN,
    GCN_PREDICTION_COLUMN,
    GCN_SENSOR_ID_COLUMN,
    LOGISTIC_MODEL_FILENAME,
    MLFLOW_TRACKING_URI,
    MODEL_CACHE_DIR,
)


# ============================================================
# MODEL AVAILABILITY
# ============================================================

@dataclass
class ModelAvailability:
    """
    Describes whether a model is enabled and whether it is
    currently usable by the prediction pipeline.
    """

    name: str
    enabled: bool
    artifact_path: Path | None
    model_uri: str | None
    available: bool
    reason: str


def _gcn_registry_available() -> tuple[bool, str]:
    """
    Check whether the configured MLflow GCN alias can be resolved.

    This does not perform a prediction; it only verifies that the
    configured registered-model URI exists and can be loaded.
    """

    if not ENABLE_GCN:
        return (
            False,
            "GCN disabled by configuration.",
        )

    try:
        mlflow.set_tracking_uri(
            MLFLOW_TRACKING_URI
        )

        mlflow.pyfunc.load_model(
            GCN_MODEL_URI
        )

        return (
            True,
            f"GCN available from MLflow: {GCN_MODEL_URI}",
        )

    except Exception as exc:
        return (
            False,
            "GCN MLflow model unavailable: "
            f"{type(exc).__name__}: {exc}",
        )


def get_model_availability() -> dict[str, ModelAvailability]:
    """
    Return availability information for all supported models.

    GCN availability is now based on the promoted MLflow model
    alias rather than a local .pt artifact.

    Logistic regression retains the previous local-artifact
    behavior because its production implementation is not ready.
    """

    gcn_legacy_path = (
        MODEL_CACHE_DIR
        / GCN_MODEL_FILENAME
    )

    logistic_path = (
        MODEL_CACHE_DIR
        / LOGISTIC_MODEL_FILENAME
    )

    gcn_available, gcn_reason = (
        _gcn_registry_available()
    )

    logistic_available = (
        ENABLE_LOGISTIC
        and logistic_path.exists()
    )

    return {
        "gcn": ModelAvailability(
            name="gcn",
            enabled=ENABLE_GCN,
            artifact_path=gcn_legacy_path,
            model_uri=GCN_MODEL_URI,
            available=gcn_available,
            reason=gcn_reason,
        ),

        "logistic": ModelAvailability(
            name="logistic",
            enabled=ENABLE_LOGISTIC,
            artifact_path=logistic_path,
            model_uri=None,
            available=logistic_available,
            reason=(
                "Logistic model available."
                if logistic_available
                else (
                    "Logistic model is not enabled."
                    if not ENABLE_LOGISTIC
                    else "Logistic model artifact not found."
                )
            ),
        ),
    }


def get_available_model_names() -> list[str]:
    """
    Return models that can currently participate in prediction.
    """

    availability = (
        get_model_availability()
    )

    return [
        model_name
        for model_name, model_info
        in availability.items()
        if model_info.available
    ]


def require_at_least_one_model() -> list[str]:
    """
    Ensure the pipeline has at least one usable prediction model.
    """

    available_models = (
        get_available_model_names()
    )

    if not available_models:
        raise RuntimeError(
            "No flood prediction models are currently available."
        )

    return available_models


# ============================================================
# GCN MODEL LOADING
# ============================================================

@lru_cache(maxsize=1)
def load_gcn_champion():
    """
    Load and cache the promoted GCN from MLflow.

    Current configured URI:

        models:/nyc-resilience-flood-gcn@champion

    The cache prevents re-downloading and reconstructing the model
    on every prediction request within the same Python process.
    """

    if not ENABLE_GCN:
        raise RuntimeError(
            "GCN is disabled by configuration."
        )

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    try:
        return mlflow.pyfunc.load_model(
            GCN_MODEL_URI
        )

    except Exception as exc:
        raise RuntimeError(
            "Unable to load promoted GCN model from MLflow. "
            f"URI={GCN_MODEL_URI}. "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def clear_model_cache() -> None:
    """
    Clear the in-process model cache.

    Use this after promoting a new MLflow champion when a long-lived
    process should reload the new version without restarting.
    """

    load_gcn_champion.cache_clear()


# ============================================================
# GCN INPUT VALIDATION
# ============================================================

def validate_gcn_snapshot(
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate one hourly GCN prediction snapshot.

    Required columns:

        deployment_id
        precip_current_hour_mm
        precip_previous_6h_mm
        daily_total_precip_mm

    Each deployment_id may appear at most once.

    Unknown deployment IDs are ultimately rejected by the registered
    MLflow model because graph membership is version-specific.
    """

    if not isinstance(
        snapshot,
        pd.DataFrame,
    ):
        raise TypeError(
            "GCN prediction input must be a pandas DataFrame."
        )

    required_columns = {
        GCN_SENSOR_ID_COLUMN,
        *GCN_FEATURE_COLUMNS,
    }

    missing = (
        required_columns
        - set(snapshot.columns)
    )

    if missing:
        raise ValueError(
            "GCN prediction input is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    if snapshot.empty:
        raise ValueError(
            "GCN prediction input is empty."
        )

    result = snapshot[
        [
            GCN_SENSOR_ID_COLUMN,
            *GCN_FEATURE_COLUMNS,
        ]
    ].copy()

    # ------------------------------------------------------------
    # SENSOR ID VALIDATION
    # ------------------------------------------------------------

    sensor_id_series = result[
        GCN_SENSOR_ID_COLUMN
    ]

    null_sensor_mask = (
        sensor_id_series.isna()
    )

    if null_sensor_mask.any():
        raise ValueError(
            "GCN prediction snapshot contains null deployment_id values."
        )

    result[
        GCN_SENSOR_ID_COLUMN
    ] = (
        sensor_id_series
        .astype(str)
        .str.strip()
    )

    blank_sensor_mask = (
        result[
            GCN_SENSOR_ID_COLUMN
        ]
        == ""
    )

    if blank_sensor_mask.any():
        raise ValueError(
            "GCN prediction snapshot contains blank deployment_id values."
        )

    duplicate_mask = result[
        GCN_SENSOR_ID_COLUMN
    ].duplicated()

    if duplicate_mask.any():

        duplicate_ids = (
            result.loc[
                duplicate_mask,
                GCN_SENSOR_ID_COLUMN,
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "GCN prediction snapshot contains duplicate deployment IDs: "
            + ", ".join(
                duplicate_ids[:20]
            )
        )

    # ------------------------------------------------------------
    # FEATURE VALIDATION
    # ------------------------------------------------------------

    for column in GCN_FEATURE_COLUMNS:

        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    invalid_mask = (
        result[
            GCN_FEATURE_COLUMNS
        ]
        .isna()
        .any(axis=1)
    )

    if invalid_mask.any():

        invalid_ids = (
            result.loc[
                invalid_mask,
                GCN_SENSOR_ID_COLUMN,
            ]
            .tolist()
        )

        raise ValueError(
            "GCN precipitation features must be numeric and non-null. "
            "Invalid deployment IDs: "
            + ", ".join(
                invalid_ids[:20]
            )
        )

    return result


# ============================================================
# GCN PREDICTION
# ============================================================

def predict_gcn_snapshot(
    snapshot: pd.DataFrame,
) -> pd.DataFrame:
    """
    Score one hourly sensor snapshot using the promoted MLflow GCN.

    Returns:

        deployment_id
        predicted_minutes_above_1inch
        predicted_event
    """

    validated = (
        validate_gcn_snapshot(
            snapshot
        )
    )

    model = (
        load_gcn_champion()
    )

    predictions = (
        model.predict(
            validated
        )
    )

    if not isinstance(
        predictions,
        pd.DataFrame,
    ):
        raise RuntimeError(
            "MLflow GCN returned an unexpected prediction type: "
            f"{type(predictions).__name__}"
        )

    required_output_columns = {
        GCN_SENSOR_ID_COLUMN,
        GCN_PREDICTION_COLUMN,
        GCN_PREDICTED_EVENT_COLUMN,
    }

    missing = (
        required_output_columns
        - set(predictions.columns)
    )

    if missing:
        raise RuntimeError(
            "MLflow GCN prediction output is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    return predictions[
        [
            GCN_SENSOR_ID_COLUMN,
            GCN_PREDICTION_COLUMN,
            GCN_PREDICTED_EVENT_COLUMN,
        ]
    ].copy()