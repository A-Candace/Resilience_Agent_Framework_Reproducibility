from __future__ import annotations

import os

from fastapi import FastAPI, Header, HTTPException

from mlops.flood.shared.config import (
    FLOOD_GCN_MODEL_ALIAS,
    FLOOD_GCN_REGISTERED_MODEL_NAME,
    GCN_MODEL_URI,
    MLFLOW_TRACKING_URI,
)
from mlops.flood.shared.model_interface import (
    clear_model_cache,
    get_model_availability,
    load_gcn_champion,
)

app = FastAPI(title="NYC Resilience Model Control")


def authorize(authorization: str | None) -> None:
    expected = os.getenv("RELOAD_API_TOKEN", "")

    if expected and authorization != f"Bearer {expected}":
        raise HTTPException(
            status_code=401,
            detail="Unauthorized",
        )


def model_metadata() -> dict:
    availability = get_model_availability()["gcn"]

    return {
        "model": "gcn",
        "enabled": availability.enabled,
        "available": availability.available,
        "reason": availability.reason,
        "tracking_uri": MLFLOW_TRACKING_URI,
        "registered_model_name": FLOOD_GCN_REGISTERED_MODEL_NAME,
        "alias": FLOOD_GCN_MODEL_ALIAS,
        "model_uri": GCN_MODEL_URI,
    }


@app.get("/health")
def health() -> dict:
    metadata = model_metadata()

    return {
        "status": "ok" if metadata["available"] else "degraded",
        "model": metadata,
    }


@app.post("/reload-model")
def reload_model(
    authorization: str | None = Header(default=None),
) -> dict:
    authorize(authorization)

    clear_model_cache()

    try:
        load_gcn_champion()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to reload MLflow champion: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc

    return {
        "status": "reloaded",
        "model": model_metadata(),
    }