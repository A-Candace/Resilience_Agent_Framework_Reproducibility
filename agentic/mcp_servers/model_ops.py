from __future__ import annotations

from typing import Any

import pandas as pd
from fastmcp import FastMCP

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
    predict_gcn_snapshot,
)

mcp = FastMCP("NYC Resilience Model Operations")


def _gcn_runtime_metadata() -> dict[str, Any]:
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


@mcp.tool
def production_model_metadata() -> dict:
    """Return metadata for the MLflow-backed production GCN alias."""
    return _gcn_runtime_metadata()


@mcp.tool
def reload_production_model() -> dict:
    """
    Clear this MCP process's cached model and reload the current MLflow champion.
    """
    clear_model_cache()
    load_gcn_champion()

    metadata = _gcn_runtime_metadata()
    metadata["status"] = "reloaded"
    return metadata


@mcp.tool
def predict_flood_snapshot(snapshot: list[dict]) -> list[dict]:
    """
    Score one hourly sensor snapshot using the promoted MLflow GCN.

    Required fields per record:
    deployment_id
    precip_current_hour_mm
    precip_previous_6h_mm
    daily_total_precip_mm
    """
    if not isinstance(snapshot, list):
        raise TypeError("snapshot must be a list of sensor dictionaries.")

    frame = pd.DataFrame(snapshot)
    predictions = predict_gcn_snapshot(frame)

    return predictions.to_dict(orient="records")


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8004,
    )