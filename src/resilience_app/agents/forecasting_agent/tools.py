

from __future__ import annotations

from resilience_app.agents.forecasting_agent.service import (
    forecast_heatmap_service,
    forecast_sensor_snapshot_service,
    get_available_forecast_hours_service,
    get_cluster_forecast_service,
    get_daypart_forecast_summary_service,
    get_flood_event_forecast_service,
    get_forecast_summary_service,
    get_grid_forecast_service,
    get_highest_risk_grids_service,
    get_hourly_forecast_service,
    get_legacy_forecast_status_service,
)


# ---------------------------------------------------------------------
# Legacy tools
# ---------------------------------------------------------------------

def forecast_heatmap_tool(
    forecast_inches: float,
) -> list[dict]:
    result = forecast_heatmap_service(forecast_inches)

    return (
        result.to_dict(orient="records")
        if hasattr(result, "to_dict")
        else result
    )


def forecast_sensor_flood_snapshot_tool(
    snapshot: list[dict],
) -> list[dict]:
    result = forecast_sensor_snapshot_service(snapshot)
    return result.to_dict(orient="records")


def get_legacy_forecast_status_tool() -> dict:
    return get_legacy_forecast_status_service()


# ---------------------------------------------------------------------
# Authoritative operational 1-km forecast tools
# ---------------------------------------------------------------------

def get_daypart_forecast_summary_tool(
    mode: str = "tomorrow",
    daypart: str = "afternoon",
    top_n: int = 20,
) -> dict:
    return get_daypart_forecast_summary_service(
        mode=mode,
        daypart=daypart,
        top_n=top_n,
    )

def get_forecast_summary_tool(
    mode: str = "tomorrow",
) -> dict:
    return get_forecast_summary_service(mode=mode)


def get_available_forecast_hours_tool(
    mode: str = "tomorrow",
) -> list[dict]:
    return get_available_forecast_hours_service(mode=mode)


def get_grid_flood_forecast_tool(
    grid_id: str,
    mode: str = "tomorrow",
) -> list[dict]:
    return get_grid_forecast_service(
        grid_id=grid_id,
        mode=mode,
    )


def get_hourly_flood_forecast_tool(
    forecast_hour: str,
    mode: str = "tomorrow",
) -> list[dict]:
    return get_hourly_forecast_service(
        forecast_hour=forecast_hour,
        mode=mode,
    )


def get_highest_risk_grids_tool(
    limit: int = 20,
    mode: str = "tomorrow",
    forecast_hour: str | None = None,
) -> list[dict]:
    return get_highest_risk_grids_service(
        limit=limit,
        mode=mode,
        forecast_hour=forecast_hour,
    )


def get_forecast_by_cluster_tool(
    cluster_number: int,
    mode: str = "tomorrow",
    forecast_hour: str | None = None,
) -> list[dict]:
    return get_cluster_forecast_service(
        cluster_number=cluster_number,
        mode=mode,
        forecast_hour=forecast_hour,
    )


def get_predicted_flood_events_tool(
    mode: str = "tomorrow",
    forecast_hour: str | None = None,
) -> list[dict]:
    return get_flood_event_forecast_service(
        mode=mode,
        forecast_hour=forecast_hour,
    )
