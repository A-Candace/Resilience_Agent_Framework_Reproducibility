

from fastmcp import FastMCP

from resilience_app.agents.forecasting_agent.tools import (
    forecast_heatmap_tool,
    forecast_sensor_flood_snapshot_tool,
    get_available_forecast_hours_tool,
    get_daypart_forecast_summary_tool,
    get_forecast_by_cluster_tool,
    get_forecast_summary_tool,
    get_grid_flood_forecast_tool,
    get_highest_risk_grids_tool,
    get_hourly_flood_forecast_tool,
    get_legacy_forecast_status_tool,
    get_predicted_flood_events_tool,
)


mcp = FastMCP("Forecasting Agent")


# ---------------------------------------------------------------------
# Authoritative operational 1-km forecast tools
# ---------------------------------------------------------------------

@mcp.tool
def get_forecast_summary(
    mode: str = "tomorrow",
) -> dict:
    """Get the authoritative NYC operational forecast summary.

    mode must be "today" or "tomorrow".
    The response reports adaptive/truncated forecast horizons explicitly.
    """
    return get_forecast_summary_tool(mode=mode)


@mcp.tool
def get_available_forecast_hours(
    mode: str = "tomorrow",
) -> list[dict]:
    """List every published forecast hour in UTC and NYC local time."""
    return get_available_forecast_hours_tool(mode=mode)

@mcp.tool
def get_daypart_forecast_summary(
    mode: str = "tomorrow",
    daypart: str = "afternoon",
    top_n: int = 20,
) -> dict:
    """Get a compact NYC-local daypart forecast summary.

    daypart must be one of:
    overnight, morning, afternoon, evening, night.

    Use this for natural-language multi-hour requests such as
    "tomorrow afternoon". It returns available-hour coverage,
    hourly predicted-flood-grid counts, unique affected grids,
    and a compact top-risk sample without returning 837 rows
    for every hour.
    """
    return get_daypart_forecast_summary_tool(
        mode=mode,
        daypart=daypart,
        top_n=top_n,
    )

@mcp.tool
def get_grid_flood_forecast(
    grid_id: str,
    mode: str = "tomorrow",
) -> list[dict]:
    """Get all available hourly forecasts for one NYC 1-km grid."""
    return get_grid_flood_forecast_tool(
        grid_id=grid_id,
        mode=mode,
    )


@mcp.tool
def get_hourly_flood_forecast(
    forecast_hour: str,
    mode: str = "tomorrow",
) -> list[dict]:
    """Get all NYC 1-km grid forecasts at one published hour.

    forecast_hour may be UTC or include an explicit timezone offset.
    """
    return get_hourly_flood_forecast_tool(
        forecast_hour=forecast_hour,
        mode=mode,
    )


@mcp.tool
def get_highest_risk_grids(
    limit: int = 20,
    mode: str = "tomorrow",
    forecast_hour: str | None = None,
) -> list[dict]:
    """Get highest-risk grid/hour forecasts for today or tomorrow."""
    return get_highest_risk_grids_tool(
        limit=limit,
        mode=mode,
        forecast_hour=forecast_hour,
    )


@mcp.tool
def get_forecast_by_cluster(
    cluster_number: int,
    mode: str = "tomorrow",
    forecast_hour: str | None = None,
) -> list[dict]:
    """Get operational forecasts for one spatial cluster."""
    return get_forecast_by_cluster_tool(
        cluster_number=cluster_number,
        mode=mode,
        forecast_hour=forecast_hour,
    )


@mcp.tool
def get_predicted_flood_events(
    mode: str = "tomorrow",
    forecast_hour: str | None = None,
) -> list[dict]:
    """Get grid/hour rows crossing the operational event threshold."""
    return get_predicted_flood_events_tool(
        mode=mode,
        forecast_hour=forecast_hour,
    )


# ---------------------------------------------------------------------
# Legacy / archive tools retained for compatibility
# ---------------------------------------------------------------------

@mcp.tool
def get_legacy_forecast_status() -> dict:
    """Report legacy forecasting availability without making it authoritative."""
    return get_legacy_forecast_status_tool()


@mcp.tool
def forecast_heatmap(
    forecast_inches: float,
) -> list[dict]:
    """Run the legacy historical rainfall-threshold heatmap heuristic."""
    return forecast_heatmap_tool(forecast_inches)


@mcp.tool
def forecast_sensor_flood_snapshot(
    snapshot: list[dict],
) -> list[dict]:
    """Run the retained direct GCN sensor-snapshot diagnostic workflow."""
    return forecast_sensor_flood_snapshot_tool(snapshot)


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8012,
    )
