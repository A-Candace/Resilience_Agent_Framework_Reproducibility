"""Compatibility server retaining the geospatial endpoint."""

from fastmcp import FastMCP

from resilience_app.agents.flood_risk_agent.mcp_server import (
    flood_grid_summary,
)


mcp = FastMCP(
    "NYC Resilience Geospatial Tools"
)

mcp.tool(
    flood_grid_summary
)


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="0.0.0.0",
        port=8001,
    )