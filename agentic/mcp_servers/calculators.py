"""Compatibility server retaining the original combined calculator endpoint."""
from fastmcp import FastMCP
from resilience_app.agents.green_roof_agent.mcp_server import green_roof_estimate
from resilience_app.agents.rain_garden_agent.mcp_server import rain_garden_estimate
mcp=FastMCP("NYC Resilience Calculator Tools")
mcp.tool(green_roof_estimate)
mcp.tool(rain_garden_estimate)
if __name__ == "__main__": mcp.run(transport="http",host="0.0.0.0",port=8002)
