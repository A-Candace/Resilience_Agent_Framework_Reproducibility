from fastmcp import FastMCP
mcp = FastMCP("Green Roof Agent")
@mcp.tool
def green_roof_estimate(roof_area_sqft: float, coverage_percent: float, rainfall_inches: float, retention_fraction: float = 0.6) -> dict:
    """Estimate the roof area actually covered (roof_area_sqft x coverage_percent/100) and the stormwater captured in gallons (covered area x rainfall_inches x 0.623 gal/sqft-inch x retention_fraction, default 0.6); returns {"covered_area_sqft": float, "captured_gallons": float}. Negative roof_area_sqft/rainfall_inches are floored to 0.0 and coverage_percent/retention_fraction are clamped to [0, 100]/[0, 1] rather than raising, but there's no upper bound on roof_area_sqft or rainfall_inches, so implausibly large values succeed silently with no sanity check."""
    # Non-numeric, null, or missing arguments are rejected by MCP schema validation
    # before this function runs — the clamping below only ever sees valid floats.
    covered=max(0.0,roof_area_sqft)*min(max(coverage_percent,0.0),100.0)/100.0
    gallons=covered*max(0.0,rainfall_inches)*0.623*min(max(retention_fraction,0.0),1.0)
    return {"covered_area_sqft":covered,"captured_gallons":gallons}
if __name__ == "__main__": mcp.run(transport="http", host="0.0.0.0", port=8013)
