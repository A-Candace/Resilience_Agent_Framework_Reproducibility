from fastmcp import FastMCP
mcp = FastMCP("Rain Garden Agent")
@mcp.tool
def rain_garden_estimate(drainage_area_sqft: float, design_rain_inches: float, capture_fraction: float = 1.0) -> dict:
    """Estimate the stormwater design volume for a rain garden from drainage_area_sqft x design_rain_inches x 0.623 gal/sqft-inch, scaled by capture_fraction (default 1.0); returns {"design_volume_gallons": float, "design_volume_cubic_feet": float} - the same volume in two units (cubic feet = gallons / 7.48052), not two distinct quantities. Negative drainage_area_sqft/design_rain_inches are floored to 0.0 and capture_fraction is clamped to [0, 1] rather than raising, but there's no upper bound, so implausibly large values succeed silently with no sanity check."""
    # Non-numeric, null, or missing arguments are rejected by MCP schema validation
    # before this function runs — the clamping below only ever sees valid floats.
    gallons=max(0.0,drainage_area_sqft)*max(0.0,design_rain_inches)*0.623*min(max(capture_fraction,0.0),1.0)
    return {"design_volume_gallons":gallons,"design_volume_cubic_feet":gallons/7.48052}
if __name__ == "__main__": mcp.run(transport="http", host="0.0.0.0", port=8014)
