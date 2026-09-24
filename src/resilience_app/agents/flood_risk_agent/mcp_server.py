from fastmcp import FastMCP
from resilience_app.core.shared import load_flood_grid
mcp = FastMCP("Flood Risk Agent")
@mcp.tool
def flood_grid_summary() -> dict:
    """Report whether the 4km historical flood-risk grid loaded and how many grid cells it contains, as {"available": bool, "rows": int}; takes no inputs. Can raise FileNotFoundError if the underlying grid data file is missing — `available: False` is never actually returned."""
    # load_flood_grid() raises FileNotFoundError instead of returning None when the
    # file is missing, so `data is not None` below is always True on success — a
    # missing file surfaces as an uncaught exception, not `available: False`.
    data = load_flood_grid()
    return {"available": data is not None, "rows": 0 if data is None else len(data)}
if __name__ == "__main__": mcp.run(transport="http", host="0.0.0.0", port=8011)
