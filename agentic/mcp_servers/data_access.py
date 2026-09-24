from pathlib import Path
from fastmcp import FastMCP
from agentic.common.aws_clients import s3_client
from agentic.common.settings import get_settings
from resilience_app.core.shared import METH_RISK, METH_UHI, METH_URBAN

mcp = FastMCP("NYC Resilience Data Tools")


@mcp.tool
def list_runtime_assets(prefix: str = "nyc-resilience/data/current/") -> list[str]:
    """List up to 200 object keys under `prefix` (default "nyc-resilience/data/current/") from the S3 bucket named by the S3_BUCKET setting. If no bucket is configured, `prefix` is ignored entirely and this instead lists bare filenames from the local `data/static/` directory. Results aren't paginated, so a bucket with more than 200 matching keys is silently truncated, and the two code paths return different shapes: full S3 keys vs. bare local filenames."""
    bucket = get_settings().s3_bucket
    if not bucket:
        return [p.name for p in Path("data/static").glob("*")]
    response = s3_client().list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=200)
    return [item["Key"] for item in response.get("Contents", [])]


_METHODOLOGY_BY_TOPIC = {
    "urban_features": METH_URBAN,
    "coastal_flood": METH_RISK["NRI Coastal"],
    "riverine_flood": METH_RISK["NRI Riverine"],
    "custom_risk": METH_RISK["My Risk Map"],
    "heat": METH_UHI,
}


@mcp.tool
def get_methodology(topic: str) -> dict:
    """Return the reference methodology text for `topic`, one of: "urban_features" (static elevation/slope/building-footprint-density features), "coastal_flood" (FEMA NRI coastal flooding, CFLD_RISKS), "riverine_flood" (FEMA NRI riverine flooding, RFLD_RISKS), "custom_risk" (in-house score_total flood risk composite), or "heat" (FEMA NRI heat-wave risk, HWAV_RISKS). Returns {"topic": topic, "methodology": str}, the exact reference text defined in resilience_app.core.shared — nothing is summarized or reformatted. Raises ValueError (listing the five valid topics) if `topic` doesn't match one of them exactly; there is no fuzzy matching or default."""
    try:
        text = _METHODOLOGY_BY_TOPIC[topic]
    except KeyError:
        valid = ", ".join(sorted(_METHODOLOGY_BY_TOPIC))
        raise ValueError(f"Unknown methodology topic {topic!r}; valid topics are: {valid}")
    return {"topic": topic, "methodology": text}


if __name__ == "__main__":
    mcp.run(transport="http", host="0.0.0.0", port=8003)
