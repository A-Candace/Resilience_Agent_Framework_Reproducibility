from io import BytesIO
from pathlib import Path
from datetime import datetime, timedelta, timezone, date
from typing import Optional

import json
import math
import os
import re

import mlflow
import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st
from dotenv import load_dotenv


# ============================================================
# Runtime data paths
# ============================================================

DATA_RUNTIME_DIR = Path(
    os.getenv("DATA_RUNTIME_DIR", "/app/data/runtime")
)

FLOOD_GRID_PATH = Path(
    os.getenv(
        "FLOOD_GRID_PATH",
        str(DATA_RUNTIME_DIR / "nyc_flood_risk_grid_4km.geojson"),
    )
)

RISK_XLSX = os.getenv(
    "RISK_XLSX",
    str(DATA_RUNTIME_DIR / "Risk_Attributes_Table_v4.xlsx"),
)


# ============================================================
# Optional geometry imports
# ============================================================

try:
    import shapely.geometry as sg
    from shapely.geometry import (
        mapping as shp_mapping,
        shape as shp_shape,
    )
    from shapely.ops import transform as shp_transform

    SHAPELY_OK = True
except Exception:
    SHAPELY_OK = False


# ============================================================
# Environment
# ============================================================

if os.getenv("ENV", "dev").lower() != "prod":
    load_dotenv()


# ---- MLflow (works with 2.17.x, and conditionally with genai if present) ----
import mlflow

def _mlflow_genai_available():
    try:
        import mlflow.genai  # noqa: F401
        return True
    except Exception:
        return False

GENAI_OK = _mlflow_genai_available()

# ---- Environment ----
if os.getenv("ENV") != "prod":
    from dotenv import load_dotenv
    load_dotenv()

# Claude / Bedrock config

#BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-sonnet-4-6")

AWS_REGION = os.getenv("AWS_REGION")
HAS_BEDROCK = bool(AWS_REGION)

# MLflow config
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "",
).strip()

MLFLOW_EXPERIMENT = os.getenv(
    "MLFLOW_EXPERIMENT",
    "nyc-resilience-agent",
)

os.environ.setdefault(
    "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING",
    "false",
)

MLFLOW_ON = bool(
    MLFLOW_TRACKING_URI
)

# Do not select/create an MLflow experiment at import time.
# Training and explicit logging workflows should choose their
# experiment when they actually start a run.
#
# Inference only needs to know which MLflow tracking server to use.
if MLFLOW_ON:
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )


# ---- App-wide UI config ----
#st.set_page_config(page_title="NYC Resilience AI Agent", page_icon="🌆", layout="wide")

# =====================================================================
# Utilities (Claude calls, MLflow GenAI logging, geometry, coloring)
# =====================================================================

def call_claude(prompt: str, system: str = None, temperature: float = 0.2, max_tokens: int = 600) -> str:
    """
    Bedrock Claude call with safe defaults. Falls back to a stub if Bedrock isn't configured.
    """
    if not HAS_BEDROCK:
        return ("(Claude disabled — add AWS credentials &amp; region)\n\n"
                "Draft explanation: Concentrations align with higher imperviousness "
                "and lower elevation. Use curb-inlet cleaning and bioswales where feasible.")

    import boto3
    from botocore.config import Config
    from botocore.exceptions import BotoCoreError, ClientError

    cfg = Config(
        region_name=AWS_REGION,
        read_timeout=20,
        connect_timeout=5,
        retries={"max_attempts": 2, "mode": "standard"},
    )
    client = boto3.client("bedrock-runtime", config=cfg)

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    if system:
        body["system"] = system

    try:
        resp = client.invoke_model(
            modelId=BEDROCK_MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body),
        )
        payload = json.loads(resp["body"].read())
        parts = []
        for blk in payload.get("content", []):
            if blk.get("type") == "text":
                parts.append(blk.get("text", ""))
        text = "".join(parts).strip()
        return text or "(no text returned)"
    except (BotoCoreError, ClientError) as e:
        return f"(Bedrock error: {e})"

def genai_log(prompt_text: str, response_text: str, meta: dict):
    """
    Log a GenAI prompt→response to MLflow.
    """

    if not MLFLOW_ON:
        return
    
    if GENAI_OK:
        import mlflow.genai as mgen
        with mgen.start_trace() as trace:
            with mgen.start_span(name=meta.get("span_name", "map_explanation")) as span:
                span.set_inputs({"prompt": prompt_text, **{f"meta_{k}": v for k, v in meta.items()}})
                span.set_outputs({"response": response_text})
        try:
            mlflow.log_params({f"explain_{k}": str(v) for k, v in meta.items()})
        except Exception:
            pass
    else:
        try:
            with mlflow.start_run(run_name=f"genai_{meta.get('span_name','explain')}"):
                mlflow.log_params({f"explain_{k}": str(v) for k, v in meta.items()})
                import tempfile
                with tempfile.TemporaryDirectory() as td:
                    pth_prompt = os.path.join(td, "prompt.txt")
                    pth_resp = os.path.join(td, "response.txt")
                    with open(pth_prompt, "w", encoding="utf-8") as f:
                        f.write(prompt_text)
                    with open(pth_resp, "w", encoding="utf-8") as f:
                        f.write(response_text)
                    mlflow.log_artifacts(td, artifact_path="genai_logs")
        except Exception:
            pass

# --- Color helpers ---
RED_YELLOW = [
    [255, 255, 178],
    [254, 217, 118],
    [254, 178, 76],
    [253, 141, 60],
    [252, 78, 42],
    [227, 26, 28],
    [177, 0, 38],
]

PALETTE4 = [
    [255, 255, 178],
    [254, 217, 118],
    [253, 141, 60],
    [227, 26, 28],
]

def compute_breaks(values: pd.Series, k: int = 7) -> list:
    v = pd.to_numeric(values, errors="coerce").dropna().to_numpy()
    if len(v) == 0:
        return [0] * (k - 1)
    qs = np.linspace(0, 1, k)
    b = np.quantile(v, qs)
    thresholds = [float(b[i]) for i in range(1, k)]
    return thresholds

def color_for_value_dynamic(val: float, breaks: list) -> list:
    idx = 0
    for b in breaks:
        if val > b:
            idx += 1
        else:
            break
    idx = min(idx, len(RED_YELLOW) - 1)
    return RED_YELLOW[idx]

def normalize_cb_id(val):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return ""
    try:
        return str(int(float(val)))
    except Exception:
        return str(val)

def get_board_display_id(board):
    attrs = board.get("attrs", {}) or {}
    tract_id = attrs.get("TRACTCE") or attrs.get("GEOID") or attrs.get("tract_id")
    if tract_id is not None and str(tract_id) != "":
        return normalize_cb_id(tract_id)
    return normalize_cb_id(board.get("unit_id", ""))

def get_attr_ci(attrs: dict, key: str):
    """Case-insensitive attribute getter"""
    if key in attrs:
        return attrs.get(key)
    lower = key.lower()
    for k, v in attrs.items():
        if (k or "").lower() == lower:
            return v
    return None

def coerce_numeric(val, default=0.0):
    """
    Convert value to numeric, handling:
    - None, NaN, empty strings → default
    - Non-numeric strings → default
    - Valid numbers → float
    """
    if val is None:
        return default
    if isinstance(val, (int, float)):
        if np.isnan(val):
            return default
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        if val == '' or val.lower() in ('nan', 'null', 'none', 'n/a', '#n/a'):
            return default
        # Remove any non-numeric characters except decimal point and minus
        cleaned = re.sub(r'[^\d.-]', '', val)
        if cleaned == '' or cleaned == '-' or cleaned == '.':
            return default
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return default
    return default

@st.cache_data
def load_flood_grid(path=None) -> pd.DataFrame:
    """
    Load the 4 km flood-risk GeoJSON.

    The default file is downloaded from S3 into DATA_RUNTIME_DIR
    before Streamlit starts.
    """
    resolved_path = Path(path) if path else FLOOD_GRID_PATH

    if not resolved_path.exists():
        raise FileNotFoundError(
            "Flood-risk grid was not found at "
            f"{resolved_path}. Confirm that scripts/s3_assets.py "
            "downloaded the S3 runtime assets successfully."
        )

    with resolved_path.open("r", encoding="utf-8") as file:
        geojson = json.load(file)

    rows = []

    for feature in geojson.get("features", []):
        properties = feature.get("properties", {}) or {}
        geometry = feature.get("geometry")

        if not geometry:
            continue

        if SHAPELY_OK:
            polygon = shp_shape(geometry)
            centroid = polygon.centroid
            longitude = float(centroid.x)
            latitude = float(centroid.y)
        else:
            coordinates = geometry.get("coordinates", [])

            if not coordinates:
                continue

            polygon_coordinates = coordinates[0]

            longitude = float(
                np.mean([point[0] for point in polygon_coordinates])
            )
            latitude = float(
                np.mean([point[1] for point in polygon_coordinates])
            )

        minimum_mm = coerce_numeric(
            properties.get("min_precip_mm"),
            default=np.nan,
        )
        mean_mm = coerce_numeric(
            properties.get("mean_precip_mm"),
            default=np.nan,
        )
        maximum_mm = coerce_numeric(
            properties.get("max_precip_mm"),
            default=np.nan,
        )

        rows.append(
            {
                "grid_id": str(properties.get("grid_id")),
                "lat": latitude,
                "lon": longitude,
                "n_events": coerce_numeric(
                    properties.get("n_events"),
                    default=0,
                ),
                "min_in": minimum_mm / 25.4,
                "mean_in": mean_mm / 25.4,
                "max_in": maximum_mm / 25.4,
                "min_percentile": coerce_numeric(
                    properties.get("min_rain_percentile"),
                    default=np.nan,
                ),
                "mean_percentile": coerce_numeric(
                    properties.get("mean_rain_percentile"),
                    default=np.nan,
                ),
                "max_percentile": coerce_numeric(
                    properties.get("max_rain_percentile"),
                    default=np.nan,
                ),
            }
        )

    return pd.DataFrame(rows)


def compute_forecast_heatmap(df, forecast_inches):
    df = df.copy()
    df["high_risk"] = forecast_inches >= df["min_in"]

    high = df[df["high_risk"]].copy()

    if high.empty:
        return high

    high["weight"] = high["n_events"].clip(lower=1)
    return high


def normalize_0_1(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0).clip(lower=0.0)
    minv = s.min()
    maxv = s.max()
    if maxv - minv <= 1e-12:
        return pd.Series(0.0, index=s.index)
    return (s - minv) / (maxv - minv)

def legend_html(title: str, labels: list, colors_rgb: list) -> str:
    rows = []
    for lab, col in zip(labels, colors_rgb):
        r, g, b = col
        rows.append(f'<div style="display:flex;align-items:center;margin:4px 0;">'
                    f'<span style="display:inline-block;width:16px;height:16px;background:rgb({r},{g},{b});margin-right:8px;border:1px solid #333;"></span>'
                    f'<span style="font-size:12px;">{lab}</span></div>')
    return (f'<div style="border:1px solid #ccc;border-radius:8px;padding:8px 10px;'
            f'background:#fff;max-width:220px;">'
            f'<div style="font-weight:600;margin-bottom:6px;">{title}</div>'
            f'{"".join(rows)}</div>')

# =====================================================================
# Excel-backed attributes (NRI + statics)
# =====================================================================



def _digits_only(x: object) -> str:
    s = "" if x is None else str(x)
    return re.sub(r"\D+", "", s)

def _pad_left(s: str, n: int) -> str:
    s = s or ""
    return s.zfill(n)

def _normalize_geoids_from_parts(state, county, tract) -> str:
    return _pad_left(_digits_only(state), 2) + _pad_left(_digits_only(county), 3) + _pad_left(_digits_only(tract), 6)

def _normalize_geoid_any(s: str) -> str:
    d = _digits_only(s)
    if len(d) >= 11:
        return d[-11:]
    return _pad_left(d, 11)

@st.cache_data
def load_risk_xlsx(path_or_bytes) -> pd.DataFrame:
    df = pd.read_excel(path_or_bytes, engine="openpyxl").copy()

    if all(c in df.columns for c in ["STATEFIPS", "COUNTYFIPS", "TRACT"]):
        df["__join_key__"] = df.apply(lambda r: _normalize_geoids_from_parts(r["STATEFIPS"], r["COUNTYFIPS"], r["TRACT"]), axis=1)
    else:
        candidates = [c for c in ["NRI_ID","GEOID","GEOID10","GEOID20","TRACTFIPS","TRACTCE","TRACT"] if c in df.columns]
        if candidates:
            col = candidates[0]
            df["__join_key__"] = df[col].map(_normalize_geoid_any)
        else:
            guess = None
            for c in df.columns:
                if re.search(r"(geoid|tract|fips)", str(c), flags=re.I):
                    guess = c; break
            if guess:
                df["__join_key__"] = df[guess].map(_normalize_geoid_any)
            else:
                df["__join_key__"] = ""

    return df

def merge_attrs_into_tracts(_tracts: list, df_excel: pd.DataFrame) -> tuple:
    """
    Merge Excel attributes onto tracts by NRI_ID.
    ONLY keeps Excel data - discards all shapefile attributes except geometry.
    Returns (tracts, coverage_ratio)
    """
    # Check if NRI_ID column exists in Excel
    if "NRI_ID" not in df_excel.columns:
        st.error("❌ Excel file must have an 'NRI_ID' column.")
        return _tracts, 0.0
    
    # Build lookup dictionary: NRI_ID -> all attributes
    df = df_excel.copy()
    df["NRI_ID"] = df["NRI_ID"].astype(str).str.strip()
    
    # Create dictionary keyed by NRI_ID
    excel_lookup = {}
    for _, row in df.iterrows():
        nri_id = str(row["NRI_ID"]).strip()
        if nri_id and nri_id != "" and nri_id.lower() != "nan":
            excel_lookup[nri_id] = row.to_dict()
    
    st.write(f"📊 Excel: {len(excel_lookup)} NRI_IDs loaded")
    
    # Match tracts to Excel using CensusTrac field
    merged = 0
    total = len(_tracts)
    sample_matches = []
    
    for idx, t in enumerate(_tracts):
        attrs = t.get("attrs", {}) or {}
        
        # Get NRI_ID from CensusTrac field
        tract_nri_id = None
        
        # Primary: Use CensusTrac field directly
        if "CensusTrac" in attrs:
            tract_nri_id = str(attrs["CensusTrac"]).strip()
        
        # Fallback: Build from FIPS components if CensusTrac missing
        if not tract_nri_id or tract_nri_id.lower() == "nan":
            state = str(attrs.get("CensusTr_3", "")).strip()
            county = str(attrs.get("CensusTr_6", "")).strip()
            tract = str(attrs.get("CensusTr_8", "")).strip()
            
            if state and county and tract:
                state = state.zfill(2)
                county = county.zfill(3)
                tract = tract.zfill(6)
                tract_nri_id = f"T{state}{county}{tract}"
        
        if not tract_nri_id:
            if idx < 3:
                sample_matches.append({
                    "tract_idx": idx,
                    "unit_id": t.get("unit_id"),
                    "status": "FAILED - no NRI_ID found"
                })
            continue
        
        # Look up in Excel
        if tract_nri_id in excel_lookup:
            excel_row = excel_lookup[tract_nri_id]
            
            # REPLACE attrs entirely with Excel data (keep only NRI_ID for reference)
            new_attrs = {
                "NRI_ID": tract_nri_id,
                # Keep only the columns we need from Excel
                "Mean_elevation": excel_row.get("Mean_elevation"),
                "Mean_slope": excel_row.get("Mean_slope"),
                "Total_ftp_area": excel_row.get("Total_ftp_area"),
                "AREA": excel_row.get("AREA"),
                "CFLD_RISKS": excel_row.get("CFLD_RISKS"),
                "RFLD_RISKS": excel_row.get("RFLD_RISKS"),
                "HWAV_RISKS": excel_row.get("HWAV_RISKS"),
                "score_precip": excel_row.get("score_precip"),
                "score_sf": excel_row.get("score_sf"),
                "score_cb": excel_row.get("score_cb"),
                "score_total": excel_row.get("score_total"),
            }
            
            ADDITIONAL_COLUMNS = [
            "HOUSEHOLDS BY TYPE - Total households",
            "HOUSEHOLDS BY TYPE - Total households - Family households (families)",
            "HOUSEHOLDS BY TYPE - Total households - Family households (families) - With own children of the householder under 18 years",
            "HOUSEHOLDS BY TYPE - Total households - Average household size",
            "HOUSEHOLDS BY TYPE - Total households - Average family size",
            "FERTILITY - Number of women 15 to 50 years old who had a birth in the past 12 months",
            "SCHOOL ENROLLMENT - Population 3 years and over enrolled in school",
            "SCHOOL ENROLLMENT - Population 3 years and over enrolled in school - Nursery school, preschool",
            "SCHOOL ENROLLMENT - Population 3 years and over enrolled in school - Kindergarten",
            "SCHOOL ENROLLMENT - Population 3 years and over enrolled in school - Elementary school (grades 1-8)",
            "SCHOOL ENROLLMENT - Population 3 years and over enrolled in school - High school (grades 9-12)",
            "SCHOOL ENROLLMENT - Population 3 years and over enrolled in school - College or graduate school",
            "EDUCATIONAL ATTAINMENT - Population 25 years and over",
            "EDUCATIONAL ATTAINMENT - Population 25 years and over - Less than 9th grade",
            "EDUCATIONAL ATTAINMENT - Population 25 years and over - 9th to 12th grade, no diploma",
            "EDUCATIONAL ATTAINMENT - Population 25 years and over - High school graduate (includes equivalency)",
            "EDUCATIONAL ATTAINMENT - Population 25 years and over - Some college, no degree",
            "EDUCATIONAL ATTAINMENT - Population 25 years and over - Associate's degree",
            "EDUCATIONAL ATTAINMENT - Population 25 years and over - Bachelor's degree",
            "EDUCATIONAL ATTAINMENT - Population 25 years and over - Graduate or professional degree",
            "EDUCATIONAL ATTAINMENT - Population 25 years and over - High school graduate or higher",
            "EDUCATIONAL ATTAINMENT - Population 25 years and over - Bachelor's degree or higher",
            "RESIDENCE 1 YEAR AGO - Population 1 year and over",
            "RESIDENCE 1 YEAR AGO - Population 1 year and over - Same house",
            "RESIDENCE 1 YEAR AGO - Population 1 year and over - Different house in the U.S.",
            "RESIDENCE 1 YEAR AGO - Population 1 year and over - Abroad",
            "U.S. CITIZENSHIP STATUS - Foreign-born population",
            "U.S. CITIZENSHIP STATUS - Foreign-born population - Naturalized U.S. citizen",
            "U.S. CITIZENSHIP STATUS - Foreign-born population - Not a U.S. citizen",
            "LANGUAGE SPOKEN AT HOME - Population 5 years and over",
            "LANGUAGE SPOKEN AT HOME - Population 5 years and over - English only",
            "LANGUAGE SPOKEN AT HOME - Population 5 years and over - Language other than English",
            "EMPLOYMENT STATUS - Population 16 years and over",
            "EMPLOYMENT STATUS - Population 16 years and over - In labor force",
            "EMPLOYMENT STATUS - Population 16 years and over - In labor force - Civilian labor force",
            "EMPLOYMENT STATUS - Population 16 years and over - In labor force - Civilian labor force - Employed",
            "EMPLOYMENT STATUS - Population 16 years and over - In labor force - Civilian labor force - Unemployed",
            "EMPLOYMENT STATUS - Population 16 years and over - In labor force - Armed Forces",
            "EMPLOYMENT STATUS - Population 16 years and over - Not in labor force",
            "EMPLOYMENT STATUS - Civilian labor force",
            "EMPLOYMENT STATUS - Civilian labor force - Unemployment Rate",
            "COMMUTING TO WORK - Workers 16 years and over",
            "COMMUTING TO WORK - Workers 16 years and over - Car, truck, or van -- drove alone",
            "COMMUTING TO WORK - Workers 16 years and over - Car, truck, or van -- carpooled",
            "COMMUTING TO WORK - Workers 16 years and over - Public transportation (excluding taxicab)",
            "COMMUTING TO WORK - Workers 16 years and over - Walked",
            "COMMUTING TO WORK - Workers 16 years and over - Other means",
            "COMMUTING TO WORK - Workers 16 years and over - Worked at home",
            "COMMUTING TO WORK - Workers 16 years and over - Mean travel time to work (minutes)",
            "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households",
            "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - Less than $10,000",
            "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - $10,000 to $14,999",
            "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - $15,000 to $24,999",
            "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - $25,000 to $34,999",
            "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - $35,000 to $49,999",
            "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - $50,000 to $74,999",
            "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - $75,000 to $99,999",
            "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - $100,000 to $149,999",
            "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - $150,000 to $199,999",
            "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - $200,000 or more",
            "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)",
            "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - Mean household income (dollars)",
            "HOUSING OCCUPANCY - Total housing units",
            "HOUSING OCCUPANCY - Total housing units - Occupied housing units",
            "HOUSING OCCUPANCY - Total housing units - Vacant housing units",
            "HOUSING OCCUPANCY - Total housing units - Homeowner vacancy rate",
            "HOUSING OCCUPANCY - Total housing units - Rental vacancy rate",
            "HOUSING TENURE - Occupied housing units",
            "HOUSING TENURE - Occupied housing units - Owner-occupied",
            "HOUSING TENURE - Occupied housing units - Renter-occupied",
            "VALUE - Owner-occupied units - Median (dollars)",
            "MORTGAGE STATUS - Owner-occupied units",
            "MORTGAGE STATUS - Owner-occupied units - Housing units with a mortgage",
            "MORTGAGE STATUS - Owner-occupied units - Housing units without a mortgage",
            "GROSS RENT - Occupied units paying rent",
            "GROSS RENT - Occupied units paying rent - Less than $500",
            "GROSS RENT - Occupied units paying rent - $500 to $999",
            "GROSS RENT - Occupied units paying rent - $1,000 to $1,499",
            "GROSS RENT - Occupied units paying rent - $1,500 to $1,999",
            "GROSS RENT - Occupied units paying rent - $2,000 to $2,499",
            "GROSS RENT - Occupied units paying rent - $2,500 to $2,999",
            "GROSS RENT - Occupied units paying rent - $3,000 or more",
            "GROSS RENT - Occupied units paying rent - Median (dollars)",
            "GROSS RENT - Occupied units paying rent - No rent paid",
            "SEX AND AGE - Total population",
            "SEX AND AGE - Total population - Male",
            "SEX AND AGE - Total population - Female",
            "SEX AND AGE - Total population - Under 5 years",
            "SEX AND AGE - Total population - 5 to 9 years",
            "SEX AND AGE - Total population - 10 to 14 years",
            "SEX AND AGE - Total population - 15 to 19 years",
            "SEX AND AGE - Total population - 20 to 24 years",
            "SEX AND AGE - Total population - 25 to 34 years",
            "SEX AND AGE - Total population - 35 to 44 years",
            "SEX AND AGE - Total population - 45 to 54 years",
            "SEX AND AGE - Total population - 55 to 59 years",
            "SEX AND AGE - Total population - 60 to 64 years",
            "SEX AND AGE - Total population - 65 to 74 years",
            "SEX AND AGE - Total population - 75 to 84 years",
            "SEX AND AGE - Total population - 85 years and over",
            "SEX AND AGE - Total population - Median age (years)",
            "RACE - Total population",
            "RACE - Total population - One race",
            "RACE - Total population - One race - White",
            "RACE - Total population - One race - Black or African American",
            "RACE - Total population - One race - American Indian and Alaska Native",
            "RACE - Total population - One race - Asian",
            "RACE - Total population - One race - Native Hawaiian and Other Pacific Islander",
            "RACE - Total population - One race - Some other race",
            "RACE - Total population - Two or more races",
            "HISPANIC OR LATINO AND RACE - Total population",
            "HISPANIC OR LATINO AND RACE - Total population - Hispanic or Latino (of any race)",
            "HISPANIC OR LATINO AND RACE - Total population - Not Hispanic or Latino",
            "HISPANIC OR LATINO AND RACE - Total population - Not Hispanic or Latino - White alone",
            "HISPANIC OR LATINO AND RACE - Total population - Not Hispanic or Latino - Black or African American alone",
            "HISPANIC OR LATINO AND RACE - Total population - Not Hispanic or Latino - American Indian and Alaska Native alone",
            "HISPANIC OR LATINO AND RACE - Total population - Not Hispanic or Latino - Asian alone",
            "HISPANIC OR LATINO AND RACE - Total population - Not Hispanic or Latino - Native Hawaiian and Other Pacific Islander alone",
            "HISPANIC OR LATINO AND RACE - Total population - Not Hispanic or Latino - Some other race alone",
            "HISPANIC OR LATINO AND RACE - Total population - Not Hispanic or Latino - Two or more races",
            "CITIZEN, VOTING AGE POPULATION - Citizen, 18 and over population",
        ]
            for col in ADDITIONAL_COLUMNS:
                new_attrs[col] = excel_row.get(col)
            # Replace attrs with ONLY Excel data
            t["attrs"] = new_attrs
            
            merged += 1
            
            if idx < 3:
                sample_matches.append({
                    "tract_idx": idx,
                    "unit_id": t.get("unit_id"),
                    "nri_id": tract_nri_id,
                    "has_elevation": new_attrs.get("Mean_elevation") is not None,
                    "has_slope": new_attrs.get("Mean_slope") is not None,
                    "has_footprint": new_attrs.get("Total_ftp_area") is not None,
                    "status": "✅ MATCHED"
                })
        else:
            if idx < 3:
                sample_matches.append({
                    "tract_idx": idx,
                    "unit_id": t.get("unit_id"),
                    "nri_id": tract_nri_id,
                    "status": "❌ Not in Excel"
                })
            # If not matched, clear attrs (keep only geometry)
            t["attrs"] = {"NRI_ID": tract_nri_id}
    
    coverage = merged / max(1, total)
    
    st.write(f"✅ Matched {merged}/{total} tracts ({coverage*100:.1f}%)")
    
    if sample_matches:
        st.write("**Sample merge attempts:**")
        import pandas as pd
        st.dataframe(pd.DataFrame(sample_matches))
    
    if coverage < 0.5:
        st.warning("⚠️ Low coverage - check if Excel NRI_IDs match shapefile CensusTrac values")
        sample_excel = list(excel_lookup.keys())[:5]
        st.write("**Sample Excel NRI_IDs:**", sample_excel)
    
    return _tracts, coverage

def get_attr_num(attrs: dict, col: str) -> float:
    """
    Safely fetch numeric from attrs[col] (case-insensitive).
    Handles all non-numeric values by converting to 0.0
    """
    raw = get_attr_ci(attrs, col)
    return coerce_numeric(raw, default=0.0)

def quantile_bins_4_with_zero(series: pd.Series) -> tuple:
    """
    Special binning for rescaled component scores:
    - Bin 0: values <= 0 (zero or negative)
    - Bins 1-3: equal intervals for positive values only
    
    Returns (bins, edges) where edges has 5 values [min, edge1, edge2, edge3, max]
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    
    if len(s) == 0:
        edges = np.array([0, 0.25, 0.5, 0.75, 1.0], dtype=float)
        bins = pd.Series(0, index=s.index, dtype=int)
        return bins, edges
    
    vmin, vmax = float(s.min()), float(s.max())
    
    # Separate zero and positive values
    zero_mask = s <= 0
    positive_mask = s > 0
    
    n_zeros = zero_mask.sum()
    n_positives = positive_mask.sum()
    
    # Initialize bins series
    bins = pd.Series(0, index=s.index, dtype=int)
    
    if n_positives == 0:
        # All zeros
        edges = np.array([vmin, 0, 0, 0, 0], dtype=float)
        return bins, edges
    
    # Get positive values only
    positive_vals = s[positive_mask]
    pos_min = float(positive_vals.min())
    pos_max = float(positive_vals.max())
    
    if np.isclose(pos_min, pos_max):
        # All positive values are identical
        edges = np.array([vmin if vmin < 0 else 0, pos_min, pos_min, pos_min, pos_max], dtype=float)
        bins[positive_mask] = 1  # Put all positive in bin 1
        return bins, edges
    
    # Create 3 equal intervals for positive values
    interval_size = (pos_max - pos_min) / 3.0
    edge1 = pos_min + interval_size
    edge2 = pos_min + 2 * interval_size
    edge3 = pos_max
    
    edges = np.array([vmin if vmin < 0 else 0, edge1, edge2, edge3, pos_max], dtype=float)
    
    # Assign bins:
    # Bin 0: <= 0
    # Bin 1: (0, edge1]
    # Bin 2: (edge1, edge2]
    # Bin 3: (edge2, pos_max]
    
    bins[zero_mask] = 0
    
    # For positive values, use equal intervals
    pos_bins = pd.Series(1, index=positive_vals.index)  # Default to bin 1
    pos_bins[positive_vals > edge1] = 2
    pos_bins[positive_vals > edge2] = 3
    
    bins[positive_mask] = pos_bins
    bins = bins.astype(int).clip(0, 3)
    
    return bins, edges

def quantile_bins_4(series: pd.Series) -> tuple:
    """
    Returns (bin[0..3], edges[5]) where edges are raw-value edges used for the legend.
    Handles duplicate edges gracefully.
    """
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)

    if len(s) == 0:
        edges = np.array([0, 0.25, 0.5, 0.75, 1.0], dtype=float)
        bins = pd.Series(0, index=s.index, dtype=int)
        return bins, edges

    vmin, vmax = float(s.min()), float(s.max())
    if not np.isfinite(vmin) or not np.isfinite(vmax):
        vmin, vmax = 0.0, 1.0

    # If all values are identical
    if np.isclose(vmin, vmax):
        edges = np.linspace(vmin, vmax + 1e-9, 5)
        bins = pd.Series(0, index=s.index, dtype=int)
        return bins, edges

    # Try quantile-based edges
    edges = np.quantile(s, [0, 0.25, 0.5, 0.75, 1.0]).astype(float)
    
    # Check for duplicate edges
    unique_edges = np.unique(edges)
    
    if len(unique_edges) < 5:
        # Fall back to equal-interval binning
        edges = np.linspace(vmin, vmax, 5)
        # Ensure strictly increasing
        for i in range(1, len(edges)):
            if edges[i] <= edges[i-1]:
                edges[i] = edges[i-1] + (vmax - vmin) * 0.01
    else:
        # Ensure strictly non-decreasing
        edges = np.maximum.accumulate(edges)
        # Add small epsilon to ensure strict inequality where needed
        for i in range(1, len(edges)):
            if np.isclose(edges[i], edges[i-1]):
                edges[i] = edges[i-1] + (vmax - vmin) * 0.001

    # Final safety check
    if np.isclose(edges[0], edges[-1]):
        edges[-1] = edges[0] + 1e-9

    labels = list(range(4))  # 0..3
    
    try:
        bins = pd.cut(s, bins=edges, labels=labels, include_lowest=True, right=True, duplicates='drop')
    except ValueError:
        # If still failing, use equal intervals
        edges = np.linspace(vmin, vmax + 1e-9, 5)
        bins = pd.cut(s, bins=edges, labels=labels, include_lowest=True, right=True, duplicates='drop')
    
    # Convert to int, handling any remaining NaN
    bins = bins.astype("Int64").fillna(0).astype(int).clip(0, 3)
    
    return bins, edges

def legend_with_ranges_html(title: str, edges: np.ndarray, colors=([255,255,178],[254,217,118],[253,141,60],[227,26,28])) -> str:
    labels = [
        f"{edges[0]:,.2f} – {edges[1]:,.2f}",
        f"{edges[1]:,.2f} – {edges[2]:,.2f}",
        f"{edges[2]:,.2f} – {edges[3]:,.2f}",
        f"{edges[3]:,.2f} – {edges[4]:,.2f}",
    ]
    rows=[]
    for lab,col in zip(labels, colors):
        r,g,b = col
        rows.append(
            f'<div style="display:flex;align-items:center;margin:4px 0;">'
            f'<span style="display:inline-block;width:16px;height:16px;background:rgb({r},{g},{b});margin-right:8px;border:1px solid #333;"></span>'
            f'<span style="font-size:12px;">{lab}</span></div>'
        )
    return (f'<div style="border:1px solid #ccc;border-radius:8px;padding:8px 10px;'
            f'background:#fff;max-width:280px;">'
            f'<div style="font-weight:600;margin-bottom:6px;">{title} (quartiles)</div>'
            f'{"".join(rows)}</div>')

def legend_with_zero_bin_html(title: str, edges: np.ndarray, colors=([255,255,178],[254,217,118],[253,141,60],[227,26,28])) -> str:
    """Legend for component scores with special zero bin"""
    labels = [
        f"Zero (≤ 0)",
        f"{edges[1]:.3f} – {edges[2]:.3f}",
        f"{edges[2]:.3f} – {edges[3]:.3f}",
        f"{edges[3]:.3f} – {edges[4]:.3f}",
    ]
    rows=[]
    for lab,col in zip(labels, colors):
        r,g,b = col
        rows.append(
            f'<div style="display:flex;align-items:center;margin:4px 0;">'
            f'<span style="display:inline-block;width:16px;height:16px;background:rgb({r},{g},{b});margin-right:8px;border:1px solid #333;"></span>'
            f'<span style="font-size:12px;">{lab}</span></div>'
        )
    return (f'<div style="border:1px solid #ccc;border-radius:8px;padding:8px 10px;'
            f'background:#fff;max-width:280px;">'
            f'<div style="font-weight:600;margin-bottom:6px;">{title}</div>'
            f'{"".join(rows)}</div>')

# =====================================================================
# Geometry load &amp; transforms
# =====================================================================

def _zipfile_from_any(obj):
    from zipfile import ZipFile
    if isinstance(obj, (str, os.PathLike)):
        return ZipFile(obj)
    if hasattr(obj, "read"):
        try:
            obj.seek(0)
        except Exception:
            pass
        data = obj.read()
        return ZipFile(BytesIO(data))
    if isinstance(obj, (bytes, bytearray)):
        return ZipFile(BytesIO(obj))
    raise TypeError("Unsupported ZIP input type for shapefile loader.")

def load_community_boards_from_json(js: dict):
    if not SHAPELY_OK:
        st.error("Community Board aggregation requires 'shapely'.")
        return None
    feats = js.get("features", [])
    polys = []
    for idx, f in enumerate(feats):
        props = f.get("properties", {}) or {}
        unit_id = props.get("BoroCD") or props.get("boro_cd") or props.get("cd") or f"CB_{idx:03d}"
        name = props.get("cd_name") or props.get("name") or unit_id
        geom_geojson = f.get("geometry")
        if not geom_geojson:
            continue
        try:
            geom = shp_shape(geom_geojson)
        except Exception:
            continue
        if geom.is_empty:
            continue
        polys.append({
            "unit_id": str(unit_id),
            "name": str(name),
            "geom": geom,
            "feature_geom": json.loads(json.dumps(geom_geojson)),
            "attrs": dict(props),
        })
    if not polys:
        st.error("No valid Polygon/MultiPolygon features found in the GeoJSON.")
    return polys or None

def load_census_tracts_from_geojson(js: dict):
    """Load census tracts from GeoJSON"""
    if not SHAPELY_OK:
        st.error("Census tract loading requires 'shapely'.")
        return None
    
    feats = js.get("features", [])
    tracts = []
    
    for idx, f in enumerate(feats):
        props = f.get("properties", {}) or {}
        
        unit_id = (props.get("TRACTCE") or 
                   props.get("GEOID") or 
                   props.get("TRACT") or 
                   props.get("tract_id") or 
                   f"TRACT_{idx:05d}")
        
        name = props.get("NAME") or props.get("name") or str(unit_id)
        
        geom_geojson = f.get("geometry")
        if not geom_geojson:
            continue
        
        try:
            geom = shp_shape(geom_geojson)
        except Exception:
            continue
        
        if geom.is_empty:
            continue
        
        clean_attrs = dict(props)
        
        tracts.append({
            "unit_id": str(unit_id),
            "name": str(name),
            "geom": geom,
            "feature_geom": json.loads(json.dumps(geom_geojson)),
            "attrs": clean_attrs,
        })
    
    if not tracts:
        st.error("No valid census tract features found in the GeoJSON.")
    else:
        st.success(f"Loaded {len(tracts)} census tracts.")
    
    return tracts or None

def load_boards_from_shapefile_path(zip_path: str):
    if not os.path.exists(zip_path):
        return None
    return _load_boards_from_shapefile_zipfilelike(zip_path)

def load_boards_from_shapefile_zip_streamlit():
    zip_file = st.file_uploader("Upload Community Board/District **Shapefile (.zip)**", type=["zip"], key="shpzip_upl")
    if zip_file is None:
        return None
    return _load_boards_from_shapefile_zipfilelike(zip_file)

def _load_boards_from_shapefile_zipfilelike(zip_filelike):
    if not SHAPELY_OK:
        st.error("Shapefile support requires 'shapely>=2.0'.")
        return None
    try:
        import shapefile
    except ImportError:
        st.error("Missing dependency: 'pyshp' (pip install pyshp).")
        return None
    try:
        from pyproj import CRS, Transformer
    except Exception:
        st.error("Shapefile reprojection requires 'pyproj>=3.6'.")
        return None

    def to_pure_geojson(geom):
        m = shp_mapping(geom)
        return json.loads(json.dumps(m))

    import tempfile
    tmpdir = tempfile.mkdtemp(prefix="cb_shp_")
    try:
        with _zipfile_from_any(zip_filelike) as zf:
            zf.extractall(tmpdir)
    except Exception as e:
        st.error(f"Could not extract shapefile ZIP: {e}")
        return None

    shp_paths = []
    for root, _, files in os.walk(tmpdir):
        for f in files:
            if f.lower().endswith(".shp"):
                shp_paths.append(os.path.join(root, f))
    if not shp_paths:
        st.error("ZIP does not contain a .shp file.")
        return None
    shp_path = shp_paths[0]
    prj_path = os.path.splitext(shp_path)[0] + ".prj"

    src_crs = None
    if os.path.exists(prj_path):
        try:
            with open(prj_path, "r") as f:
                wkt = f.read()
            src_crs = CRS.from_wkt(wkt)
        except Exception:
            src_crs = None

    transformer = None
    if src_crs and (src_crs.to_epsg() != 4326):
        try:
            transformer = Transformer.from_crs(src_crs, CRS.from_epsg(4326), always_xy=True)
        except Exception:
            transformer = None

    try:
        r = shapefile.Reader(shp_path)
    except Exception as e:
        st.error(f"Could not read shapefile: {e}")
        return None

    fields = [f[0] for f in r.fields if f[0] != "DeletionFlag"]

    def looks_like_lonlat(x, y):
        return (-180.0 <= x <= 180.0) and (-90.0 <= y <= 90.0)

    polys = []
    for idx, sr in enumerate(r.shapeRecords()):
        rec = {fields[i]: sr.record[i] for i in range(len(fields))}
        
        unit_id = rec.get("TRACTCE") or rec.get("GEOID") or rec.get("BoroCD") or rec.get("boro_cd") or rec.get("cd") or f"TRACT_{idx:05d}"
        name = rec.get("NAME") or rec.get("cd_name") or rec.get("name") or unit_id
        
        geom_geojson = sr.shape.__geo_interface__
        try:
            geom = shp_shape(geom_geojson)
        except Exception as e:
            st.warning(f"Could not parse geometry for {unit_id}: {e}")
            continue
        if geom.is_empty:
            continue

        if transformer is None:
            try:
                if geom.geom_type == 'Polygon':
                    x0, y0 = list(geom.exterior.coords)[0]
                elif geom.geom_type == 'MultiPolygon':
                    x0, y0 = list(list(geom.geoms)[0].exterior.coords)[0]
                else:
                    x0, y0 = (0, 0)
                if not looks_like_lonlat(x0, y0):
                    from pyproj import Transformer
                    transformer = Transformer.from_crs(2263, 4326, always_xy=True)
            except Exception:
                pass
        if transformer is not None:
            try:
                geom = shp_transform(lambda x, y, z=None: transformer.transform(x, y), geom)
            except Exception as e:
                st.warning(f"Could not transform geometry for {unit_id}: {e}")
                continue

        if geom.geom_type not in ['Polygon', 'MultiPolygon']:
            st.warning(f"Skipping {unit_id}: geometry is {geom.geom_type}, not Polygon/MultiPolygon")
            continue

        feature_geom = to_pure_geojson(geom)
        clean_attrs = dict(rec)

        polys.append({
            "unit_id": str(unit_id),
            "name": str(name),
            "geom": geom,
            "feature_geom": feature_geom,
            "attrs": clean_attrs,
        })

    if not polys:
        st.error("No valid polygon features found in the shapefile.")
    else:
        st.success(f"✅ Loaded {len(polys)} census tracts from shapefile.")
    return polys or None

# =====================================================================
# Data merges and helpers
# =====================================================================

def load_and_merge_board_data(
    boards: list,
    csv_path: str | None = None,
) -> list:
    """
    Legacy function kept for compatibility.

    DataForBoxPlots.csv is not required by the current census-tract
    application. If used later, its path can be supplied through
    DATA_FOR_BOX_PLOTS_PATH.
    """
    csv_path = csv_path or os.getenv(
        "DATA_FOR_BOX_PLOTS_PATH",
        str(DATA_RUNTIME_DIR / "DataForBoxPlots.csv"),
    )

    if not os.path.exists(csv_path):
        return boards
    try:
        csv_df = pd.read_csv(csv_path)
        csv_cols = ['CB_id', 'Buildings', 'Elevation', 'Slope', 'Commuting',
                    'Imperv', 'Footprint', 'BLDperArea', 'FTPperArea']
        missing = [c for c in csv_cols if c not in csv_df.columns]
        if missing:
            st.warning(f"Static CSV missing columns: {missing}")
            return boards
        csv_df = csv_df[csv_cols].copy()
        csv_df['CB_id'] = csv_df['CB_id'].apply(normalize_cb_id)
        lookup = csv_df.set_index('CB_id').to_dict('index')

        out = []
        for b in boards:
            boro_cd_raw = b.get('attrs', {}).get('BoroCD') or b.get('attrs', {}).get('boro_cd') or b.get('attrs', {}).get('cd')
            boro_cd = normalize_cb_id(boro_cd_raw)
            filtered_attrs = {
                'boro_cd': boro_cd,
                'shape_area': b.get('attrs', {}).get('shape_area'),
                'shape_leng': b.get('attrs', {}).get('shape_leng'),
                'Area': b.get('attrs', {}).get('Area'),
            }
            if boro_cd in lookup:
                filtered_attrs.update(lookup[boro_cd])
            b['attrs'] = filtered_attrs
            out.append(b)
        return out
    except Exception as e:
        st.warning(f"Error merging CSV: {e}")
        return boards

def boards_bbox(fc):
    minx, miny, maxx, maxy = 999, 999, -999, -999

    def update_bbox(coords, is_multi):
        nonlocal minx, miny, maxx, maxy
        if is_multi:
            for poly in coords:
                for ring in poly:
                    for x, y in ring:
                        minx = min(minx, x); maxx = max(maxx, x)
                        miny = min(miny, y); maxy = max(maxy, y)
        else:
            for ring in coords:
                for x, y in ring:
                    minx = min(minx, x); maxx = max(maxx, x)
                    miny = min(miny, y); maxy = max(maxy, y)

    for f in fc.get("features", []):
        geom = f.get("geometry", {})
        if not geom:
            continue
        t = geom.get("type")
        c = geom.get("coordinates", [])
        if t == "Polygon":
            update_bbox(c, False)
        elif t == "MultiPolygon":
            update_bbox(c, True)

    if minx > maxx or miny > maxy:
        return 40.7128, -74.0060, 10.0
    lon = (minx + maxx) / 2.0
    lat = (miny + maxy) / 2.0
    span = max(maxx - minx, maxy - miny)
    zoom = 10.5
    if span > 2.0: zoom = 9.0
    elif span > 1.0: zoom = 9.8
    elif span > 0.5: zoom = 10.2
    return lat, lon, zoom

def cb_label_layer_from_boards(boards, text_size=12):
    rows = []
    for b in boards:
        cb_num = get_board_display_id(b)
        if not cb_num or not SHAPELY_OK:
            continue
        p = b["geom"].representative_point()
        rows.append({"lon": float(p.x), "lat": float(p.y), "cb": cb_num})
    if not rows:
        return None
    return pdk.Layer(
        "TextLayer",
        data=rows,
        get_position='[lon, lat]',
        get_text='cb',
        get_size=text_size,
        get_angle=0,
        get_color=[20, 20, 20],
        get_alignment_baseline='"center"'
    )

# =====================================================================
# PNG exporter (matplotlib) for choropleths
# =====================================================================

def save_choropleth_png(boards: list, values_by_board: pd.Series, title: str, palette=None, breaks=None) -> bytes:
    """Render a simple static choropleth using matplotlib and return PNG bytes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    s = pd.to_numeric(values_by_board, errors="coerce").fillna(0.0)
    if breaks is None:
        breaks = compute_breaks(s, k=(len(palette) if palette else len(RED_YELLOW)))
    colorset = palette if palette else RED_YELLOW

    def rgb_norm(c):
        return (c[0]/255.0, c[1]/255.0, c[2]/255.0)

    fig, ax = plt.subplots(figsize=(8, 8))
    patches = []
    colors = []

    for b in boards:
        uid = b["unit_id"]
        v = float(s.get(uid, 0.0))
        if palette is not None and (v in [0.0, 1.0, 2.0, 3.0]):
            col = colorset[int(v)]
        else:
            col = color_for_value_dynamic(v, breaks)
        col = rgb_norm(col)

        geom = b["geom"]
        if geom.geom_type == "Polygon":
            rings = [np.asarray(geom.exterior.coords)]
            for r in rings:
                from matplotlib.patches import Polygon
                patches.append(Polygon(r, closed=True))
                colors.append(col)
        elif geom.geom_type == "MultiPolygon":
            for g in geom.geoms:
                r = np.asarray(g.exterior.coords)
                patches.append(Polygon(r, closed=True))
                colors.append(col)

    pc = PatchCollection(
        patches,
        facecolor=colors,
        edgecolor=(0.2, 0.2, 0.2),
        linewidths=0.5,
        alpha=0.9
    )
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect('equal', adjustable='box')
    ax.axis("off")
    ax.set_title(title, fontsize=12)

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    return buf.read()

# =====================================================================
# Methodology text
# =====================================================================

# =====================================================================
# Cost references (Green Roof + Rain Garden)
# =====================================================================

GREEN_ROOF_COST_USD_PER_SQFT = (5.0, 25.0)     # EPA: Vegetated Roofs $5-$25/sf
RAIN_GARDEN_COST_USD_PER_SQFT = (5.0, 40.0)    # EPA example notes typical range $5-$40/sf

COST_SOURCES = {
    "green_roof": {
        "label": "EPA (Vegetated Roofs): $5–$25 per sq ft",
        "url": "https://www.epa.gov/sites/default/files/2016-01/documents/bf_revitalization_climate_vulnerable_areas_012616_508_v2_web.pdf"
    },
    "rain_garden": {
        "label": "EPA (Rain gardens): range $5–$40 per sq ft (example project narrative)",
        "url": "https://www.epa.gov/sites/default/files/2015-10/documents/umd_narrative.pdf"
    }
}

METH_RISK_URBAN = {
    "Urban Combined": """**Urban Flooding - Combined Risk Score (methodology)**

**Overview:**  
This in-house flood risk score captures **pluvial (rainfall-driven) urban flooding** by combining three community-based and sensor-based indicators. Unlike NRI coastal/riverine scores, this focuses on localized stormwater flooding across NYC census tracts.

**Data Sources:**
- **FloodNet water-depth sensors**: High-frequency depth measurements at 100+ locations across NYC
- **NOAA MRMS radar precipitation**: Hourly rainfall at ~1km resolution
- **NYC 311 service requests**: Community-reported street flooding and catch basin issues

**Three Component Scores:**

1. **score_precip** (Precipitation Sensitivity, range 0-5):
   - Derived from Negative Binomial GLM modeling sensor depth response to daily rainfall
   - Sensors with significant positive rain response (p<0.05, coef>0) score 4-5
   - Measures: *How much do water levels rise when it rains?*

2. **score_sf** (Street Flooding Reports, range 0-4):
   - Based on average daily 311 street flooding complaints within 0.5km of sensors on wet days
   - Normalized across all sensors with positive complaint rates
   - Measures: *How often do residents report street flooding during rain events?*

3. **score_cb** (Catch Basin Issues, range 0-4):
   - Based on average daily 311 catch basin complaints on wet days
   - Includes clogged, damaged, or malfunctioning drainage infrastructure reports
   - Measures: *How often do residents report drainage infrastructure problems?*

**score_total = score_precip + score_sf + score_cb** (range 0-13)

**Spatial Propagation:**
- Scores computed at 155 FloodNet sensor locations
- 500 synthetic "virtual sensors" fill spatial gaps using topographic similarity (elevation, slope, building proximity)
- Each census tract assigned scores from nearest sensor (real or synthetic) via spatial join

**Interpretation:**
- **High score_total (red)**: Areas with confirmed sensor-detected flooding AND high community reporting
- **High score_sf**: Community-identified street flooding hotspots (infrastructure may be overwhelmed)
- **High score_cb**: Areas with chronic drainage infrastructure issues (maintenance priority)
- **High score_precip**: Hydrologically sensitive locations (water accumulates during storms)

**Use Cases:**
- **score_total**: Overall urban flood vulnerability for capital planning
- **score_sf**: Prioritize emergency response and flood warning systems
- **score_cb**: Target catch basin cleaning, repair, and green infrastructure retrofits
- **score_precip**: Validate sensor network expansion locations

**Limitations:**
- Reflects 2020-2023 conditions (sensor deployment period)
- Underrepresents areas >0.5km from sensors (despite synthetic interpolation)
- 311 reports are biased toward higher-income, digitally-connected communities
- Does not include coastal/tidal flooding (see NRI CFLD_RISKS for that)

**Reference:** See Methods_Risk_Map.docx for full statistical workflow
""",
    
    "Urban Precip": """**Urban Flooding - Precipitation Sensitivity Score (methodology)**

**What it measures:**  
How strongly water depth at FloodNet sensors responds to rainfall, derived from daily-scale statistical modeling.

**Method:**
1. **Data**: 155 FloodNet sensors with hourly depth measurements merged with NOAA MRMS radar precipitation
2. **Aggregation**: Sum hourly depth and precipitation to daily totals per sensor
3. **Statistical Model**: Negative Binomial GLM per sensor:
   - Dependent variable: daily depth (mm) as count data
   - Predictors: daily precipitation (mm) + 311 complaint counts
   - Extract rain coefficient (`coef_precip`) and p-value
4. **Scoring**:
   - p-value ≥ 0.05 or coef ≤ 0 → **score = 0** (not rain-sensitive)
   - coef = 0 (exactly) → **score = 3** (neutral)
   - p < 0.05 and coef > 0 → normalize positive coefficients to **range 4-5** (most sensitive)

**Spatial Coverage:**  
Scores assigned to 500 synthetic sensors using topographic nearest-neighbor matching (elevation, slope, building distance), then propagated to census tracts via spatial join.

**Interpretation:**
- **High scores (red, 4-5)**: Locations where rain consistently causes measurable water accumulation
- **Zero scores (yellow)**: Areas with no detected rain-flood relationship (may be well-drained or sensors are dry)
- **~55 sensors** (out of 155) are statistically rain-sensitive

**Use for:** 
- Identifying chronic drainage problem areas
- Validating whether green infrastructure reduces rain response over time
- Sensor network optimization (place sensors where rain sensitivity is unknown)

**Limitation:** Does not capture human perception—use score_sf for community-reported impacts
""",
    
    "Urban SF": """**Urban Flooding - Street Flooding Reports Score (methodology)**

**What it measures:**  
Community-reported street flooding intensity during wet weather, based on NYC 311 service requests.

**Method:**
1. **311 Data Filtering**:
   - Select complaints with descriptor containing both "street" AND "flood" (case-insensitive)
   - Convert to hourly timestamps, filter to valid lat/lon coordinates
2. **Spatial Linking**:
   - Use BallTree (haversine distance) to assign each complaint to all FloodNet sensors within **0.5 km radius**
   - Aggregate to hourly complaint counts per sensor
3. **Wet-Day Focus**:
   - Merge with MRMS daily precipitation
   - Retain only days with precipitation > 0 mm
   - Compute **mean daily street flooding complaints per sensor on wet days**
4. **Scoring**:
   - Sensors with zero complaints → **score = 0**
   - Normalize positive values to **range 3-4** (linear scaling)

**Spatial Propagation:**  
Assigned to 500 synthetic sensors via topographic similarity, then to census tracts via nearest sensor.

**Interpretation:**
- **High scores (orange/red, 3.5-4)**: Areas where residents frequently report street flooding during rain
- **Zero scores**: Either well-drained OR underreporting (check against score_precip to distinguish)
- Captures **community perception** and **localized impacts** that sensors may miss

**Use for:**
- Emergency response prioritization during storms
- Identifying flood-prone intersections for signage/barricades
- Validating sensor placement (do sensors detect flooding where residents report it?)

**Limitations:**
- Reporting bias: wealthier, more engaged communities file more 311 requests
- Does not measure flood depth—only frequency of reports
- 0.5km radius may group unrelated flooding locations in dense areas

**Complementary to score_precip:** Sensors detect water, 311 detects *impacts on people*
""",
    
    "Urban CB": """**Urban Flooding - Catch Basin Issues Score (methodology)**

**What it measures:**  
Frequency of resident-reported catch basin problems (clogs, damage, capacity issues) during wet weather—an indicator of **drainage infrastructure failure**.

**Method:**
1. **311 Data Aggregation**:
   - Group all catch basin-related complaint descriptors into "Catch_Basin_Issues" category
   - Examples: "Clogged Catch Basin," "Broken Catch Basin," "Request Catch Basin Cleaning"
2. **Daily Complaint Counts**:
   - Merge 311 complaints with MRMS daily precipitation at sensor locations
   - Retain only **wet days** (daily precipitation > 0 mm)
3. **Scoring**:
   - Compute **mean daily catch basin complaints per sensor on wet days**
   - Sensors with zero complaints → **score = 0**
   - Normalize positive values to **range 3-4** (linear scaling)

**Spatial Propagation:**  
Assigned to 500 synthetic sensors via topographic nearest-neighbor, then to census tracts via spatial join.

**Interpretation:**
- **High scores (orange/red, 3.5-4)**: Chronic drainage infrastructure problems—catch basins are clogged, undersized, or broken
- **Zero scores**: Either well-maintained infrastructure OR underreporting
- Unlike score_sf (which reports *flooding*), this reports *the cause* (failed drainage)

**Use for:**
- **Infrastructure maintenance planning**: Prioritize catch basin cleaning and repair
- **Capital projects**: Identify areas needing upsized or additional catch basins
- **Green infrastructure targeting**: Where traditional gray infrastructure is failing, consider bioswales, permeable pavement
- **Real-time operations**: During storms, preemptively dispatch crews to high-score areas

**Limitations:**
- Reporting bias (same as score_sf)
- Does not measure actual catch basin condition—only community complaints
- Some complaints may be duplicate reports of the same basin
- Does not distinguish clog severity (partial vs. complete blockage)

**Complementary to score_sf:**  
- High score_cb + High score_sf = flooding caused by infrastructure failure → **fix catch basins**
- Low score_cb + High score_sf = flooding despite working drains → **inadequate capacity** → need upsizing or green infrastructure
""",
}

METH_RISK = {
    "NRI Coastal": """**NRI Coastal Flooding Risk (methodology)**
- **Source:** FEMA National Risk Index (NRI), tract-level download.
- **Metric used:** `CFLD_RISKS` from spreadsheet (raw NRI composite risk score; higher = higher risk).
- **Join:** Tract IDs normalized to digits-only and matched to feature attributes.
- **Mapping:** Colors are **quartiles of the raw NRI values**, not normalized; legend shows **actual value ranges**.
- **Interpretation:** The NRI "Risk" combines Expected Annual Loss, Social Vulnerability, and Community Resilience.
- **Caveats:** NRI scores are comparative; use alongside local knowledge, drainage and land-use data.
""",
    "NRI Riverine": """**NRI Riverine Flooding Risk (methodology)**
- **Source:** FEMA NRI (tract level).
- **Metric used:** `RFLD_RISKS` raw values from spreadsheet.
- **Join &amp; mapping:** Same as coastal — **quartiles on raw values** with range legend.
- **Interpretation/Caveats:** Same as coastal.
""",
    "My Risk Map": """**In-House Flood Risk (methodology)**
- **Source columns:** `score_total` in the spreadsheet (custom composite).
- **Intended inputs:** static features such as Elevation, Slope, Imperviousness, Footprint and any learned weights.
- **Mapping:** **Quartiles of raw `score_total`**, legend shows value ranges.
- **Next steps:** Document weighting scheme; add validation against observed flooding or 311 events.
"""
}

METH_FORECAST = {
    "Water sensors": """**TODO – Next-Day Forecast (Water Sensors)**
- TODO: sensors &amp; units
- TODO: aggregation to boards
- TODO: baseline/persistence definition
- TODO: known limitations
""",
    "Street flooding": """**TODO – Next-Day Forecast (Street Flooding)**
- TODO: 311 ingestion and cleaning
- TODO: temporal aggregation window
- TODO: baseline/persistence definition
- TODO: known limitations
""",
    "Catch basins": """**TODO – Next-Day Forecast (Catch Basins)**
- TODO: clog/cleaning data source
- TODO: aggregation to boards &amp; cadence
- TODO: baseline/persistence definition
- TODO: known limitations
"""
}

METH_URBAN = """**Static Urban Features (context)**
- **Elevation**: Mean elevation [from NYC Open Data] aggregated per census tract via ArcGIS Pro.
- **Slope**: Mean slope [from NYC Open Data] aggregated per census tract via ArcGIS Pro.
- **Building Footprint**: Total building footprint area (square kilometers) [from NYC Open Data] aggregated per census tract via ArcGIS Pro.
- **Building Footprint Density**: Ratio of building footprint area to total tract area (both in sq km). Higher values indicate more built-up areas.
- **Data Processing**: All spatial data aggregated to census tract level using ArcGIS Pro statistical tooling.
- **Interpretation**: Higher footprint density, smaller slope and lower elevation typically correlate with increased flood vulnerability.
"""

METH_UHI = """**Urban Heat Island / Heat-Wave Risk (methodology)**
- **Source:** FEMA NRI heat-wave component.
- **Metric used:** `HWAV_RISKS` from the spreadsheet (raw values).
- **Join:** Same normalized tract-ID match as flood layers.
- **Mapping:** **Quartiles of raw values**; legend shows value ranges (not z-scores).
- **Use:** Prioritize cooling centers, tree planting, reflective surfaces in higher-risk tracts.
"""

# ===================================================================== 
# Socio-Demographic Field Categories
# ===================================================================== 

DEMOGRAPHIC_CATEGORIES = {
    "Household Characteristics": {
        "Total households": "HOUSEHOLDS BY TYPE - Total households",
        "Family households": "HOUSEHOLDS BY TYPE - Total households - Family households (families)",
        "Families with children": "HOUSEHOLDS BY TYPE - Total households - Family households (families) - With own children of the householder under 18 years",
        "Average household size": "HOUSEHOLDS BY TYPE - Total households - Average household size",
        "Average family size": "HOUSEHOLDS BY TYPE - Total households - Average family size",
    },
    
    "Age Distribution": {
        "Total population": "SEX AND AGE - Total population",
        "Under 5 years": "SEX AND AGE - Total population - Under 5 years",
        "5 to 9 years": "SEX AND AGE - Total population - 5 to 9 years",
        "10 to 14 years": "SEX AND AGE - Total population - 10 to 14 years",
        "15 to 19 years": "SEX AND AGE - Total population - 15 to 19 years",
        "20 to 24 years": "SEX AND AGE - Total population - 20 to 24 years",
        "25 to 34 years": "SEX AND AGE - Total population - 25 to 34 years",
        "35 to 44 years": "SEX AND AGE - Total population - 35 to 44 years",
        "45 to 54 years": "SEX AND AGE - Total population - 45 to 54 years",
        "55 to 59 years": "SEX AND AGE - Total population - 55 to 59 years",
        "60 to 64 years": "SEX AND AGE - Total population - 60 to 64 years",
        "65 to 74 years (Elderly)": "SEX AND AGE - Total population - 65 to 74 years",
        "75 to 84 years (Elderly)": "SEX AND AGE - Total population - 75 to 84 years",
        "85 years and over (Elderly)": "SEX AND AGE - Total population - 85 years and over",
        "Median age": "SEX AND AGE - Total population - Median age (years)",
    },
    
    "Gender": {
        "Male population": "SEX AND AGE - Total population - Male",
        "Female population": "SEX AND AGE - Total population - Female",
    },
    
    "Race and Ethnicity": {
        "Total population": "RACE - Total population",
        "White": "RACE - Total population - One race - White",
        "Black or African American": "RACE - Total population - One race - Black or African American",
        "American Indian and Alaska Native": "RACE - Total population - One race - American Indian and Alaska Native",
        "Asian": "RACE - Total population - One race - Asian",
        "Native Hawaiian and Pacific Islander": "RACE - Total population - One race - Native Hawaiian and Other Pacific Islander",
        "Some other race": "RACE - Total population - One race - Some other race",
        "Two or more races": "RACE - Total population - Two or more races",
        "Hispanic or Latino (any race)": "HISPANIC OR LATINO AND RACE - Total population - Hispanic or Latino (of any race)",
        "Not Hispanic or Latino": "HISPANIC OR LATINO AND RACE - Total population - Not Hispanic or Latino",
        "White alone (not Hispanic)": "HISPANIC OR LATINO AND RACE - Total population - Not Hispanic or Latino - White alone",
    },
    
    "Education": {
        "Population 25+ years": "EDUCATIONAL ATTAINMENT - Population 25 years and over",
        "Less than 9th grade": "EDUCATIONAL ATTAINMENT - Population 25 years and over - Less than 9th grade",
        "9th to 12th grade, no diploma": "EDUCATIONAL ATTAINMENT - Population 25 years and over - 9th to 12th grade, no diploma",
        "High school graduate": "EDUCATIONAL ATTAINMENT - Population 25 years and over - High school graduate (includes equivalency)",
        "Some college, no degree": "EDUCATIONAL ATTAINMENT - Population 25 years and over - Some college, no degree",
        "Associate's degree": "EDUCATIONAL ATTAINMENT - Population 25 years and over - Associate's degree",
        "Bachelor's degree": "EDUCATIONAL ATTAINMENT - Population 25 years and over - Bachelor's degree",
        "Graduate or professional degree": "EDUCATIONAL ATTAINMENT - Population 25 years and over - Graduate or professional degree",
        "High school graduate or higher (%)": "EDUCATIONAL ATTAINMENT - Population 25 years and over - High school graduate or higher",
        "Bachelor's degree or higher (%)": "EDUCATIONAL ATTAINMENT - Population 25 years and over - Bachelor's degree or higher",
    },
    
    "School Enrollment": {
        "Population 3+ enrolled": "SCHOOL ENROLLMENT - Population 3 years and over enrolled in school",
        "Nursery/preschool": "SCHOOL ENROLLMENT - Population 3 years and over enrolled in school - Nursery school, preschool",
        "Kindergarten": "SCHOOL ENROLLMENT - Population 3 years and over enrolled in school - Kindergarten",
        "Elementary (grades 1-8)": "SCHOOL ENROLLMENT - Population 3 years and over enrolled in school - Elementary school (grades 1-8)",
        "High school (grades 9-12)": "SCHOOL ENROLLMENT - Population 3 years and over enrolled in school - High school (grades 9-12)",
        "College or graduate school": "SCHOOL ENROLLMENT - Population 3 years and over enrolled in school - College or graduate school",
    },
    
    "Income": {
        "Median household income": "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)",
        "Mean household income": "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - Mean household income (dollars)",
        "Less than \$10,000": "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - Less than \$10,000",
        "\$10,000 to \$14,999": "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - \$10,000 to \$14,999",
        "\$15,000 to \$24,999": "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - \$15,000 to \$24,999",
        "\$25,000 to \$34,999": "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - \$25,000 to \$34,999",
        "\$35,000 to \$49,999": "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - \$35,000 to \$49,999",
        "\$50,000 to \$74,999": "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - \$50,000 to \$74,999",
        "\$75,000 to \$99,999": "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - \$75,000 to \$99,999",
        "\$100,000 to \$149,999": "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - \$100,000 to \$149,999",
        "\$150,000 to \$199,999": "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - \$150,000 to \$199,999",
        "\$200,000 or more": "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - \$200,000 or more",
    },
    
    "Employment": {
        "Population 16+ years": "EMPLOYMENT STATUS - Population 16 years and over",
        "In labor force": "EMPLOYMENT STATUS - Population 16 years and over - In labor force",
        "Civilian labor force": "EMPLOYMENT STATUS - Population 16 years and over - In labor force - Civilian labor force",
        "Employed": "EMPLOYMENT STATUS - Population 16 years and over - In labor force - Civilian labor force - Employed",
        "Unemployed": "EMPLOYMENT STATUS - Population 16 years and over - In labor force - Civilian labor force - Unemployed",
        "Not in labor force": "EMPLOYMENT STATUS - Population 16 years and over - Not in labor force",
        "Unemployment rate (%)": "EMPLOYMENT STATUS - Civilian labor force - Unemployment Rate",
    },
    
    "Commuting": {
        "Workers 16+ years": "COMMUTING TO WORK - Workers 16 years and over",
        "Drove alone": "COMMUTING TO WORK - Workers 16 years and over - Car, truck, or van -- drove alone",
        "Carpooled": "COMMUTING TO WORK - Workers 16 years and over - Car, truck, or van -- carpooled",
        "Public transportation": "COMMUTING TO WORK - Workers 16 years and over - Public transportation (excluding taxicab)",
        "Walked": "COMMUTING TO WORK - Workers 16 years and over - Walked",
        "Other means": "COMMUTING TO WORK - Workers 16 years and over - Other means",
        "Worked at home": "COMMUTING TO WORK - Workers 16 years and over - Worked at home",
        "Mean travel time (minutes)": "COMMUTING TO WORK - Workers 16 years and over - Mean travel time to work (minutes)",
    },
    
    "Housing": {
        "Total housing units": "HOUSING OCCUPANCY - Total housing units",
        "Occupied units": "HOUSING OCCUPANCY - Total housing units - Occupied housing units",
        "Vacant units": "HOUSING OCCUPANCY - Total housing units - Vacant housing units",
        "Owner-occupied": "HOUSING TENURE - Occupied housing units - Owner-occupied",
        "Renter-occupied": "HOUSING TENURE - Occupied housing units - Renter-occupied",
        "Median home value": "VALUE - Owner-occupied units - Median (dollars)",
        "Units with mortgage": "MORTGAGE STATUS - Owner-occupied units - Housing units with a mortgage",
        "Units without mortgage": "MORTGAGE STATUS - Owner-occupied units - Housing units without a mortgage",
        "Median gross rent": "GROSS RENT - Occupied units paying rent - Median (dollars)",
    },
    
    "Language and Citizenship": {
        "Population 5+ years": "LANGUAGE SPOKEN AT HOME - Population 5 years and over",
        "English only": "LANGUAGE SPOKEN AT HOME - Population 5 years and over - English only",
        "Language other than English": "LANGUAGE SPOKEN AT HOME - Population 5 years and over - Language other than English",
        "Foreign-born population": "U.S. CITIZENSHIP STATUS - Foreign-born population",
        "Naturalized U.S. citizen": "U.S. CITIZENSHIP STATUS - Foreign-born population - Naturalized U.S. citizen",
        "Not a U.S. citizen": "U.S. CITIZENSHIP STATUS - Foreign-born population - Not a U.S. citizen",
    },
    
    "Mobility": {
        "Population 1+ year": "RESIDENCE 1 YEAR AGO - Population 1 year and over",
        "Same house": "RESIDENCE 1 YEAR AGO - Population 1 year and over - Same house",
        "Different house in U.S.": "RESIDENCE 1 YEAR AGO - Population 1 year and over - Different house in the U.S.",
        "Abroad": "RESIDENCE 1 YEAR AGO - Population 1 year and over - Abroad U.S.",
    },
    
    "Voting Age Population": {
        "Citizen, 18 and over": "CITIZEN, VOTING AGE POPULATION - Citizen, 18 and over population",
    },
}

# Methodology text for demographics
METH_DEMOGRAPHICS = """**Socio-Demographic Data (methodology)**

- **Source:** U.S. Census Bureau American Community Survey (ACS) 5-Year Estimates (2014-2018)
- **Geography:** Census tract level for NYC
- **Join:** Merged with census tract geometries via NRI_ID/GEOID matching
- **Mapping:** Colors represent **quartiles** (4 equal groups) of the selected metric
  - **Red (Q4):** Highest 25% of values
  - **Orange (Q3):** 50-75th percentile
  - **Yellow (Q2):** 25-50th percentile  
  - **Light Yellow (Q1):** Lowest 25% of values

**Key Vulnerability Indicators:**
- **Elderly Population (65+):** Higher vulnerability to heat waves and flooding (mobility constraints)
- **Low Income (<;\$25k):** Limited resources for evacuation, recovery, or adaptation
- **Renter-Occupied Housing:** Less control over building improvements, higher displacement risk
- **Non-English Speakers:** May miss emergency warnings and assistance programs
- **Low Educational Attainment:** Correlated with lower awareness of climate risks
- **High Unemployment:** Reduced economic resilience to climate shocks

**Use Cases:**
- **Environmental Justice:** Identify tracts with high climate risk AND high social vulnerability
- **Resource Allocation:** Prioritize cooling centers, flood warnings, and assistance in vulnerable areas
- **Community Engagement:** Target outreach to non-English speaking or elderly populations
- **Evacuation Planning:** Account for mobility constraints and vehicle access

**Limitations:**
- Data from 2018 (may not reflect recent demographic shifts)
- Tract-level aggregation masks within-tract variation
- Does not capture informal housing or undocumented populations

**Reference:** U.S. Census Bureau ACS 2014-2018 5-Year Estimates
"""
# =====================================================================
# Map builders
# =====================================================================

def fc_from_boards_and_values(boards: list, values: pd.Series, prop_name: str, color_series: Optional[pd.Series] = None) -> dict:
    """Build a FeatureCollection with proper color handling"""
    feats = []
    breaks = None
    if color_series is None:
        breaks = compute_breaks(values, k=len(RED_YELLOW))

    for b in boards:
        uid = b["unit_id"]
        v = float(pd.to_numeric(values.get(uid, 0.0), errors="coerce") or 0.0)

        if color_series is not None:
            col = color_series.get(uid)
            if not isinstance(col, (list, tuple)) or len(col) != 3:
                col = [255, 255, 178]
            fill = [int(col[0]), int(col[1]), int(col[2])]
        else:
            fill = color_for_value_dynamic(v, breaks)

        feats.append({
            "type": "Feature",
            "properties": {
                "unit_id": uid,
                "cb": get_board_display_id(b),
                prop_name: v,
                prop_name + "_formatted": f"{v:.4f}",
                "fill_color": fill
            },
            "geometry": b["feature_geom"]
        })
    return {"type": "FeatureCollection", "features": feats}

def pydeck_choropleth(fc: dict, lat_c: float, lon_c: float, zoom_c: float, prop_name: str, tooltip_label: str, opacity=0.85, label_layer=None):
    layer = pdk.Layer(
        "GeoJsonLayer",
        data=fc,
        pickable=True, stroked=True, filled=True,
        get_fill_color="properties.fill_color",
        get_line_color=[40, 40, 40], lineWidthMinPixels=2, opacity=opacity
    )
    layers = [
        pdk.Layer(
            "TileLayer",
            data="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
            minZoom=0, maxZoom=19, tileSize=256
        ),
        layer
    ]
    # Only add label layer if explicitly provided (which we won't do anymore)
    if label_layer:
        layers.append(label_layer)
    
    formatted_prop = prop_name + "_formatted"
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=pdk.ViewState(latitude=lat_c, longitude=lon_c, zoom=zoom_c),
        tooltip={
            "html": "<b>Tract #:</b> {cb}<br><b>" + tooltip_label + ":</b> {" + formatted_prop + "}",
            "style": {"backgroundColor": "white", "color": "black"}
        }
    )
    return deck

# =====================================================================
# NEW: Query Engine for Multi-Criteria Selection
# =====================================================================

def parse_user_query_with_claude(user_query: str, boards: list) -> dict:
    """Use Claude to parse natural language into structured criteria"""
    
    # Build available fields from all categories
    available_fields = {
        # Risk fields
        "CFLD_RISKS": "Coastal flooding risk (NRI)",
        "RFLD_RISKS": "Riverine flooding risk (NRI)",
        "HWAV_RISKS": "Heat wave risk (NRI)",
        "score_total": "In-house flood risk score",
        
        # Urban features
        "Mean_elevation": "Mean elevation (meters)",
        "Mean_slope": "Mean slope (degrees)",
        "Total_ftp_area": "Building footprint area (sq km)",
        "_COMPUTED_DENSITY_": "Building footprint density",
        
        # Demographics - add key fields
        "SEX AND AGE - Total population - 65 to 74 years": "Elderly population 65-74",
        "SEX AND AGE - Total population - 75 to 84 years": "Elderly population 75-84",
        "SEX AND AGE - Total population - 85 years and over": "Elderly population 85+",
        "INCOME AND BENEFITS (IN 2018 INFLATION-ADJUSTED DOLLARS) - Total households - Median household income (dollars)": "Median household income",
        "HOUSING TENURE - Occupied housing units - Renter-occupied": "Renter-occupied housing",
        "LANGUAGE SPOKEN AT HOME - Population 5 years and over - Language other than English": "Non-English speakers",
        "EDUCATIONAL ATTAINMENT - Population 25 years and over - Less than 9th grade": "Less than 9th grade education",
        "EMPLOYMENT STATUS - Civilian labor force - Unemployment Rate": "Unemployment rate",
    }
    
    system = (
        "You are a query parser for NYC census tract data. "
        "Parse natural language queries into JSON criteria. "
        "Return ONLY valid JSON with no explanation.\n\n"
        "Available fields:\n" + 
        "\n".join([f"- {k}: {v}" for k, v in available_fields.items()]) +
        "\n\nBin meanings (equal intervals):\n"
        "- bin 3 = highest 25% (red)\n"
        "- bin 2 = 50-75% (orange)\n"
        "- bin 1 = 25-50% (yellow)\n"
        "- bin 0 = lowest 25% (light yellow)\n\n"
        "For demographic queries:\n"
        "- 'high elderly population' → use elderly age fields, bin 3\n"
        "- 'low income' → use median income, bin 0\n"
        "- 'high renters' → use renter-occupied, bin 3"
    )
    
    prompt = f"""Parse this query into JSON: "{user_query}"

Return format:
{{
  "criteria": [
    {{"field": "RFLD_RISKS", "bin": 3, "description": "highest riverine risk"}},
    {{"field": "SEX AND AGE - Total population - 65 to 74 years", "bin": 3, "description": "high elderly population"}}
  ]
}}

Keywords:
- "highest", "high", "red" → bin 3
- "high-medium" → bin 2  
- "low-medium" → bin 1
- "lowest", "low" → bin 0

Return ONLY the JSON object."""

    response = call_claude(prompt, system=system, temperature=0.1, max_tokens=800)
    
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        return {"criteria": []}
    except Exception as e:
        st.error(f"Parse error: {e}")
        return {"criteria": []}

def filter_tracts_by_criteria(boards: list, criteria: list) -> list:
    """Filter tracts meeting ALL criteria"""
    if not criteria:
        return []
    
    # Compute bins for all relevant fields
    field_bins = {}
    for crit in criteria:
        field = crit.get("field")
        if not field:
            continue
        
        # Handle computed density
        if field == "_COMPUTED_DENSITY_":
            vals = {}
            for b in boards:
                attrs = b.get("attrs", {})
                ftp = coerce_numeric(get_attr_ci(attrs, "Total_ftp_area"), default=0.0)
                area_sqmi = coerce_numeric(get_attr_ci(attrs, "AREA"), default=0.0)
                area_sqkm = area_sqmi * 2.58999
                if area_sqkm > 0:
                    vals[b["unit_id"]] = ftp / area_sqkm
                else:
                    vals[b["unit_id"]] = 0.0
        else:
            vals = {b["unit_id"]: get_attr_num(b.get("attrs", {}), field) for b in boards}
        
        series = pd.Series(vals)
        bins, _ = quantile_bins_4(series)
        field_bins[field] = bins
    
    # Filter tracts
    matching = []
    for b in boards:
        uid = b["unit_id"]
        meets_all = True
        
        for crit in criteria:
            field = crit.get("field")
            target_bin = crit.get("bin")
            
            if field not in field_bins:
                meets_all = False
                break
            
            tract_bin = field_bins[field].get(uid, 0)
            
            if tract_bin != target_bin:
                meets_all = False
                break
        
        if meets_all:
            matching.append(b)
    
    return matching

# =====================================================================
