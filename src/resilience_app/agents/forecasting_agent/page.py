"""Operational NYC 1-km day-ahead flood forecasting dashboard."""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pydeck as pdk
import streamlit as st


# =====================================================================
# Paths
# =====================================================================

FORECAST_ROOT = Path(
    os.getenv(
        "GRID_FLOOD_FORECAST_ROOT",
        "/app/artifacts/flood/forecasting",
    )
)

SUPPORTED_FORECAST_MODES = (
    "today",
    "tomorrow",
)

FORECAST_FILENAME = "grid_flood_forecast.parquet"

FORECAST_SUMMARY_FILENAME = (
    "grid_flood_forecast_summary.json"
)

PRECIPITATION_SUMMARY_FILENAME = (
    "grid_precipitation_forecast_summary.json"
)

DEFAULT_GRID_PATH = Path(
    os.getenv(
        "GRID_IMPUTATION_GEOJSON_PATH",
        "/app/artifacts/flood/spatial/grid_imputation_reference.geojson",
    )
)


# =====================================================================
# Constants
# =====================================================================

NYC_LATITUDE = 40.7128
NYC_LONGITUDE = -74.0060
NYC_TIMEZONE = ZoneInfo("America/New_York")

EXPECTED_GRID_COUNT = 837

LOGISTIC_MODEL_NAME = "logistic"
GCN_MODEL_NAME = "gcn"

PRECIPITATION_OPTIONS = {
    "Current-hour precipitation": "precip_current_hour_mm",
    "Previous 6-hour precipitation": "precip_previous_6h_mm",
    "Trailing 24-hour precipitation": "daily_total_precip_mm",
}

PRECIPITATION_SHORT_LABELS = {
    "precip_current_hour_mm": "Current hour",
    "precip_previous_6h_mm": "Previous 6h",
    "daily_total_precip_mm": "Trailing 24h",
}


# =====================================================================
# Data loading
# =====================================================================


@st.cache_data(show_spinner=False)
def load_forecast_data(
    forecast_path: str,
    modified_time: float,
) -> pd.DataFrame:
    """Load and validate the operational grid/hour forecast artifact."""

    del modified_time

    dataframe = pd.read_parquet(forecast_path)

    required_columns = {
        "grid_id",
        "forecast_hour",
        "grid_lat",
        "grid_lon",
        "cluster_number",
        "precip_current_hour_mm",
        "precip_previous_6h_mm",
        "daily_total_precip_mm",
        "target_date_utc",
        "forecast_initialization",
        "hrrr_cycle_id",
        "hrrr_distance_km",
        "support_sensor_id",
        "support_sensor_model",
        "support_sensor_f1",
        "support_sensor_distance_km",
        "has_eligible_sensor_in_grid",
        "imputation_required",
        "is_imputed_grid",
        "imputation_method",
        "support_scope",
        "logistic_event_probability",
        "gcn_predicted_minutes_above_1inch",
        "predicted_flood_event",
        "native_prediction_type",
        "native_prediction_value",
        "classification_threshold",
        "forecast_generated_at",
        "operational_prediction_source",
    }

    missing = sorted(
        required_columns - set(dataframe.columns)
    )

    if missing:
        raise ValueError(
            "Operational forecast artifact is missing required columns: "
            + ", ".join(missing)
        )

    dataframe = dataframe.copy()

    dataframe["grid_id"] = (
        dataframe["grid_id"]
        .astype(str)
    )

    dataframe["forecast_hour"] = pd.to_datetime(
        dataframe["forecast_hour"],
        utc=True,
        errors="coerce",
    )

    if dataframe["forecast_hour"].isna().any():
        raise ValueError(
            "One or more forecast_hour values could not be parsed."
        )

    dataframe["support_sensor_model"] = (
        dataframe["support_sensor_model"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    dataframe["predicted_flood_event"] = (
        pd.to_numeric(
            dataframe["predicted_flood_event"],
            errors="coerce",
        )
        .fillna(0)
        .astype(int)
        .astype(bool)
    )

    duplicate_count = dataframe.duplicated(
        ["grid_id", "forecast_hour"]
    ).sum()

    if duplicate_count:
        raise ValueError(
            "Forecast artifact contains "
            f"{duplicate_count:,} duplicate grid/hour rows."
        )

    return dataframe


@st.cache_data(show_spinner=False)
def load_grid_geojson(
    grid_path: str,
    modified_time: float,
) -> dict:
    """Load the authoritative NYC 1-km GeoJSON."""

    del modified_time

    with Path(grid_path).open(
        "r",
        encoding="utf-8",
    ) as file:
        geojson = json.load(file)

    features = geojson.get(
        "features",
        [],
    )

    if not features:
        raise ValueError(
            "The 1-km grid GeoJSON contains no features."
        )

    return geojson


# =====================================================================
# Basic helpers
# =====================================================================

def resolve_forecast_paths(
    mode: str,
) -> tuple[Path, Path, Path]:
    """Resolve mode-specific operational forecast artifacts."""

    mode = str(mode).strip().lower()

    if mode not in SUPPORTED_FORECAST_MODES:
        raise ValueError(
            "Unsupported forecast mode: "
            f"{mode!r}"
        )

    mode_dir = FORECAST_ROOT / mode

    return (
        mode_dir / FORECAST_FILENAME,
        mode_dir / FORECAST_SUMMARY_FILENAME,
        mode_dir / PRECIPITATION_SUMMARY_FILENAME,
    )


@st.cache_data(show_spinner=False)
def load_json_file(
    path: str,
    modified_time: float,
) -> dict:
    """Load an optional operational JSON metadata artifact."""

    del modified_time

    file_path = Path(path)

    if not file_path.exists():
        return {}

    with file_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        value = json.load(handle)

    return (
        value
        if isinstance(value, dict)
        else {}
    )


def load_optional_json(
    path: Path,
) -> dict:
    """Load cached JSON metadata when the artifact exists."""

    if not path.exists():
        return {}

    return load_json_file(
        str(path),
        path.stat().st_mtime,
    )


def local_target_date(
    forecast: pd.DataFrame,
) -> str:
    """Determine the NYC-local target date."""

    if "target_date_nyc" in forecast.columns:
        values = (
            forecast["target_date_nyc"]
            .dropna()
            .astype(str)
            .unique()
        )

        if len(values):
            return values[0]

    first_hour = pd.Timestamp(
        forecast["forecast_hour"].min()
    )

    return str(
        first_hour
        .tz_convert(NYC_TIMEZONE)
        .date()
    )


def forecast_available_through(
    forecast: pd.DataFrame,
    precipitation_summary: dict,
) -> pd.Timestamp:
    """Determine the final published forecast hour."""

    value = precipitation_summary.get(
        "available_through_nyc"
    )

    if value:
        return (
            pd.Timestamp(value)
            .tz_convert(
                NYC_TIMEZONE
            )
        )

    final_hour = pd.Timestamp(
        forecast["forecast_hour"].max()
    )

    return final_hour.tz_convert(
        NYC_TIMEZONE
    )


def requested_forecast_hours(
    precipitation_summary: dict,
) -> int:
    """Return the intended product horizon."""

    try:
        return int(
            precipitation_summary.get(
                "requested_target_hour_count",
                24,
            )
        )
    except (TypeError, ValueError):
        return 24

def safe_float(value) -> float | None:
    """Convert a value to float safely."""

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_round(
    value,
    digits: int = 3,
):
    """Round numeric values while preserving nulls."""

    value = safe_float(value)

    if value is None:
        return None

    return round(
        value,
        digits,
    )


def format_utc_hour(
    timestamp: pd.Timestamp,
) -> str:
    """Format operational UTC time."""

    timestamp = pd.Timestamp(timestamp)

    return timestamp.strftime(
        "%Y-%m-%d %H:%M UTC"
    )


def format_local_hour(
    timestamp: pd.Timestamp,
) -> str:
    """Format operational time in NYC local time."""

    timestamp = pd.Timestamp(timestamp)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            "UTC"
        )

    local = timestamp.tz_convert(
        NYC_TIMEZONE
    )

    return local.strftime(
        "%Y-%m-%d %I:%M %p %Z"
    )


def format_hour_control(
    timestamp: pd.Timestamp,
) -> str:
    """Compact UTC + NYC label for the hour controller."""

    timestamp = pd.Timestamp(timestamp)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(
            "UTC"
        )

    local = timestamp.tz_convert(
        NYC_TIMEZONE
    )

    return (
        f"{timestamp.strftime('%H:%M UTC')}"
        f"  ·  "
        f"{local.strftime('%I:%M %p %Z')}"
    )


# =====================================================================
# Contract validation
# =====================================================================


def validate_forecast_contract(
    forecast: pd.DataFrame,
    grid_geojson: dict,
) -> None:
    """Validate forecast completeness and geometry alignment."""

    geometry_ids = {
        str(
            feature.get(
                "properties",
                {},
            ).get(
                "grid_id",
                "",
            )
        )
        for feature in grid_geojson.get(
            "features",
            [],
        )
    }

    forecast_ids = set(
        forecast["grid_id"]
    )

    missing_geometry = (
        forecast_ids
        - geometry_ids
    )

    if missing_geometry:
        examples = sorted(
            missing_geometry
        )[:10]

        raise ValueError(
            "Forecast grid IDs are missing from the authoritative "
            "1-km geometry. Examples: "
            + ", ".join(examples)
        )

    grid_count = forecast[
        "grid_id"
    ].nunique()

    hour_count = forecast[
        "forecast_hour"
    ].nunique()

    expected_rows = (
        grid_count * hour_count
    )

    if len(forecast) != expected_rows:
        raise ValueError(
            "Forecast is not a complete grid/hour matrix. "
            f"Expected {expected_rows:,} rows from "
            f"{grid_count:,} grids × {hour_count:,} hours; "
            f"found {len(forecast):,}."
        )


# =====================================================================
# Flood susceptibility intensity
# =====================================================================


FLOOD_INTENSITY_LABELS = {
    0: "Below operational threshold",
    1: "Lower",
    2: "Moderate",
    3: "Elevated",
    4: "High",
    5: "Highest",
}


def flood_susceptibility_level(
    model: str,
    logistic_probability: float | None,
    gcn_minutes: float | None,
    predicted_event: bool,
) -> int:
    """
    Convert the selected model's native prediction to a common
    five-level display scale.

    This function changes visualization only. The authoritative
    operational event decision remains predicted_flood_event.
    Logistic intensity is based on event probability; GCN intensity
    is based on predicted minutes above the one-inch target.
    """

    if not predicted_event:
        return 0

    model = str(model).strip().lower()

    if model == LOGISTIC_MODEL_NAME:
        value = safe_float(
            logistic_probability
        )

        if value is None:
            return 0
        if value < 0.60:
            return 1
        if value < 0.70:
            return 2
        if value < 0.80:
            return 3
        if value < 0.90:
            return 4
        return 5

    if model == GCN_MODEL_NAME:
        value = safe_float(
            gcn_minutes
        )

        if value is None:
            return 0
        if value <= 1.50:
            return 1
        if value <= 2.00:
            return 2
        if value <= 2.50:
            return 3
        if value <= 3.00:
            return 4
        return 5

    return 0


def flood_fill_color(
    susceptibility_level: int,
) -> list[int]:
    """Flood-susceptibility polygon fill color."""

    colors = {
        0: [70, 125, 180, 28],
        1: [255, 235, 180, 125],
        2: [255, 195, 110, 155],
        3: [245, 135, 70, 180],
        4: [220, 70, 55, 205],
        5: [145, 20, 30, 230],
    }

    return colors.get(
        susceptibility_level,
        colors[0],
    )


def flood_line_color(
    susceptibility_level: int,
) -> list[int]:
    """Flood-susceptibility polygon border color."""

    if susceptibility_level == 0:
        return [
            70,
            90,
            110,
            75,
        ]

    if susceptibility_level >= 4:
        return [
            105,
            15,
            20,
            235,
        ]

    return [
        145,
        80,
        35,
        210,
    ]


def precipitation_fill_color(
    value: float | None,
) -> list[int]:
    """Discrete operational precipitation color scale."""

    if value is None:
        return [
            180,
            180,
            180,
            25,
        ]

    if value <= 0:
        return [
            225,
            235,
            245,
            25,
        ]

    if value < 0.1:
        return [
            198,
            219,
            239,
            75,
        ]

    if value < 1:
        return [
            158,
            202,
            225,
            110,
        ]

    if value < 5:
        return [
            107,
            174,
            214,
            145,
        ]

    if value < 10:
        return [
            49,
            130,
            189,
            170,
        ]

    if value < 25:
        return [
            84,
            39,
            143,
            190,
        ]

    if value < 50:
        return [
            129,
            15,
            124,
            215,
        ]

    return [
        165,
        15,
        21,
        235,
    ]


# =====================================================================
# Hourly GeoJSON builders
# =====================================================================


def build_flood_geojson(
    base_geojson: dict,
    hour_dataframe: pd.DataFrame,
) -> dict:
    """Attach operational flood predictions to 1-km polygons."""

    geojson = copy.deepcopy(
        base_geojson
    )

    lookup = (
        hour_dataframe
        .set_index("grid_id")
        .to_dict(orient="index")
    )

    for feature in geojson.get(
        "features",
        [],
    ):
        properties = feature.setdefault(
            "properties",
            {},
        )

        grid_id = str(
            properties.get(
                "grid_id",
                "",
            )
        )

        row = lookup.get(
            grid_id
        )

        if row is None:
            properties.update(
                {
                    "has_forecast": False,
                    "predicted_flood_event": False,
                    "forecast_status": "Forecast unavailable",
                    "fill_color": [
                        180,
                        180,
                        180,
                        25,
                    ],
                    "line_color": [
                        100,
                        100,
                        100,
                        50,
                    ],
                }
            )

            continue

        predicted = bool(
            row["predicted_flood_event"]
        )

        model = str(
            row["support_sensor_model"]
        )

        native_type = str(
            row.get(
                "native_prediction_type",
                "",
            )
        )

        native_value = safe_round(
            row.get(
                "native_prediction_value"
            ),
            4,
        )

        susceptibility_level = (
            flood_susceptibility_level(
                model=model,
                logistic_probability=row.get(
                    "logistic_event_probability"
                ),
                gcn_minutes=row.get(
                    "gcn_predicted_minutes_above_1inch"
                ),
                predicted_event=predicted,
            )
        )

        susceptibility_label = (
            FLOOD_INTENSITY_LABELS[
                susceptibility_level
            ]
        )

        properties.update(
            {
                "has_forecast": True,
                "grid_id": grid_id,
                "forecast_status": (
                    "Predicted flood-susceptible"
                    if predicted
                    else "Below flood-susceptibility threshold"
                ),
                "predicted_flood_event": predicted,
                "flood_susceptibility_level": (
                    susceptibility_level
                ),
                "flood_susceptibility_label": (
                    susceptibility_label
                ),
                "forecast_hour_utc": format_utc_hour(
                    row["forecast_hour"]
                ),
                "forecast_hour_nyc": format_local_hour(
                    row["forecast_hour"]
                ),
                "support_sensor_model": model.upper(),
                "support_sensor_id": str(
                    row["support_sensor_id"]
                ),
                "support_sensor_f1": safe_round(
                    row["support_sensor_f1"],
                    3,
                ),
                "is_imputed_grid": bool(
                    row["is_imputed_grid"]
                ),
                "imputation_method": str(
                    row["imputation_method"]
                ),
                "support_scope": str(
                    row["support_scope"]
                ),
                "support_sensor_distance_km": safe_round(
                    row["support_sensor_distance_km"],
                    2,
                ),
                "logistic_event_probability": safe_round(
                    row["logistic_event_probability"],
                    4,
                ),
                "gcn_predicted_minutes_above_1inch": safe_round(
                    row["gcn_predicted_minutes_above_1inch"],
                    3,
                ),
                "native_prediction_type": native_type,
                "native_prediction_value": native_value,
                "classification_threshold": safe_round(
                    row["classification_threshold"],
                    3,
                ),
                "precip_current_hour_mm": safe_round(
                    row["precip_current_hour_mm"],
                    2,
                ),
                "precip_previous_6h_mm": safe_round(
                    row["precip_previous_6h_mm"],
                    2,
                ),
                "daily_total_precip_mm": safe_round(
                    row["daily_total_precip_mm"],
                    2,
                ),
                "fill_color": flood_fill_color(
                    susceptibility_level
                ),
                "line_color": flood_line_color(
                    susceptibility_level
                ),
            }
        )

    return geojson


def build_precipitation_geojson(
    base_geojson: dict,
    hour_dataframe: pd.DataFrame,
    precipitation_column: str,
) -> dict:
    """Attach operational HRRR precipitation to 1-km polygons."""

    geojson = copy.deepcopy(
        base_geojson
    )

    lookup = (
        hour_dataframe
        .set_index("grid_id")
        .to_dict(orient="index")
    )

    short_label = (
        PRECIPITATION_SHORT_LABELS[
            precipitation_column
        ]
    )

    for feature in geojson.get(
        "features",
        [],
    ):
        properties = feature.setdefault(
            "properties",
            {},
        )

        grid_id = str(
            properties.get(
                "grid_id",
                "",
            )
        )

        row = lookup.get(
            grid_id
        )

        if row is None:
            properties.update(
                {
                    "has_forecast": False,
                    "precipitation_mm": None,
                    "precipitation_label": short_label,
                    "fill_color": [
                        180,
                        180,
                        180,
                        25,
                    ],
                    "line_color": [
                        100,
                        100,
                        100,
                        45,
                    ],
                }
            )

            continue

        precipitation = safe_float(
            row[
                precipitation_column
            ]
        )

        properties.update(
            {
                "has_forecast": True,
                "grid_id": grid_id,
                "forecast_hour_utc": format_utc_hour(
                    row["forecast_hour"]
                ),
                "forecast_hour_nyc": format_local_hour(
                    row["forecast_hour"]
                ),
                "precipitation_label": short_label,
                "precipitation_mm": safe_round(
                    precipitation,
                    2,
                ),
                "precip_current_hour_mm": safe_round(
                    row["precip_current_hour_mm"],
                    2,
                ),
                "precip_previous_6h_mm": safe_round(
                    row["precip_previous_6h_mm"],
                    2,
                ),
                "daily_total_precip_mm": safe_round(
                    row["daily_total_precip_mm"],
                    2,
                ),
                "hrrr_cycle_id": str(
                    row["hrrr_cycle_id"]
                ),
                "hrrr_distance_km": safe_round(
                    row["hrrr_distance_km"],
                    2,
                ),
                "fill_color": precipitation_fill_color(
                    precipitation
                ),
                "line_color": [
                    70,
                    90,
                    110,
                    65,
                ],
            }
        )

    return geojson


# =====================================================================
# Operational summaries
# =====================================================================

def build_hourly_summary(
    forecast: pd.DataFrame,
) -> pd.DataFrame:
    """Create one operational summary row per published forecast hour."""

    summary = (
        forecast
        .groupby(
            "forecast_hour",
            as_index=False,
        )
        .agg(
            predicted_flood_grids=(
                "predicted_flood_event",
                "sum",
            ),
            max_current_hour_precip_mm=(
                "precip_current_hour_mm",
                "max",
            ),
            mean_current_hour_precip_mm=(
                "precip_current_hour_mm",
                "mean",
            ),
            max_previous_6h_mm=(
                "precip_previous_6h_mm",
                "max",
            ),
            mean_previous_6h_mm=(
                "precip_previous_6h_mm",
                "mean",
            ),
            max_centered_rolling24_mm=(
                "daily_total_precip_mm",
                "max",
            ),
            mean_centered_rolling24_mm=(
                "daily_total_precip_mm",
                "mean",
            ),
        )
        .sort_values(
            "forecast_hour"
        )
        .reset_index(drop=True)
    )

    # Keep the actual timestamp for plotting.
    # This preserves chronological order.
    summary["forecast_hour_nyc"] = (
        summary[
            "forecast_hour"
        ]
        .dt.tz_convert(
            NYC_TIMEZONE
        )
    )

    # These are display-only strings.
    summary["hour_utc"] = (
        summary[
            "forecast_hour"
        ]
        .dt.strftime(
            "%H:%M"
        )
    )

    summary["hour_nyc"] = (
        summary[
            "forecast_hour_nyc"
        ]
        .dt.strftime(
            "%I:%M %p"
        )
    )

    return summary


def build_daily_summary(
    forecast: pd.DataFrame,
) -> pd.DataFrame:
    """Collapse the published forecast horizon to one record per grid."""

    working = forecast.copy()

    working[
        "predicted_event_hour"
    ] = (
        working[
            "forecast_hour"
        ]
        .where(
            working[
                "predicted_flood_event"
            ]
        )
    )

    summary = (
        working
        .groupby(
            "grid_id",
            as_index=False,
        )
        .agg(
            any_flood_predicted=(
                "predicted_flood_event",
                "max",
            ),
            flood_event_hours=(
                "predicted_flood_event",
                "sum",
            ),
            first_predicted_event_hour=(
                "predicted_event_hour",
                "min",
            ),
            peak_hourly_precip_mm=(
                "precip_current_hour_mm",
                "max",
            ),
            peak_previous_6h_mm=(
                "precip_previous_6h_mm",
                "max",
            ),
            peak_centered_rolling24_precip_mm=(
                "daily_total_precip_mm",
                "max",
            ),
            support_sensor_model=(
                "support_sensor_model",
                "first",
            ),
            support_sensor_id=(
                "support_sensor_id",
                "first",
            ),
            is_imputed_grid=(
                "is_imputed_grid",
                "first",
            ),
            cluster_number=(
                "cluster_number",
                "first",
            ),
        )
    )

    return summary


# =====================================================================
# Maps
# =====================================================================


def flood_map(
    geojson: dict,
) -> pdk.Deck:
    """Build the synchronized flood prediction map."""

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill_color",
        get_line_color="properties.line_color",
        get_line_width=1,
        line_width_min_pixels=0.5,
    )

    tooltip = {
        "html": """
        <b>Grid:</b> {grid_id}<br/>
        <b>Status:</b> {forecast_status}<br/>
        <b>Flood susceptibility:</b>
        {flood_susceptibility_label}<br/>
        <b>Intensity level:</b>
        {flood_susceptibility_level} / 5<br/>
        <b>UTC:</b> {forecast_hour_utc}<br/>
        <b>NYC:</b> {forecast_hour_nyc}<br/>
        <hr/>
        <b>Selected model:</b> {support_sensor_model}<br/>
        <b>Native prediction type:</b>
        {native_prediction_type}<br/>
        <b>Native prediction value:</b>
        {native_prediction_value}<br/>
        <b>Operational threshold:</b>
        {classification_threshold}<br/>
        <b>Logistic flood-event probability:</b>
        {logistic_event_probability}<br/>
        <b>GCN predicted minutes above 1-inch target:</b>
        {gcn_predicted_minutes_above_1inch}<br/>
        <hr/>
        <b>Current-hour HRRR:</b> {precip_current_hour_mm} mm<br/>
        <b>Previous 6h HRRR:</b> {precip_previous_6h_mm} mm<br/>
        <b>Trailing 24h HRRR:</b> {daily_total_precip_mm} mm<br/>
        <hr/>
        <b>Support sensor:</b> {support_sensor_id}<br/>
        <b>Support sensor F1:</b> {support_sensor_f1}<br/>
        <b>Imputed grid:</b> {is_imputed_grid}<br/>
        <b>Imputation:</b> {imputation_method}<br/>
        <b>Support distance:</b> {support_sensor_distance_km} km
        """,
        "style": {
            "backgroundColor": "white",
            "color": "black",
            "fontSize": "12px",
        },
    }

    return pdk.Deck(
        layers=[
            layer,
        ],
        initial_view_state=pdk.ViewState(
            latitude=NYC_LATITUDE,
            longitude=NYC_LONGITUDE,
            zoom=9.6,
            pitch=0,
        ),
        tooltip=tooltip,
        map_style=(
            "mapbox://styles/mapbox/light-v10"
        ),
    )


def precipitation_map(
    geojson: dict,
) -> pdk.Deck:
    """Build the synchronized HRRR precipitation map."""

    layer = pdk.Layer(
        "GeoJsonLayer",
        data=geojson,
        pickable=True,
        stroked=True,
        filled=True,
        get_fill_color="properties.fill_color",
        get_line_color="properties.line_color",
        get_line_width=1,
        line_width_min_pixels=0.5,
    )

    tooltip = {
        "html": """
        <b>Grid:</b> {grid_id}<br/>
        <b>UTC:</b> {forecast_hour_utc}<br/>
        <b>NYC:</b> {forecast_hour_nyc}<br/>
        <hr/>
        <b>Displayed precipitation:</b>
        {precipitation_label}<br/>
        <b>Amount:</b> {precipitation_mm} mm<br/>
        <hr/>
        <b>Current hour:</b> {precip_current_hour_mm} mm<br/>
        <b>Previous 6h:</b> {precip_previous_6h_mm} mm<br/>
        <b>Trailing 24h:</b> {daily_total_precip_mm} mm<br/>
        <hr/>
        <b>HRRR cycle:</b> {hrrr_cycle_id}<br/>
        <b>Nearest HRRR point:</b> {hrrr_distance_km} km
        """,
        "style": {
            "backgroundColor": "white",
            "color": "black",
            "fontSize": "12px",
        },
    }

    return pdk.Deck(
        layers=[
            layer,
        ],
        initial_view_state=pdk.ViewState(
            latitude=NYC_LATITUDE,
            longitude=NYC_LONGITUDE,
            zoom=9.6,
            pitch=0,
        ),
        tooltip=tooltip,
        map_style=(
            "mapbox://styles/mapbox/light-v10"
        ),
    )


# =====================================================================
# Animation state
# =====================================================================


def initialize_animation_state(
    hour_count: int,
) -> None:
    """Initialize synchronized forecast-hour controls."""

    if (
        "forecast_hour_index"
        not in st.session_state
    ):
        st.session_state[
            "forecast_hour_index"
        ] = 0

    if (
        "forecast_animation_running"
        not in st.session_state
    ):
        st.session_state[
            "forecast_animation_running"
        ] = False

    if (
        st.session_state[
            "forecast_hour_index"
        ]
        >= hour_count
    ):
        st.session_state[
            "forecast_hour_index"
        ] = 0


# =====================================================================
# Main page
# =====================================================================


def forecasting_page() -> None:
    """Render the operational NYC flood forecasting dashboard."""

    st.title(
        "NYC Operational Flood Vulnerability Forecast"
    )

    st.caption(
        "Hourly NYC 1-km flood vulnerability predictions using "
        "HRRR forecast precipitation and dynamically selected "
        "sensor-trained models. Historical model training uses "
        "MRMS observations."
    )

    # -------------------------------------------------------------
    # Forecast product selector
    # -------------------------------------------------------------

    selector_left, selector_right = st.columns(
        [2, 3]
    )

    with selector_left:
        forecast_mode = st.radio(
            "Forecast product",
            options=[
                "today",
                "tomorrow",
            ],
            horizontal=True,
            format_func=lambda value: (
                "Today"
                if value == "today"
                else "Tomorrow"
            ),
            key="operational_forecast_mode",
        )

    if (
        st.session_state.get(
            "forecast_previous_mode"
        )
        != forecast_mode
    ):
        st.session_state[
            "forecast_hour_index"
        ] = 0

        st.session_state[
            "forecast_animation_running"
        ] = False

        st.session_state[
            "forecast_previous_mode"
        ] = forecast_mode

    (
        forecast_path,
        forecast_summary_path,
        precipitation_summary_path,
    ) = resolve_forecast_paths(
        forecast_mode
    )

    grid_path = DEFAULT_GRID_PATH

    # -------------------------------------------------------------
    # Input existence
    # -------------------------------------------------------------

    if not forecast_path.exists():
        st.warning(
            f"The operational **{forecast_mode}** forecast "
            "is not currently available."
        )

        st.caption(
            f"Expected artifact: `{forecast_path}`"
        )

        st.stop()

    if not grid_path.exists():
        st.error(
            "Authoritative 1-km grid geometry not found at "
            f"`{grid_path}`."
        )
        st.stop()

    # -------------------------------------------------------------
    # Load + validate
    # -------------------------------------------------------------

    try:
        forecast = load_forecast_data(
            str(forecast_path),
            forecast_path.stat().st_mtime,
        )

        grid_geojson = load_grid_geojson(
            str(grid_path),
            grid_path.stat().st_mtime,
        )

        validate_forecast_contract(
            forecast,
            grid_geojson,
        )

        forecast_summary_metadata = (
            load_optional_json(
                forecast_summary_path
            )
        )

        precipitation_summary_metadata = (
            load_optional_json(
                precipitation_summary_path
            )
        )

    except Exception as exc:
        st.error(
            f"Could not load operational forecast: {exc}"
        )
        st.stop()

    forecast_hours = [
        pd.Timestamp(value)
        for value in sorted(
            forecast[
                "forecast_hour"
            ].unique()
        )
    ]

    grid_count = int(
        forecast["grid_id"].nunique()
    )

    hour_count = len(
        forecast_hours
    )

    target_date = local_target_date(
        forecast
    )

    intended_hour_count = (
        requested_forecast_hours(
            precipitation_summary_metadata
        )
    )

    forecast_truncated = bool(
        precipitation_summary_metadata.get(
            "forecast_horizon_truncated",
            hour_count < intended_hour_count,
        )
    )

    available_through = (
        forecast_available_through(
            forecast,
            precipitation_summary_metadata,
        )
    )

    hrrr_cycles = (
        forecast[
            "hrrr_cycle_id"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    hrrr_cycle = (
        ", ".join(
            sorted(hrrr_cycles)
        )
        if hrrr_cycles
        else "Unavailable"
    )

    generated_values = pd.to_datetime(
        forecast[
            "forecast_generated_at"
        ],
        utc=True,
        errors="coerce",
    ).dropna()

    generated_at = (
        generated_values.max()
        if not generated_values.empty
        else None
    )

    # -------------------------------------------------------------
    # Forecast-horizon status
    # -------------------------------------------------------------

    with selector_right:
        st.markdown(
            f"**Target date:** {target_date}"
        )

        st.markdown(
            "**Forecast available through:** "
            f"{available_through.strftime('%I:%M %p %Z')}"
        )

        st.caption(
            f"{hour_count} of "
            f"{intended_hour_count} requested forecast hours "
            "currently published."
        )

    if forecast_truncated:
        st.info(
            "This forecast currently has an **adaptive horizon**. "
            f"Operational predictions are available through "
            f"**{available_through.strftime('%I:%M %p %Z')}** "
            f"on {available_through.strftime('%A, %B %d')}. "
            "Hours after that time are unavailable—not "
            "zero-risk forecasts."
        )
    else:
        st.success(
            "The complete requested operational forecast horizon "
            "is currently available."
        )

    # -----------------------------------------------------------------
    # Daily metrics
    # -----------------------------------------------------------------

    daily_summary = build_daily_summary(
        forecast
    )

    unique_flood_grids = int(
        daily_summary[
            "any_flood_predicted"
        ].sum()
    )

    total_event_rows = int(
        forecast[
            "predicted_flood_event"
        ].sum()
    )

    hourly_summary = build_hourly_summary(
        forecast
    )

    peak_hour_row = (
        hourly_summary
        .sort_values(
            "predicted_flood_grids",
            ascending=False,
        )
        .iloc[0]
    )

    # -----------------------------------------------------------------
    # Header status
    # -----------------------------------------------------------------

    metric_one, metric_two, metric_three, metric_four = (
        st.columns(4)
    )
    metric_one.metric(
    "Forecast product",
    (
        "Today"
        if forecast_mode == "today"
        else "Tomorrow"
    ),
    )

    metric_two.metric(
        "Available hours",
        f"{hour_count} / {intended_hour_count}",
    )

    metric_three.metric(
        "Affected grids",
        f"{unique_flood_grids:,}",
    )

    metric_four.metric(
        "Predicted grid-hours",
        f"{total_event_rows:,}",
    )

    st.caption(
        f"HRRR cycle: **{hrrr_cycle}**"
        + (
            " · Forecast generated: "
            f"**{format_utc_hour(generated_at)}**"
            if generated_at is not None
            else ""
        )
    )



    # -----------------------------------------------------------------
    # Timeline
    # -----------------------------------------------------------------

    st.markdown(
        "### Available forecast timeline"
    )

    timeline_left, timeline_right = (
        st.columns(2)
    )

    with timeline_left:
        st.caption(
            "Predicted flood susceptible grids by NYC-local forecast hour"
        )

        chart_data = (
            hourly_summary[
                [
                    "forecast_hour_nyc",
                    "predicted_flood_grids",
                ]
            ]
            .sort_values(
                "forecast_hour_nyc"
            )
            .set_index(
                "forecast_hour_nyc"
            )
        )

        st.bar_chart(
            chart_data,
            use_container_width=True,
        )


    with timeline_right:
        st.caption(
            "Maximum HRRR precipitation context by NYC-local hour"
        )

        rain_chart_data = (
            hourly_summary[
                [
                    "forecast_hour_nyc",
                    "max_current_hour_precip_mm",
                    "max_previous_6h_mm",
                    "max_centered_rolling24_mm",
                ]
            ]
            .sort_values(
                "forecast_hour_nyc"
            )
            .rename(
                columns={
                    "max_current_hour_precip_mm":
                        "Current hour",
                    "max_previous_6h_mm":
                        "Previous 6h",
                    "max_centered_rolling24_mm":
                        "Trailing 24h",
                }
            )
            .set_index(
                "forecast_hour_nyc"
            )
        )

        st.line_chart(
            rain_chart_data,
            use_container_width=True,
        )

    st.caption(
        "Peak affected-grid hour: "
        f"**{format_local_hour(peak_hour_row['forecast_hour'])}** "
        f"with **{int(peak_hour_row['predicted_flood_grids']):,}** "
        "predicted flood grids."
    )


    # -----------------------------------------------------------------
    # Shared synchronized time controls
    # -----------------------------------------------------------------

    initialize_animation_state(
        hour_count
    )

    # -------------------------------------------------------------
    # Animation advance
    #
    # IMPORTANT:
    # forecast_hour_index must be modified BEFORE the slider using
    # that key is instantiated.
    # -------------------------------------------------------------

    if st.session_state[
        "forecast_animation_running"
    ]:
        current_index = int(
            st.session_state[
                "forecast_hour_index"
            ]
        )

        if current_index >= hour_count - 1:
            st.session_state[
                "forecast_animation_running"
            ] = False

        else:
            st.session_state[
                "forecast_hour_index"
            ] = (
                current_index + 1
            )

    st.markdown(
        "### Explore the forecast by hour"
    )

    control_left, control_middle, control_right = (
        st.columns(
            [
                1,
                2,
                1,
            ]
        )
    )

    with control_left:
        previous_clicked = st.button(
            "◀ Previous",
            use_container_width=True,
            disabled=(
                st.session_state[
                    "forecast_hour_index"
                ]
                <= 0
            ),
        )

        if previous_clicked:
            st.session_state[
                "forecast_animation_running"
            ] = False

            st.session_state[
                "forecast_hour_index"
            ] = max(
                0,
                st.session_state[
                    "forecast_hour_index"
                ]
                - 1,
            )

    with control_middle:
        if st.session_state[
            "forecast_animation_running"
        ]:
            if st.button(
                "⏸ Pause animation",
                use_container_width=True,
            ):
                st.session_state[
                    "forecast_animation_running"
                ] = False

        else:
            if st.button(
                "▶ Play forecast animation",
                use_container_width=True,
            ):
                st.session_state[
                    "forecast_animation_running"
                ] = True

                st.rerun()

    with control_right:
        next_clicked = st.button(
            "Next ▶",
            use_container_width=True,
            disabled=(
                st.session_state[
                    "forecast_hour_index"
                ]
                >= hour_count - 1
            ),
        )

        if next_clicked:
            st.session_state[
                "forecast_animation_running"
            ] = False

            st.session_state[
                "forecast_hour_index"
            ] = min(
                hour_count - 1,
                st.session_state[
                    "forecast_hour_index"
                ]
                + 1,
            )

    st.slider(
        "Forecast hour",
        min_value=0,
        max_value=hour_count - 1,
        step=1,
        key="forecast_hour_index",
    )

    selected_index = int(
        st.session_state[
            "forecast_hour_index"
        ]
    )

    selected_hour = (
        forecast_hours[
            selected_index
        ]
    )

    st.caption(
        "Selected forecast hour: "
        f"**{format_local_hour(selected_hour)}** "
        f"({format_utc_hour(selected_hour)})"
    )

    hour_dataframe = (
        forecast.loc[
            forecast[
                "forecast_hour"
            ]
            == selected_hour
        ]
        .copy()
        .sort_values(
            "grid_id"
        )
    )

    st.markdown(
        f"#### {format_local_hour(selected_hour)}"
    )

    st.caption(
        f"Operational UTC hour: **{format_utc_hour(selected_hour)}**"
    )

    # -----------------------------------------------------------------
    # Selected-hour metrics
    # -----------------------------------------------------------------

    selected_flood_count = int(
        hour_dataframe[
            "predicted_flood_event"
        ].sum()
    )

    logistic_events = int(
        hour_dataframe.loc[
            hour_dataframe[
                "support_sensor_model"
            ]
            == LOGISTIC_MODEL_NAME,
            "predicted_flood_event",
        ].sum()
    )

    gcn_events = int(
        hour_dataframe.loc[
            hour_dataframe[
                "support_sensor_model"
            ]
            == GCN_MODEL_NAME,
            "predicted_flood_event",
        ].sum()
    )

    max_hourly_rain = float(
        hour_dataframe[
            "precip_current_hour_mm"
        ].max()
    )

    max_previous_6h_rain = float(
        hour_dataframe[
            "precip_previous_6h_mm"
        ].max()
    )

    max_rolling24_rain = float(
        hour_dataframe[
            "daily_total_precip_mm"
        ].max()
    )

    (
        selected_one,
        selected_two,
        selected_three,
        selected_four,
        selected_five,
        selected_six,
    ) = st.columns(6)

    selected_one.metric(
        "Flood grids",
        f"{selected_flood_count:,}",
    )

    selected_two.metric(
        "Logistic events",
        f"{logistic_events:,}",
    )

    selected_three.metric(
        "GCN events",
        f"{gcn_events:,}",
    )

    selected_four.metric(
        "Current-hour rain",
        f"{max_hourly_rain:.2f} mm",
    )

    selected_five.metric(
        "Previous 6h",
        f"{max_previous_6h_rain:.2f} mm",
    )

    selected_six.metric(
        "Rolling 24h",
        f"{max_rolling24_rain:.2f} mm",
    )

    # -----------------------------------------------------------------
    # Precipitation selector
    # -----------------------------------------------------------------

    precipitation_label = st.selectbox(
        "Precipitation map variable",
        options=list(
            PRECIPITATION_OPTIONS.keys()
        ),
        index=0,
        help=(
            "All three operational predictors are HRRR forecast "
            "precipitation. Current-hour precipitation describes hour H; "
            "previous-6-hour precipitation summarizes H-6 through H-1; "
            "the trailing 24-hour context summarizes H-23 through H "
            "through H+11 and therefore changes as the selected "
            "forecast hour changes."
        ),
    )

    precipitation_column = (
        PRECIPITATION_OPTIONS[
            precipitation_label
        ]
    )

    precip_values = pd.to_numeric(
        hour_dataframe[
            precipitation_column
        ],
        errors="coerce",
    )

    precip_max = float(
        precip_values.max()
    )

    precip_mean = float(
        precip_values.mean()
    )

    # -----------------------------------------------------------------
    # Build synchronized maps
    # -----------------------------------------------------------------

    flood_geojson = build_flood_geojson(
        grid_geojson,
        hour_dataframe,
    )

    precip_geojson = build_precipitation_geojson(
        grid_geojson,
        hour_dataframe,
        precipitation_column,
    )

    flood_column, precip_column = (
        st.columns(2)
    )

    with flood_column:
        st.markdown(
            "### Forecasted Flood-Susceptible Grids"
        )

        st.caption(
            "Color intensity represents forecasted flood "
            "susceptibility among grids classified as operational "
            "flood events. Logistic grids are graduated using "
            "flood-event probability; GCN grids are graduated using "
            "predicted minutes above the one-inch target."
        )

        st.pydeck_chart(
            flood_map(
                flood_geojson
            ),
            use_container_width=True,
        )

        st.caption(
            f"Predicted flood-susceptible grids at this hour: "
            f"**{selected_flood_count:,} / {grid_count:,}**"
        )

        st.markdown(
            """
**Forecasted Flood Susceptibility Intensity**

| Level | Interpretation | Logistic | GCN |
|---|---|---:|---:|
| **1** | Lower | 0.50–<0.60 | >1.0–1.5 min |
| **2** | Moderate | 0.60–<0.70 | >1.5–2.0 min |
| **3** | Elevated | 0.70–<0.80 | >2.0–2.5 min |
| **4** | High | 0.80–<0.90 | >2.5–3.0 min |
| **5** | Highest | ≥0.90 | >3.0 min |

*Blue/light grids are below the operational event threshold.
The Level 1–5 scale is a visualization layer and does not
change the underlying model classification.*
            """
        )

    with precip_column:
        st.markdown(
            "### HRRR precipitation forecast"
        )

        st.caption(
            f"Displayed variable: **{precipitation_label}**"
        )

        st.pydeck_chart(
            precipitation_map(
                precip_geojson
            ),
            use_container_width=True,
        )

        precip_metric_one, precip_metric_two = (
            st.columns(2)
        )

        precip_metric_one.metric(
            "Maximum",
            f"{precip_max:.2f} mm",
        )

        precip_metric_two.metric(
            "Mean",
            f"{precip_mean:.2f} mm",
        )

        st.caption(
            "Precipitation color guide: "
            "very light = dry/trace · blue = light/moderate · "
            "purple = heavy · dark red = very heavy."
        )

    # -----------------------------------------------------------------
    # Selected-hour detail
    # -----------------------------------------------------------------

    with st.expander(
        "Selected-hour grid details",
        expanded=False,
    ):
        detail_columns = [
            "grid_id",
            "predicted_flood_event",
            "support_sensor_model",
            "native_prediction_type",
            "native_prediction_value",
            "classification_threshold",
            "logistic_event_probability",
            "gcn_predicted_minutes_above_1inch",
            "precip_current_hour_mm",
            "precip_previous_6h_mm",
            "daily_total_precip_mm",
            "is_imputed_grid",
            "imputation_method",
            "support_sensor_id",
            "support_sensor_distance_km",
            "cluster_number",
        ]

        st.dataframe(
            hour_dataframe[
                detail_columns
            ].sort_values(
                [
                    "predicted_flood_event",
                    "precip_current_hour_mm",
                ],
                ascending=[
                    False,
                    False,
                ],
            ),
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------------------
    # Daily summary
    # -----------------------------------------------------------------

    with st.expander(
        "Day-ahead grid summary",
        expanded=False,
    ):
        display_daily = (
            daily_summary.copy()
        )

        display_daily[
            "first_predicted_event_hour"
        ] = (
            display_daily[
                "first_predicted_event_hour"
            ]
            .apply(
                lambda value: (
                    format_utc_hour(
                        pd.Timestamp(value)
                    )
                    if pd.notna(value)
                    else None
                )
            )
        )

        st.dataframe(
            display_daily.sort_values(
                [
                    "any_flood_predicted",
                    "flood_event_hours",
                    "peak_hourly_precip_mm",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            ),
            use_container_width=True,
            hide_index=True,
        )

    # -----------------------------------------------------------------
    # Provenance
    # -----------------------------------------------------------------

    with st.expander(
        "Forecast methodology and provenance",
        expanded=False,
    ):
        st.markdown(
            """
**Operational geography**

The forecast covers the authoritative NYC 1-km grid.

**Historical model training**

Sensor models are trained and evaluated using MRMS precipitation
observations.

**Operational meteorology**

Operational rainfall predictors are sourced from HRRR forecast data.
Each target 1-km grid uses its nearest HRRR forecast point.

**Operational precipitation features**

Each forecast grid/hour receives three HRRR predictors:

- `precip_current_hour_mm`: precipitation at forecast hour H.
- `precip_previous_6h_mm`: precipitation during H-6 through H-1.
- `daily_total_precip_mm`: retained as a legacy model column name,
  but operationally represents the trailing 24-hour
  precipitation context H-23 through H.

The trailing 24-hour value may therefore change from hour to hour.

**Support sensors and imputation**

Each grid is routed through the selected model associated with its
support sensor. A grid without an eligible in-grid sensor may use an
imputed support sensor according to the established spatial hierarchy.

The support sensor supplies the model relationship. HRRR precipitation
still belongs to the target grid; precipitation is not copied from the
support-sensor location.

**Logistic model**

`logistic_event_probability` is an event probability. The operational
classification threshold is stored in `classification_threshold`.

**GCN model**

`gcn_predicted_minutes_above_1inch` is predicted minutes above the
one-inch flood target. It is not a probability.

**Common event indicator**

`predicted_flood_event` is the authoritative operational event flag
across both model families.

**Interpretation**

These are model predictions, not confirmed or observed flood events.
HRRR precipitation values shown here are forecasts, not measured rain.
            """
        )

    # -----------------------------------------------------------------
    # Operational health
    # -----------------------------------------------------------------

    with st.expander(
        "Operational forecast health",
        expanded=False,
    ):
        health = {
            "forecast_rows": len(
                forecast
            ),
            "grid_cells": grid_count,
            "forecast_hours": hour_count,
            "expected_rows_for_published_horizon": (
                grid_count * hour_count
            ),
            "requested_forecast_hours": (
                intended_hour_count
            ),
            "forecast_horizon_truncated": (
                forecast_truncated
            ),
            "forecast_available_through_nyc": (
                available_through.isoformat()
            ),
            "forecast_mode": forecast_mode,
            "duplicate_grid_hours": int(
                forecast.duplicated(
                    [
                        "grid_id",
                        "forecast_hour",
                    ]
                ).sum()
            ),
            "support_sensors": int(
                forecast[
                    "support_sensor_id"
                ].nunique()
            ),
            "imputed_rows": int(
                forecast[
                    "is_imputed_grid"
                ].sum()
            ),
            "direct_rows": int(
                (
                    ~forecast[
                        "is_imputed_grid"
                    ]
                ).sum()
            ),
            "logistic_rows": int(
                (
                    forecast[
                        "support_sensor_model"
                    ]
                    == LOGISTIC_MODEL_NAME
                ).sum()
            ),
            "gcn_rows": int(
                (
                    forecast[
                        "support_sensor_model"
                    ]
                    == GCN_MODEL_NAME
                ).sum()
            ),
            "predicted_flood_grid_hours": (
                total_event_rows
            ),
            "unique_predicted_flood_grids": (
                unique_flood_grids
            ),
            "hrrr_cycle": hrrr_cycle,
            "target_date": str(
                target_date
            ),
        }

        st.json(
            health
        )

    # -----------------------------------------------------------------
    # Continue animation
    # -----------------------------------------------------------------

    if st.session_state[
        "forecast_animation_running"
    ]:
        time.sleep(
            1.25
        )

        st.rerun()