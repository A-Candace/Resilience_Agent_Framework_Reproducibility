"""
Sensor-to-MRMS grid mapping utilities.

Purpose
-------
Map each FloodNet sensor to the nearest MRMS radar grid cell.

Existing mappings are preserved whenever possible so that historical
and future precipitation features remain spatially consistent.

Newly deployed FloodNet sensors are assigned to their nearest valid
MRMS grid point.

The resulting mapping can be persisted to:

    s3://<bucket>/mlops/flood/reference/
        sensor_mrms_grid_map.parquet

Expected mapping schema
-----------------------
deployment_id
sensor_lat
sensor_lon
mrms_lat
mrms_lon
distance_km
mapping_source
mapping_version
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import boto3
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_S3_BUCKET = os.getenv(
    "DATA_S3_BUCKET",
    "nyc-resilience-data",
)

GRID_MAP_KEY = os.getenv(
    "FLOOD_SENSOR_MRMS_MAP_KEY",
    (
        "mlops/flood/reference/"
        "sensor_mrms_grid_map.parquet"
    ),
)

MAPPING_VERSION = os.getenv(
    "FLOOD_SENSOR_MRMS_MAPPING_VERSION",
    "v1",
)


# ============================================================
# COLUMN DEFINITIONS
# ============================================================

SENSOR_ID = "deployment_id"

SENSOR_LAT = "sensor_lat"

SENSOR_LON = "sensor_lon"

MRMS_LAT = "mrms_lat"

MRMS_LON = "mrms_lon"


REQUIRED_SENSOR_COLUMNS = {
    SENSOR_ID,
    SENSOR_LAT,
    SENSOR_LON,
}

REQUIRED_GRID_COLUMNS = {
    MRMS_LAT,
    MRMS_LON,
}


# ============================================================
# DATA CONTAINER
# ============================================================

@dataclass(frozen=True)
class SensorGridAssignment:
    """
    One FloodNet sensor → MRMS grid-cell assignment.
    """

    deployment_id: str

    sensor_lat: float

    sensor_lon: float

    mrms_lat: float

    mrms_lon: float

    distance_km: float

    mapping_source: str

    mapping_version: str


# ============================================================
# DISTANCE
# ============================================================

def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Great-circle distance between two latitude/longitude points.

    Returns
    -------
    float
        Distance in kilometers.
    """

    earth_radius_km = 6371.0088

    phi1 = math.radians(
        float(lat1)
    )

    phi2 = math.radians(
        float(lat2)
    )

    delta_phi = math.radians(
        float(lat2) - float(lat1)
    )

    delta_lambda = math.radians(
        float(lon2) - float(lon1)
    )

    a = (
        math.sin(
            delta_phi / 2.0
        ) ** 2
        +
        math.cos(phi1)
        * math.cos(phi2)
        * math.sin(
            delta_lambda / 2.0
        ) ** 2
    )

    c = (
        2.0
        * math.atan2(
            math.sqrt(a),
            math.sqrt(1.0 - a),
        )
    )

    return (
        earth_radius_km
        * c
    )


# ============================================================
# SENSOR VALIDATION
# ============================================================

def validate_sensor_inventory(
    sensors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and normalize FloodNet sensor metadata.
    """

    missing = (
        REQUIRED_SENSOR_COLUMNS
        - set(sensors.columns)
    )

    if missing:
        raise ValueError(
            "Sensor inventory is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    result = sensors[
        [
            SENSOR_ID,
            SENSOR_LAT,
            SENSOR_LON,
        ]
    ].copy()

    result[SENSOR_ID] = (
        result[SENSOR_ID]
        .astype(str)
    )

    result[SENSOR_LAT] = pd.to_numeric(
        result[SENSOR_LAT],
        errors="coerce",
    )

    result[SENSOR_LON] = pd.to_numeric(
        result[SENSOR_LON],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            SENSOR_ID,
            SENSOR_LAT,
            SENSOR_LON,
        ]
    )

    result = (
        result.drop_duplicates(
            subset=[
                SENSOR_ID,
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# MRMS GRID VALIDATION
# ============================================================

def validate_mrms_grid(
    grid: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and normalize a collection of MRMS grid points.

    The grid should contain unique latitude/longitude pairs.
    """

    missing = (
        REQUIRED_GRID_COLUMNS
        - set(grid.columns)
    )

    if missing:
        raise ValueError(
            "MRMS grid is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    result = grid[
        [
            MRMS_LAT,
            MRMS_LON,
        ]
    ].copy()

    result[MRMS_LAT] = pd.to_numeric(
        result[MRMS_LAT],
        errors="coerce",
    )

    result[MRMS_LON] = pd.to_numeric(
        result[MRMS_LON],
        errors="coerce",
    )

    result = result.dropna(
        subset=[
            MRMS_LAT,
            MRMS_LON,
        ]
    )

    result = (
        result.drop_duplicates(
            subset=[
                MRMS_LAT,
                MRMS_LON,
            ]
        )
        .sort_values(
            [
                MRMS_LAT,
                MRMS_LON,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if result.empty:
        raise ValueError(
            "MRMS grid contains no valid points."
        )

    return result


# ============================================================
# EXISTING MAPPING
# ============================================================

def mapping_s3_uri() -> str:
    """
    Return the configured mapping S3 URI.
    """

    return (
        f"s3://{DATA_S3_BUCKET}/"
        f"{GRID_MAP_KEY}"
    )


def mapping_exists() -> bool:
    """
    Check whether a persisted mapping already exists in S3.
    """

    client = boto3.client(
        "s3"
    )

    try:
        client.head_object(
            Bucket=DATA_S3_BUCKET,
            Key=GRID_MAP_KEY,
        )

        return True

    except client.exceptions.ClientError as exc:

        code = (
            exc.response
            .get("Error", {})
            .get("Code")
        )

        if code in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return False

        raise


def load_existing_mapping() -> pd.DataFrame:
    """
    Load the existing mapping from S3.

    Returns an empty DataFrame when no mapping exists yet.
    """

    if not mapping_exists():

        return pd.DataFrame(
            columns=[
                SENSOR_ID,
                SENSOR_LAT,
                SENSOR_LON,
                MRMS_LAT,
                MRMS_LON,
                "distance_km",
                "mapping_source",
                "mapping_version",
            ]
        )

    mapping = pd.read_parquet(
        mapping_s3_uri()
    )

    return mapping


# ============================================================
# EXISTING-MAPPING ELIGIBILITY
# ============================================================

def reusable_existing_mapping(
    sensor: pd.Series,
    existing_mapping: pd.DataFrame,
) -> pd.Series | None:
    """
    Return an existing mapping for a sensor when available.

    Existing sensor assignments are intentionally retained to
    preserve continuity with historical model predictors.

    We do not require exact equality between historical and current
    sensor coordinates because metadata may contain tiny floating-
    point differences.
    """

    if existing_mapping.empty:
        return None

    matches = existing_mapping[
        existing_mapping[SENSOR_ID].astype(str)
        ==
        str(sensor[SENSOR_ID])
    ]

    if matches.empty:
        return None

    row = matches.iloc[-1]

    required_values = [
        row.get(MRMS_LAT),
        row.get(MRMS_LON),
    ]

    if any(
        pd.isna(value)
        for value in required_values
    ):
        return None

    return row


# ============================================================
# NEAREST MRMS POINT
# ============================================================

def find_nearest_mrms_point(
    sensor_lat: float,
    sensor_lon: float,
    grid: pd.DataFrame,
) -> tuple[float, float, float]:
    """
    Find the nearest available MRMS grid point.

    The vectorized calculation is intentionally lightweight because
    the NYC historical grid contains only a few thousand points.
    """

    lat1 = np.radians(
        float(sensor_lat)
    )

    lon1 = np.radians(
        float(sensor_lon)
    )

    lat2 = np.radians(
        grid[MRMS_LAT]
        .to_numpy(
            dtype=float
        )
    )

    lon2 = np.radians(
        grid[MRMS_LON]
        .to_numpy(
            dtype=float
        )
    )

    delta_lat = (
        lat2
        - lat1
    )

    delta_lon = (
        lon2
        - lon1
    )

    a = (
        np.sin(
            delta_lat / 2.0
        ) ** 2
        +
        np.cos(lat1)
        * np.cos(lat2)
        * np.sin(
            delta_lon / 2.0
        ) ** 2
    )

    central_angle = (
        2.0
        * np.arctan2(
            np.sqrt(a),
            np.sqrt(
                1.0 - a
            ),
        )
    )

    distances = (
        6371.0088
        * central_angle
    )

    nearest_index = int(
        np.argmin(
            distances
        )
    )

    nearest = grid.iloc[
        nearest_index
    ]

    return (
        float(
            nearest[MRMS_LAT]
        ),
        float(
            nearest[MRMS_LON]
        ),
        float(
            distances[
                nearest_index
            ]
        ),
    )


# ============================================================
# ASSIGN ONE SENSOR
# ============================================================

def assign_sensor(
    sensor: pd.Series,
    grid: pd.DataFrame,
    existing_mapping: pd.DataFrame,
) -> SensorGridAssignment:
    """
    Assign one FloodNet sensor to an MRMS grid cell.
    """

    deployment_id = str(
        sensor[SENSOR_ID]
    )

    sensor_lat = float(
        sensor[SENSOR_LAT]
    )

    sensor_lon = float(
        sensor[SENSOR_LON]
    )

    existing = (
        reusable_existing_mapping(
            sensor,
            existing_mapping,
        )
    )

    # --------------------------------------------------------
    # Reuse historical assignment
    # --------------------------------------------------------

    if existing is not None:

        mrms_lat = float(
            existing[MRMS_LAT]
        )

        mrms_lon = float(
            existing[MRMS_LON]
        )

        distance = haversine_km(
            sensor_lat,
            sensor_lon,
            mrms_lat,
            mrms_lon,
        )

        return SensorGridAssignment(
            deployment_id=deployment_id,
            sensor_lat=sensor_lat,
            sensor_lon=sensor_lon,
            mrms_lat=mrms_lat,
            mrms_lon=mrms_lon,
            distance_km=distance,
            mapping_source=(
                "existing"
            ),
            mapping_version=(
                MAPPING_VERSION
            ),
        )

    # --------------------------------------------------------
    # New sensor
    # --------------------------------------------------------

    (
        mrms_lat,
        mrms_lon,
        distance,
    ) = find_nearest_mrms_point(
        sensor_lat,
        sensor_lon,
        grid,
    )

    return SensorGridAssignment(
        deployment_id=deployment_id,
        sensor_lat=sensor_lat,
        sensor_lon=sensor_lon,
        mrms_lat=mrms_lat,
        mrms_lon=mrms_lon,
        distance_km=distance,
        mapping_source="nearest",
        mapping_version=(
            MAPPING_VERSION
        ),
    )


# ============================================================
# BUILD COMPLETE MAPPING
# ============================================================

def build_sensor_grid_mapping(
    sensors: pd.DataFrame,
    mrms_grid: pd.DataFrame,
    existing_mapping: (
        pd.DataFrame
        | None
    ) = None,
) -> pd.DataFrame:
    """
    Build the complete FloodNet → MRMS grid mapping.

    Existing sensor mappings are reused. New sensors are assigned
    to the nearest MRMS grid point.
    """

    sensors = (
        validate_sensor_inventory(
            sensors
        )
    )

    mrms_grid = (
        validate_mrms_grid(
            mrms_grid
        )
    )

    if existing_mapping is None:

        existing_mapping = (
            load_existing_mapping()
        )

    assignments = []

    for _, sensor in (
        sensors.iterrows()
    ):

        assignment = assign_sensor(
            sensor,
            mrms_grid,
            existing_mapping,
        )

        assignments.append(
            {
                SENSOR_ID: (
                    assignment.deployment_id
                ),
                SENSOR_LAT: (
                    assignment.sensor_lat
                ),
                SENSOR_LON: (
                    assignment.sensor_lon
                ),
                MRMS_LAT: (
                    assignment.mrms_lat
                ),
                MRMS_LON: (
                    assignment.mrms_lon
                ),
                "distance_km": (
                    assignment.distance_km
                ),
                "mapping_source": (
                    assignment.mapping_source
                ),
                "mapping_version": (
                    assignment.mapping_version
                ),
            }
        )

    result = pd.DataFrame(
        assignments
    )

    result = (
        result.sort_values(
            SENSOR_ID
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# UNIQUE RADAR CELLS
# ============================================================

def unique_required_mrms_cells(
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the unique MRMS cells required by all mapped sensors.

    This is the small collection of radar locations the ingestion
    pipeline actually needs to retain.
    """

    required = mapping[
        [
            MRMS_LAT,
            MRMS_LON,
        ]
    ].copy()

    required = (
        required.drop_duplicates()
        .sort_values(
            [
                MRMS_LAT,
                MRMS_LON,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return required


# ============================================================
# SAVE MAPPING
# ============================================================

def save_mapping_to_s3(
    mapping: pd.DataFrame,
) -> str:
    """
    Persist the canonical sensor → MRMS mapping to S3.
    """

    mapping.to_parquet(
        mapping_s3_uri(),
        index=False,
    )

    return mapping_s3_uri()


# ============================================================
# SUMMARY
# ============================================================

def mapping_summary(
    mapping: pd.DataFrame,
) -> dict:
    """
    Return basic mapping statistics.
    """

    if mapping.empty:

        return {
            "sensors": 0,
            "unique_mrms_cells": 0,
            "existing_assignments": 0,
            "new_assignments": 0,
            "mean_distance_km": None,
            "max_distance_km": None,
        }

    return {
        "sensors": int(
            mapping[
                SENSOR_ID
            ].nunique()
        ),

        "unique_mrms_cells": int(
            mapping[
                [
                    MRMS_LAT,
                    MRMS_LON,
                ]
            ]
            .drop_duplicates()
            .shape[0]
        ),

        "existing_assignments": int(
            (
                mapping[
                    "mapping_source"
                ]
                == "existing"
            ).sum()
        ),

        "new_assignments": int(
            (
                mapping[
                    "mapping_source"
                ]
                == "nearest"
            ).sum()
        ),

        "mean_distance_km": float(
            mapping[
                "distance_km"
            ].mean()
        ),

        "max_distance_km": float(
            mapping[
                "distance_km"
            ].max()
        ),
    }
