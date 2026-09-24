"""
MRMS hourly precipitation ingestion for the NYC flood MLOps pipeline.

This module reads NOAA MRMS hourly QPE files, extracts only the
MRMS grid cells required by the FloodNet sensor network, and returns
hourly precipitation values suitable for downstream feature building.

IMPORTANT TIME CONVENTION
-------------------------
Historical model features are labeled by the START of the rainfall
hour.

Example:

    feature hour:
        2026-03-12 04:00 UTC

    represents precipitation during:
        04:00 <= time < 05:00

The NOAA MultiSensor_QPE_01H_Pass2 file representing that rainfall
window is timestamped at the END of the accumulation period:

    NOAA source hour:
        2026-03-12 05:00 UTC

Therefore:

    source_hour = feature_hour + 1 hour

The output remains labeled with feature_hour so that the new data is
compatible with the historical GCN training feature convention.

NOAA source
-----------
Bucket:
    s3://noaa-mrms-pds

Product:
    CONUS/MultiSensor_QPE_01H_Pass2_00.00/

Canonical FloodNet -> MRMS mapping:
    s3://nyc-resilience-data/mlops/flood/reference/
        sensor_mrms_grid_map.parquet
"""

from __future__ import annotations

import gzip
import os
import tempfile
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import xarray as xr

from botocore import UNSIGNED
from botocore.config import Config


# ============================================================
# CONFIGURATION
# ============================================================

DATA_BUCKET = os.getenv(
    "DATA_S3_BUCKET",
    "nyc-resilience-data",
)

SENSOR_MAP_KEY = os.getenv(
    "FLOOD_SENSOR_MRMS_MAP_KEY",
    "mlops/flood/reference/sensor_mrms_grid_map.parquet",
)

NOAA_MRMS_BUCKET = os.getenv(
    "NOAA_MRMS_BUCKET",
    "noaa-mrms-pds",
)

NOAA_MRMS_REGION = os.getenv(
    "NOAA_MRMS_REGION",
    "us-east-1",
)

MRMS_PRODUCT = (
    "MultiSensor_QPE_01H_Pass2_00.00"
)

MRMS_PREFIX = (
    f"CONUS/{MRMS_PRODUCT}"
)

SOURCE_HOUR_OFFSET = pd.Timedelta(
    hours=1
)


# ============================================================
# COLUMN DEFINITIONS
# ============================================================

SENSOR_ID = "deployment_id"

FEATURE_HOUR = "hour"

SOURCE_HOUR = "mrms_source_hour"

MRMS_LAT = "mrms_lat"

MRMS_LON = "mrms_lon"

PRECIP_COLUMN = "precip_current_hour_mm"

PRODUCT_COLUMN = "mrms_product"


# ============================================================
# S3 CLIENT
# ============================================================

def get_public_mrms_s3_client():
    """
    Anonymous client for NOAA's public MRMS bucket.
    """

    return boto3.client(
        "s3",
        region_name=NOAA_MRMS_REGION,
        config=Config(
            signature_version=UNSIGNED,
            retries={
                "max_attempts": 5,
                "mode": "standard",
            },
        ),
    )


# ============================================================
# TIME CONVENTION
# ============================================================

def normalize_feature_hour(
    hour,
) -> pd.Timestamp:
    """
    Normalize an ML feature hour to UTC.
    """

    return pd.to_datetime(
        hour,
        utc=True,
        errors="raise",
    ).floor("h")


def feature_hour_to_source_hour(
    feature_hour,
) -> pd.Timestamp:
    """
    Convert historical/model feature-hour convention to NOAA's
    hourly QPE source timestamp.

    Example
    -------
    feature hour:
        2026-03-12 04:00 UTC

    NOAA source:
        2026-03-12 05:00 UTC
    """

    feature_hour = (
        normalize_feature_hour(
            feature_hour
        )
    )

    return (
        feature_hour
        + SOURCE_HOUR_OFFSET
    )


# ============================================================
# LOAD SENSOR MAPPING
# ============================================================

def load_sensor_grid_mapping() -> pd.DataFrame:
    """
    Load the canonical FloodNet -> MRMS mapping.
    """

    uri = (
        f"s3://{DATA_BUCKET}/"
        f"{SENSOR_MAP_KEY}"
    )

    mapping = pd.read_parquet(
        uri
    )

    required = {
        SENSOR_ID,
        MRMS_LAT,
        MRMS_LON,
    }

    missing = (
        required
        - set(mapping.columns)
    )

    if missing:
        raise ValueError(
            "Sensor MRMS mapping missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    mapping = mapping.dropna(
        subset=[
            SENSOR_ID,
            MRMS_LAT,
            MRMS_LON,
        ]
    ).copy()

    mapping[SENSOR_ID] = (
        mapping[SENSOR_ID]
        .astype(str)
    )

    mapping[MRMS_LAT] = pd.to_numeric(
        mapping[MRMS_LAT],
        errors="coerce",
    )

    mapping[MRMS_LON] = pd.to_numeric(
        mapping[MRMS_LON],
        errors="coerce",
    )

    mapping = mapping.dropna(
        subset=[
            MRMS_LAT,
            MRMS_LON,
        ]
    )

    return mapping


def required_mrms_cells(
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return unique MRMS cells required by FloodNet.
    """

    return (
        mapping[
            [
                MRMS_LAT,
                MRMS_LON,
            ]
        ]
        .drop_duplicates()
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


# ============================================================
# NOAA OBJECT KEY
# ============================================================

def mrms_object_key(
    source_hour,
) -> str:
    """
    Construct the NOAA MRMS object key for one NOAA source hour.

    IMPORTANT:
    source_hour is the NOAA END-of-accumulation timestamp,
    not the historical feature-hour label.
    """

    timestamp = pd.to_datetime(
        source_hour,
        utc=True,
        errors="raise",
    ).floor("h")

    date_string = (
        timestamp.strftime(
            "%Y%m%d"
        )
    )

    timestamp_string = (
        timestamp.strftime(
            "%Y%m%d-%H0000"
        )
    )

    filename = (
        f"MRMS_{MRMS_PRODUCT}_"
        f"{timestamp_string}.grib2.gz"
    )

    return (
        f"{MRMS_PREFIX}/"
        f"{date_string}/"
        f"{filename}"
    )


# ============================================================
# OBJECT EXISTENCE
# ============================================================

def mrms_source_object_exists(
    source_hour,
) -> bool:
    """
    Check whether one NOAA MRMS source object exists.
    """

    key = mrms_object_key(
        source_hour
    )

    client = (
        get_public_mrms_s3_client()
    )

    try:
        client.head_object(
            Bucket=NOAA_MRMS_BUCKET,
            Key=key,
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


def mrms_feature_hour_exists(
    feature_hour,
) -> bool:
    """
    Check whether the NOAA object needed for a feature hour exists.
    """

    source_hour = (
        feature_hour_to_source_hour(
            feature_hour
        )
    )

    return mrms_source_object_exists(
        source_hour
    )


# ============================================================
# DOWNLOAD
# ============================================================

def download_mrms_source_object(
    source_hour,
) -> bytes:
    """
    Download one compressed NOAA MRMS hourly QPE object.
    """

    key = mrms_object_key(
        source_hour
    )

    client = (
        get_public_mrms_s3_client()
    )

    response = client.get_object(
        Bucket=NOAA_MRMS_BUCKET,
        Key=key,
    )

    return (
        response["Body"]
        .read()
    )


# Backward-compatible helper for earlier validation commands.
def download_mrms_object(
    source_hour,
) -> bytes:
    return download_mrms_source_object(
        source_hour
    )


# ============================================================
# OPEN GRIB2
# ============================================================

def open_grib2(
    compressed_bytes: bytes,
):
    """
    Decompress and load one MRMS GRIB2 object.
    """

    raw_grib = gzip.decompress(
        compressed_bytes
    )

    with tempfile.NamedTemporaryFile(
        suffix=".grib2",
        delete=False,
    ) as tmp:

        tmp.write(
            raw_grib
        )

        tmp_path = Path(
            tmp.name
        )

    try:

        dataset = xr.open_dataset(
            tmp_path,
            engine="cfgrib",
            backend_kwargs={
                "indexpath": "",
            },
        )

        dataset.load()

        return dataset

    finally:

        try:
            tmp_path.unlink()
        except OSError:
            pass


# ============================================================
# PRECIPITATION VARIABLE
# ============================================================

def find_precip_variable(
    dataset,
) -> str:
    """
    Identify the MRMS QPE variable.

    ecCodes currently exposes this MRMS-specific parameter as
    'unknown' on this environment, so a one-variable dataset is
    accepted.
    """

    variables = list(
        dataset.data_vars
    )

    if not variables:
        raise RuntimeError(
            "MRMS dataset contains no data variables."
        )

    if len(variables) == 1:
        return variables[0]

    preferred = [
        "precip",
        "qpe",
        "tp",
        "unknown",
    ]

    for token in preferred:

        for variable in variables:

            if token in variable.lower():
                return variable

    raise RuntimeError(
        "Could not identify MRMS precipitation variable. "
        f"Variables found: {variables}"
    )


# ============================================================
# GRID COORDINATES
# ============================================================

def get_coordinate_arrays(
    dataset,
) -> tuple[
    np.ndarray,
    np.ndarray,
]:
    """
    Extract latitude and longitude arrays.
    """

    latitude = None
    longitude = None

    for candidate in [
        "latitude",
        "lat",
    ]:

        if candidate in dataset.coords:

            latitude = np.asarray(
                dataset.coords[
                    candidate
                ].values,
                dtype=float,
            )

            break

    for candidate in [
        "longitude",
        "lon",
    ]:

        if candidate in dataset.coords:

            longitude = np.asarray(
                dataset.coords[
                    candidate
                ].values,
                dtype=float,
            )

            break

    if latitude is None:
        raise RuntimeError(
            "MRMS latitude coordinate not found."
        )

    if longitude is None:
        raise RuntimeError(
            "MRMS longitude coordinate not found."
        )

    return (
        latitude,
        longitude,
    )


# ============================================================
# NEAREST MRMS INDEX
# ============================================================

def nearest_grid_index(
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    target_lat: float,
    target_lon: float,
) -> tuple[int, int]:
    """
    Return nearest CONUS MRMS index for one target coordinate.
    """

    # --------------------------------------------------------
    # Regular 1-D lat/lon grid
    # --------------------------------------------------------

    if (
        latitude.ndim == 1
        and longitude.ndim == 1
    ):

        lat_index = int(
            np.abs(
                latitude
                - target_lat
            ).argmin()
        )

        lon_target = float(
            target_lon
        )

        # NOAA GRIB longitude is 0-360 while our historical
        # mapping uses conventional negative western longitude.
        if (
            np.nanmax(
                longitude
            )
            > 180
            and lon_target < 0
        ):
            lon_target = (
                lon_target
                % 360
            )

        lon_index = int(
            np.abs(
                longitude
                - lon_target
            ).argmin()
        )

        return (
            lat_index,
            lon_index,
        )

    # --------------------------------------------------------
    # General 2-D grid fallback
    # --------------------------------------------------------

    lon_target = float(
        target_lon
    )

    if (
        np.nanmax(
            longitude
        )
        > 180
        and lon_target < 0
    ):
        lon_target = (
            lon_target
            % 360
        )

    distance_squared = (
        (
            latitude
            - target_lat
        ) ** 2
        +
        (
            longitude
            - lon_target
        ) ** 2
    )

    flat_index = int(
        np.nanargmin(
            distance_squared
        )
    )

    return tuple(
        int(value)
        for value in np.unravel_index(
            flat_index,
            distance_squared.shape,
        )
    )


# ============================================================
# EXTRACT REQUIRED CELLS
# ============================================================

def extract_required_cells(
    dataset,
    cells: pd.DataFrame,
    *,
    feature_hour,
    source_hour,
) -> pd.DataFrame:
    """
    Extract hourly precipitation at required MRMS cells.

    Output is labeled using feature_hour, while source_hour is
    retained explicitly for provenance.
    """

    variable_name = (
        find_precip_variable(
            dataset
        )
    )

    field = np.asarray(
        dataset[
            variable_name
        ].values,
        dtype=float,
    )

    (
        latitude,
        longitude,
    ) = get_coordinate_arrays(
        dataset
    )

    feature_hour = (
        normalize_feature_hour(
            feature_hour
        )
    )

    source_hour = pd.to_datetime(
        source_hour,
        utc=True,
    ).floor("h")

    rows = []

    for _, cell in cells.iterrows():

        target_lat = float(
            cell[
                MRMS_LAT
            ]
        )

        target_lon = float(
            cell[
                MRMS_LON
            ]
        )

        index = nearest_grid_index(
            latitude,
            longitude,
            target_lat=target_lat,
            target_lon=target_lon,
        )

        value = float(
            field[
                index
            ]
        )

        rows.append(
            {
                FEATURE_HOUR: (
                    feature_hour
                ),
                SOURCE_HOUR: (
                    source_hour
                ),
                PRODUCT_COLUMN: (
                    MRMS_PRODUCT
                ),
                MRMS_LAT: (
                    target_lat
                ),
                MRMS_LON: (
                    target_lon
                ),
                PRECIP_COLUMN: (
                    value
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# MAP CELLS BACK TO SENSORS
# ============================================================

def allocate_precip_to_sensors(
    cell_precip: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Allocate each MRMS-cell precipitation value to every FloodNet
    sensor mapped to that cell.
    """

    result = mapping.merge(
        cell_precip,
        on=[
            MRMS_LAT,
            MRMS_LON,
        ],
        how="left",
        validate="many_to_one",
    )

    return result[
        [
            SENSOR_ID,
            FEATURE_HOUR,
            SOURCE_HOUR,
            PRODUCT_COLUMN,
            PRECIP_COLUMN,
            MRMS_LAT,
            MRMS_LON,
        ]
    ].copy()


# ============================================================
# PROCESS ONE FEATURE HOUR
# ============================================================

def process_hour(
    hour,
) -> pd.DataFrame:
    """
    Build MRMS precipitation for one MODEL FEATURE hour.

    Parameters
    ----------
    hour:
        Historical/model feature hour.

        For example:

            hour = 2026-03-12 04:00 UTC

        causes NOAA object:

            2026-03-12 05:00 UTC

        to be downloaded.

    Returns
    -------
    pd.DataFrame
        One row per currently mapped FloodNet sensor.

        The returned `hour` remains the requested model feature hour.
    """

    feature_hour = (
        normalize_feature_hour(
            hour
        )
    )

    source_hour = (
        feature_hour_to_source_hour(
            feature_hour
        )
    )

    key = mrms_object_key(
        source_hour
    )

    print(
        f"Feature hour: "
        f"{feature_hour}"
    )

    print(
        f"MRMS source hour: "
        f"{source_hour}"
    )

    print(
        "NOAA object:"
    )

    print(
        f"s3://{NOAA_MRMS_BUCKET}/{key}"
    )

    mapping = (
        load_sensor_grid_mapping()
    )

    cells = (
        required_mrms_cells(
            mapping
        )
    )

    print(
        f"FloodNet sensors: "
        f"{mapping[SENSOR_ID].nunique():,}"
    )

    print(
        f"Required MRMS cells: "
        f"{len(cells):,}"
    )

    compressed = (
        download_mrms_source_object(
            source_hour
        )
    )

    print(
        "Compressed object size: "
        f"{len(compressed) / 1_000_000:.2f} MB"
    )

    dataset = open_grib2(
        compressed
    )

    cell_precip = (
        extract_required_cells(
            dataset,
            cells,
            feature_hour=feature_hour,
            source_hour=source_hour,
        )
    )

    sensor_precip = (
        allocate_precip_to_sensors(
            cell_precip,
            mapping,
        )
    )

    return sensor_precip


# ============================================================
# SMOKE TEST
# ============================================================

def main() -> None:
    """
    Validate the feature-hour/source-hour convention.
    """

    feature_hour = pd.Timestamp(
        "2026-03-12 04:00:00",
        tz="UTC",
    )

    source_hour = (
        feature_hour_to_source_hour(
            feature_hour
        )
    )

    print(
        "MRMS FEATURE-HOUR EXTRACTION TEST"
    )

    print(
        "================================="
    )

    print(
        f"Feature hour: "
        f"{feature_hour}"
    )

    print(
        f"Expected NOAA source hour: "
        f"{source_hour}"
    )

    if not mrms_source_object_exists(
        source_hour
    ):

        raise RuntimeError(
            "Expected MRMS source object "
            "does not exist."
        )

    result = process_hour(
        feature_hour
    )

    print(
        "\n=== Result ==="
    )

    print(
        f"Rows: "
        f"{len(result):,}"
    )

    print(
        f"Sensors: "
        f"{result[SENSOR_ID].nunique():,}"
    )

    print(
        "Feature hour(s):"
    )

    print(
        result[
            FEATURE_HOUR
        ].unique()
    )

    print(
        "MRMS source hour(s):"
    )

    print(
        result[
            SOURCE_HOUR
        ].unique()
    )

    print(
        "Missing precipitation: "
        f"{result[PRECIP_COLUMN].isna().sum():,}"
    )

    print(
        "\nPrecipitation distribution:"
    )

    print(
        result[
            PRECIP_COLUMN
        ].describe()
    )

    print(
        "\nSample:"
    )

    print(
        result.head(
            10
        ).to_string(
            index=False
        )
    )

    print(
        "\nMRMS FEATURE-HOUR TEST COMPLETE"
    )


if __name__ == "__main__":
    main()