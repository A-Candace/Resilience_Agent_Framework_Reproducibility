from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd


# ============================================================
# S3 CONFIGURATION
# ============================================================

DATA_S3_BUCKET = os.getenv(
    "DATA_S3_BUCKET",
    "nyc-resilience-data",
)

BOOTSTRAP_PREFIX = os.getenv(
    "FLOOD_BOOTSTRAP_PREFIX",
    "mlops/flood/bootstrap",
)

MERGED_MRMS_FLOODNET_KEY = (
    f"{BOOTSTRAP_PREFIX}/merged_mrms_floodnet.parquet"
)

HOURLY_MODEL_KEY = (
    f"{BOOTSTRAP_PREFIX}/all_sensor_hourly_model_1mile.parquet"
)


def s3_uri(key: str) -> str:
    return f"s3://{DATA_S3_BUCKET}/{key}"


# ============================================================
# DATA CONTAINERS
# ============================================================

@dataclass
class FloodBootstrapData:
    """
    Historical bootstrap datasets used by the flood MLOps pipeline.
    """

    merged_sensor_radar: pd.DataFrame
    hourly_model_features: pd.DataFrame


# ============================================================
# MERGED SENSOR + RADAR DATA
# ============================================================

def load_merged_sensor_radar(
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load the historical minute-level merged FloodNet + MRMS dataset.

    Default location:
        s3://nyc-resilience-data/
        mlops/flood/bootstrap/
        merged_mrms_floodnet.parquet

    Parameters
    ----------
    columns:
        Optional list of columns to read.

        Supplying columns is strongly recommended because the
        historical merged dataset contains roughly 194 million rows.

    Returns
    -------
    pd.DataFrame
    """

    uri = s3_uri(
        MERGED_MRMS_FLOODNET_KEY
    )

    return pd.read_parquet(
        uri,
        columns=columns,
    )


# ============================================================
# HOURLY MODEL FEATURE TABLE
# ============================================================

def load_hourly_model_features(
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load the historical hourly model-ready feature table.

    This table currently contains all 334 historical sensors and
    spans approximately:

        2020-10-14 through 2026-03-13

    It includes precipitation features, sensor coordinates,
    response fields, and additional variables used by other
    historical experiments.

    Parameters
    ----------
    columns:
        Optional columns to load.

    Returns
    -------
    pd.DataFrame
    """

    uri = s3_uri(
        HOURLY_MODEL_KEY
    )

    return pd.read_parquet(
        uri,
        columns=columns,
    )


# ============================================================
# GCN-SPECIFIC HISTORICAL INPUT
# ============================================================

def load_gcn_hourly_features() -> pd.DataFrame:
    """
    Load only the columns required by the current precipitation-only
    GCN workflow.

    This prevents the GCN from unnecessarily loading historical
    variables that are not part of its current scientific design.
    """

    columns = [
        "deployment_id",
        "hour",
        "hourly_max_depth_mm",
        "response_observed",
        "precip_current_hour_mm",
        "precip_previous_6h_mm",
        "daily_total_precip_mm",
        "sensor_lat",
        "sensor_lon",
    ]

    df = load_hourly_model_features(
        columns=columns,
    )

    df["hour"] = pd.to_datetime(
        df["hour"],
        utc=True,
        errors="coerce",
    )

    return df


# ============================================================
# RAW RESPONSE INPUT
# ============================================================

def load_gcn_response_source() -> pd.DataFrame:
    """
    Load the minute-level FloodNet response fields used by the
    original GCN workflow to rebuild the hourly flood-duration
    response.

    Only the required columns are loaded because the full merged
    dataset is very large.
    """

    columns = [
        "deployment_id",
        "time",
        "depth_proc_mm",
    ]

    df = load_merged_sensor_radar(
        columns=columns,
    )

    df["time"] = pd.to_datetime(
        df["time"],
        utc=True,
        errors="coerce",
    )

    return df


# ============================================================
# FULL BOOTSTRAP LOAD
# ============================================================

def load_bootstrap_data() -> FloodBootstrapData:
    """
    Load both historical datasets.

    This function is primarily useful for diagnostics and pipeline
    setup. Training code should generally use the model-specific
    loader functions above to avoid loading unnecessary columns.
    """

    return FloodBootstrapData(
        merged_sensor_radar=(
            load_gcn_response_source()
        ),
        hourly_model_features=(
            load_gcn_hourly_features()
        ),
    )