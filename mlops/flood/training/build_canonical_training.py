"""
Build the canonical hourly flood-training dataset.

This combines:

1. Predictor-side hourly features from:
   all_sensor_hourly_model_1mile.parquet

2. The validated 1-inch flood response reconstructed from:
   merged_mrms_floodnet.parquet

The resulting canonical dataset is intended to become the common
historical training source for GCN, Logistic Regression, and future
candidate models.

The original bootstrap parquet files are left unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

import boto3
import pandas as pd

from mlops.flood.shared.config import (
    FLOOD_ARTIFACT_DIR,
)

from mlops.flood.shared.s3_bootstrap import (
    load_hourly_model_features,
)

from mlops.flood.training.build_historical_response import (
    build_historical_response,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_S3_BUCKET = os.getenv(
    "DATA_S3_BUCKET",
    "nyc-resilience-data",
)

OUTPUT_KEY = os.getenv(
    "FLOOD_CANONICAL_TRAINING_KEY",
    (
        "mlops/flood/processed/"
        "flood_hourly_training_canonical_test.parquet"
    ),
)

LOCAL_OUTPUT = (
    FLOOD_ARTIFACT_DIR
    / "flood_hourly_training_canonical_test.parquet"
)


# ============================================================
# TEST RANGE
# ============================================================

# Keep this aligned with the current small-range historical
# response test before running a full historical build.

TEST_START = os.getenv(
    "FLOOD_CANONICAL_TEST_START",
    "2020-11-16",
)

TEST_END = os.getenv(
    "FLOOD_CANONICAL_TEST_END",
    "2020-11-17",
)


# ============================================================
# PREDICTOR COLUMNS
# ============================================================

# For now, retain the predictor-side columns needed by the
# precipitation-only GCN plus sensor coordinates.
#
# We deliberately do NOT reuse the old response columns from
# all_sensor_hourly_model_1mile.parquet.

PREDICTOR_COLUMNS = [
    "deployment_id",
    "hour",
    "precip_current_hour_mm",
    "precip_previous_6h_mm",
    "daily_total_precip_mm",
    "sensor_lat",
    "sensor_lon",
]


# ============================================================
# LOAD PREDICTORS
# ============================================================

def load_predictors_for_test_range() -> pd.DataFrame:
    """
    Load predictor-side hourly data from S3 for the configured
    test date range.
    """

    df = load_hourly_model_features(
        columns=PREDICTOR_COLUMNS,
    )

    df["hour"] = pd.to_datetime(
        df["hour"],
        utc=True,
        errors="coerce",
    )

    start_time = pd.Timestamp(
        TEST_START,
        tz="UTC",
    )

    end_time = pd.Timestamp(
        TEST_END,
        tz="UTC",
    )

    df = df[
        (df["hour"] >= start_time)
        &
        (df["hour"] < end_time)
    ].copy()

    df = df.dropna(
        subset=[
            "deployment_id",
            "hour",
            "sensor_lat",
            "sensor_lon",
        ]
    )

    df = (
        df.sort_values(
            [
                "deployment_id",
                "hour",
            ]
        )
        .drop_duplicates(
            subset=[
                "deployment_id",
                "hour",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    return df


# ============================================================
# BUILD RESPONSE
# ============================================================

def load_response_for_test_range() -> pd.DataFrame:
    """
    Build the validated 1-inch hourly response for the same
    configured test range.

    The underlying response builder reads from the large
    merged FloodNet/MRMS bootstrap parquet.
    """

    response = build_historical_response()

    return response[
        [
            "deployment_id",
            "hour",
            "observed_5min_bins",
            "valid_depth_5min_bins",
            "high_depth_5min_bins",
            "hourly_max_depth_mm",
            "response_observed",
            "minutes_above_1p5_inch",
        ]
    ].copy()


# ============================================================
# BUILD CANONICAL DATASET
# ============================================================

def build_canonical_training() -> pd.DataFrame:
    """
    Join predictor-side features with the validated 1-inch
    response.

    The join key is:

        deployment_id + hour
    """

    predictors = (
        load_predictors_for_test_range()
    )

    response = (
        load_response_for_test_range()
    )

    canonical = predictors.merge(
        response,
        on=[
            "deployment_id",
            "hour",
        ],
        how="left",
        validate="one_to_one",
    )

    canonical = canonical.sort_values(
        [
            "deployment_id",
            "hour",
        ]
    ).reset_index(drop=True)

    return canonical


# ============================================================
# VALIDATION
# ============================================================

def validate_canonical(
    df: pd.DataFrame,
) -> None:
    """
    Validate basic properties of the canonical training table.
    """

    required_columns = {
        "deployment_id",
        "hour",
        "precip_current_hour_mm",
        "precip_previous_6h_mm",
        "daily_total_precip_mm",
        "sensor_lat",
        "sensor_lon",
        "observed_5min_bins",
        "valid_depth_5min_bins",
        "high_depth_5min_bins",
        "hourly_max_depth_mm",
        "response_observed",
        "minutes_above_1p5_inch",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Canonical training dataset is missing columns: "
            + ", ".join(sorted(missing))
        )

    duplicates = df.duplicated(
        subset=[
            "deployment_id",
            "hour",
        ]
    ).sum()

    if duplicates:
        raise ValueError(
            "Canonical training dataset contains "
            f"{duplicates:,} duplicate sensor-hour rows."
        )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print validation information for the canonical dataset.
    """

    print(
        "\n=== Canonical Training Test Summary ==="
    )

    print(
        f"Rows: {len(df):,}"
    )

    print(
        "Sensors: "
        f"{df['deployment_id'].nunique():,}"
    )

    print(
        f"Start: {df['hour'].min()}"
    )

    print(
        f"End:   {df['hour'].max()}"
    )

    print(
        "Rows with observed response: "
        f"{int((df['response_observed'] == 1).sum()):,}"
    )

    print(
        "Rows with positive flood duration: "
        f"{int((df['minutes_above_1p5_inch'] > 0).sum()):,}"
    )

    print(
        "Rows missing reconstructed response: "
        f"{int(df['response_observed'].isna().sum()):,}"
    )

    print(
        "\nColumns:"
    )

    for column in df.columns:
        print(
            f"  - {column}"
        )


# ============================================================
# SAVE LOCALLY
# ============================================================

def save_local(
    df: pd.DataFrame,
) -> Path:
    """
    Save the canonical test dataset locally.
    """

    LOCAL_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        LOCAL_OUTPUT,
        index=False,
    )

    return LOCAL_OUTPUT


# ============================================================
# UPLOAD TO S3
# ============================================================

def upload_to_s3(
    local_path: Path,
) -> None:
    """
    Upload the canonical test dataset to S3.
    """

    client = boto3.client(
        "s3"
    )

    client.upload_file(
        str(local_path),
        DATA_S3_BUCKET,
        OUTPUT_KEY,
    )


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:
    print(
        "Building canonical flood training dataset "
        "for SMALL TEST RANGE..."
    )

    print(
        f"\nTest range:"
        f"\n{TEST_START} <= hour < {TEST_END}"
    )

    canonical = (
        build_canonical_training()
    )

    validate_canonical(
        canonical
    )

    print_summary(
        canonical
    )

    local_path = save_local(
        canonical
    )

    print(
        f"\nSaved locally:"
        f"\n{local_path}"
    )

    upload_to_s3(
        local_path
    )

    print(
        f"\nUploaded test canonical dataset to:"
        f"\ns3://{DATA_S3_BUCKET}/{OUTPUT_KEY}"
    )

    print(
        "\nCANONICAL TEST BUILD COMPLETE"
    )


if __name__ == "__main__":
    main()