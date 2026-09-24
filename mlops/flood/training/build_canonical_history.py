"""
Build the full canonical historical flood-training dataset month by month.

Each monthly partition combines:

1. Predictor-side hourly features from:
   all_sensor_hourly_model_1mile.parquet

2. Reconstructed 1-inch flood response from:
   merged_mrms_floodnet.parquet

Outputs are stored in partitioned S3 paths:

    mlops/flood/processed/canonical/year=YYYY/month=MM/part.parquet

The build is resumable and auditable:

- existing monthly partitions are skipped
- successful months are recorded as completed
- skipped months are recorded as skipped
- failed months are recorded with their error
- progress is persisted to an S3 manifest
"""

from __future__ import annotations

import os
from pathlib import Path

import boto3
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.fs as pafs

from mlops.flood.shared.config import (
    FLOOD_ARTIFACT_DIR,
)

from mlops.flood.shared.s3_bootstrap import (
    load_hourly_model_features,
)

from mlops.flood.training.history_manifest import (
    load_manifest,
    month_key,
    persist_manifest,
    update_partition,
)

from mlops.flood.training.response_builder import (
    build_hourly_flood_response,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_S3_BUCKET = os.getenv(
    "DATA_S3_BUCKET",
    "nyc-resilience-data",
)

MERGED_SOURCE_KEY = os.getenv(
    "FLOOD_MERGED_BOOTSTRAP_KEY",
    "mlops/flood/bootstrap/merged_mrms_floodnet.parquet",
)

CANONICAL_PREFIX = os.getenv(
    "FLOOD_CANONICAL_PREFIX",
    "mlops/flood/processed/canonical",
)

HISTORY_START = os.getenv(
    "FLOOD_HISTORY_START",
    "2020-10-14",
)

HISTORY_END = os.getenv(
    "FLOOD_HISTORY_END",
    "2026-03-14",
)

BATCH_ROWS = int(
    os.getenv(
        "FLOOD_RESPONSE_BATCH_ROWS",
        "1000000",
    )
)

OVERWRITE_EXISTING = (
    os.getenv(
        "FLOOD_CANONICAL_OVERWRITE",
        "false",
    ).lower()
    == "true"
)


# ============================================================
# PREDICTOR COLUMNS
# ============================================================

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
# S3 CLIENTS
# ============================================================

def get_boto3_s3_client():
    """
    Return an S3 client using the active AWS credential chain.
    """

    return boto3.client(
        "s3"
    )


def get_pyarrow_s3_filesystem() -> pafs.S3FileSystem:
    """
    Create a PyArrow S3 filesystem with explicit AWS region,
    longer network timeouts, and retry support.
    """

    region = os.getenv(
        "AWS_REGION",
        os.getenv(
            "AWS_DEFAULT_REGION",
            "us-east-2",
        ),
    )

    return pafs.S3FileSystem(
        region=region,
        connect_timeout=30.0,
        request_timeout=120.0,
        retry_strategy=pafs.AwsStandardS3RetryStrategy(
            max_attempts=5,
        ),
    )


# ============================================================
# DATE UTILITIES
# ============================================================

def iter_month_ranges():
    """
    Yield monthly [start, end) ranges across the historical
    baseline.

    The first month may begin mid-month because the historical
    source begins on 2020-10-14.
    """

    overall_start = pd.Timestamp(
        HISTORY_START,
        tz="UTC",
    )

    overall_end = pd.Timestamp(
        HISTORY_END,
        tz="UTC",
    )

    current = overall_start

    while current < overall_end:

        first_of_next_month = (
            current
            + pd.offsets.MonthBegin(1)
        ).normalize()

        if first_of_next_month <= current:
            first_of_next_month = (
                current
                + pd.offsets.MonthBegin(2)
            ).normalize()

        month_end = min(
            first_of_next_month,
            overall_end,
        )

        yield current, month_end

        current = month_end


# ============================================================
# OUTPUT PATHS
# ============================================================

def output_key_for_month(
    start_time: pd.Timestamp,
) -> str:
    """
    Return the partitioned S3 object key for a month.
    """

    return (
        f"{CANONICAL_PREFIX}/"
        f"year={start_time.year:04d}/"
        f"month={start_time.month:02d}/"
        "part.parquet"
    )


def local_output_for_month(
    start_time: pd.Timestamp,
) -> Path:
    """
    Return the equivalent local temporary artifact path.
    """

    return (
        FLOOD_ARTIFACT_DIR
        / "canonical_history"
        / f"year={start_time.year:04d}"
        / f"month={start_time.month:02d}"
        / "part.parquet"
    )


# ============================================================
# S3 EXISTENCE CHECK
# ============================================================

def s3_object_exists(
    key: str,
) -> bool:
    """
    Return True when a canonical monthly partition already
    exists in S3.
    """

    client = get_boto3_s3_client()

    try:
        client.head_object(
            Bucket=DATA_S3_BUCKET,
            Key=key,
        )

        return True

    except client.exceptions.ClientError as exc:

        error_code = (
            exc.response
            .get("Error", {})
            .get("Code")
        )

        if error_code in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return False

        raise


# ============================================================
# LOAD MONTHLY PREDICTORS
# ============================================================

def load_predictors_for_month(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load predictor-side hourly rows for one month.

    The old response columns in the bootstrap hourly parquet
    are deliberately not loaded.
    """

    df = load_hourly_model_features(
        columns=PREDICTOR_COLUMNS,
    )

    df["hour"] = pd.to_datetime(
        df["hour"],
        utc=True,
        errors="coerce",
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
# STREAM MONTHLY RESPONSE SOURCE
# ============================================================

def iter_response_source_batches(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
):
    """
    Stream only one month's raw FloodNet response observations
    from the large historical merged parquet.
    """

    filesystem = (
        get_pyarrow_s3_filesystem()
    )

    source_path = (
        f"{DATA_S3_BUCKET}/"
        f"{MERGED_SOURCE_KEY}"
    )

    dataset = ds.dataset(
        source_path,
        filesystem=filesystem,
        format="parquet",
    )

    scanner = dataset.scanner(
        columns=[
            "deployment_id",
            "time",
            "depth_proc_mm",
        ],
        filter=(
            (
                ds.field("time")
                >= start_time
            )
            &
            (
                ds.field("time")
                < end_time
            )
        ),
        batch_size=BATCH_ROWS,
        use_threads=True,
    )

    for record_batch in scanner.to_batches():

        yield (
            record_batch
            .to_pandas()
        )


# ============================================================
# BUILD MONTHLY RESPONSE
# ============================================================

def build_response_for_month(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    """
    Reconstruct the canonical 1-inch hourly response for one
    month.
    """

    parts: list[pd.DataFrame] = []

    carry = pd.DataFrame()

    batch_number = 0

    for batch in iter_response_source_batches(
        start_time,
        end_time,
    ):

        batch_number += 1

        if batch.empty:
            continue

        batch["time"] = pd.to_datetime(
            batch["time"],
            utc=True,
            errors="coerce",
        )

        batch = batch.dropna(
            subset=[
                "deployment_id",
                "time",
            ]
        )

        if batch.empty:
            continue

        if not carry.empty:

            batch = pd.concat(
                [
                    carry,
                    batch,
                ],
                ignore_index=True,
            )

        batch = (
            batch.sort_values(
                [
                    "deployment_id",
                    "time",
                ]
            )
            .reset_index(
                drop=True
            )
        )

        batch["_hour"] = (
            batch["time"]
            .dt.floor("h")
        )

        # ----------------------------------------------------
        # Preserve the last sensor-hour across batch boundaries
        # ----------------------------------------------------

        last_sensor = (
            batch["deployment_id"]
            .iloc[-1]
        )

        last_hour = (
            batch["_hour"]
            .iloc[-1]
        )

        carry_mask = (
            (
                batch["deployment_id"]
                == last_sensor
            )
            &
            (
                batch["_hour"]
                == last_hour
            )
        )

        carry = (
            batch.loc[
                carry_mask,
                [
                    "deployment_id",
                    "time",
                    "depth_proc_mm",
                ],
            ]
            .copy()
        )

        process_now = (
            batch.loc[
                ~carry_mask,
                [
                    "deployment_id",
                    "time",
                    "depth_proc_mm",
                ],
            ]
            .copy()
        )

        if not process_now.empty:

            hourly = (
                build_hourly_flood_response(
                    process_now
                )
            )

            parts.append(
                hourly
            )

        print(
            f"    response batch "
            f"{batch_number:,} "
            f"| rows={len(batch):,} "
            f"| carry={len(carry):,}"
        )

    # --------------------------------------------------------
    # Process final carried sensor-hour
    # --------------------------------------------------------

    if not carry.empty:

        parts.append(
            build_hourly_flood_response(
                carry
            )
        )

    if not parts:

        return pd.DataFrame(
            columns=[
                "deployment_id",
                "hour",
                "observed_5min_bins",
                "valid_depth_5min_bins",
                "high_depth_5min_bins",
                "hourly_max_depth_mm",
                "response_observed",
                "minutes_above_1p5_inch",
            ]
        )

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    result = (
        result.sort_values(
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
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# BUILD MONTHLY CANONICAL PARTITION
# ============================================================

def build_canonical_month(
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
) -> pd.DataFrame:
    """
    Combine monthly predictors with the rebuilt 1-inch response.
    """

    predictors = (
        load_predictors_for_month(
            start_time,
            end_time,
        )
    )

    response = (
        build_response_for_month(
            start_time,
            end_time,
        )
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

    canonical = (
        canonical.sort_values(
            [
                "deployment_id",
                "hour",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return canonical


# ============================================================
# VALIDATE MONTH
# ============================================================

def validate_canonical_month(
    df: pd.DataFrame,
) -> None:
    """
    Validate required schema and unique sensor-hour keys.
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
            "Canonical monthly partition "
            "is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    duplicate_count = (
        df.duplicated(
            subset=[
                "deployment_id",
                "hour",
            ]
        )
        .sum()
    )

    if duplicate_count:

        raise ValueError(
            "Canonical monthly partition "
            f"contains {duplicate_count:,} "
            "duplicate sensor-hour rows."
        )


# ============================================================
# MONTH METRICS
# ============================================================

def month_metrics(
    df: pd.DataFrame,
) -> dict:
    """
    Return metrics stored in the build manifest.
    """

    return {
        "rows": int(
            len(df)
        ),

        "sensors": int(
            df[
                "deployment_id"
            ].nunique()
        ),

        "observed_response_rows": int(
            (
                df["response_observed"]
                == 1
            ).sum()
        ),

        "positive_duration_rows": int(
            (
                df[
                    "minutes_above_1p5_inch"
                ]
                > 0
            ).sum()
        ),

        "missing_response_rows": int(
            df[
                "response_observed"
            ]
            .isna()
            .sum()
        ),
    }


# ============================================================
# SAVE MONTH
# ============================================================

def save_month(
    df: pd.DataFrame,
    start_time: pd.Timestamp,
) -> tuple[Path, str]:
    """
    Save a monthly canonical partition locally and upload it
    to S3.
    """

    local_path = (
        local_output_for_month(
            start_time
        )
    )

    local_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_parquet(
        local_path,
        index=False,
    )

    s3_key = (
        output_key_for_month(
            start_time
        )
    )

    get_boto3_s3_client().upload_file(
        str(local_path),
        DATA_S3_BUCKET,
        s3_key,
    )

    return (
        local_path,
        s3_key,
    )


# ============================================================
# PRINT MONTH SUMMARY
# ============================================================

def print_month_summary(
    metrics: dict,
) -> None:
    """
    Print monthly validation metrics.
    """

    print(
        f"    rows: "
        f"{metrics['rows']:,}"
    )

    print(
        f"    sensors: "
        f"{metrics['sensors']:,}"
    )

    print(
        "    observed response rows: "
        f"{metrics['observed_response_rows']:,}"
    )

    print(
        "    positive duration rows: "
        f"{metrics['positive_duration_rows']:,}"
    )

    print(
        "    missing reconstructed response: "
        f"{metrics['missing_response_rows']:,}"
    )


# ============================================================
# MAIN HISTORICAL LOOP
# ============================================================

def main() -> None:
    """
    Build all canonical historical monthly partitions.

    The S3 manifest is updated after each month so progress is
    preserved even if the process stops midway.
    """

    print(
        "Building canonical flood history "
        "month by month..."
    )

    print(
        f"\nHistorical range:"
        f"\n{HISTORY_START} <= time < "
        f"{HISTORY_END}"
    )

    manifest = load_manifest()

    completed = 0
    skipped = 0
    failed = 0

    for (
        start_time,
        end_time,
    ) in iter_month_ranges():

        partition = month_key(
            start_time.year,
            start_time.month,
        )

        s3_key = (
            output_key_for_month(
                start_time
            )
        )

        print(
            f"\n=== {partition} ==="
        )

        # ====================================================
        # EXISTING PARTITION
        # ====================================================

        if (
            not OVERWRITE_EXISTING
            and s3_object_exists(
                s3_key
            )
        ):

            print(
                "    existing partition found; "
                "skipping"
            )

            manifest = update_partition(
                manifest,
                partition=partition,
                status="skipped",
                start_time=(
                    start_time.isoformat()
                ),
                end_time=(
                    end_time.isoformat()
                ),
                s3_key=s3_key,
            )

            persist_manifest(
                manifest
            )

            skipped += 1

            continue

        # ====================================================
        # BUILD PARTITION
        # ====================================================

        try:

            print(
                f"    range: "
                f"{start_time} → "
                f"{end_time}"
            )

            canonical = (
                build_canonical_month(
                    start_time,
                    end_time,
                )
            )

            if canonical.empty:

                raise RuntimeError(
                    "Canonical month contained "
                    "no rows."
                )

            validate_canonical_month(
                canonical
            )

            metrics = (
                month_metrics(
                    canonical
                )
            )

            print_month_summary(
                metrics
            )

            (
                local_path,
                uploaded_key,
            ) = save_month(
                canonical,
                start_time,
            )

            print(
                f"    saved locally: "
                f"{local_path}"
            )

            print(
                f"    uploaded: "
                f"s3://{DATA_S3_BUCKET}/"
                f"{uploaded_key}"
            )

            manifest = update_partition(
                manifest,
                partition=partition,
                status="completed",
                start_time=(
                    start_time.isoformat()
                ),
                end_time=(
                    end_time.isoformat()
                ),
                s3_key=uploaded_key,
                rows=metrics["rows"],
                sensors=metrics["sensors"],
                observed_response_rows=(
                    metrics[
                        "observed_response_rows"
                    ]
                ),
                positive_duration_rows=(
                    metrics[
                        "positive_duration_rows"
                    ]
                ),
                missing_response_rows=(
                    metrics[
                        "missing_response_rows"
                    ]
                ),
            )

            persist_manifest(
                manifest
            )

            completed += 1

        # ====================================================
        # FAILED PARTITION
        # ====================================================

        except Exception as exc:

            failed += 1

            print(
                f"    ERROR: {exc}"
            )

            manifest = update_partition(
                manifest,
                partition=partition,
                status="failed",
                start_time=(
                    start_time.isoformat()
                ),
                end_time=(
                    end_time.isoformat()
                ),
                s3_key=s3_key,
                error=str(exc),
            )

            persist_manifest(
                manifest
            )

            print(
                "    failure recorded in manifest; "
                "continuing to next month"
            )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n=================================="
    )

    print(
        "CANONICAL HISTORY BUILD COMPLETE"
    )

    print(
        f"Completed partitions: "
        f"{completed}"
    )

    print(
        f"Skipped existing partitions: "
        f"{skipped}"
    )

    print(
        f"Failed partitions: "
        f"{failed}"
    )

    print(
        "\nManifest:"
    )

    print(
        f"s3://{DATA_S3_BUCKET}/"
        "mlops/flood/manifests/"
        "canonical_history_manifest.json"
    )

    print(
        "=================================="
    )


if __name__ == "__main__":
    main()