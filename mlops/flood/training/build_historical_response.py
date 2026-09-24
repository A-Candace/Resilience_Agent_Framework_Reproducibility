"""
Build the historical hourly flood-response table from the large
minute-level merged FloodNet + MRMS parquet stored in S3.

This version is configured for a SMALL DATE-RANGE TEST first.

The full historical source contains roughly 193 million rows, so the
pipeline should be validated on a narrow date range before removing
the test filter.

Future continuous ingestion should write new daily/hourly partitions
rather than rebuilding the complete historical response every time.
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

from mlops.flood.training.response_builder import (
    build_hourly_flood_response,
)


# ============================================================
# S3 CONFIGURATION
# ============================================================

DATA_S3_BUCKET = os.getenv(
    "DATA_S3_BUCKET",
    "nyc-resilience-data",
)

SOURCE_KEY = os.getenv(
    "FLOOD_MERGED_BOOTSTRAP_KEY",
    (
        "mlops/flood/bootstrap/"
        "merged_mrms_floodnet.parquet"
    ),
)

OUTPUT_KEY = os.getenv(
    "FLOOD_HISTORICAL_RESPONSE_KEY",
    (
        "mlops/flood/processed/"
        "historical_hourly_response_test.parquet"
    ),
)


# ============================================================
# LOCAL OUTPUT
# ============================================================

LOCAL_OUTPUT = (
    FLOOD_ARTIFACT_DIR
    / "historical_hourly_response_test.parquet"
)


# ============================================================
# STREAMING CONFIGURATION
# ============================================================

BATCH_ROWS = int(
    os.getenv(
        "FLOOD_RESPONSE_BATCH_ROWS",
        "1000000",
    )
)


# ============================================================
# TEMPORARY TEST DATE RANGE
# ============================================================

# IMPORTANT:
#
# We are intentionally testing only one day before allowing the
# script to process the complete 2020-2026 historical dataset.
#
# These environment variables let us change the range without
# editing code.
#
TEST_START = os.getenv(
    "FLOOD_RESPONSE_TEST_START",
    "2020-11-16",
)

TEST_END = os.getenv(
    "FLOOD_RESPONSE_TEST_END",
    "2020-11-17",
)


# ============================================================
# S3 FILESYSTEM
# ============================================================

def get_s3_filesystem() -> pafs.S3FileSystem:
    """
    Create a PyArrow S3 filesystem using the current AWS
    credential chain.

    Explicit region and longer network timeouts are used because
    the historical parquet is large and local Windows development
    can otherwise hit Arrow's short default S3 timeouts.
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
# BATCHED SOURCE READER
# ============================================================

def iter_source_batches():
    """
    Stream selected columns from the historical parquet.

    Only rows inside TEST_START <= time < TEST_END are read.

    This allows us to validate the transformation on a small
    date range before processing the full historical dataset.

    Yields
    ------
    pd.DataFrame
        Batches containing:
        - deployment_id
        - time
        - depth_proc_mm
    """

    filesystem = get_s3_filesystem()

    source_path = (
        f"{DATA_S3_BUCKET}/{SOURCE_KEY}"
    )

    dataset = ds.dataset(
        source_path,
        filesystem=filesystem,
        format="parquet",
    )

    start_time = pd.Timestamp(
        TEST_START,
        tz="UTC",
    )

    end_time = pd.Timestamp(
        TEST_END,
        tz="UTC",
    )

    scanner = dataset.scanner(
        columns=[
            "deployment_id",
            "time",
            "depth_proc_mm",
        ],
        filter=(
            (ds.field("time") >= start_time)
            &
            (ds.field("time") < end_time)
        ),
        batch_size=BATCH_ROWS,
        use_threads=True,
    )

    for record_batch in scanner.to_batches():
        yield record_batch.to_pandas()


# ============================================================
# HISTORICAL RESPONSE BUILD
# ============================================================

def build_historical_response() -> pd.DataFrame:
    """
    Build the hourly flood-response table for the configured
    test date range.

    The source is read in batches so that the entire historical
    parquet does not need to be loaded into memory at once.
    """

    completed_parts: list[pd.DataFrame] = []

    carry = pd.DataFrame()

    batch_number = 0

    for batch in iter_source_batches():
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

        batch = batch.sort_values(
            [
                "deployment_id",
                "time",
            ]
        ).reset_index(drop=True)

        batch["_hour"] = (
            batch["time"]
            .dt.floor("h")
        )

        # ====================================================
        # CARRY FINAL SENSOR-HOUR INTO NEXT BATCH
        # ====================================================

        last_sensor = (
            batch["deployment_id"].iloc[-1]
        )

        last_hour = (
            batch["_hour"].iloc[-1]
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

            completed_parts.append(
                hourly
            )

        print(
            f"Processed batch {batch_number:,} "
            f"| input rows={len(batch):,} "
            f"| carry rows={len(carry):,}"
        )

    # ========================================================
    # PROCESS FINAL CARRY
    # ========================================================

    if not carry.empty:
        final_hourly = (
            build_hourly_flood_response(
                carry
            )
        )

        completed_parts.append(
            final_hourly
        )

    if not completed_parts:
        raise RuntimeError(
            "No hourly response rows were created "
            "for the selected test date range."
        )

    result = pd.concat(
        completed_parts,
        ignore_index=True,
    )

    # ========================================================
    # FINAL SAFETY DEDUPLICATION
    # ========================================================

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
        .reset_index(drop=True)
    )

    return result


# ============================================================
# SAVE LOCALLY
# ============================================================

def save_local(
    response: pd.DataFrame,
) -> Path:
    """
    Save the derived test response locally before S3 upload.
    """

    LOCAL_OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    response.to_parquet(
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
    Upload the derived test artifact to S3.
    """

    client = boto3.client("s3")

    client.upload_file(
        str(local_path),
        DATA_S3_BUCKET,
        OUTPUT_KEY,
    )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    response: pd.DataFrame,
) -> None:
    """
    Print basic validation information for the generated response.
    """

    print(
        "\n=== Historical Response Test Summary ==="
    )

    print(
        f"Rows: {len(response):,}"
    )

    print(
        "Sensors: "
        f"{response['deployment_id'].nunique():,}"
    )

    print(
        f"Start: {response['hour'].min()}"
    )

    print(
        f"End:   {response['hour'].max()}"
    )

    print(
        "Observed responses: "
        f"{int(response['response_observed'].sum()):,}"
    )

    print(
        "Unobserved responses: "
        f"{int((response['response_observed'] == 0).sum()):,}"
    )

    print(
        "Rows with positive flood duration: "
        f"{int((response['minutes_above_1p5_inch'] > 0).sum()):,}"
    )

    print(
        "\nFlood-duration distribution:"
    )

    print(
        response[
            "minutes_above_1p5_inch"
        ]
        .describe()
    )


# ============================================================
# ENTRY POINT
# ============================================================

def main() -> None:
    print(
        "Building historical hourly flood response "
        "for SMALL TEST RANGE..."
    )

    print(
        f"Source:"
        f"\ns3://{DATA_S3_BUCKET}/{SOURCE_KEY}"
    )

    print(
        f"\nTest range:"
        f"\n{TEST_START} <= time < {TEST_END}"
    )

    response = (
        build_historical_response()
    )

    print_summary(
        response
    )

    local_path = save_local(
        response
    )

    print(
        f"\nSaved locally:"
        f"\n{local_path}"
    )

    upload_to_s3(
        local_path
    )

    print(
        f"\nUploaded test artifact to:"
        f"\ns3://{DATA_S3_BUCKET}/{OUTPUT_KEY}"
    )

    print(
        "\nTEST BUILD COMPLETE"
    )


if __name__ == "__main__":
    main()