"""
Forward-only MRMS backfill for NYC flood MLOps.

This script starts strictly AFTER the immutable historical cutoff:

    historical feature data ends:
        2026-03-13 06:00 UTC

    first forward feature hour:
        2026-03-13 07:00 UTC

Historical bootstrap and canonical datasets are NEVER overwritten.

For each feature hour H:

    NOAA MRMS source hour = H + 1 hour

The script:
1. Processes one feature hour at a time.
2. Writes daily forward MRMS partitions to S3.
3. Skips days already completed.
4. Can resume safely after interruption.
5. Refuses to write any hour at or before the historical cutoff.
"""

from __future__ import annotations

import argparse
import io
import os

import boto3
import pandas as pd

from mlops.flood.ingestion.mrms import (
    FEATURE_HOUR,
    SENSOR_ID,
    PRECIP_COLUMN,
    SOURCE_HOUR,
    PRODUCT_COLUMN,
    MRMS_LAT,
    MRMS_LON,
    process_hour,
)


# ============================================================
# IMMUTABLE HISTORICAL BOUNDARY
# ============================================================

HISTORICAL_CUTOFF = pd.Timestamp(
    "2026-03-13 06:00:00",
    tz="UTC",
)

FORWARD_START = (
    HISTORICAL_CUTOFF
    + pd.Timedelta(hours=1)
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_BUCKET = os.getenv(
    "DATA_S3_BUCKET",
    "nyc-resilience-data",
)

FORWARD_PREFIX = os.getenv(
    "FLOOD_FORWARD_MRMS_PREFIX",
    "mlops/flood/processed/forward/mrms",
)

def prior_week_end() -> pd.Timestamp:
    """
    Exclusive ingestion boundary.

    Returns 00:00 UTC Monday of the current week,
    meaning ingestion only includes fully completed
    Monday-Sunday weeks.
    """
    now = pd.Timestamp.now(tz="UTC")

    return (
        now.normalize()
        - pd.Timedelta(days=now.weekday())
    )


DEFAULT_END = prior_week_end()


# ============================================================
# S3 HELPERS
# ============================================================

def s3_client():
    return boto3.client("s3")


def partition_key(
    day: pd.Timestamp,
) -> str:
    """
    Return the canonical S3 key for one forward MRMS day.
    """

    day = pd.to_datetime(
        day,
        utc=True,
    ).normalize()

    return (
        f"{FORWARD_PREFIX}/"
        f"year={day.year:04d}/"
        f"month={day.month:02d}/"
        f"day={day.day:02d}/"
        "part.parquet"
    )


def partition_uri(
    day: pd.Timestamp,
) -> str:
    return (
        f"s3://{DATA_BUCKET}/"
        f"{partition_key(day)}"
    )


def partition_exists(
    day: pd.Timestamp,
) -> bool:
    """
    Return True when the daily forward partition already exists.
    """

    client = s3_client()

    try:
        client.head_object(
            Bucket=DATA_BUCKET,
            Key=partition_key(day),
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


def upload_dataframe(
    df: pd.DataFrame,
    day: pd.Timestamp,
) -> str:
    """
    Write one daily MRMS partition to S3 as parquet.
    """

    buffer = io.BytesIO()

    df.to_parquet(
        buffer,
        index=False,
    )

    buffer.seek(0)

    s3_client().put_object(
        Bucket=DATA_BUCKET,
        Key=partition_key(day),
        Body=buffer.getvalue(),
    )

    return partition_uri(day)


# ============================================================
# RANGE SAFETY
# ============================================================

def normalize_start(
    value,
) -> pd.Timestamp:
    """
    Normalize and enforce the immutable historical boundary.
    """

    start = pd.to_datetime(
        value,
        utc=True,
        errors="raise",
    ).floor("h")

    if start < FORWARD_START:
        raise ValueError(
            "Forward MRMS ingestion may not start before "
            f"{FORWARD_START}. "
            f"Requested: {start}"
        )

    return start


def normalize_end(
    value,
) -> pd.Timestamp:
    """
    Normalize exclusive end hour.
    """

    return pd.to_datetime(
        value,
        utc=True,
        errors="raise",
    ).floor("h")


# ============================================================
# BUILD ONE DAY
# ============================================================

def build_day(
    day: pd.Timestamp,
    *,
    start_hour: pd.Timestamp,
    end_hour: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build all requested feature hours for one UTC day.

    end_hour is exclusive.
    """

    day_start = day.normalize()

    day_end = (
        day_start
        + pd.Timedelta(days=1)
    )

    effective_start = max(
        day_start,
        start_hour,
        FORWARD_START,
    )

    effective_end = min(
        day_end,
        end_hour,
    )

    if effective_start >= effective_end:
        return pd.DataFrame()

    hours = pd.date_range(
        start=effective_start,
        end=effective_end,
        freq="h",
        inclusive="left",
        tz="UTC",
    )

    parts = []

    print(
        f"    feature hours: "
        f"{effective_start} -> {effective_end} "
        f"(exclusive)"
    )

    for index, feature_hour in enumerate(
        hours,
        start=1,
    ):
        print(
            f"    [{index:02d}/{len(hours):02d}] "
            f"{feature_hour}"
        )

        try:
            result = process_hour(
                feature_hour
            )

        except Exception as exc:
            print(
                f"        ERROR: {exc}"
            )
            raise

        parts.append(
            result
        )

    if not parts:
        return pd.DataFrame()

    result = pd.concat(
        parts,
        ignore_index=True,
    )

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_partition(
    df: pd.DataFrame,
) -> None:
    """
    Validate one daily forward MRMS partition before upload.
    """

    if df.empty:
        raise ValueError(
            "Cannot persist an empty MRMS partition."
        )

    required_columns = {
        SENSOR_ID,
        FEATURE_HOUR,
        SOURCE_HOUR,
        PRODUCT_COLUMN,
        PRECIP_COLUMN,
        MRMS_LAT,
        MRMS_LON,
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Forward MRMS partition missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    df[FEATURE_HOUR] = pd.to_datetime(
        df[FEATURE_HOUR],
        utc=True,
        errors="raise",
    )

    df[SOURCE_HOUR] = pd.to_datetime(
        df[SOURCE_HOUR],
        utc=True,
        errors="raise",
    )

    if (
        df[FEATURE_HOUR]
        <= HISTORICAL_CUTOFF
    ).any():
        raise ValueError(
            "Forward partition contains feature hours "
            "inside the immutable historical period."
        )

    expected_source_hour = (
        df[FEATURE_HOUR]
        + pd.Timedelta(hours=1)
    )

    if not (
        df[SOURCE_HOUR]
        == expected_source_hour
    ).all():
        raise ValueError(
            "MRMS source-hour alignment is invalid."
        )

    duplicate_count = (
        df.duplicated(
            subset=[
                SENSOR_ID,
                FEATURE_HOUR,
            ]
        )
        .sum()
    )

    if duplicate_count:
        raise ValueError(
            f"Detected {duplicate_count:,} duplicate "
            "sensor-hour rows."
        )

    missing_precip = (
        df[PRECIP_COLUMN]
        .isna()
        .sum()
    )

    if missing_precip:
        raise ValueError(
            f"Detected {missing_precip:,} missing "
            "precipitation values."
        )


# ============================================================
# SUMMARY
# ============================================================

def print_partition_summary(
    df: pd.DataFrame,
) -> None:
    print(
        f"    rows: {len(df):,}"
    )

    print(
        f"    sensors: "
        f"{df[SENSOR_ID].nunique():,}"
    )

    print(
        f"    feature hours: "
        f"{df[FEATURE_HOUR].nunique():,}"
    )

    print(
        f"    start: "
        f"{df[FEATURE_HOUR].min()}"
    )

    print(
        f"    end:   "
        f"{df[FEATURE_HOUR].max()}"
    )

    print(
        f"    mean precip: "
        f"{df[PRECIP_COLUMN].mean():.4f} mm"
    )

    print(
        f"    max precip: "
        f"{df[PRECIP_COLUMN].max():.4f} mm"
    )


# ============================================================
# BACKFILL
# ============================================================

def run_backfill(
    *,
    start,
    end,
) -> None:
    """
    Run resumable forward-only MRMS ingestion.

    start is inclusive.
    end is exclusive.
    """

    start_hour = normalize_start(
        start
    )

    end_hour = normalize_end(
        end
    )

    if end_hour <= start_hour:
        raise ValueError(
            "End must be later than start."
        )

    print(
        "FORWARD MRMS BACKFILL"
    )

    print(
        "====================="
    )

    print(
        f"Immutable historical cutoff: "
        f"{HISTORICAL_CUTOFF}"
    )

    print(
        f"Forward start: "
        f"{FORWARD_START}"
    )

    print(
        "\nRequested range:"
    )

    print(
        f"{start_hour} <= feature hour < "
        f"{end_hour}"
    )

    days = pd.date_range(
        start=start_hour.normalize(),
        end=(
            end_hour
            - pd.Timedelta(
                microseconds=1
            )
        ).normalize(),
        freq="D",
        tz="UTC",
    )

    completed = 0
    skipped = 0
    failed = 0

    for day in days:

        print(
            "\n=================================="
        )

        print(
            f"{day.date()}"
        )

        print(
            "=================================="
        )

        if partition_exists(
            day
        ):
            print(
                "    existing forward partition "
                "found; skipping"
            )

            skipped += 1
            continue

        try:
            df = build_day(
                day,
                start_hour=start_hour,
                end_hour=end_hour,
            )

            if df.empty:
                print(
                    "    no requested hours for day"
                )
                continue

            validate_partition(
                df
            )

            print_partition_summary(
                df
            )

            uri = upload_dataframe(
                df,
                day,
            )

            print(
                "    uploaded:"
            )

            print(
                f"    {uri}"
            )

            completed += 1

        except Exception as exc:
            failed += 1

            print(
                f"    FAILED: {exc}"
            )

            raise

    print(
        "\n=================================="
    )

    print(
        "FORWARD MRMS BACKFILL COMPLETE"
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
        "=================================="
    )


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Forward-only NOAA MRMS backfill "
            "for NYC flood MLOps."
        )
    )

    parser.add_argument(
        "--start",
        default=FORWARD_START.isoformat(),
        help=(
            "Inclusive feature-hour start. "
            "May not precede the historical cutoff."
        ),
    )

    parser.add_argument(
        "--end",
        default=DEFAULT_END.isoformat(),
        help=(
            "Exclusive feature-hour end."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_backfill(
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()