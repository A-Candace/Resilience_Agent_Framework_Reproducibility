"""
Forward-only FloodNet depth backfill for NYC flood MLOps.

Historical data is immutable.

Historical cutoff:
    2026-03-13 06:00 UTC

First forward feature hour:
    2026-03-13 07:00 UTC

This script:
1. Pulls the current FloodNet deployment inventory.
2. Requests only depth observations after the historical cutoff.
3. Uses the start of the current UTC week as the default exclusive end.
4. Writes one raw FloodNet parquet partition per UTC day.
5. Skips partitions that already exist.
6. Can safely resume after interruption.
7. Never modifies historical bootstrap/canonical datasets.

The raw forward FloodNet layer will later feed the shared
1-inch hourly response builder.
"""

from __future__ import annotations

import argparse
import io
import os

import boto3
import pandas as pd

from mlops.flood.ingestion.floodnet import (
    DEPTH_PROCESSED_COLUMN,
    DEPTH_RAW_COLUMN,
    SENSOR_ID,
    TIME_COLUMN,
    fetch_depth_for_all_sensors,
    get_deployments,
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
    "FLOOD_FORWARD_FLOODNET_PREFIX",
    "mlops/flood/processed/forward/floodnet",
)


# ============================================================
# PRIOR-COMPLETED-WEEK POLICY
# ============================================================

def prior_week_end() -> pd.Timestamp:
    """
    Exclusive ingestion boundary.

    Returns Monday 00:00 UTC of the current week, so only fully
    completed Monday-Sunday weeks are ingested.
    """

    now = pd.Timestamp.now(
        tz="UTC"
    )

    return (
        now.normalize()
        - pd.Timedelta(
            days=now.weekday()
        )
    )


DEFAULT_END = prior_week_end()


# ============================================================
# S3 HELPERS
# ============================================================

def s3_client():
    return boto3.client(
        "s3"
    )


def partition_key(
    day: pd.Timestamp,
) -> str:
    """
    Return S3 key for one raw forward FloodNet day.
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
    Check whether a daily FloodNet partition already exists.
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
    Upload one daily FloodNet parquet partition.
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

    return partition_uri(
        day
    )


# ============================================================
# RANGE NORMALIZATION
# ============================================================

def normalize_start(
    value,
) -> pd.Timestamp:
    """
    Enforce the immutable historical boundary.
    """

    start = pd.to_datetime(
        value,
        utc=True,
        errors="raise",
    ).floor("h")

    if start < FORWARD_START:
        raise ValueError(
            "Forward FloodNet ingestion may not start before "
            f"{FORWARD_START}. Requested: {start}"
        )

    return start


def normalize_end(
    value,
) -> pd.Timestamp:
    """
    Normalize exclusive end timestamp.
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
    deployments: pd.DataFrame,
) -> pd.DataFrame:
    """
    Retrieve raw FloodNet depth observations for one UTC day.

    end_hour is exclusive.
    """

    day_start = (
        day.normalize()
    )

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

    if (
        effective_start
        >= effective_end
    ):
        return pd.DataFrame()

    print(
        "    request window:"
    )

    print(
        f"    {effective_start} "
        f"-> {effective_end} "
        "(exclusive)"
    )

    result = (
        fetch_depth_for_all_sensors(
            deployments,
            global_start=effective_start,
            global_end=effective_end,
        )
    )

    if result.empty:
        return result

    result[TIME_COLUMN] = (
        pd.to_datetime(
            result[TIME_COLUMN],
            utc=True,
            errors="coerce",
        )
    )

    result = result[
        (
            result[TIME_COLUMN]
            >= effective_start
        )
        &
        (
            result[TIME_COLUMN]
            < effective_end
        )
    ].copy()

    return result


# ============================================================
# VALIDATION
# ============================================================

def validate_partition(
    df: pd.DataFrame,
    *,
    day: pd.Timestamp,
) -> None:
    """
    Validate one raw forward FloodNet partition.
    """

    if df.empty:
        raise ValueError(
            "Cannot persist an empty FloodNet partition."
        )

    required_columns = {
        SENSOR_ID,
        TIME_COLUMN,
        DEPTH_PROCESSED_COLUMN,
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Forward FloodNet partition missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    df[TIME_COLUMN] = (
        pd.to_datetime(
            df[TIME_COLUMN],
            utc=True,
            errors="raise",
        )
    )

    if (
        df[TIME_COLUMN]
        <= HISTORICAL_CUTOFF
    ).any():
        raise ValueError(
            "Forward FloodNet partition contains observations "
            "inside the immutable historical period."
        )

    day_start = (
        day.normalize()
    )

    day_end = (
        day_start
        + pd.Timedelta(days=1)
    )

    if not (
        (
            df[TIME_COLUMN]
            >= day_start
        )
        &
        (
            df[TIME_COLUMN]
            < day_end
        )
    ).all():
        raise ValueError(
            "FloodNet partition contains timestamps "
            "outside its UTC day."
        )

    duplicates = (
        df.duplicated(
            subset=[
                SENSOR_ID,
                TIME_COLUMN,
            ]
        )
        .sum()
    )

    if duplicates:
        raise ValueError(
            f"Detected {duplicates:,} duplicate "
            "FloodNet sensor/timestamp rows."
        )


# ============================================================
# SUMMARY
# ============================================================

def print_partition_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print basic partition diagnostics.
    """

    print(
        f"    rows: "
        f"{len(df):,}"
    )

    print(
        f"    sensors with observations: "
        f"{df[SENSOR_ID].nunique():,}"
    )

    print(
        f"    start: "
        f"{df[TIME_COLUMN].min()}"
    )

    print(
        f"    end:   "
        f"{df[TIME_COLUMN].max()}"
    )

    missing_depth = (
        df[
            DEPTH_PROCESSED_COLUMN
        ]
        .isna()
        .sum()
    )

    print(
        "    missing processed depth: "
        f"{missing_depth:,}"
    )

    if (
        DEPTH_RAW_COLUMN
        in df.columns
    ):

        missing_raw = (
            df[
                DEPTH_RAW_COLUMN
            ]
            .isna()
            .sum()
        )

        print(
            "    missing raw depth: "
            f"{missing_raw:,}"
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
    Run resumable forward-only FloodNet ingestion.
    """

    start_hour = (
        normalize_start(
            start
        )
    )

    end_hour = (
        normalize_end(
            end
        )
    )

    if (
        end_hour
        <= start_hour
    ):
        raise ValueError(
            "End must be later than start."
        )

    print(
        "FORWARD FLOODNET BACKFILL"
    )

    print(
        "========================="
    )

    print(
        "Immutable historical cutoff:"
    )

    print(
        HISTORICAL_CUTOFF
    )

    print(
        "\nForward start:"
    )

    print(
        FORWARD_START
    )

    print(
        "\nRequested range:"
    )

    print(
        f"{start_hour} <= time < "
        f"{end_hour}"
    )

    # --------------------------------------------------------
    # Inventory once
    # --------------------------------------------------------

    print(
        "\nRetrieving FloodNet deployment inventory..."
    )

    deployments = (
        get_deployments()
    )

    print(
        f"Deployments available: "
        f"{len(deployments):,}"
    )

    print(
        "Deployment IDs:"
    )

    print(
        f"{deployments[SENSOR_ID].nunique():,}"
    )

    # --------------------------------------------------------
    # Days
    # --------------------------------------------------------

    last_requested_instant = (
        end_hour
        - pd.Timedelta(
            microseconds=1
        )
    )

    days = pd.date_range(
        start=(
            start_hour.normalize()
        ),
        end=(
            last_requested_instant
            .normalize()
        ),
        freq="D",
        tz="UTC",
    )

    completed = 0
    skipped = 0
    empty = 0
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
                deployments=deployments,
            )

            if df.empty:

                print(
                    "    no FloodNet observations returned"
                )

                empty += 1
                continue

            validate_partition(
                df,
                day=day,
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
                f"    FAILED: "
                f"{exc}"
            )

            raise

    print(
        "\n=================================="
    )

    print(
        "FORWARD FLOODNET BACKFILL COMPLETE"
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
        f"Empty days: "
        f"{empty}"
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
    parser = (
        argparse.ArgumentParser(
            description=(
                "Forward-only FloodNet depth backfill "
                "for NYC flood MLOps."
            )
        )
    )

    parser.add_argument(
        "--start",
        default=(
            FORWARD_START
            .isoformat()
        ),
        help=(
            "Inclusive UTC start. "
            "May not precede the immutable historical cutoff."
        ),
    )

    parser.add_argument(
        "--end",
        default=(
            DEFAULT_END
            .isoformat()
        ),
        help=(
            "Exclusive UTC end. Defaults to the start "
            "of the current UTC week."
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