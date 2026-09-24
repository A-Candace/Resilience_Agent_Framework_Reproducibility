"""
Build the forward hourly FloodNet response dataset.

This module converts the append-only raw FloodNet forward partitions
into the hourly flood-response representation used by the GCN
training pipeline.

Historical data is immutable.

Historical cutoff
-----------------
The historical feature/response dataset ends at:

    2026-03-13 06:00 UTC

The first forward feature/response hour is therefore:

    2026-03-13 07:00 UTC

Input
-----
Raw forward FloodNet observations:

    s3://nyc-resilience-data/
        mlops/flood/processed/forward/floodnet/
        year=YYYY/month=MM/day=DD/part.parquet

Output
------
Hourly forward response:

    s3://nyc-resilience-data/
        mlops/flood/processed/forward/response/
        year=YYYY/month=MM/day=DD/part.parquet

Scientific methodology
----------------------
The response is constructed by the already validated shared function:

    build_hourly_flood_response()

That function preserves the research methodology, including:

    - 5-minute depth bins
    - 25.4 mm (1-inch) flood threshold
    - minimum valid-bin requirement
    - hourly maximum depth
    - response_observed
    - flood-duration minutes

IMPORTANT:
The historical target column still carries the legacy name:

    minutes_above_1p5_inch

but the ACTIVE scientific threshold is 1 inch = 25.4 mm.

This script deliberately preserves that column name for training
compatibility while preserving the validated 1-inch behavior.
"""

from __future__ import annotations

import argparse
import io
import os

import boto3
import pandas as pd

from mlops.flood.training.response_builder import (
    build_hourly_flood_response,
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
# COLUMN DEFINITIONS
# ============================================================

SENSOR_ID = "deployment_id"

RAW_TIME_COLUMN = "time"

RAW_DEPTH_COLUMN = "depth_proc_mm"

HOUR_COLUMN = "hour"

OBSERVED_BINS_COLUMN = "observed_5min_bins"

VALID_BINS_COLUMN = "valid_depth_5min_bins"

HIGH_BINS_COLUMN = "high_depth_5min_bins"

MAX_DEPTH_COLUMN = "hourly_max_depth_mm"

RESPONSE_OBSERVED_COLUMN = "response_observed"

TARGET_COLUMN = "minutes_above_1p5_inch"


# ============================================================
# CONFIGURATION
# ============================================================

DATA_BUCKET = os.getenv(
    "DATA_S3_BUCKET",
    "nyc-resilience-data",
)

RAW_FLOODNET_PREFIX = os.getenv(
    "FLOOD_FORWARD_FLOODNET_PREFIX",
    "mlops/flood/processed/forward/floodnet",
)

FORWARD_RESPONSE_PREFIX = os.getenv(
    "FLOOD_FORWARD_RESPONSE_PREFIX",
    "mlops/flood/processed/forward/response",
)


# ============================================================
# PRIOR-COMPLETED-WEEK POLICY
# ============================================================

def prior_week_end() -> pd.Timestamp:
    """
    Return the exclusive forward-processing boundary.

    The result is Monday 00:00 UTC of the current week.

    Therefore only fully completed Monday-Sunday weeks are
    processed.
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
# S3 CLIENT
# ============================================================

def s3_client():
    """
    Return the S3 client used for the private flood-data bucket.
    """

    return boto3.client(
        "s3"
    )


# ============================================================
# RAW INPUT PARTITIONS
# ============================================================

def raw_partition_key(
    day: pd.Timestamp,
) -> str:
    """
    Return the raw FloodNet S3 key for one UTC day.
    """

    day = pd.to_datetime(
        day,
        utc=True,
    ).normalize()

    return (
        f"{RAW_FLOODNET_PREFIX}/"
        f"year={day.year:04d}/"
        f"month={day.month:02d}/"
        f"day={day.day:02d}/"
        "part.parquet"
    )


def raw_partition_uri(
    day: pd.Timestamp,
) -> str:
    """
    Return the raw FloodNet S3 URI for one day.
    """

    return (
        f"s3://{DATA_BUCKET}/"
        f"{raw_partition_key(day)}"
    )


def raw_partition_exists(
    day: pd.Timestamp,
) -> bool:
    """
    Return True when the raw forward FloodNet partition exists.
    """

    client = s3_client()

    try:

        client.head_object(
            Bucket=DATA_BUCKET,
            Key=raw_partition_key(day),
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


# ============================================================
# RESPONSE OUTPUT PARTITIONS
# ============================================================

def response_partition_key(
    day: pd.Timestamp,
) -> str:
    """
    Return the hourly response S3 key for one UTC day.
    """

    day = pd.to_datetime(
        day,
        utc=True,
    ).normalize()

    return (
        f"{FORWARD_RESPONSE_PREFIX}/"
        f"year={day.year:04d}/"
        f"month={day.month:02d}/"
        f"day={day.day:02d}/"
        "part.parquet"
    )


def response_partition_uri(
    day: pd.Timestamp,
) -> str:
    """
    Return the hourly response S3 URI for one UTC day.
    """

    return (
        f"s3://{DATA_BUCKET}/"
        f"{response_partition_key(day)}"
    )


def response_partition_exists(
    day: pd.Timestamp,
) -> bool:
    """
    Return True when the forward response partition exists.
    """

    client = s3_client()

    try:

        client.head_object(
            Bucket=DATA_BUCKET,
            Key=response_partition_key(day),
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


# ============================================================
# LOAD RAW DAY
# ============================================================

def load_raw_day(
    day: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load one raw forward FloodNet daily partition.
    """

    uri = raw_partition_uri(
        day
    )

    df = pd.read_parquet(
        uri
    )

    required_columns = {
        SENSOR_ID,
        RAW_TIME_COLUMN,
        RAW_DEPTH_COLUMN,
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Raw FloodNet partition is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    df = df.copy()

    df[SENSOR_ID] = (
        df[SENSOR_ID]
        .astype(str)
    )

    df[RAW_TIME_COLUMN] = pd.to_datetime(
        df[RAW_TIME_COLUMN],
        utc=True,
        errors="coerce",
    )

    df[RAW_DEPTH_COLUMN] = pd.to_numeric(
        df[RAW_DEPTH_COLUMN],
        errors="coerce",
    )

    # Invalid timestamps cannot participate in hourly response
    # construction.
    df = df.dropna(
        subset=[
            RAW_TIME_COLUMN,
        ]
    )

    df = (
        df.sort_values(
            [
                SENSOR_ID,
                RAW_TIME_COLUMN,
            ]
        )
        .drop_duplicates(
            subset=[
                SENSOR_ID,
                RAW_TIME_COLUMN,
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# RANGE NORMALIZATION
# ============================================================

def normalize_start(
    value,
) -> pd.Timestamp:
    """
    Normalize start hour and enforce historical immutability.
    """

    start = pd.to_datetime(
        value,
        utc=True,
        errors="raise",
    ).floor("h")

    if start < FORWARD_START:

        raise ValueError(
            "Forward response construction may not start before "
            f"{FORWARD_START}. Requested: {start}"
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

def build_response_day(
    day: pd.Timestamp,
    *,
    start_hour: pd.Timestamp,
    end_hour: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build hourly response rows for one UTC day.

    The shared response builder performs the actual scientific
    reconstruction.

    end_hour is exclusive.
    """

    day = pd.to_datetime(
        day,
        utc=True,
    ).normalize()

    day_end = (
        day
        + pd.Timedelta(days=1)
    )

    effective_start = max(
        day,
        start_hour,
        FORWARD_START,
    )

    effective_end = min(
        day_end,
        end_hour,
    )

    if effective_start >= effective_end:

        return pd.DataFrame()

    if not raw_partition_exists(
        day
    ):

        raise FileNotFoundError(
            "Required raw forward FloodNet partition does not exist: "
            f"{raw_partition_uri(day)}"
        )

    print(
        "    loading raw:"
    )

    print(
        f"    {raw_partition_uri(day)}"
    )

    raw = load_raw_day(
        day
    )

    print(
        f"    raw rows: "
        f"{len(raw):,}"
    )

    print(
        f"    raw sensors: "
        f"{raw[SENSOR_ID].nunique():,}"
    )

    if raw.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Restrict raw observations to requested window
    # --------------------------------------------------------

    raw = raw[
        (
            raw[RAW_TIME_COLUMN]
            >= effective_start
        )
        &
        (
            raw[RAW_TIME_COLUMN]
            < effective_end
        )
    ].copy()

    if raw.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # Shared validated scientific response builder
    # --------------------------------------------------------

    response = (
        build_hourly_flood_response(
            raw
        )
    )

    if response.empty:

        return response

    response[HOUR_COLUMN] = pd.to_datetime(
        response[HOUR_COLUMN],
        utc=True,
        errors="coerce",
    )

    response = response[
        (
            response[HOUR_COLUMN]
            >= effective_start
        )
        &
        (
            response[HOUR_COLUMN]
            < effective_end
        )
    ].copy()

    response = (
        response.sort_values(
            [
                HOUR_COLUMN,
                SENSOR_ID,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return response


# ============================================================
# VALIDATION
# ============================================================

def validate_response_partition(
    df: pd.DataFrame,
    *,
    day: pd.Timestamp,
    start_hour: pd.Timestamp,
    end_hour: pd.Timestamp,
) -> None:
    """
    Validate one hourly forward response partition before upload.
    """

    if df.empty:

        raise ValueError(
            "Cannot persist an empty forward response partition."
        )

    required_columns = {
        SENSOR_ID,
        HOUR_COLUMN,
        OBSERVED_BINS_COLUMN,
        VALID_BINS_COLUMN,
        HIGH_BINS_COLUMN,
        MAX_DEPTH_COLUMN,
        RESPONSE_OBSERVED_COLUMN,
        TARGET_COLUMN,
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Forward response partition missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    df[HOUR_COLUMN] = pd.to_datetime(
        df[HOUR_COLUMN],
        utc=True,
        errors="raise",
    )

    # --------------------------------------------------------
    # Historical immutability
    # --------------------------------------------------------

    if (
        df[HOUR_COLUMN]
        <= HISTORICAL_CUTOFF
    ).any():

        raise ValueError(
            "Forward response contains hours inside the "
            "immutable historical period."
        )

    # --------------------------------------------------------
    # Partition-day containment
    # --------------------------------------------------------

    day_start = pd.to_datetime(
        day,
        utc=True,
    ).normalize()

    day_end = (
        day_start
        + pd.Timedelta(days=1)
    )

    valid_day = (
        (
            df[HOUR_COLUMN]
            >= day_start
        )
        &
        (
            df[HOUR_COLUMN]
            < day_end
        )
    )

    if not valid_day.all():

        raise ValueError(
            "Response partition contains hours outside its UTC day."
        )

    # --------------------------------------------------------
    # Requested range containment
    # --------------------------------------------------------

    if not (
        (
            df[HOUR_COLUMN]
            >= start_hour
        )
        &
        (
            df[HOUR_COLUMN]
            < end_hour
        )
    ).all():

        raise ValueError(
            "Response partition contains hours outside "
            "the requested forward range."
        )

    # --------------------------------------------------------
    # One sensor/hour row
    # --------------------------------------------------------

    duplicate_count = (
        df.duplicated(
            subset=[
                SENSOR_ID,
                HOUR_COLUMN,
            ]
        )
        .sum()
    )

    if duplicate_count:

        raise ValueError(
            f"Detected {duplicate_count:,} duplicate "
            "sensor-hour response rows."
        )

    # --------------------------------------------------------
    # Bin-count sanity
    # --------------------------------------------------------

    for column in [
        OBSERVED_BINS_COLUMN,
        VALID_BINS_COLUMN,
        HIGH_BINS_COLUMN,
    ]:

        invalid = (
            df[column]
            .dropna()
            .lt(0)
            .any()
        )

        if invalid:

            raise ValueError(
                f"Negative values detected in {column}."
            )

    # At most 12 five-minute bins exist within an hour.
    if (
        df[OBSERVED_BINS_COLUMN]
        .dropna()
        .gt(12)
        .any()
    ):

        raise ValueError(
            "Observed 5-minute bin count exceeds 12."
        )

    if (
        df[VALID_BINS_COLUMN]
        .dropna()
        .gt(
            df[OBSERVED_BINS_COLUMN]
        )
        .any()
    ):

        raise ValueError(
            "Valid depth-bin count exceeds observed-bin count."
        )

    if (
        df[HIGH_BINS_COLUMN]
        .dropna()
        .gt(
            df[VALID_BINS_COLUMN]
        )
        .any()
    ):

        raise ValueError(
            "High-depth bin count exceeds valid-depth bin count."
        )

    # --------------------------------------------------------
    # Flood-duration sanity
    # --------------------------------------------------------

    observed = (
        df[RESPONSE_OBSERVED_COLUMN]
        .astype(bool)
    )

    observed_duration = (
        df.loc[
            observed,
            TARGET_COLUMN,
        ]
    )

    if observed_duration.isna().any():

        raise ValueError(
            "Observed responses contain missing flood duration."
        )

    if (
        observed_duration
        .lt(0)
        .any()
    ):

        raise ValueError(
            "Negative flood-duration values detected."
        )

    if (
        observed_duration
        .gt(60)
        .any()
    ):

        raise ValueError(
            "Flood duration exceeds 60 minutes in an hour."
        )


# ============================================================
# UPLOAD
# ============================================================

def upload_response_partition(
    df: pd.DataFrame,
    day: pd.Timestamp,
) -> str:
    """
    Persist one daily forward-response parquet partition.
    """

    buffer = io.BytesIO()

    df.to_parquet(
        buffer,
        index=False,
    )

    buffer.seek(0)

    s3_client().put_object(
        Bucket=DATA_BUCKET,
        Key=response_partition_key(day),
        Body=buffer.getvalue(),
    )

    return response_partition_uri(
        day
    )


# ============================================================
# SUMMARY
# ============================================================

def print_response_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print useful diagnostics for one response partition.
    """

    observed_mask = (
        df[
            RESPONSE_OBSERVED_COLUMN
        ]
        .astype(bool)
    )

    positive_mask = (
        observed_mask
        &
        (
            df[TARGET_COLUMN]
            > 0
        )
    )

    print(
        f"    rows: "
        f"{len(df):,}"
    )

    print(
        f"    sensors: "
        f"{df[SENSOR_ID].nunique():,}"
    )

    print(
        f"    hours: "
        f"{df[HOUR_COLUMN].nunique():,}"
    )

    print(
        f"    start: "
        f"{df[HOUR_COLUMN].min()}"
    )

    print(
        f"    end:   "
        f"{df[HOUR_COLUMN].max()}"
    )

    print(
        "    observed responses: "
        f"{observed_mask.sum():,}"
    )

    print(
        "    unobserved responses: "
        f"{(~observed_mask).sum():,}"
    )

    print(
        "    positive flood-duration rows: "
        f"{positive_mask.sum():,}"
    )

    if observed_mask.any():

        print(
            "    mean observed flood duration: "
            f"{df.loc[observed_mask, TARGET_COLUMN].mean():.4f} min"
        )

        print(
            "    max observed flood duration: "
            f"{df.loc[observed_mask, TARGET_COLUMN].max():.4f} min"
        )


# ============================================================
# FORWARD BUILD
# ============================================================

def run_forward_response_build(
    *,
    start,
    end,
) -> None:
    """
    Build resumable daily forward response partitions.

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
        "FORWARD FLOOD RESPONSE BUILD"
    )

    print(
        "============================"
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
        f"{start_hour} <= hour < "
        f"{end_hour}"
    )

    # --------------------------------------------------------
    # Daily processing range
    # --------------------------------------------------------

    last_requested_instant = (
        end_hour
        - pd.Timedelta(
            microseconds=1
        )
    )

    days = pd.date_range(
        start=start_hour.normalize(),
        end=last_requested_instant.normalize(),
        freq="D",
        tz="UTC",
    )

    completed = 0

    skipped = 0

    missing_raw = 0

    empty = 0

    failed = 0

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for day in days:

        print(
            "\n=================================="
        )

        print(
            day.date()
        )

        print(
            "=================================="
        )

        # ----------------------------------------------------
        # Resumability
        # ----------------------------------------------------

        if response_partition_exists(
            day
        ):

            print(
                "    existing forward response partition "
                "found; skipping"
            )

            skipped += 1

            continue

        # ----------------------------------------------------
        # Input requirement
        # ----------------------------------------------------

        if not raw_partition_exists(
            day
        ):

            print(
                "    raw FloodNet partition not found:"
            )

            print(
                f"    {raw_partition_uri(day)}"
            )

            missing_raw += 1

            continue

        # ----------------------------------------------------
        # Build
        # ----------------------------------------------------

        try:

            response = build_response_day(
                day,
                start_hour=start_hour,
                end_hour=end_hour,
            )

            if response.empty:

                print(
                    "    no hourly response rows produced"
                )

                empty += 1

                continue

            validate_response_partition(
                response,
                day=day,
                start_hour=start_hour,
                end_hour=end_hour,
            )

            print_response_summary(
                response
            )

            uri = upload_response_partition(
                response,
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

    # --------------------------------------------------------
    # Completion summary
    # --------------------------------------------------------

    print(
        "\n=================================="
    )

    print(
        "FORWARD FLOOD RESPONSE BUILD COMPLETE"
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
        f"Missing raw partitions: "
        f"{missing_raw}"
    )

    print(
        f"Empty response days: "
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
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Build forward-only hourly FloodNet responses "
            "without modifying historical data."
        )
    )

    parser.add_argument(
        "--start",
        default=(
            FORWARD_START.isoformat()
        ),
        help=(
            "Inclusive response-hour start. "
            "May not precede the historical cutoff."
        ),
    )

    parser.add_argument(
        "--end",
        default=(
            DEFAULT_END.isoformat()
        ),
        help=(
            "Exclusive response-hour end. "
            "Defaults to the beginning of the current UTC week."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    CLI entry point.
    """

    args = parse_args()

    run_forward_response_build(
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()