"""
Build the forward canonical flood training dataset.

Historical training data is immutable.

Historical cutoff
-----------------
The existing historical dataset ends at:

    2026-03-13 06:00 UTC

Forward canonical data begins at:

    2026-03-13 07:00 UTC

Inputs
------
Forward MRMS hourly precipitation:

    s3://nyc-resilience-data/
        mlops/flood/processed/forward/mrms/
        year=YYYY/month=MM/day=DD/part.parquet

Forward FloodNet hourly response:

    s3://nyc-resilience-data/
        mlops/flood/processed/forward/response/
        year=YYYY/month=MM/day=DD/part.parquet

Lifecycle-aware sensor mapping:

    s3://nyc-resilience-data/
        mlops/flood/reference/sensor_mrms_grid_map.parquet

Read-only historical precipitation context:

    load_gcn_hourly_features()

Output
------
Forward canonical training partitions:

    s3://nyc-resilience-data/
        mlops/flood/processed/forward/canonical/
        year=YYYY/month=MM/day=DD/part.parquet

Canonical columns
-----------------
    deployment_id
    hour
    precip_current_hour_mm
    precip_previous_6h_mm
    daily_total_precip_mm
    sensor_lat
    sensor_lon
    observed_5min_bins
    valid_depth_5min_bins
    high_depth_5min_bins
    hourly_max_depth_mm
    response_observed
    minutes_above_1p5_inch

IMPORTANT
---------
The target column retains its historical legacy name
"minutes_above_1p5_inch", but the validated response builder uses
the active 1-inch = 25.4 mm scientific threshold.
"""

from __future__ import annotations

import argparse
import io
import os

import boto3
import numpy as np
import pandas as pd

from mlops.flood.shared.s3_bootstrap import (
    load_gcn_hourly_features,
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

HOUR_COLUMN = "hour"

CURRENT_PRECIP = "precip_current_hour_mm"

PREVIOUS_6H_PRECIP = "precip_previous_6h_mm"

DAILY_PRECIP = "daily_total_precip_mm"

SENSOR_LAT = "sensor_lat"

SENSOR_LON = "sensor_lon"

DATE_DEPLOYED = "date_deployed"

DATE_DOWN = "date_down"

OBSERVED_BINS = "observed_5min_bins"

VALID_BINS = "valid_depth_5min_bins"

HIGH_BINS = "high_depth_5min_bins"

MAX_DEPTH = "hourly_max_depth_mm"

RESPONSE_OBSERVED = "response_observed"

TARGET_COLUMN = "minutes_above_1p5_inch"


CANONICAL_COLUMNS = [
    SENSOR_ID,
    HOUR_COLUMN,
    CURRENT_PRECIP,
    PREVIOUS_6H_PRECIP,
    DAILY_PRECIP,
    SENSOR_LAT,
    SENSOR_LON,
    OBSERVED_BINS,
    VALID_BINS,
    HIGH_BINS,
    MAX_DEPTH,
    RESPONSE_OBSERVED,
    TARGET_COLUMN,
]


# ============================================================
# CONFIGURATION
# ============================================================

DATA_BUCKET = os.getenv(
    "DATA_S3_BUCKET",
    "nyc-resilience-data",
)

FORWARD_MRMS_PREFIX = os.getenv(
    "FLOOD_FORWARD_MRMS_PREFIX",
    "mlops/flood/processed/forward/mrms",
)

FORWARD_RESPONSE_PREFIX = os.getenv(
    "FLOOD_FORWARD_RESPONSE_PREFIX",
    "mlops/flood/processed/forward/response",
)

FORWARD_CANONICAL_PREFIX = os.getenv(
    "FLOOD_FORWARD_CANONICAL_PREFIX",
    "mlops/flood/processed/forward/canonical",
)

SENSOR_MAP_KEY = os.getenv(
    "FLOOD_SENSOR_MRMS_MAP_KEY",
    "mlops/flood/reference/sensor_mrms_grid_map.parquet",
)


# ============================================================
# PRIOR-COMPLETED-WEEK POLICY
# ============================================================

def prior_week_end() -> pd.Timestamp:
    """
    Return Monday 00:00 UTC of the current week.

    The end timestamp is exclusive, therefore only fully completed
    Monday-Sunday weeks are included.
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


def object_exists(
    key: str,
) -> bool:
    """
    Check whether an object exists in the private flood bucket.
    """

    client = s3_client()

    try:

        client.head_object(
            Bucket=DATA_BUCKET,
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


# ============================================================
# PARTITION PATHS
# ============================================================

def daily_partition_key(
    prefix: str,
    day: pd.Timestamp,
) -> str:
    """
    Build one daily partition key.
    """

    day = pd.to_datetime(
        day,
        utc=True,
    ).normalize()

    return (
        f"{prefix}/"
        f"year={day.year:04d}/"
        f"month={day.month:02d}/"
        f"day={day.day:02d}/"
        "part.parquet"
    )


def mrms_partition_key(
    day: pd.Timestamp,
) -> str:

    return daily_partition_key(
        FORWARD_MRMS_PREFIX,
        day,
    )


def response_partition_key(
    day: pd.Timestamp,
) -> str:

    return daily_partition_key(
        FORWARD_RESPONSE_PREFIX,
        day,
    )


def canonical_partition_key(
    day: pd.Timestamp,
) -> str:

    return daily_partition_key(
        FORWARD_CANONICAL_PREFIX,
        day,
    )


def key_uri(
    key: str,
) -> str:

    return (
        f"s3://{DATA_BUCKET}/"
        f"{key}"
    )


def mrms_partition_uri(
    day: pd.Timestamp,
) -> str:

    return key_uri(
        mrms_partition_key(day)
    )


def response_partition_uri(
    day: pd.Timestamp,
) -> str:

    return key_uri(
        response_partition_key(day)
    )


def canonical_partition_uri(
    day: pd.Timestamp,
) -> str:

    return key_uri(
        canonical_partition_key(day)
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
            "Forward canonical construction may not start before "
            f"{FORWARD_START}. Requested: {start}"
        )

    return start


def normalize_end(
    value,
) -> pd.Timestamp:

    return pd.to_datetime(
        value,
        utc=True,
        errors="raise",
    ).floor("h")


# ============================================================
# SENSOR LIFECYCLE MAPPING
# ============================================================

def load_sensor_mapping() -> pd.DataFrame:
    """
    Load lifecycle-aware canonical sensor mapping.
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
        SENSOR_LAT,
        SENSOR_LON,
        DATE_DEPLOYED,
        DATE_DOWN,
    }

    missing = (
        required
        - set(mapping.columns)
    )

    if missing:

        raise ValueError(
            "Lifecycle-aware sensor mapping is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    mapping = mapping.copy()

    mapping[SENSOR_ID] = (
        mapping[SENSOR_ID]
        .astype(str)
    )

    mapping[DATE_DEPLOYED] = pd.to_datetime(
        mapping[DATE_DEPLOYED],
        utc=True,
        errors="coerce",
    )

    mapping[DATE_DOWN] = pd.to_datetime(
        mapping[DATE_DOWN],
        utc=True,
        errors="coerce",
    )

    return mapping


def apply_lifecycle_filter(
    df: pd.DataFrame,
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep predictor rows only when the sensor existed at that hour.

    Active-at-hour definition:

        date_deployed <= hour

    and:

        date_down is null
        OR
        hour < date_down
    """

    metadata = mapping[
        [
            SENSOR_ID,
            SENSOR_LAT,
            SENSOR_LON,
            DATE_DEPLOYED,
            DATE_DOWN,
        ]
    ].drop_duplicates(
        subset=[
            SENSOR_ID,
        ],
        keep="last",
    )

    result = df.merge(
        metadata,
        on=SENSOR_ID,
        how="left",
        validate="many_to_one",
    )

    known_start = (
        result[DATE_DEPLOYED]
        .notna()
    )

    active = (
        known_start
        &
        (
            result[DATE_DEPLOYED]
            <= result[HOUR_COLUMN]
        )
        &
        (
            result[DATE_DOWN].isna()
            |
            (
                result[HOUR_COLUMN]
                < result[DATE_DOWN]
            )
        )
    )

    result = (
        result.loc[
            active
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# LOAD FORWARD MRMS DAY
# ============================================================

def load_mrms_day(
    day: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load one forward MRMS daily partition.
    """

    key = mrms_partition_key(
        day
    )

    if not object_exists(
        key
    ):

        raise FileNotFoundError(
            "Forward MRMS partition not found: "
            f"{key_uri(key)}"
        )

    df = pd.read_parquet(
        key_uri(key)
    )

    required = {
        SENSOR_ID,
        HOUR_COLUMN,
        CURRENT_PRECIP,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Forward MRMS partition missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    df = df[
        [
            SENSOR_ID,
            HOUR_COLUMN,
            CURRENT_PRECIP,
        ]
    ].copy()

    df[SENSOR_ID] = (
        df[SENSOR_ID]
        .astype(str)
    )

    df[HOUR_COLUMN] = pd.to_datetime(
        df[HOUR_COLUMN],
        utc=True,
        errors="raise",
    )

    df[CURRENT_PRECIP] = pd.to_numeric(
        df[CURRENT_PRECIP],
        errors="coerce",
    )

    return df


# ============================================================
# LOAD RESPONSE DAY
# ============================================================

def load_response_day(
    day: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load one forward hourly response partition.
    """

    key = response_partition_key(
        day
    )

    if not object_exists(
        key
    ):

        raise FileNotFoundError(
            "Forward response partition not found: "
            f"{key_uri(key)}"
        )

    df = pd.read_parquet(
        key_uri(key)
    )

    required = {
        SENSOR_ID,
        HOUR_COLUMN,
        OBSERVED_BINS,
        VALID_BINS,
        HIGH_BINS,
        MAX_DEPTH,
        RESPONSE_OBSERVED,
        TARGET_COLUMN,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Forward response partition missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    df = df[
        [
            SENSOR_ID,
            HOUR_COLUMN,
            OBSERVED_BINS,
            VALID_BINS,
            HIGH_BINS,
            MAX_DEPTH,
            RESPONSE_OBSERVED,
            TARGET_COLUMN,
        ]
    ].copy()

    df[SENSOR_ID] = (
        df[SENSOR_ID]
        .astype(str)
    )

    df[HOUR_COLUMN] = pd.to_datetime(
        df[HOUR_COLUMN],
        utc=True,
        errors="raise",
    )

    return df


# ============================================================
# HISTORICAL PRECIPITATION CONTEXT
# ============================================================

_HISTORICAL_PRECIP_CACHE: (
    pd.DataFrame
    | None
) = None


def historical_precip_context() -> pd.DataFrame:
    """
    Read historical precipitation values for feature context only.

    Historical data is NEVER modified or written by this script.

    Only deployment_id, hour and precip_current_hour_mm are retained.
    """

    global _HISTORICAL_PRECIP_CACHE

    if _HISTORICAL_PRECIP_CACHE is not None:

        return _HISTORICAL_PRECIP_CACHE

    print(
        "Loading immutable historical precipitation context..."
    )

    historical = (
        load_gcn_hourly_features()
    )

    required = {
        SENSOR_ID,
        HOUR_COLUMN,
        CURRENT_PRECIP,
    }

    missing = (
        required
        - set(historical.columns)
    )

    if missing:

        raise ValueError(
            "Historical precipitation context missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    historical = historical[
        [
            SENSOR_ID,
            HOUR_COLUMN,
            CURRENT_PRECIP,
        ]
    ].copy()

    historical[SENSOR_ID] = (
        historical[SENSOR_ID]
        .astype(str)
    )

    historical[HOUR_COLUMN] = pd.to_datetime(
        historical[HOUR_COLUMN],
        utc=True,
        errors="coerce",
    )

    historical[CURRENT_PRECIP] = pd.to_numeric(
        historical[CURRENT_PRECIP],
        errors="coerce",
    )

    historical = historical[
        historical[HOUR_COLUMN]
        <= HISTORICAL_CUTOFF
    ].copy()

    _HISTORICAL_PRECIP_CACHE = (
        historical
    )

    return (
        _HISTORICAL_PRECIP_CACHE
    )


# ============================================================
# LOAD PRECIPITATION CONTEXT FOR ONE DAY
# ============================================================

def load_precip_context(
    day: pd.Timestamp,
) -> pd.DataFrame:
    """
    Load enough precipitation information to calculate:

        precip_previous_6h_mm
        daily_total_precip_mm

    The context includes:

        - the current UTC day
        - up to 23 preceding feature hours

    On the first forward day, missing preceding hours are read from
    the immutable historical dataset.
    """

    day = pd.to_datetime(
        day,
        utc=True,
    ).normalize()

    lookback_start = (
        day
        - pd.Timedelta(hours=6)
    )

    day_end = (
        day
        + pd.Timedelta(days=1)
    )

    pieces = []

    # --------------------------------------------------------
    # Previous forward day when available
    # --------------------------------------------------------

    previous_day = (
        day
        - pd.Timedelta(days=1)
    )

    previous_key = (
        mrms_partition_key(
            previous_day
        )
    )

    if object_exists(
        previous_key
    ):

        previous = (
            load_mrms_day(
                previous_day
            )
        )

        previous = previous[
            (
                previous[HOUR_COLUMN]
                >= lookback_start
            )
            &
            (
                previous[HOUR_COLUMN]
                < day
            )
        ].copy()

        pieces.append(
            previous
        )

    # --------------------------------------------------------
    # Historical context when needed
    # --------------------------------------------------------

    if lookback_start <= HISTORICAL_CUTOFF:

        historical = (
            historical_precip_context()
        )

        historical_slice = historical[
            (
                historical[HOUR_COLUMN]
                >= lookback_start
            )
            &
            (
                historical[HOUR_COLUMN]
                < day
            )
        ].copy()

        pieces.append(
            historical_slice
        )

    # --------------------------------------------------------
    # Current forward day
    # --------------------------------------------------------

    current = load_mrms_day(
        day
    )

    current = current[
        (
            current[HOUR_COLUMN]
            >= day
        )
        &
        (
            current[HOUR_COLUMN]
            < day_end
        )
    ].copy()

    pieces.append(
        current
    )

    context = pd.concat(
        pieces,
        ignore_index=True,
    )

    context = (
        context.drop_duplicates(
            subset=[
                SENSOR_ID,
                HOUR_COLUMN,
            ],
            keep="last",
        )
        .sort_values(
            [
                SENSOR_ID,
                HOUR_COLUMN,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return context


# ============================================================
# PRECIPITATION FEATURE ENGINEERING
# ============================================================

def add_precipitation_features(
    context: pd.DataFrame,
    *,
    day: pd.Timestamp,
) -> pd.DataFrame:
    """
    Build precipitation features for the requested UTC day.

    Previous-6-hour definition
    --------------------------
    For feature hour H:

        sum precipitation for H-6 through H-1

    The current hour is intentionally excluded.

    Daily total definition
    ----------------------
    Sum precip_current_hour_mm across the UTC calendar day for each
    deployment.

    For the first partial forward day, immutable historical hours from
    earlier March 13 are used when available so the calendar-day total
    remains continuous across the historical/forward boundary.
    """

    day = pd.to_datetime(
        day,
        utc=True,
    ).normalize()

    day_end = (
        day
        + pd.Timedelta(days=1)
    )

    context = context.copy()

    context = (
        context.sort_values(
            [
                SENSOR_ID,
                HOUR_COLUMN,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Previous six feature hours
    # --------------------------------------------------------

    context[
        PREVIOUS_6H_PRECIP
    ] = (
        context.groupby(
            SENSOR_ID,
            sort=False,
        )[CURRENT_PRECIP]
        .transform(
            lambda series: (
                series
                .shift(1)
                .rolling(
                    window=6,
                    min_periods=1,
                )
                .sum()
            )
        )
    )

    # --------------------------------------------------------
    # Calendar-day total
    # --------------------------------------------------------

    current_day_mask = (
        (
            context[HOUR_COLUMN]
            >= day
        )
        &
        (
            context[HOUR_COLUMN]
            < day_end
        )
    )

    current_day = (
        context.loc[
            current_day_mask,
            [
                SENSOR_ID,
                HOUR_COLUMN,
                CURRENT_PRECIP,
                PREVIOUS_6H_PRECIP,
            ],
        ]
        .copy()
    )

    # For the first forward day, append historical hours earlier
    # on that same calendar day before calculating the day total.
    daily_source_parts = [
        current_day[
            [
                SENSOR_ID,
                HOUR_COLUMN,
                CURRENT_PRECIP,
            ]
        ]
    ]

    if day <= HISTORICAL_CUTOFF.normalize():

        historical = (
            historical_precip_context()
        )

        earlier_same_day = historical[
            (
                historical[HOUR_COLUMN]
                >= day
            )
            &
            (
                historical[HOUR_COLUMN]
                <= HISTORICAL_CUTOFF
            )
        ][
            [
                SENSOR_ID,
                HOUR_COLUMN,
                CURRENT_PRECIP,
            ]
        ].copy()

        daily_source_parts.append(
            earlier_same_day
        )

    daily_source = pd.concat(
        daily_source_parts,
        ignore_index=True,
    )

    daily_source = (
        daily_source.drop_duplicates(
            subset=[
                SENSOR_ID,
                HOUR_COLUMN,
            ],
            keep="last",
        )
    )

    daily_totals = (
        daily_source.groupby(
            SENSOR_ID,
            as_index=False,
        )[CURRENT_PRECIP]
        .sum(
            min_count=1
        )
        .rename(
            columns={
                CURRENT_PRECIP: (
                    DAILY_PRECIP
                )
            }
        )
    )

    current_day = current_day.merge(
        daily_totals,
        on=SENSOR_ID,
        how="left",
        validate="many_to_one",
    )

    return current_day


# ============================================================
# RESPONSE DEFAULTS FOR NO OBSERVATIONS
# ============================================================

def fill_missing_response(
    canonical: pd.DataFrame,
) -> pd.DataFrame:
    """
    Represent an active sensor-hour with no FloodNet response row as
    unobserved rather than silently deleting its predictor row.

    Active sensors without sufficient FloodNet observations remain in
    the canonical feature dataset with:

        response_observed = False

    Their flood-duration target and hourly maximum depth remain NaN,
    because those values were not observed.
    """

    result = canonical.copy()

    # Normalize response_observed to Pandas nullable Boolean before
    # assigning False. This avoids assigning a Boolean into a float
    # column created by the left join.
    result[RESPONSE_OBSERVED] = (
        result[RESPONSE_OBSERVED]
        .astype("boolean")
    )

    missing_response = (
        result[RESPONSE_OBSERVED]
        .isna()
    )

    # A completely missing response row means there were zero
    # reconstructed observations/bins for that active sensor-hour.
    for column in [
        OBSERVED_BINS,
        VALID_BINS,
        HIGH_BINS,
    ]:
        result.loc[
            missing_response,
            column,
        ] = 0

    result.loc[
        missing_response,
        RESPONSE_OBSERVED,
    ] = False

    # Preserve missing hourly_max_depth_mm and target duration for
    # unobserved rows. Zero would incorrectly imply that we observed
    # the sensor and confirmed no flooding.
    result[RESPONSE_OBSERVED] = (
        result[RESPONSE_OBSERVED]
        .fillna(False)
        .astype(bool)
    )

    return result


# ============================================================
# BUILD ONE CANONICAL DAY
# ============================================================

def build_canonical_day(
    day: pd.Timestamp,
    *,
    start_hour: pd.Timestamp,
    end_hour: pd.Timestamp,
    sensor_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build one forward canonical UTC-day partition.
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

    # --------------------------------------------------------
    # Precipitation
    # --------------------------------------------------------

    print(
        "    loading precipitation context..."
    )

    precip_context = (
        load_precip_context(
            day
        )
    )

    precip = (
        add_precipitation_features(
            precip_context,
            day=day,
        )
    )

    precip = precip[
        (
            precip[HOUR_COLUMN]
            >= effective_start
        )
        &
        (
            precip[HOUR_COLUMN]
            < effective_end
        )
    ].copy()

    print(
        f"    predictor rows before lifecycle filter: "
        f"{len(precip):,}"
    )

    # --------------------------------------------------------
    # Lifecycle
    # --------------------------------------------------------

    precip = apply_lifecycle_filter(
        precip,
        sensor_mapping,
    )

    print(
        f"    predictor rows after lifecycle filter: "
        f"{len(precip):,}"
    )

    print(
        f"    active sensors represented: "
        f"{precip[SENSOR_ID].nunique():,}"
    )

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    response = load_response_day(
        day
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

    # --------------------------------------------------------
    # Join
    # --------------------------------------------------------

    canonical = precip.merge(
        response,
        on=[
            SENSOR_ID,
            HOUR_COLUMN,
        ],
        how="left",
        validate="one_to_one",
    )

    canonical = (
        fill_missing_response(
            canonical
        )
    )

    canonical = canonical[
        CANONICAL_COLUMNS
    ].copy()

    canonical = (
        canonical.sort_values(
            [
                HOUR_COLUMN,
                SENSOR_ID,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return canonical


# ============================================================
# VALIDATE CANONICAL PARTITION
# ============================================================

def validate_canonical_partition(
    df: pd.DataFrame,
) -> None:
    """
    Validate one forward canonical partition before upload.
    """

    if df.empty:

        raise ValueError(
            "Cannot persist an empty forward canonical partition."
        )

    missing = (
        set(CANONICAL_COLUMNS)
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            "Canonical partition missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    if (
        df[HOUR_COLUMN]
        <= HISTORICAL_CUTOFF
    ).any():

        raise ValueError(
            "Forward canonical partition contains historical hours."
        )

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
            "canonical sensor-hour rows."
        )

    if (
        df[CURRENT_PRECIP]
        .isna()
        .any()
    ):

        raise ValueError(
            "Canonical rows contain missing current-hour precipitation."
        )

    if (
        df[CURRENT_PRECIP]
        .lt(0)
        .any()
    ):

        raise ValueError(
            "Canonical rows contain negative precipitation."
        )

    if (
        df[PREVIOUS_6H_PRECIP]
        .lt(0)
        .any()
    ):

        raise ValueError(
            "Canonical rows contain negative previous-6-hour precipitation."
        )

    if (
        df[DAILY_PRECIP]
        .lt(0)
        .any()
    ):

        raise ValueError(
            "Canonical rows contain negative daily precipitation."
        )

    observed = (
        df[RESPONSE_OBSERVED]
        .astype(bool)
    )

    if (
        df.loc[
            observed,
            TARGET_COLUMN,
        ]
        .isna()
        .any()
    ):

        raise ValueError(
            "Observed response rows contain missing flood duration."
        )


# ============================================================
# UPLOAD
# ============================================================

def upload_canonical_partition(
    df: pd.DataFrame,
    day: pd.Timestamp,
) -> str:
    """
    Persist one daily forward canonical parquet partition.
    """

    buffer = io.BytesIO()

    df.to_parquet(
        buffer,
        index=False,
    )

    buffer.seek(0)

    key = canonical_partition_key(
        day
    )

    s3_client().put_object(
        Bucket=DATA_BUCKET,
        Key=key,
        Body=buffer.getvalue(),
    )

    return key_uri(
        key
    )


# ============================================================
# SUMMARY
# ============================================================

def print_canonical_summary(
    df: pd.DataFrame,
) -> None:
    """
    Print one partition's canonical diagnostics.
    """

    observed = (
        df[RESPONSE_OBSERVED]
        .astype(bool)
    )

    positive = (
        observed
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
        f"    observed responses: "
        f"{observed.sum():,}"
    )

    print(
        f"    unobserved responses: "
        f"{(~observed).sum():,}"
    )

    print(
        "    positive flood-duration rows: "
        f"{positive.sum():,}"
    )

    print(
        "    mean current-hour precip: "
        f"{df[CURRENT_PRECIP].mean():.4f} mm"
    )

    print(
        "    max current-hour precip: "
        f"{df[CURRENT_PRECIP].max():.4f} mm"
    )

    print(
        "    mean previous-6h precip: "
        f"{df[PREVIOUS_6H_PRECIP].mean():.4f} mm"
    )

    print(
        "    max daily precip: "
        f"{df[DAILY_PRECIP].max():.4f} mm"
    )


# ============================================================
# FORWARD CANONICAL BUILD
# ============================================================

def run_forward_canonical_build(
    *,
    start,
    end,
) -> None:
    """
    Build resumable append-only forward canonical partitions.
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
        "FORWARD CANONICAL TRAINING BUILD"
    )

    print(
        "================================"
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

    print(
        "\nLoading lifecycle-aware sensor mapping..."
    )

    sensor_mapping = (
        load_sensor_mapping()
    )

    print(
        f"Canonical mapped sensors: "
        f"{sensor_mapping[SENSOR_ID].nunique():,}"
    )

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

    missing_inputs = 0

    empty = 0

    failed = 0

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

        canonical_key = (
            canonical_partition_key(
                day
            )
        )

        if object_exists(
            canonical_key
        ):

            print(
                "    existing forward canonical partition "
                "found; skipping"
            )

            skipped += 1

            continue

        # ----------------------------------------------------
        # Both forward source layers must exist.
        # ----------------------------------------------------

        required_keys = [
            mrms_partition_key(day),
            response_partition_key(day),
        ]

        missing = [
            key
            for key in required_keys
            if not object_exists(key)
        ]

        if missing:

            print(
                "    required source partition(s) missing:"
            )

            for key in missing:

                print(
                    f"    {key_uri(key)}"
                )

            missing_inputs += 1

            continue

        try:

            canonical = build_canonical_day(
                day,
                start_hour=start_hour,
                end_hour=end_hour,
                sensor_mapping=sensor_mapping,
            )

            if canonical.empty:

                print(
                    "    no canonical rows produced"
                )

                empty += 1

                continue

            validate_canonical_partition(
                canonical
            )

            print_canonical_summary(
                canonical
            )

            uri = upload_canonical_partition(
                canonical,
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
        "FORWARD CANONICAL BUILD COMPLETE"
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
        f"Missing source partitions: "
        f"{missing_inputs}"
    )

    print(
        f"Empty canonical days: "
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
    Parse CLI arguments.
    """

    parser = argparse.ArgumentParser(
        description=(
            "Build append-only forward canonical flood "
            "training partitions."
        )
    )

    parser.add_argument(
        "--start",
        default=(
            FORWARD_START.isoformat()
        ),
        help=(
            "Inclusive feature-hour start. "
            "May not precede the historical cutoff."
        ),
    )

    parser.add_argument(
        "--end",
        default=(
            DEFAULT_END.isoformat()
        ),
        help=(
            "Exclusive end. Defaults to the beginning "
            "of the current UTC week."
        ),
    )

    return parser.parse_args()


def main() -> None:
    """
    CLI entry point.
    """

    args = parse_args()

    run_forward_canonical_build(
        start=args.start,
        end=args.end,
    )


if __name__ == "__main__":
    main()