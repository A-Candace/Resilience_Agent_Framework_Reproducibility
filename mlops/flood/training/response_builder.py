"""
Shared flood-response construction for NYC flood models.

This module reconstructs the hourly response from minute-level
FloodNet depth observations.

The goal is to preserve the scientific target definition from the
original GCN notebook so that GCN, Logistic Regression, and any future
candidate models can be evaluated against the same response.
"""

from __future__ import annotations

import pandas as pd

from mlops.flood.training.train_gcn import (
    BIN_FREQUENCY,
    BIN_MINUTES,
    DEPTH_THRESHOLD_MM,
    MIN_VALID_DEPTH_BINS_PER_HOUR,
    SENSOR_ID,
    SOURCE_TARGET_COLUMN,
)

TARGET_COLUMN = SOURCE_TARGET_COLUMN

# ---------------------------------------------------------------------
# Raw FloodNet observation schema
# ---------------------------------------------------------------------
#
# These columns belong to the raw-response construction layer, not the
# GCN training layer.  The GCN consumes canonical hourly training data
# after the response has already been reconstructed.
#
# Keep these names aligned with build_forward_response.py.

RAW_TIME_COLUMN = "time"
RAW_DEPTH_COLUMN = "depth_proc_mm"

# ============================================================
# RESPONSE CONSTRUCTION
# ============================================================

def build_hourly_flood_response(
    raw_depth: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert minute-level FloodNet depth observations into the
    hourly flood-duration response used by the GCN workflow.

    Scientific logic
    ----------------
    1. Parse timestamps in UTC.
    2. Aggregate raw depth readings into 5-minute bins.
    3. Use the maximum observed depth within each 5-minute bin.
    4. Mark a bin as above threshold when:

           depth_proc_mm > DEPTH_THRESHOLD_MM

    5. Aggregate to sensor-hour level.
    6. Require at least MIN_VALID_DEPTH_BINS_PER_HOUR valid
       5-minute bins for the hourly response to be considered
       observed.
    7. Count high-depth bins and multiply by BIN_MINUTES to
       produce flood duration in minutes.

    Notes
    -----
    The active GCN notebook currently uses a threshold of:

        25.4 mm = 1 inch

    even though TARGET_COLUMN is still named
    "minutes_above_1p5_inch".

    This module preserves the active behavior rather than silently
    changing the scientific definition.
    """

    required_columns = {
        SENSOR_ID,
        RAW_TIME_COLUMN,
        RAW_DEPTH_COLUMN,
    }

    missing = required_columns - set(
        raw_depth.columns
    )

    if missing:
        raise ValueError(
            "Raw FloodNet data is missing required columns: "
            + ", ".join(sorted(missing))
        )

    df = raw_depth[
        [
            SENSOR_ID,
            RAW_TIME_COLUMN,
            RAW_DEPTH_COLUMN,
        ]
    ].copy()

    df[RAW_TIME_COLUMN] = pd.to_datetime(
        df[RAW_TIME_COLUMN],
        utc=True,
        errors="coerce",
    )

    df[RAW_DEPTH_COLUMN] = pd.to_numeric(
        df[RAW_DEPTH_COLUMN],
        errors="coerce",
    )

    df = df.dropna(
        subset=[
            SENSOR_ID,
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
        .reset_index(drop=True)
    )    
    # ========================================================
    # 5-MINUTE SENSOR BINS
    # ========================================================

    df["bin_5min"] = (
        df[RAW_TIME_COLUMN]
        .dt.floor(BIN_FREQUENCY)
    )

    five_minute = (
        df.groupby(
            [
                SENSOR_ID,
                "bin_5min",
            ],
            as_index=False,
        )
        .agg(
            depth_5min_mm=(
                RAW_DEPTH_COLUMN,
                "max",
            )
        )
    )

    five_minute["depth_valid"] = (
        five_minute[
            "depth_5min_mm"
        ].notna()
    ).astype(int)

    five_minute[
        "above_threshold"
    ] = (
        five_minute[
            "depth_5min_mm"
        ]
        > DEPTH_THRESHOLD_MM
    ).astype(int)

    # ========================================================
    # HOURLY SENSOR AGGREGATION
    # ========================================================

    five_minute["hour"] = (
        five_minute["bin_5min"]
        .dt.floor("h")
    )

    hourly = (
        five_minute.groupby(
            [
                SENSOR_ID,
                "hour",
            ],
            as_index=False,
        )
        .agg(
            observed_5min_bins=(
                "bin_5min",
                "size",
            ),
            valid_depth_5min_bins=(
                "depth_valid",
                "sum",
            ),
            high_depth_5min_bins=(
                "above_threshold",
                "sum",
            ),
            hourly_max_depth_mm=(
                "depth_5min_mm",
                "max",
            ),
        )
    )

    # ========================================================
    # RESPONSE OBSERVATION MASK
    # ========================================================

    hourly["response_observed"] = (
        hourly[
            "valid_depth_5min_bins"
        ]
        >= MIN_VALID_DEPTH_BINS_PER_HOUR
    ).astype(int)

    # ========================================================
    # FLOOD-DURATION RESPONSE
    # ========================================================

    hourly[TARGET_COLUMN] = (
        hourly[
            "high_depth_5min_bins"
        ]
        * BIN_MINUTES
    ).astype(float)

    # If response quality is insufficient, the target should
    # not participate in supervised training.
    hourly.loc[
        hourly["response_observed"] == 0,
        TARGET_COLUMN,
    ] = pd.NA

    # The physical upper bound for one hour is 60 minutes.
    hourly[TARGET_COLUMN] = (
        pd.to_numeric(
            hourly[TARGET_COLUMN],
            errors="coerce",
        )
        .clip(
            lower=0,
            upper=60,
        )
    )

    return hourly.sort_values(
        [
            "hour",
            SENSOR_ID,
        ]
    ).reset_index(drop=True)