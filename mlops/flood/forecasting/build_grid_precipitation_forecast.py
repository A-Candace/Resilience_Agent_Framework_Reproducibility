"""
Build HRRR precipitation predictors for NYC 1-km flood forecasting.

Supported operational forecast modes
------------------------------------

today
    HRRR-based same-day forecast for the complete current
    America/New_York calendar day.

tomorrow
    HRRR-based day-ahead forecast for the complete next
    America/New_York calendar day.

Important spatial contract
--------------------------

The operational prediction geography is the NYC 1-km grid.

HRRR remains at its native meteorological resolution. Each target
1-km grid is assigned its nearest HRRR forecast point. We do NOT
interpolate HRRR into artificial 1-km meteorological detail.

The output therefore retains:

    hrrr_lat
    hrrr_lon
    hrrr_distance_km

to make the true forcing resolution/provenance explicit.

Important model-feature contract
--------------------------------

The selected flood models consume:

    precip_current_hour_mm
    precip_previous_6h_mm
    daily_total_precip_mm

daily_total_precip_mm retains its legacy downstream name, but for
operational forecasting it is defined as a centered rolling 24-hour
HRRR precipitation context from H-12 through H+11.

The operational valid-time fields may be stitched from multiple
extended HRRR cycles. When the full target day cannot yet receive a
complete H-12 through H+11 context, the published horizon may be
truncated at the latest scientifically complete hour. For each valid time, the newest available cycle
that can supply a positive-lead one-hour APCP field is selected.

All operational timestamps remain timezone-aware and UTC internally.

Training / retraining continues to use MRMS observations.
These HRRR grid forecast rows are operational inference inputs only.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import (
    date,
    datetime,
    timedelta,
    timezone,
)
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from mlops.flood.forecasting.hrrr import (
    HRRRCycle,
    build_flood_precipitation_predictors,
    candidate_extended_cycles,
    create_unsigned_s3_client,
    cycle_supports_forecast_hour,
    retrieve_grid_hourly_precipitation,
)


# ============================================================
# PATHS
# ============================================================

DEFAULT_GRID_PATH = Path(
    "data/raw/geospatial/grid_clusters.parquet"
)

DEFAULT_PROCESSED_OUTPUT_ROOT = Path(
    "data/processed/forecasting"
)

DEFAULT_ARTIFACT_OUTPUT_ROOT = Path(
    "artifacts/flood/forecasting"
)

OUTPUT_PARQUET = (
    "grid_precipitation_forecast.parquet"
)

OUTPUT_CSV = (
    "grid_precipitation_forecast.csv"
)

CONTEXT_PARQUET = (
    "hrrr_hourly_grid_context.parquet"
)

SUMMARY_JSON = (
    "grid_precipitation_forecast_summary.json"
)


# ============================================================
# OPERATIONAL CONTRACT
# ============================================================

NYC_TIMEZONE_NAME = (
    "America/New_York"
)

NYC_TIMEZONE = ZoneInfo(
    NYC_TIMEZONE_NAME
)

UTC = timezone.utc

PREVIOUS_6H_HOURS = 6

ROLLING_24H_PAST_HOURS = 23
ROLLING_24H_FUTURE_HOURS = 0

GRID_ID = "grid_id"

GRID_LAT = "grid_lat"

GRID_LON = "grid_lon"

VALID_TIME = (
    "forecast_hour_valid_time"
)

HOURLY_PRECIP = (
    "precip_hourly_mm"
)

MODEL_FEATURES = [
    "precip_current_hour_mm",
    "precip_previous_6h_mm",
    "daily_total_precip_mm",
]

SUPPORTED_MODES = {
    "today",
    "tomorrow",
}


# ============================================================
# FORECAST WINDOW
# ============================================================


@dataclass(
    frozen=True
)
class ForecastWindow:
    """
    Operational NYC-local forecast product and HRRR predictor window.

    The published product is the complete target America/New_York
    calendar day.

    The weather retrieval window extends beyond the published target
    hours because the operational precipitation features require:

        precip_previous_6h_mm
            H-6 through H-1

        daily_total_precip_mm
            centered rolling 24-hour precipitation context:
            H-12 through H+11

    Extra weather-context hours are predictor inputs only and are
    never published as additional forecast rows.
    """

    mode: str

    target_date_nyc: date

    target_start_nyc: datetime

    next_day_start_nyc: datetime

    target_start_utc: datetime

    target_end_utc: datetime

    context_start_utc: datetime

    retrieval_end_utc: datetime

    target_hour_count: int


# ============================================================
# TIME HELPERS
# ============================================================


def ensure_utc(
    value: datetime,
) -> datetime:
    """Return a timezone-aware UTC datetime."""

    if value.tzinfo is None:
        return value.replace(
            tzinfo=UTC
        )

    return value.astimezone(
        UTC
    )


def ensure_nyc(
    value: datetime,
) -> datetime:
    """Return a timezone-aware NYC datetime."""

    if value.tzinfo is None:
        value = value.replace(
            tzinfo=UTC
        )

    return value.astimezone(
        NYC_TIMEZONE
    )


def parse_reference_time(
    value: str | None,
) -> datetime | None:
    """
    Parse an optional reproducibility/reference timestamp.

    Naive values are interpreted as UTC.
    """

    if value is None:
        return None

    normalized = (
        value.strip()
        .replace(
            "Z",
            "+00:00",
        )
    )

    try:
        parsed = datetime.fromisoformat(
            normalized
        )

    except ValueError as exc:
        raise ValueError(
            "--reference-time must be an ISO-8601 datetime"
        ) from exc

    return ensure_utc(
        parsed
    )


def local_today(
    reference_time: datetime | None = None,
) -> date:
    """
    Resolve today's America/New_York calendar date.
    """

    if reference_time is None:
        reference_time = datetime.now(
            UTC
        )

    return ensure_nyc(
        reference_time
    ).date()


def resolve_target_date(
    *,
    mode: str,
    target_date_override: str | None,
    reference_time: datetime | None,
) -> date:
    """
    Resolve the operational NYC calendar date.

    --target-date is an explicit NYC calendar-date override.
    """

    mode = (
        mode
        .strip()
        .lower()
    )

    if mode not in SUPPORTED_MODES:
        raise ValueError(
            "Unsupported forecast mode: "
            f"{mode}"
        )

    if target_date_override is not None:

        try:
            return date.fromisoformat(
                target_date_override
            )

        except ValueError as exc:
            raise ValueError(
                "--target-date must use YYYY-MM-DD"
            ) from exc

    today = local_today(
        reference_time
    )

    if mode == "today":
        return today

    return (
        today
        + timedelta(
            days=1
        )
    )


def build_forecast_window(
    *,
    mode: str,
    target_date: date,
) -> ForecastWindow:
    """
    Build a complete America/New_York calendar-day window.

    The corresponding UTC interval may cross UTC calendar dates.

    This also correctly allows 23-hour or 25-hour local days during
    daylight-saving transitions.
    """

    target_start_nyc = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        hour=0,
        minute=0,
        second=0,
        tzinfo=NYC_TIMEZONE,
    )

    next_date = (
        target_date
        + timedelta(
            days=1
        )
    )

    next_day_start_nyc = datetime(
        next_date.year,
        next_date.month,
        next_date.day,
        hour=0,
        minute=0,
        second=0,
        tzinfo=NYC_TIMEZONE,
    )

    target_start_utc = (
        target_start_nyc
        .astimezone(
            UTC
        )
    )

    next_day_start_utc = (
        next_day_start_nyc
        .astimezone(
            UTC
        )
    )

    target_end_utc = (
        next_day_start_utc
        - timedelta(
            hours=1
        )
    )
    # The centered operational 24-hour precipitation context for
    # the first published hour needs H-12.  This is also more
    # conservative than the six-hour antecedent requirement.
    context_start_utc = (
        target_start_utc
        - timedelta(
            hours=ROLLING_24H_PAST_HOURS
        )
    )

    # The final published hour needs H+11 to complete its centered
    # 24-hour precipitation context.  These post-target HRRR hours
    # are predictor context only and are never published as flood
    # forecast rows.
    retrieval_end_utc = (
        target_end_utc
        + timedelta(
            hours=ROLLING_24H_FUTURE_HOURS
        )
    )

    target_times = pd.date_range(
        start=target_start_utc,
        end=target_end_utc,
        freq="h",
        tz="UTC",
    )

    return ForecastWindow(
        mode=mode,
        target_date_nyc=target_date,
        target_start_nyc=target_start_nyc,
        next_day_start_nyc=next_day_start_nyc,
        target_start_utc=target_start_utc,
        target_end_utc=target_end_utc,
        context_start_utc=context_start_utc,
        retrieval_end_utc=retrieval_end_utc,
        target_hour_count=len(
            target_times
        ),
    )


# ============================================================
# GRID
# ============================================================


def load_target_grid(
    path: Path,
) -> pd.DataFrame:
    """Load the canonical operational NYC 1-km grid."""

    actual_path = path

    if (
        not actual_path.exists()
        and actual_path.suffix.lower()
        in {
            ".parquet",
            ".pq",
        }
    ):

        csv_fallback = (
            actual_path
            .with_suffix(
                ".csv"
            )
        )

        if csv_fallback.exists():
            actual_path = (
                csv_fallback
            )

    if not actual_path.exists():

        raise FileNotFoundError(
            f"Target grid table not found: {path}"
        )

    suffix = (
        actual_path
        .suffix
        .lower()
    )

    if suffix in {
        ".parquet",
        ".pq",
    }:

        frame = pd.read_parquet(
            actual_path
        )

    elif suffix == ".csv":

        frame = pd.read_csv(
            actual_path
        )

    else:

        raise ValueError(
            "Target grid must be CSV or Parquet"
        )

    required = {
        GRID_ID,
        GRID_LAT,
        GRID_LON,
    }

    missing = (
        required
        - set(
            frame.columns
        )
    )

    if missing:

        raise ValueError(
            "Target grid is missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    result = frame[
        [
            GRID_ID,
            GRID_LAT,
            GRID_LON,
        ]
    ].copy()

    result[
        GRID_ID
    ] = (
        result[
            GRID_ID
        ]
        .astype(str)
        .str.strip()
    )

    for column in [
        GRID_LAT,
        GRID_LON,
    ]:

        result[
            column
        ] = pd.to_numeric(
            result[
                column
            ],
            errors="coerce",
        )

    if result[
        [
            GRID_LAT,
            GRID_LON,
        ]
    ].isna().any().any():

        raise ValueError(
            "Target grid contains invalid coordinates"
        )

    if result[
        GRID_ID
    ].duplicated().any():

        raise ValueError(
            "Target grid contains duplicate grid_id values"
        )

    if result.empty:

        raise ValueError(
            "Target grid is empty"
        )

    return (
        result
        .sort_values(
            GRID_ID
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# HRRR LEAD HOURS
# ============================================================


def lead_hour(
    cycle: HRRRCycle,
    valid_time: datetime,
) -> int:
    """Calculate integer HRRR lead hour."""

    hours = (
        ensure_utc(
            valid_time
        )
        -
        ensure_utc(
            cycle.initialization_time
        )
    ).total_seconds() / 3600.0

    if not hours.is_integer():

        raise ValueError(
            "Forecast valid time is not aligned "
            "to an integer HRRR lead"
        )

    return int(
        hours
    )


def required_forecast_hours(
    cycle: HRRRCycle,
    window: ForecastWindow,
) -> list[int]:
    """
    Return the HRRR forecast lead hours required to cover the
    complete operational predictor window.

    Retrieval begins 12 hours before the first target hour and
    extends 11 hours after the final target hour so every published
    hour can receive its H-12 through H+11 rolling precipitation
    context. Extra context hours are not published as forecast rows.
    """

    minimum = lead_hour(
        cycle,
        window.context_start_utc,
    )

    maximum = lead_hour(
        cycle,
        window.retrieval_end_utc,
    )

    if minimum < 1:

        raise ValueError(
            "Cycle is too recent to provide the first required "
            "one-hour APCP context field"
        )

    if maximum < minimum:

        raise ValueError(
            "Invalid HRRR forecast window"
        )

    return list(
        range(
            minimum,
            maximum + 1,
        )
    )


def _select_plan_entry_for_valid_time(
    *,
    s3_client,
    cycles: list[HRRRCycle],
    valid_timestamp: pd.Timestamp,
) -> dict | None:
    """Select the newest available HRRR cycle for one valid time."""

    valid_time = valid_timestamp.to_pydatetime()

    for cycle in cycles:
        try:
            hour = lead_hour(
                cycle,
                valid_time,
            )
        except ValueError:
            continue

        # F00 is not used because this workflow requires a
        # one-hour APCP accumulation field.
        if hour < 1:
            continue

        if cycle_supports_forecast_hour(
            s3_client=s3_client,
            cycle=cycle,
            forecast_hour=hour,
        ):
            return {
                "valid_time": valid_timestamp,
                "cycle": cycle,
                "forecast_hour": int(hour),
            }

    return None


def build_hrrr_retrieval_plan(
    *,
    s3_client,
    window: ForecastWindow,
    reference_time: datetime | None = None,
) -> list[dict]:
    """
    Build a strict valid-time-aware HRRR retrieval plan.

    Every hour from context_start_utc through retrieval_end_utc must
    be available. Use resolve_available_forecast_window() when an
    adaptive target horizon is acceptable.
    """

    required_times = expected_context_times(window)
    cycles = candidate_extended_cycles(
        reference_time=reference_time,
        lookback_cycles=16,
    )

    if not cycles:
        raise RuntimeError(
            "No candidate extended HRRR cycles are available"
        )

    plan: list[dict] = []
    unresolved: list[str] = []

    for valid_timestamp in required_times:
        entry = _select_plan_entry_for_valid_time(
            s3_client=s3_client,
            cycles=cycles,
            valid_timestamp=valid_timestamp,
        )

        if entry is None:
            unresolved.append(
                valid_timestamp.isoformat()
            )
            continue

        plan.append(entry)

    if unresolved:
        raise RuntimeError(
            "Unable to assemble complete HRRR coverage for the "
            "centered rolling-24 predictor window. Missing valid "
            "times: "
            + ", ".join(unresolved[:20])
        )

    if len(plan) != len(required_times):
        raise RuntimeError(
            "HRRR retrieval plan does not contain exactly one "
            "entry per required valid time"
        )

    return plan


def resolve_available_forecast_window(
    *,
    s3_client,
    window: ForecastWindow,
    reference_time: datetime | None = None,
    require_full_day: bool = False,
) -> tuple[ForecastWindow, list[dict], dict]:
    """
    Resolve the maximum publishable target horizon with complete
    centered rolling-24 HRRR context.

    The requested NYC product is initially a full local calendar day.
    If the final H+11 context hours are not yet available, this function
    truncates only the END of the published target period. No missing
    HRRR hours are filled with zero and no partial rolling-24 totals are
    permitted.

    The context must be continuous beginning at context_start_utc. If a
    gap occurs before enough context exists to publish even the first
    target hour, the function fails.
    """

    required_times = expected_context_times(window)
    cycles = candidate_extended_cycles(
        reference_time=reference_time,
        lookback_cycles=16,
    )

    if not cycles:
        raise RuntimeError(
            "No candidate extended HRRR cycles are available"
        )

    contiguous_plan: list[dict] = []
    first_missing: pd.Timestamp | None = None

    for valid_timestamp in required_times:
        entry = _select_plan_entry_for_valid_time(
            s3_client=s3_client,
            cycles=cycles,
            valid_timestamp=valid_timestamp,
        )

        if entry is None:
            first_missing = valid_timestamp
            break

        contiguous_plan.append(entry)

    full_available = (
        first_missing is None
        and len(contiguous_plan) == len(required_times)
    )

    if full_available:
        metadata = {
            "requested_target_hour_count": int(window.target_hour_count),
            "published_target_hour_count": int(window.target_hour_count),
            "forecast_horizon_truncated": False,
            "first_missing_context_time_utc": None,
        }
        return window, contiguous_plan, metadata

    if require_full_day:
        missing_text = (
            first_missing.isoformat()
            if first_missing is not None
            else "unknown"
        )
        raise RuntimeError(
            "Full-day HRRR coverage is required but unavailable. "
            f"First missing context valid time: {missing_text}"
        )

    if not contiguous_plan:
        raise RuntimeError(
            "No continuous HRRR predictor context is available"
        )

    last_available_context = pd.Timestamp(
        contiguous_plan[-1]["valid_time"]
    )

    latest_publishable_target = (
        last_available_context
        - pd.Timedelta(
            hours=ROLLING_24H_FUTURE_HOURS
        )
    )

    requested_start = pd.Timestamp(
        window.target_start_utc
    )
    requested_end = pd.Timestamp(
        window.target_end_utc
    )

    effective_end = min(
        requested_end,
        latest_publishable_target,
    )

    if effective_end < requested_start:
        raise RuntimeError(
            "HRRR context does not yet support even the first target "
            "hour with a complete H-12 through H+11 precipitation "
            "window"
        )

    effective_retrieval_end = (
        effective_end
        + pd.Timedelta(
            hours=ROLLING_24H_FUTURE_HOURS
        )
    )

    truncated_plan = [
        entry
        for entry in contiguous_plan
        if pd.Timestamp(entry["valid_time"])
        <= effective_retrieval_end
    ]

    target_times = pd.date_range(
        start=requested_start,
        end=effective_end,
        freq="h",
        tz="UTC",
    )

    effective_window = replace(
        window,
        target_end_utc=effective_end.to_pydatetime(),
        retrieval_end_utc=effective_retrieval_end.to_pydatetime(),
        target_hour_count=len(target_times),
    )

    metadata = {
        "requested_target_hour_count": int(window.target_hour_count),
        "published_target_hour_count": int(effective_window.target_hour_count),
        "forecast_horizon_truncated": True,
        "first_missing_context_time_utc": (
            first_missing.isoformat()
            if first_missing is not None
            else None
        ),
    }

    return effective_window, truncated_plan, metadata


def retrieve_planned_hourly_precipitation(
    *,
    grid: pd.DataFrame,
    plan: list[dict],
    s3_client,
    keep_grib: bool,
) -> pd.DataFrame:
    """Retrieve and stitch HRRR fields from the retrieval plan."""

    if not plan:
        raise ValueError("HRRR retrieval plan is empty")

    grouped: dict[str, dict] = {}

    for entry in plan:
        cycle = entry["cycle"]
        group = grouped.setdefault(
            cycle.cycle_id,
            {"cycle": cycle, "forecast_hours": []},
        )
        group["forecast_hours"].append(
            int(entry["forecast_hour"])
        )

    frames: list[pd.DataFrame] = []

    print()
    print("HRRR MULTI-CYCLE RETRIEVAL PLAN")
    print("---------------------------------")

    for group in grouped.values():
        cycle = group["cycle"]
        hours = sorted(set(group["forecast_hours"]))

        print(
            f"{cycle.cycle_id}: "
            f"{len(hours)} field(s), "
            f"F{hours[0]:02d} -> F{hours[-1]:02d}"
        )

        frame = retrieve_grid_hourly_precipitation(
            grid=grid,
            cycle=cycle,
            forecast_hours=hours,
            s3_client=s3_client,
            keep_grib=keep_grib,
        )

        frame = frame.copy()
        frame["retrieval_hrrr_cycle"] = cycle.cycle_id
        frame["retrieval_hrrr_initialization_time"] = ensure_utc(
            cycle.initialization_time
        )

        frames.append(frame)

    combined = pd.concat(
        frames,
        ignore_index=True,
    )

    combined[VALID_TIME] = pd.to_datetime(
        combined[VALID_TIME],
        utc=True,
        errors="coerce",
    )

    if combined[VALID_TIME].isna().any():
        raise RuntimeError(
            "Stitched HRRR retrieval contains invalid valid times"
        )

    expected_pairs = {
        (entry["valid_time"], entry["cycle"].cycle_id)
        for entry in plan
    }

    combined = combined.loc[
        combined.apply(
            lambda row: (
                row[VALID_TIME],
                row["retrieval_hrrr_cycle"],
            ) in expected_pairs,
            axis=1,
        )
    ].copy()

    return combined.sort_values(
        [VALID_TIME, GRID_ID]
    ).reset_index(drop=True)


# ============================================================
# CONTEXT VALIDATION
# ============================================================


def expected_context_times(
    window: ForecastWindow,
) -> pd.DatetimeIndex:
    """
    Expected hourly UTC timestamps required for predictor
    construction.

    This includes:

        centered rolling-24 antecedent context (H-12)
        +
        NYC target forecast hours
        +
        centered rolling-24 future context (through H+11).
    """

    return pd.date_range(
        start=window.context_start_utc,
        end=window.retrieval_end_utc,
        freq="h",
        tz="UTC",
    )


def expected_target_times(
    window: ForecastWindow,
) -> pd.DatetimeIndex:
    """Expected hourly timestamps for the NYC-local target day."""

    return pd.date_range(
        start=window.target_start_utc,
        end=window.target_end_utc,
        freq="h",
        tz="UTC",
    )


def validate_hourly_context(
    *,
    frame: pd.DataFrame,
    grid: pd.DataFrame,
    window: ForecastWindow,
) -> pd.DataFrame:
    """
    Require exact hourly and spatial HRRR coverage.
    """

    result = frame.copy()

    required_columns = {
        GRID_ID,
        VALID_TIME,
        GRID_LAT,
        GRID_LON,
        HOURLY_PRECIP,
    }

    missing_columns = (
        required_columns
        - set(
            result.columns
        )
    )

    if missing_columns:

        raise ValueError(
            "HRRR context is missing required columns: "
            + ", ".join(
                sorted(
                    missing_columns
                )
            )
        )

    result[
        VALID_TIME
    ] = pd.to_datetime(
        result[
            VALID_TIME
        ],
        utc=True,
        errors="coerce",
    )

    if result[
        VALID_TIME
    ].isna().any():

        raise ValueError(
            "HRRR context contains invalid valid times"
        )

    result[
        HOURLY_PRECIP
    ] = pd.to_numeric(
        result[
            HOURLY_PRECIP
        ],
        errors="coerce",
    )

    if result[
        HOURLY_PRECIP
    ].isna().any():

        raise ValueError(
            "HRRR context contains invalid hourly precipitation"
        )

    if (
        result[
            HOURLY_PRECIP
        ]
        < 0
    ).any():

        raise ValueError(
            "HRRR context contains negative precipitation"
        )

    expected_times = (
        expected_context_times(
            window
        )
    )

    observed_times = pd.DatetimeIndex(
        sorted(
            result[
                VALID_TIME
            ].unique()
        )
    )

    if not observed_times.equals(
        expected_times
    ):

        missing = sorted(
            set(
                expected_times
            )
            -
            set(
                observed_times
            )
        )

        extra = sorted(
            set(
                observed_times
            )
            -
            set(
                expected_times
            )
        )

        raise ValueError(
            "HRRR context does not exactly cover context + "
            "target window. "
            f"Missing={missing[:10]}, "
            f"extra={extra[:10]}"
        )

    expected_ids = set(
        grid[
            GRID_ID
        ]
    )

    for valid_time in expected_times:

        subset = result.loc[
            result[
                VALID_TIME
            ]
            == valid_time
        ]

        observed_ids = set(
            subset[
                GRID_ID
            ]
        )

        if (
            len(
                subset
            )
            != len(
                expected_ids
            )
            or observed_ids
            != expected_ids
        ):

            raise ValueError(
                "HRRR context grid coverage failed at "
                f"{valid_time}: expected "
                f"{len(expected_ids)}, got "
                f"{len(subset)}"
            )

    if result.duplicated(
        [
            GRID_ID,
            VALID_TIME,
        ]
    ).any():

        raise ValueError(
            "HRRR context contains duplicate grid/time rows"
        )

    return (
        result
        .sort_values(
            [
                VALID_TIME,
                GRID_ID,
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# MODEL PREDICTORS
# ============================================================


def compute_centered_rolling24_total(
    *,
    hourly_context: pd.DataFrame,
    window: ForecastWindow,
) -> pd.DataFrame:
    """
    Calculate the operational centered rolling 24-hour
    precipitation context for every published target grid/hour.

    The legacy downstream feature name is retained:

        daily_total_precip_mm

    Operationally, however, the value at forecast hour H is:

        H-12 through H+11

    which is exactly 24 hourly HRRR one-hour APCP values.

    This feature is intentionally different from the static
    calendar-day MRMS total used by the historical training
    pipeline.  The training pipeline is not modified here.

    Missing or non-contiguous HRRR hours are never interpreted as
    zero; they cause the operational build to fail explicitly.
    """

    required_columns = {
        GRID_ID,
        VALID_TIME,
        HOURLY_PRECIP,
    }

    missing = required_columns - set(hourly_context.columns)
    if missing:
        raise RuntimeError(
            "Cannot compute centered rolling-24 precipitation. "
            "Hourly HRRR context is missing required columns: "
            + ", ".join(sorted(missing))
        )

    frame = hourly_context[
        [GRID_ID, VALID_TIME, HOURLY_PRECIP]
    ].copy()

    frame[GRID_ID] = frame[GRID_ID].astype(str).str.strip()
    if frame[GRID_ID].eq("").any():
        raise RuntimeError(
            "Cannot compute centered rolling-24 precipitation "
            "because one or more grid_id values are empty"
        )

    frame[VALID_TIME] = pd.to_datetime(
        frame[VALID_TIME],
        utc=True,
        errors="coerce",
    )
    if frame[VALID_TIME].isna().any():
        raise RuntimeError(
            "Cannot compute centered rolling-24 precipitation "
            "because valid_time contains invalid timestamps"
        )

    frame[HOURLY_PRECIP] = pd.to_numeric(
        frame[HOURLY_PRECIP],
        errors="coerce",
    )
    if frame[HOURLY_PRECIP].isna().any():
        raise RuntimeError(
            "Cannot compute centered rolling-24 precipitation "
            "because hourly precipitation contains null values"
        )
    if (frame[HOURLY_PRECIP] < 0).any():
        raise RuntimeError(
            "Cannot compute centered rolling-24 precipitation "
            "because hourly precipitation contains negative values"
        )

    duplicate_mask = frame.duplicated(
        subset=[GRID_ID, VALID_TIME],
        keep=False,
    )
    if duplicate_mask.any():
        sample = (
            frame.loc[duplicate_mask, [GRID_ID, VALID_TIME]]
            .head(10)
            .to_dict(orient="records")
        )
        raise RuntimeError(
            "Duplicate grid/hour HRRR rows found while computing "
            "centered rolling-24 precipitation. "
            f"Examples: {sample}"
        )

    frame = (
        frame
        .sort_values([GRID_ID, VALID_TIME])
        .reset_index(drop=True)
    )

    # Build exactly 24 aligned hourly values for every row:
    # H-12, ..., H-1, H, H+1, ..., H+11.
    groups = frame.groupby(
        GRID_ID,
        sort=False,
        group_keys=False,
    )

    precip_parts: list[pd.Series] = []
    valid_parts: list[pd.Series] = []

    for offset in range(
        -ROLLING_24H_PAST_HOURS,
        ROLLING_24H_FUTURE_HOURS + 1,
    ):
        shifted_precip = groups[HOURLY_PRECIP].shift(-offset)
        shifted_time = groups[VALID_TIME].shift(-offset)
        expected_time = (
            frame[VALID_TIME]
            + pd.to_timedelta(offset, unit="h")
        )
        valid = shifted_time.eq(expected_time)
        precip_parts.append(shifted_precip.where(valid))
        valid_parts.append(valid)

    if len(precip_parts) != 24:
        raise RuntimeError(
            "Centered rolling precipitation definition must contain "
            f"exactly 24 hours; got {len(precip_parts)}"
        )

    precip_matrix = pd.concat(precip_parts, axis=1)
    valid_matrix = pd.concat(valid_parts, axis=1)
    complete = (
        valid_matrix.all(axis=1)
        & precip_matrix.notna().all(axis=1)
    )

    frame["daily_total_precip_mm"] = np.nan
    frame.loc[complete, "daily_total_precip_mm"] = (
        precip_matrix.loc[complete].sum(axis=1)
    )

    target = frame.loc[
        (frame[VALID_TIME] >= pd.Timestamp(window.target_start_utc))
        & (frame[VALID_TIME] <= pd.Timestamp(window.target_end_utc)),
        [GRID_ID, VALID_TIME, "daily_total_precip_mm"],
    ].copy()

    expected_rows = (
        frame[GRID_ID].nunique()
        * window.target_hour_count
    )
    if len(target) != expected_rows:
        raise RuntimeError(
            "Unexpected centered rolling-24 target row count: "
            f"expected {expected_rows:,}, got {len(target):,}"
        )

    if target["daily_total_precip_mm"].isna().any():
        sample = (
            target.loc[
                target["daily_total_precip_mm"].isna(),
                [GRID_ID, VALID_TIME],
            ]
            .head(10)
            .to_dict(orient="records")
        )
        raise RuntimeError(
            "Centered rolling-24 precipitation is incomplete for "
            "one or more published target rows. "
            f"Examples: {sample}"
        )

    if (target["daily_total_precip_mm"] < 0).any():
        raise RuntimeError(
            "Centered rolling-24 precipitation contains negative values"
        )

    if target.duplicated(subset=[GRID_ID, VALID_TIME]).any():
        raise RuntimeError(
            "Duplicate grid/hour rows were created while assigning "
            "centered rolling-24 precipitation"
        )

    return (
        target
        .sort_values([VALID_TIME, GRID_ID])
        .reset_index(drop=True)
    )

def build_target_day_predictors(
    *,
    hourly_context: pd.DataFrame,
    grid: pd.DataFrame,
    window: ForecastWindow,
) -> pd.DataFrame:
    """
    Build the three model predictors for each target grid/hour.

    current-hour and previous-6h predictors continue to use the
    established HRRR predictor-engineering implementation.

    daily_total_precip_mm is explicitly overwritten with the
    complete NYC-local calendar-day total.
    """

    engineered = (
        build_flood_precipitation_predictors(
            hourly_context
        )
    )

    engineered[
        VALID_TIME
    ] = pd.to_datetime(
        engineered[
            VALID_TIME
        ],
        utc=True,
        errors="coerce",
    )

    if engineered[
        VALID_TIME
    ].isna().any():

        raise RuntimeError(
            "Engineered HRRR predictors contain invalid timestamps"
        )

    target = engineered.loc[
        (
            engineered[
                VALID_TIME
            ]
            >= pd.Timestamp(
                window.target_start_utc
            )
        )
        &
        (
            engineered[
                VALID_TIME
            ]
            <= pd.Timestamp(
                window.target_end_utc
            )
        )
    ].copy()

    expected_rows = (
        len(
            grid
        )
        *
        window.target_hour_count
    )

    if len(
        target
    ) != expected_rows:

        raise RuntimeError(
            "Unexpected target-day row count: "
            f"expected {expected_rows:,}, "
            f"got {len(target):,}"
        )

    # --------------------------------------------------------
    # Validate predictors that do not depend on calendar-day
    # definition.
    # --------------------------------------------------------

    for feature in [
        "precip_current_hour_mm",
        "precip_previous_6h_mm",
    ]:

        if feature not in target.columns:

            raise RuntimeError(
                "HRRR predictor engineering did not produce "
                f"{feature}"
            )

        target[
            feature
        ] = pd.to_numeric(
            target[
                feature
            ],
            errors="coerce",
        )

        if target[
            feature
        ].isna().any():

            raise RuntimeError(
                f"{feature} contains null values"
            )

        if (
            target[
                feature
            ]
            < 0
        ).any():

            raise RuntimeError(
                f"{feature} contains negative values"
            )
    # --------------------------------------------------------
    # Operational centered rolling 24-hour precipitation context
    # --------------------------------------------------------

    daily_total = (
        compute_centered_rolling24_total(
            hourly_context=hourly_context,
            window=window,
        )
    )

    if (
        "daily_total_precip_mm"
        in target.columns
    ):

        target = target.drop(
            columns=[
                "daily_total_precip_mm"
            ]
        )

    target = target.merge(
        daily_total,
        on=[
            GRID_ID,
            VALID_TIME,
        ],
        how="left",
        validate="one_to_one",
    )

    if target[
        "daily_total_precip_mm"
    ].isna().any():

        raise RuntimeError(
            "Centered rolling-24 precipitation is missing "
            "for one or more target rows"
        )

    if (
        target[
            "daily_total_precip_mm"
        ]
        < 0
    ).any():

        raise RuntimeError(
            "daily_total_precip_mm contains negative values"
        )

    # --------------------------------------------------------
    # Temporal/product provenance
    # --------------------------------------------------------

    target[
        "forecast_hour"
    ] = target[
        VALID_TIME
    ]

    target[
        "forecast_hour_nyc"
    ] = (
        target[
            "forecast_hour"
        ]
        .dt.tz_convert(
            NYC_TIMEZONE_NAME
        )
    )

    target[
        "forecast_mode"
    ] = (
        window.mode
    )

    target[
        "target_date_nyc"
    ] = (
        window
        .target_date_nyc
        .isoformat()
    )

    # This is the actual UTC calendar date of each row.
    # It may differ across rows because one NYC calendar day
    # can span two UTC dates.
    target[
        "target_date_utc"
    ] = (
        target[
            "forecast_hour"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    target[
        "target_timezone"
    ] = (
        NYC_TIMEZONE_NAME
    )

    target[
        "hrrr_source"
    ] = (
        "noaa-hrrr-bdp-pds"
    )

    target[
        "weather_product"
    ] = (
        "HRRR CONUS surface 1-hour APCP"
    )

    target[
        "meteorological_native_resolution_km"
    ] = 3.0

    target[
        "operational_prediction_grid_km"
    ] = 1.0

    target[
        "spatial_assignment_method"
    ] = (
        "nearest_hrrr_point_to_target_grid"
    )

    target[
        "generated_at"
    ] = pd.Timestamp.now(
        tz="UTC"
    )

    output_columns = [
        GRID_ID,
        "forecast_hour",
        "forecast_hour_nyc",
        GRID_LAT,
        GRID_LON,

        "forecast_mode",
        "target_date_nyc",
        "target_date_utc",
        "target_timezone",

        "forecast_initialization",
        "hrrr_cycle_id",

        "hrrr_lat",
        "hrrr_lon",
        "hrrr_distance_km",

        "precip_current_hour_mm",
        "precip_previous_6h_mm",
        "daily_total_precip_mm",

        "hrrr_source",
        "weather_product",

        "meteorological_native_resolution_km",
        "operational_prediction_grid_km",
        "spatial_assignment_method",

        "hrrr_apcp_description",
        "generated_at",
    ]

    missing_output = [
        column
        for column in output_columns
        if column not in target.columns
    ]

    if missing_output:

        raise RuntimeError(
            "Final HRRR predictor table is missing columns: "
            + ", ".join(
                missing_output
            )
        )

    result = (
        target[
            output_columns
        ]
        .sort_values(
            [
                "forecast_hour",
                GRID_ID,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if result.duplicated(
        [
            GRID_ID,
            "forecast_hour",
        ]
    ).any():

        raise RuntimeError(
            "Final HRRR predictor output contains duplicate "
            "grid/hour rows"
        )

    return result


# ============================================================
# SUMMARY
# ============================================================


def build_summary(
    *,
    target: pd.DataFrame,
    context: pd.DataFrame,
    window: ForecastWindow,
    retrieval_plan: list[dict],
    horizon_metadata: dict,
) -> dict:
    """Build operational weather-product metadata."""

    target_start_nyc = (
        window.target_start_nyc
        .isoformat()
    )

    target_end_nyc = (
        pd.Timestamp(
            window.target_end_utc
        )
        .tz_convert(
            NYC_TIMEZONE_NAME
        )
        .isoformat()
    )

    return {
        "forecast_mode":
            window.mode,

        "target_timezone":
            NYC_TIMEZONE_NAME,

        "target_date_nyc":
            window.target_date_nyc.isoformat(),

        "target_start_nyc":
            target_start_nyc,

        "target_end_nyc":
            target_end_nyc,

        "target_start_utc":
            window.target_start_utc.isoformat(),

        "target_end_utc":
            window.target_end_utc.isoformat(),

        "context_start_utc":
            window.context_start_utc.isoformat(),

        "retrieval_end_utc":
            window.retrieval_end_utc.isoformat(),
        "previous_6h_hours":
            PREVIOUS_6H_HOURS,

        "rolling_24h_past_hours":
            ROLLING_24H_PAST_HOURS,

        "rolling_24h_future_hours":
            ROLLING_24H_FUTURE_HOURS,

        "target_hour_count":
            window.target_hour_count,

        "requested_target_hour_count":
            int(horizon_metadata["requested_target_hour_count"]),

        "published_target_hour_count":
            int(horizon_metadata["published_target_hour_count"]),

        "forecast_horizon_truncated":
            bool(horizon_metadata["forecast_horizon_truncated"]),

        "first_missing_context_time_utc":
            horizon_metadata["first_missing_context_time_utc"],

        "available_through_nyc": (
            pd.Timestamp(window.target_end_utc)
            .tz_convert(NYC_TIMEZONE_NAME)
            .isoformat()
        ),

        "hrrr_retrieval_strategy":
            "latest available extended HRRR cycle per valid time",

        "hrrr_cycles_used":
            sorted(
                {
                    entry["cycle"].cycle_id
                    for entry in retrieval_plan
                }
            ),

        "hrrr_cycle_count":
            len(
                {
                    entry["cycle"].cycle_id
                    for entry in retrieval_plan
                }
            ),

        "minimum_forecast_lead_hour":
            int(
                min(
                    entry["forecast_hour"]
                    for entry in retrieval_plan
                )
            ),

        "maximum_forecast_lead_hour":
            int(
                max(
                    entry["forecast_hour"]
                    for entry in retrieval_plan
                )
            ),

        "grid_cells":
            int(
                target[
                    GRID_ID
                ].nunique()
            ),

        "target_forecast_hours":
            int(
                target[
                    "forecast_hour"
                ].nunique()
            ),

        "target_rows":
            int(
                len(
                    target
                )
            ),

        "context_rows":
            int(
                len(
                    context
                )
            ),

        "unique_hrrr_points_used":
            int(
                target[
                    [
                        "hrrr_lat",
                        "hrrr_lon",
                    ]
                ]
                .drop_duplicates()
                .shape[
                    0
                ]
            ),

        "hrrr_distance_km": {
            "mean":
                float(
                    target[
                        "hrrr_distance_km"
                    ].mean()
                ),

            "median":
                float(
                    target[
                        "hrrr_distance_km"
                    ].median()
                ),

            "max":
                float(
                    target[
                        "hrrr_distance_km"
                    ].max()
                ),
        },

        "precip_current_hour_mm": {
            "mean":
                float(
                    target[
                        "precip_current_hour_mm"
                    ].mean()
                ),

            "max":
                float(
                    target[
                        "precip_current_hour_mm"
                    ].max()
                ),

            "wet_rows":
                int(
                    (
                        target[
                            "precip_current_hour_mm"
                        ]
                        > 0
                    ).sum()
                ),
        },

        "daily_total_precip_mm": {
            "mean":
                float(
                    target[
                        "daily_total_precip_mm"
                    ].mean()
                ),

            "max":
                float(
                    target[
                        "daily_total_precip_mm"
                    ].max()
                ),
        },

        "model_features":
            MODEL_FEATURES,

        "daily_total_definition": (
            "operational centered rolling 24-hour HRRR precipitation "
            "context from H-12 through H+11; legacy column name "
            "daily_total_precip_mm retained for downstream model "
            "compatibility"
        ),

        "previous_6h_definition": (
            "sum of six preceding hourly HRRR one-hour APCP "
            "values, excluding current hour"
        ),

        "training_source":
            "MRMS observed precipitation",

        "operational_forecast_source":
            "HRRR forecast precipitation",

        "hrrr_native_resolution_km":
            3.0,

        "operational_prediction_grid_km":
            1.0,

        "hrrr_spatial_assignment": (
            "nearest HRRR forecast point to each target "
            "NYC 1-km grid"
        ),

        "hrrr_interpolated_to_1km":
            False,

        "synthetic_grid_rows_used_for_retraining":
            False,
    }


# ============================================================
# AUDIT
# ============================================================


def print_audit(
    *,
    target: pd.DataFrame,
    context: pd.DataFrame,
    window: ForecastWindow,
    retrieval_plan: list[dict],
    horizon_metadata: dict,
) -> None:
    """Print human-readable operational weather audit."""

    print()

    print(
        "HRRR FORECAST AUDIT"
    )

    print(
        "==================="
    )

    print(
        "Forecast mode:",
        window.mode,
    )

    print(
        "Target NYC date:",
        window.target_date_nyc,
    )

    print(
        "Target timezone:",
        NYC_TIMEZONE_NAME,
    )

    print(
        "Target NYC start:",
        window.target_start_nyc,
    )

    print(
        "Target UTC window:",
        window.target_start_utc,
        "->",
        window.target_end_utc,
    )

    print(
        "Published target hours:",
        window.target_hour_count,
    )

    print(
        "Forecast horizon truncated:",
        horizon_metadata["forecast_horizon_truncated"],
    )

    print(
        "Available through NYC:",
        pd.Timestamp(window.target_end_utc)
        .tz_convert(NYC_TIMEZONE_NAME),
    )

    print(
        "Antecedent context begins:",
        window.context_start_utc,
    )

    cycles_used = sorted(
        {
            entry["cycle"].cycle_id
            for entry in retrieval_plan
        }
    )

    lead_hours = [
        int(entry["forecast_hour"])
        for entry in retrieval_plan
    ]

    print(
        "HRRR retrieval strategy:",
        "latest available cycle per valid time",
    )

    print(
        "HRRR cycles used:",
        ", ".join(cycles_used),
    )

    print(
        "Retrieved lead-hour range:",
        f"F{min(lead_hours):02d} -> F{max(lead_hours):02d}",
    )

    print(
        "Context rows:",
        f"{len(context):,}",
    )

    print(
        "Published target rows:",
        f"{len(target):,}",
    )

    print(
        "Target grid cells:",
        f"{target[GRID_ID].nunique():,}",
    )

    print(
        "Target forecast hours:",
        f"{target['forecast_hour'].nunique():,}",
    )

    print()

    print(
        "SPATIAL RESOLUTION CONTRACT"
    )

    print(
        "---------------------------"
    )

    print(
        "HRRR native meteorological grid: ~3 km"
    )

    print(
        "Operational prediction grid:     1 km"
    )

    print(
        "Assignment: nearest HRRR point to target grid"
    )

    print(
        "Artificial HRRR 1-km interpolation: NO"
    )

    print()

    print(
        "HRRR -> 1-KM GRID DISTANCE (KM)"
    )

    print(
        "--------------------------------"
    )

    print(
        target[
            "hrrr_distance_km"
        ]
        .describe()
        .to_string()
    )

    print()

    print(
        "CURRENT-HOUR PRECIPITATION (MM)"
    )

    print(
        "-------------------------------"
    )

    print(
        target[
            "precip_current_hour_mm"
        ]
        .describe()
        .to_string()
    )

    print()

    print(
        "PREVIOUS-6H PRECIPITATION (MM)"
    )

    print(
        "------------------------------"
    )

    print(
        target[
            "precip_previous_6h_mm"
        ]
        .describe()
        .to_string()
    )

    print()

    print(
        "CENTERED ROLLING-24 PRECIPITATION (MM)"
    )

    print(
        "-------------------------------------"
    )

    print(
        target[
            "daily_total_precip_mm"
        ]
        .describe()
        .to_string()
    )


# ============================================================
# OUTPUTS
# ============================================================


def resolve_output_dirs(
    *,
    mode: str,
    processed_override: Path | None,
    artifact_override: Path | None,
) -> tuple[
    Path,
    Path,
]:
    """
    Keep Today and Tomorrow products physically separate.

    Explicit CLI directory overrides are used exactly as provided.
    """

    if processed_override is None:

        processed = (
            DEFAULT_PROCESSED_OUTPUT_ROOT
            / mode
        )

    else:

        processed = (
            processed_override
        )

    if artifact_override is None:

        artifact = (
            DEFAULT_ARTIFACT_OUTPUT_ROOT
            / mode
        )

    else:

        artifact = (
            artifact_override
        )

    return (
        processed,
        artifact,
    )


def save_outputs(
    *,
    target: pd.DataFrame,
    context: pd.DataFrame,
    summary: dict,
    processed_output_dir: Path,
    artifact_output_dir: Path,
) -> None:
    """Save mode-specific HRRR weather products."""

    processed_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    parquet_path = (
        processed_output_dir
        / OUTPUT_PARQUET
    )

    csv_path = (
        processed_output_dir
        / OUTPUT_CSV
    )

    context_path = (
        artifact_output_dir
        / CONTEXT_PARQUET
    )

    summary_path = (
        artifact_output_dir
        / SUMMARY_JSON
    )

    target.to_parquet(
        parquet_path,
        index=False,
    )

    target.to_csv(
        csv_path,
        index=False,
    )

    context.to_parquet(
        context_path,
        index=False,
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()

    print(
        "SAVED HRRR PRECIPITATION FORECAST"
    )

    print(
        "---------------------------------"
    )

    print(
        parquet_path
    )

    print(
        csv_path
    )

    print(
        context_path
    )

    print(
        summary_path
    )


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    """Parse operational weather-product arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Build NYC-local same-day or day-ahead HRRR "
            "precipitation predictors for NYC 1-km grids"
        )
    )

    parser.add_argument(
        "--mode",
        choices=[
            "today",
            "tomorrow",
        ],
        default="tomorrow",
        help=(
            "Operational forecast product. "
            "'today' uses the current NYC calendar day; "
            "'tomorrow' uses the next NYC calendar day."
        ),
    )

    parser.add_argument(
        "--grid",
        type=Path,
        default=DEFAULT_GRID_PATH,
    )

    parser.add_argument(
        "--target-date",
        type=str,
        default=None,
        help=(
            "Optional explicit America/New_York target "
            "calendar date YYYY-MM-DD. Overrides the date "
            "normally resolved from --mode."
        ),
    )

    parser.add_argument(
        "--reference-time",
        type=str,
        default=None,
        help=(
            "Optional ISO-8601 UTC/reference timestamp used "
            "for reproducibility and HRRR cycle selection."
        ),
    )

    parser.add_argument(
        "--processed-output-dir",
        type=Path,
        default=None,
        help=(
            "Optional exact processed-output directory. "
            "Default: data/processed/forecasting/<mode>"
        ),
    )

    parser.add_argument(
        "--artifact-output-dir",
        type=Path,
        default=None,
        help=(
            "Optional exact artifact-output directory. "
            "Default: artifacts/flood/forecasting/<mode>"
        ),
    )

    parser.add_argument(
        "--require-full-day",
        action="store_true",
        help=(
            "Fail instead of truncating the published target horizon "
            "when complete H-12 through H+11 HRRR context is not yet "
            "available for all target hours."
        ),
    )

    parser.add_argument(
        "--keep-grib",
        action="store_true",
        help=(
            "Keep APCP-only GRIB files in the HRRR cache"
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    """Build one operational HRRR forecast product."""

    args = parse_args()

    reference_time = (
        parse_reference_time(
            args.reference_time
        )
    )

    target_date = (
        resolve_target_date(
            mode=args.mode,
            target_date_override=args.target_date,
            reference_time=reference_time,
        )
    )

    window = (
        build_forecast_window(
            mode=args.mode,
            target_date=target_date,
        )
    )

    (
        processed_output_dir,
        artifact_output_dir,
    ) = resolve_output_dirs(
        mode=args.mode,
        processed_override=(
            args.processed_output_dir
        ),
        artifact_override=(
            args.artifact_output_dir
        ),
    )

    print(
        "BUILD NYC HRRR PRECIPITATION FORECAST"
    )

    print(
        "====================================="
    )

    print(
        "Forecast mode:                 ",
        args.mode,
    )

    print(
        "Training precipitation source:  MRMS observations"
    )

    print(
        "Operational forecast source:    HRRR"
    )

    print(
        "Operational target geography:   NYC 1-km grids"
    )

    print(
        "HRRR native resolution:         ~3 km"
    )

    print(
        "HRRR spatial assignment:        nearest HRRR point"
    )

    print(
        "Artificial 1-km interpolation:  NO"
    )

    print(
        "Target calendar convention:     America/New_York"
    )

    print(
        "Target NYC date:               ",
        window.target_date_nyc,
    )

    print(
        "Published NYC target UTC:      ",
        window.target_start_utc,
        "->",
        window.target_end_utc,
    )

    print(
        "Required UTC weather window:   ",
        window.context_start_utc,
        "->",
        window.retrieval_end_utc,
    )

    print(
        "Processed output directory:    ",
        processed_output_dir,
    )

    print(
        "Artifact output directory:     ",
        artifact_output_dir,
    )

    grid = load_target_grid(
        args.grid
    )

    print()

    print(
        "TARGET GRID"
    )

    print(
        "-----------"
    )

    print(
        "Grid cells:",
        f"{len(grid):,}",
    )

    s3_client = (
        create_unsigned_s3_client()
    )

    print()

    print(
        "Building valid-time-aware multi-cycle HRRR retrieval plan..."
    )

    requested_window = window

    (
        window,
        retrieval_plan,
        horizon_metadata,
    ) = resolve_available_forecast_window(
        s3_client=s3_client,
        window=requested_window,
        reference_time=reference_time,
        require_full_day=args.require_full_day,
    )

    if horizon_metadata["forecast_horizon_truncated"]:
        print()
        print(
            "ADAPTIVE FORECAST HORIZON"
        )
        print(
            "-------------------------"
        )
        print(
            "Requested target hours:",
            horizon_metadata["requested_target_hour_count"],
        )
        print(
            "Published target hours:",
            horizon_metadata["published_target_hour_count"],
        )
        print(
            "Available through NYC:",
            pd.Timestamp(window.target_end_utc)
            .tz_convert(NYC_TIMEZONE_NAME),
        )
        print(
            "First unavailable context hour UTC:",
            horizon_metadata["first_missing_context_time_utc"],
        )

    cycles_used = sorted(
        {
            entry["cycle"].cycle_id
            for entry in retrieval_plan
        }
    )

    print(
        "Required valid-time fields:",
        len(retrieval_plan),
    )

    print(
        "HRRR cycles required:",
        len(cycles_used),
    )

    print(
        "Cycles:",
        ", ".join(cycles_used),
    )

    context = retrieve_planned_hourly_precipitation(
        grid=grid,
        plan=retrieval_plan,
        s3_client=s3_client,
        keep_grib=args.keep_grib,
    )

    context = (
        validate_hourly_context(
            frame=context,
            grid=grid,
            window=window,
        )
    )

    target = (
        build_target_day_predictors(
            hourly_context=context,
            grid=grid,
            window=window,
        )
    )

    summary = (
        build_summary(
            target=target,
            context=context,
            window=window,
            retrieval_plan=retrieval_plan,
            horizon_metadata=horizon_metadata,
        )
    )

    print_audit(
        target=target,
        context=context,
        window=window,
        retrieval_plan=retrieval_plan,
        horizon_metadata=horizon_metadata,
    )

    save_outputs(
        target=target,
        context=context,
        summary=summary,
        processed_output_dir=(
            processed_output_dir
        ),
        artifact_output_dir=(
            artifact_output_dir
        ),
    )

    print()

    print(
        "HRRR PRECIPITATION FORECAST COMPLETE"
    )

    print(
        f"Mode '{args.mode}' is ready for "
        "build_grid_forecast_inputs.py"
    )


if __name__ == "__main__":
    main()