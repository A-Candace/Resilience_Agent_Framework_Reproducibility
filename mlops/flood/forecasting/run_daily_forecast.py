from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


# ============================================================
# ROOT PATHS
# ============================================================

FORECAST_ARTIFACT_DIR = (
    Path("artifacts")
    / "flood"
    / "forecasting"
)

WEATHER_OUTPUT_DIR = (
    Path("data")
    / "processed"
    / "forecasting"
)

ARCHIVE_ROOT = (
    FORECAST_ARTIFACT_DIR
    / "archive"
)

FAILED_RUNS_ROOT = (
    FORECAST_ARTIFACT_DIR
    / "failed_runs"
)

DAILY_RUN_FILENAME = (
    "daily_forecast_run.json"
)


# ============================================================
# OPERATIONAL CONTRACT
# ============================================================

NYC_TIMEZONE_NAME = "America/New_York"
NYC_TIMEZONE = ZoneInfo(NYC_TIMEZONE_NAME)

EXPECTED_GRID_CELLS = 837
MAX_FORECAST_HOURS = 24

MIN_PRECISION = 0.25
MIN_RECALL = 0.30

LOGISTIC_THRESHOLD = 0.5
GCN_DURATION_THRESHOLD_MINUTES = 1.0

SUPPORTED_MODES = (
    "today",
    "tomorrow",
)


# ============================================================
# MODE PATHS
# ============================================================

@dataclass(frozen=True)
class ModePaths:
    mode: str
    weather_dir: Path
    artifact_dir: Path

    precipitation_parquet: Path
    precipitation_csv: Path
    precipitation_summary: Path

    forecast_inputs_parquet: Path
    forecast_inputs_csv: Path
    forecast_inputs_summary: Path

    final_forecast_parquet: Path
    final_forecast_csv: Path
    final_forecast_summary: Path


def mode_paths(mode: str) -> ModePaths:
    if mode not in SUPPORTED_MODES:
        raise ValueError(
            f"Unsupported forecast mode: {mode!r}. "
            f"Expected one of {SUPPORTED_MODES}."
        )

    weather_dir = (
        WEATHER_OUTPUT_DIR
        / mode
    )

    artifact_dir = (
        FORECAST_ARTIFACT_DIR
        / mode
    )

    return ModePaths(
        mode=mode,
        weather_dir=weather_dir,
        artifact_dir=artifact_dir,

        precipitation_parquet=(
            weather_dir
            / "grid_precipitation_forecast.parquet"
        ),
        precipitation_csv=(
            weather_dir
            / "grid_precipitation_forecast.csv"
        ),
        precipitation_summary=(
            artifact_dir
            / "grid_precipitation_forecast_summary.json"
        ),

        forecast_inputs_parquet=(
            artifact_dir
            / "grid_forecast_inputs.parquet"
        ),
        forecast_inputs_csv=(
            artifact_dir
            / "grid_forecast_inputs.csv"
        ),
        forecast_inputs_summary=(
            artifact_dir
            / "grid_forecast_inputs_summary.json"
        ),

        final_forecast_parquet=(
            artifact_dir
            / "grid_flood_forecast.parquet"
        ),
        final_forecast_csv=(
            artifact_dir
            / "grid_flood_forecast.csv"
        ),
        final_forecast_summary=(
            artifact_dir
            / "grid_flood_forecast_summary.json"
        ),
    )


# ============================================================
# TIME HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def nyc_now() -> datetime:
    return utc_now().astimezone(
        NYC_TIMEZONE
    )


def default_today_target() -> date:
    """
    User-facing forecast dates are NYC-local calendar dates.
    """

    return nyc_now().date()


def parse_base_target_date(
    value: str | None,
) -> date:
    """
    Resolve the NYC-local date represented by mode='today'.

    If --target-date is omitted:
        today    = current America/New_York date
        tomorrow = today + 1 day

    If --target-date is supplied:
        today    = supplied date
        tomorrow = supplied date + 1 day
    """

    if value is None:
        return default_today_target()

    try:
        return date.fromisoformat(
            value
        )
    except ValueError as exc:
        raise ValueError(
            "--target-date must use YYYY-MM-DD."
        ) from exc


def target_date_for_mode(
    *,
    base_today: date,
    mode: str,
) -> date:
    if mode == "today":
        return base_today

    if mode == "tomorrow":
        return (
            base_today
            + timedelta(days=1)
        )

    raise ValueError(
        f"Unsupported mode: {mode}"
    )


# ============================================================
# JSON HELPERS
# ============================================================

def json_safe(value):
    if isinstance(
        value,
        np.integer,
    ):
        return int(value)

    if isinstance(
        value,
        np.floating,
    ):
        return float(value)

    if isinstance(
        value,
        np.bool_,
    ):
        return bool(value)

    if isinstance(
        value,
        (
            pd.Timestamp,
            datetime,
        ),
    ):
        return value.isoformat()

    if isinstance(
        value,
        date,
    ):
        return value.isoformat()

    return value


def write_json(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            {
                key: json_safe(value)
                for key, value
                in payload.items()
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )


# ============================================================
# SUBPROCESS EXECUTION
# ============================================================

def run_module(
    module: str,
    *,
    arguments: list[str] | None = None,
) -> None:
    command = [
        sys.executable,
        "-m",
        module,
    ]

    if arguments:
        command.extend(
            arguments
        )

    print()
    print("=" * 72)
    print("RUNNING:")
    print(
        " ".join(
            command
        )
    )
    print("=" * 72)
    print()

    subprocess.run(
        command,
        check=True,
    )


# ============================================================
# BACKUP / ROLLBACK
# ============================================================

def latest_final_files(
    paths: ModePaths,
) -> list[Path]:
    return [
        paths.final_forecast_parquet,
        paths.final_forecast_csv,
        paths.final_forecast_summary,
    ]


def create_latest_backup(
    *,
    paths: ModePaths,
    backup_dir: Path,
) -> dict[str, bool]:
    backup_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    existed: dict[str, bool] = {}

    for source in latest_final_files(
        paths
    ):
        existed[
            str(source)
        ] = source.exists()

        if source.exists():
            shutil.copy2(
                source,
                backup_dir
                / source.name,
            )

    return existed


def restore_latest_backup(
    *,
    paths: ModePaths,
    backup_dir: Path,
    existed: dict[str, bool],
) -> None:
    print()
    print(
        "RESTORING LAST KNOWN-GOOD "
        f"{paths.mode.upper()} FORECAST"
    )
    print(
        "-" * 42
    )

    for destination in latest_final_files(
        paths
    ):
        had_previous = existed.get(
            str(destination),
            False,
        )

        backup = (
            backup_dir
            / destination.name
        )

        if (
            had_previous
            and backup.exists()
        ):
            shutil.copy2(
                backup,
                destination,
            )

            print(
                "Restored:",
                destination,
            )

        elif (
            not had_previous
            and destination.exists()
        ):
            destination.unlink()

            print(
                "Removed failed partial output:",
                destination,
            )


# ============================================================
# TIME VALIDATION
# ============================================================

def normalize_forecast_hour(
    frame: pd.DataFrame,
) -> pd.Series:
    values = pd.to_datetime(
        frame["forecast_hour"],
        utc=True,
        errors="coerce",
    )

    if values.isna().any():
        raise RuntimeError(
            "Forecast contains invalid "
            "forecast_hour values."
        )

    return values


def validate_nyc_target_day(
    *,
    forecast_hours: pd.Series,
    target_date: date,
    label: str,
) -> None:
    """
    A NYC-local day spans two UTC dates during EDT/EST.

    Therefore validation must occur after converting UTC valid
    timestamps to America/New_York rather than requiring a
    single UTC calendar date.
    """

    nyc_hours = (
        forecast_hours
        .dt.tz_convert(
            NYC_TIMEZONE_NAME
        )
    )

    actual_dates = set(
        nyc_hours
        .dt.date
        .unique()
        .tolist()
    )

    if actual_dates != {
        target_date
    }:
        raise RuntimeError(
            f"{label} target NYC date does not "
            "match request. "
            f"Expected={target_date}; "
            f"observed={sorted(actual_dates)}"
        )


def validate_population(
    *,
    frame: pd.DataFrame,
    label: str,
) -> tuple[int, int, int, int]:
    rows = len(frame)

    grids = int(
        frame["grid_id"]
        .nunique()
    )

    hours = int(
        frame["forecast_hour"]
        .nunique()
    )

    duplicates = int(
        frame.duplicated(
            [
                "grid_id",
                "forecast_hour",
            ]
        ).sum()
    )

    if grids != EXPECTED_GRID_CELLS:
        raise RuntimeError(
            f"{label} grid count failed validation. "
            f"Expected={EXPECTED_GRID_CELLS}; "
            f"observed={grids}."
        )

    if not (
        1
        <= hours
        <= MAX_FORECAST_HOURS
    ):
        raise RuntimeError(
            f"{label} forecast-hour count failed validation. "
            f"Expected between 1 and "
            f"{MAX_FORECAST_HOURS}; "
            f"observed={hours}."
        )

    expected_rows = (
        EXPECTED_GRID_CELLS
        * hours
    )

    if rows != expected_rows:
        raise RuntimeError(
            f"{label} row count failed validation. "
            f"Expected={expected_rows:,}; "
            f"observed={rows:,}."
        )

    if duplicates:
        raise RuntimeError(
            f"{label} contains duplicate "
            "grid/hour rows: "
            f"{duplicates:,}"
        )

    return (
        rows,
        grids,
        hours,
        duplicates,
    )


# ============================================================
# PRECIPITATION VALIDATION
# ============================================================

def validate_precipitation_forecast(
    *,
    paths: ModePaths,
    target_date: date,
) -> dict:
    if not paths.precipitation_parquet.exists():
        raise FileNotFoundError(
            "Missing precipitation forecast: "
            f"{paths.precipitation_parquet}"
        )

    frame = pd.read_parquet(
        paths.precipitation_parquet
    )

    required = [
        "grid_id",
        "forecast_hour",
        "grid_lat",
        "grid_lon",
        "hrrr_lat",
        "hrrr_lon",
        "hrrr_distance_km",
        "precip_current_hour_mm",
        "precip_previous_6h_mm",
        "daily_total_precip_mm",
        "hrrr_cycle_id",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            "Precipitation forecast is missing "
            "required columns: "
            + ", ".join(
                missing
            )
        )

    frame = frame.copy()

    frame[
        "forecast_hour"
    ] = normalize_forecast_hour(
        frame
    )

    validate_nyc_target_day(
        forecast_hours=frame[
            "forecast_hour"
        ],
        target_date=target_date,
        label=(
            f"{paths.mode} precipitation forecast"
        ),
    )

    (
        rows,
        grids,
        hours,
        duplicates,
    ) = validate_population(
        frame=frame,
        label=(
            f"{paths.mode} precipitation forecast"
        ),
    )

    feature_columns = [
        "precip_current_hour_mm",
        "precip_previous_6h_mm",
        "daily_total_precip_mm",
    ]

    if (
        frame[
            feature_columns
        ]
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError(
            "Precipitation forecast contains "
            "missing model predictors."
        )

    if (
        frame[
            feature_columns
        ]
        < 0
    ).any().any():
        raise RuntimeError(
            "Precipitation forecast contains "
            "negative precipitation."
        )

    cycle_values = sorted(
        frame[
            "hrrr_cycle_id"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not cycle_values:
        raise RuntimeError(
            "Precipitation forecast contains "
            "no HRRR cycle identifiers."
        )

    return {
        "precipitation_rows": rows,
        "precipitation_grid_cells": grids,
        "precipitation_forecast_hours": hours,
        "precipitation_duplicate_rows": duplicates,
        "forecast_horizon_truncated": (
            hours
            < MAX_FORECAST_HOURS
        ),
        "hrrr_cycle_count": len(
            cycle_values
        ),
        "hrrr_cycle_ids": cycle_values,
        "hrrr_distance_mean_km": float(
            frame[
                "hrrr_distance_km"
            ].mean()
        ),
        "hrrr_distance_max_km": float(
            frame[
                "hrrr_distance_km"
            ].max()
        ),
        "maximum_current_hour_precip_mm": float(
            frame[
                "precip_current_hour_mm"
            ].max()
        ),
        "maximum_previous_6h_precip_mm": float(
            frame[
                "precip_previous_6h_mm"
            ].max()
        ),
        "maximum_daily_total_precip_mm": float(
            frame[
                "daily_total_precip_mm"
            ].max()
        ),
        "forecast_start_utc": (
            frame[
                "forecast_hour"
            ].min()
        ),
        "forecast_end_utc": (
            frame[
                "forecast_hour"
            ].max()
        ),
    }


# ============================================================
# GRID INPUT VALIDATION
# ============================================================

def validate_grid_forecast_inputs(
    *,
    paths: ModePaths,
    target_date: date,
) -> dict:
    if not paths.forecast_inputs_parquet.exists():
        raise FileNotFoundError(
            "Missing grid forecast inputs: "
            f"{paths.forecast_inputs_parquet}"
        )

    frame = pd.read_parquet(
        paths.forecast_inputs_parquet
    )

    required = [
        "grid_id",
        "forecast_hour",
        "cluster_number",
        "support_sensor_id",
        "support_sensor_model",
        "support_sensor_f1",
        "support_sensor_precision",
        "support_sensor_recall",
        "imputation_method",
        "precip_current_hour_mm",
        "precip_previous_6h_mm",
        "daily_total_precip_mm",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            "Grid forecast inputs are missing "
            "required columns: "
            + ", ".join(
                missing
            )
        )

    frame = frame.copy()

    frame[
        "forecast_hour"
    ] = normalize_forecast_hour(
        frame
    )

    validate_nyc_target_day(
        forecast_hours=frame[
            "forecast_hour"
        ],
        target_date=target_date,
        label=(
            f"{paths.mode} grid forecast inputs"
        ),
    )

    (
        rows,
        grids,
        hours,
        duplicates,
    ) = validate_population(
        frame=frame,
        label=(
            f"{paths.mode} grid forecast inputs"
        ),
    )

    if (
        frame[
            "support_sensor_id"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Some forecast rows are missing "
            "support_sensor_id."
        )

    if (
        frame[
            "support_sensor_model"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Some forecast rows are missing "
            "support_sensor_model."
        )

    metric_columns = [
        "support_sensor_f1",
        "support_sensor_precision",
        "support_sensor_recall",
    ]

    if (
        frame[
            metric_columns
        ]
        .isna()
        .any()
        .any()
    ):
        raise RuntimeError(
            "Some forecast rows are missing "
            "support-sensor performance metrics."
        )

    minimum_precision_observed = float(
        frame[
            "support_sensor_precision"
        ].min()
    )

    minimum_recall_observed = float(
        frame[
            "support_sensor_recall"
        ].min()
    )

    if (
        minimum_precision_observed
        < MIN_PRECISION
    ):
        raise RuntimeError(
            "A support sensor no longer satisfies "
            "the minimum precision contract. "
            "Minimum observed="
            f"{minimum_precision_observed:.6f}."
        )

    if (
        minimum_recall_observed
        < MIN_RECALL
    ):
        raise RuntimeError(
            "A support sensor no longer satisfies "
            "the minimum recall contract. "
            "Minimum observed="
            f"{minimum_recall_observed:.6f}."
        )

    models = (
        frame[
            "support_sensor_model"
        ]
        .value_counts()
        .to_dict()
    )

    imputation = (
        frame[
            "imputation_method"
        ]
        .value_counts()
        .to_dict()
    )

    return {
        "forecast_input_rows": rows,
        "forecast_input_grid_cells": grids,
        "forecast_input_hours": hours,
        "forecast_input_duplicate_rows": duplicates,
        "unique_support_sensors": int(
            frame[
                "support_sensor_id"
            ].nunique()
        ),
        "logistic_rows": int(
            models.get(
                "logistic",
                0,
            )
        ),
        "gcn_rows": int(
            models.get(
                "gcn",
                0,
            )
        ),
        "direct_grid_rows": int(
            imputation.get(
                "direct_grid_primary_sensor",
                0,
            )
        ),
        "imputed_grid_rows": int(
            imputation.get(
                "same_cluster_primary_sensor",
                0,
            )
        ),
        "minimum_support_precision": (
            minimum_precision_observed
        ),
        "minimum_support_recall": (
            minimum_recall_observed
        ),
    }


# ============================================================
# FINAL FORECAST VALIDATION
# ============================================================

def validate_final_forecast(
    *,
    paths: ModePaths,
    target_date: date,
) -> dict:
    if not paths.final_forecast_parquet.exists():
        raise FileNotFoundError(
            "Missing final flood forecast: "
            f"{paths.final_forecast_parquet}"
        )

    frame = pd.read_parquet(
        paths.final_forecast_parquet
    )

    required = [
        "grid_id",
        "forecast_hour",
        "cluster_number",
        "support_sensor_id",
        "support_sensor_model",
        "support_sensor_f1",
        "support_sensor_precision",
        "support_sensor_recall",
        "imputation_method",
        "logistic_event_probability",
        "gcn_predicted_minutes_above_1inch",
        "predicted_flood_event",
        "native_prediction_type",
        "native_prediction_value",
        "classification_threshold",
        "hrrr_cycle_id",
    ]

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise RuntimeError(
            "Final forecast is missing "
            "required columns: "
            + ", ".join(
                missing
            )
        )

    frame = frame.copy()

    frame[
        "forecast_hour"
    ] = normalize_forecast_hour(
        frame
    )

    validate_nyc_target_day(
        forecast_hours=frame[
            "forecast_hour"
        ],
        target_date=target_date,
        label=(
            f"{paths.mode} final forecast"
        ),
    )

    (
        rows,
        grids,
        hours,
        duplicates,
    ) = validate_population(
        frame=frame,
        label=(
            f"{paths.mode} final forecast"
        ),
    )

    if (
        frame[
            "predicted_flood_event"
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Final forecast contains missing "
            "predicted_flood_event."
        )

    logistic_mask = (
        frame[
            "support_sensor_model"
        ]
        == "logistic"
    )

    gcn_mask = (
        frame[
            "support_sensor_model"
        ]
        == "gcn"
    )

    if (
        frame.loc[
            logistic_mask,
            "logistic_event_probability",
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Some logistic forecast rows are "
            "missing probability."
        )

    if (
        frame.loc[
            gcn_mask,
            "gcn_predicted_minutes_above_1inch",
        ]
        .isna()
        .any()
    ):
        raise RuntimeError(
            "Some GCN forecast rows are missing "
            "predicted duration."
        )

    if (
        frame.loc[
            logistic_mask,
            "gcn_predicted_minutes_above_1inch",
        ]
        .notna()
        .any()
    ):
        raise RuntimeError(
            "Logistic rows unexpectedly contain "
            "GCN native output."
        )

    if (
        frame.loc[
            gcn_mask,
            "logistic_event_probability",
        ]
        .notna()
        .any()
    ):
        raise RuntimeError(
            "GCN rows unexpectedly contain "
            "logistic native output."
        )

    logistic_thresholds = set(
        frame.loc[
            logistic_mask,
            "classification_threshold",
        ]
        .dropna()
        .astype(float)
        .unique()
        .tolist()
    )

    if (
        logistic_mask.any()
        and logistic_thresholds
        != {
            LOGISTIC_THRESHOLD
        }
    ):
        raise RuntimeError(
            "Logistic classification threshold "
            "differs from expected "
            f"{LOGISTIC_THRESHOLD}. "
            f"Observed={logistic_thresholds}"
        )

    gcn_thresholds = set(
        frame.loc[
            gcn_mask,
            "classification_threshold",
        ]
        .dropna()
        .astype(float)
        .unique()
        .tolist()
    )

    if (
        gcn_mask.any()
        and gcn_thresholds
        != {
            GCN_DURATION_THRESHOLD_MINUTES
        }
    ):
        raise RuntimeError(
            "GCN classification threshold differs "
            "from expected "
            f"{GCN_DURATION_THRESHOLD_MINUTES}. "
            f"Observed={gcn_thresholds}"
        )

    cycle_values = sorted(
        frame[
            "hrrr_cycle_id"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    if not cycle_values:
        raise RuntimeError(
            "Final forecast contains no "
            "HRRR cycle identifiers."
        )

    return {
        "final_rows": rows,
        "final_grid_cells": grids,
        "final_forecast_hours": hours,
        "final_duplicate_grid_hour_rows": (
            duplicates
        ),
        "predicted_flood_rows": int(
            frame[
                "predicted_flood_event"
            ].sum()
        ),
        "predicted_flood_grids": int(
            frame.loc[
                frame[
                    "predicted_flood_event"
                ],
                "grid_id",
            ].nunique()
        ),
        "logistic_probability_min": (
            float(
                frame.loc[
                    logistic_mask,
                    "logistic_event_probability",
                ].min()
            )
            if logistic_mask.any()
            else None
        ),
        "logistic_probability_max": (
            float(
                frame.loc[
                    logistic_mask,
                    "logistic_event_probability",
                ].max()
            )
            if logistic_mask.any()
            else None
        ),
        "gcn_predicted_minutes_min": (
            float(
                frame.loc[
                    gcn_mask,
                    "gcn_predicted_minutes_above_1inch",
                ].min()
            )
            if gcn_mask.any()
            else None
        ),
        "gcn_predicted_minutes_max": (
            float(
                frame.loc[
                    gcn_mask,
                    "gcn_predicted_minutes_above_1inch",
                ].max()
            )
            if gcn_mask.any()
            else None
        ),
        "hrrr_cycle_count": len(
            cycle_values
        ),
        "hrrr_cycle_ids": cycle_values,
        "forecast_start_utc": (
            frame[
                "forecast_hour"
            ].min()
        ),
        "forecast_end_utc": (
            frame[
                "forecast_hour"
            ].max()
        ),
    }


# ============================================================
# ARCHIVING
# ============================================================

def archive_successful_mode_run(
    *,
    paths: ModePaths,
    target_date: date,
    run_metadata: dict,
) -> Path:
    archive_dir = (
        ARCHIVE_ROOT
        / target_date.isoformat()
        / paths.mode
    )

    archive_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    files_to_archive = [
        paths.precipitation_parquet,
        paths.precipitation_csv,
        paths.precipitation_summary,
        paths.forecast_inputs_parquet,
        paths.forecast_inputs_csv,
        paths.forecast_inputs_summary,
        paths.final_forecast_parquet,
        paths.final_forecast_csv,
        paths.final_forecast_summary,
    ]

    for source in files_to_archive:
        if not source.exists():
            raise FileNotFoundError(
                "Cannot archive successful "
                f"{paths.mode} forecast because "
                "required artifact is missing: "
                f"{source}"
            )

        shutil.copy2(
            source,
            archive_dir
            / source.name,
        )

    write_json(
        archive_dir
        / DAILY_RUN_FILENAME,
        run_metadata,
    )

    return archive_dir


# ============================================================
# ONE MODE
# ============================================================

def run_one_mode(
    *,
    mode: str,
    target_date: date,
    run_started_at: datetime,
    parent_run_id: str,
    keep_grib: bool,
) -> dict:
    paths = mode_paths(
        mode
    )

    paths.weather_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths.artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mode_run_id = (
        parent_run_id
        + "__"
        + mode
    )

    rollback_dir = (
        FORECAST_ARTIFACT_DIR
        / ".daily_run_backup"
        / mode_run_id
    )

    previous_latest = create_latest_backup(
        paths=paths,
        backup_dir=rollback_dir,
    )

    metadata: dict = {
        "run_id": mode_run_id,
        "parent_run_id": parent_run_id,
        "mode": mode,
        "status": "running",
        "target_date_nyc": (
            target_date.isoformat()
        ),
        "target_timezone": (
            NYC_TIMEZONE_NAME
        ),
        "started_at": (
            utc_now().isoformat()
        ),
        "expected_grid_cells": (
            EXPECTED_GRID_CELLS
        ),
        "maximum_forecast_hours": (
            MAX_FORECAST_HOURS
        ),
        "minimum_precision": (
            MIN_PRECISION
        ),
        "minimum_recall": (
            MIN_RECALL
        ),
        "logistic_threshold": (
            LOGISTIC_THRESHOLD
        ),
        "gcn_duration_threshold_minutes": (
            GCN_DURATION_THRESHOLD_MINUTES
        ),
        "training_precipitation_source": (
            "MRMS"
        ),
        "operational_precipitation_source": (
            "HRRR"
        ),
        "weather_output_dir": str(
            paths.weather_dir
        ),
        "artifact_output_dir": str(
            paths.artifact_dir
        ),
    }

    try:
        precipitation_arguments = [
            "--mode",
            mode,
            "--target-date",
            target_date.isoformat(),
        ]

        if keep_grib:
            precipitation_arguments.append(
                "--keep-grib"
            )

        # ----------------------------------------------------
        # STAGE 1: HRRR
        # ----------------------------------------------------

        run_module(
            "mlops.flood.forecasting."
            "build_grid_precipitation_forecast",
            arguments=precipitation_arguments,
        )

        precipitation_validation = (
            validate_precipitation_forecast(
                paths=paths,
                target_date=target_date,
            )
        )

        metadata[
            "precipitation"
        ] = precipitation_validation

        print()
        print(
            f"{mode.upper()} STAGE 1 "
            "VALIDATION PASSED"
        )
        print(
            "-" * 36
        )
        print(
            "Rows:",
            f"{precipitation_validation['precipitation_rows']:,}",
        )
        print(
            "Forecast hours:",
            precipitation_validation[
                "precipitation_forecast_hours"
            ],
        )
        print(
            "HRRR cycles:",
            precipitation_validation[
                "hrrr_cycle_count"
            ],
        )

        # ----------------------------------------------------
        # STAGE 2: MODEL INPUTS
        # ----------------------------------------------------

        run_module(
            "mlops.flood.forecasting."
            "build_grid_forecast_inputs",
            arguments=[
                "--weather",
                str(
                    paths.precipitation_parquet
                ),
                "--output-dir",
                str(
                    paths.artifact_dir
                ),
            ],
        )

        input_validation = (
            validate_grid_forecast_inputs(
                paths=paths,
                target_date=target_date,
            )
        )

        metadata[
            "forecast_inputs"
        ] = input_validation

        print()
        print(
            f"{mode.upper()} STAGE 2 "
            "VALIDATION PASSED"
        )
        print(
            "-" * 36
        )
        print(
            "Rows:",
            f"{input_validation['forecast_input_rows']:,}",
        )
        print(
            "Support sensors:",
            input_validation[
                "unique_support_sensors"
            ],
        )
        print(
            "Logistic rows:",
            f"{input_validation['logistic_rows']:,}",
        )
        print(
            "GCN rows:",
            f"{input_validation['gcn_rows']:,}",
        )

        # ----------------------------------------------------
        # STAGE 3: SELECTED-MODEL INFERENCE
        # ----------------------------------------------------

        run_module(
            "mlops.flood.forecasting."
            "run_grid_flood_forecast",
            arguments=[
                "--mode",
                mode,
            ],
        )

        final_validation = (
            validate_final_forecast(
                paths=paths,
                target_date=target_date,
            )
        )

        metadata[
            "final_forecast"
        ] = final_validation

        completed_at = utc_now()

        metadata[
            "status"
        ] = "success"

        metadata[
            "completed_at"
        ] = completed_at.isoformat()

        metadata[
            "duration_seconds"
        ] = (
            completed_at
            - run_started_at
        ).total_seconds()

        archive_dir = (
            archive_successful_mode_run(
                paths=paths,
                target_date=target_date,
                run_metadata=metadata,
            )
        )

        metadata[
            "archive_dir"
        ] = str(
            archive_dir
        )

        write_json(
            paths.artifact_dir
            / DAILY_RUN_FILENAME,
            metadata,
        )

        print()
        print(
            f"{mode.upper()} FORECAST SUCCESS"
        )
        print(
            "=" * 36
        )
        print(
            "NYC target date:",
            target_date,
        )
        print(
            "Rows:",
            f"{final_validation['final_rows']:,}",
        )
        print(
            "Grids:",
            final_validation[
                "final_grid_cells"
            ],
        )
        print(
            "Hours:",
            final_validation[
                "final_forecast_hours"
            ],
        )
        print(
            "Predicted flood rows:",
            final_validation[
                "predicted_flood_rows"
            ],
        )
        print(
            "Latest forecast:",
            paths.final_forecast_parquet,
        )
        print(
            "Archived forecast:",
            archive_dir,
        )

        return metadata

    except Exception as exc:
        failed_at = utc_now()

        metadata[
            "status"
        ] = "failed"

        metadata[
            "failed_at"
        ] = failed_at.isoformat()

        metadata[
            "duration_seconds"
        ] = (
            failed_at
            - run_started_at
        ).total_seconds()

        metadata[
            "error_type"
        ] = type(
            exc
        ).__name__

        metadata[
            "error_message"
        ] = str(
            exc
        )

        metadata[
            "traceback"
        ] = traceback.format_exc()

        failed_run_dir = (
            FAILED_RUNS_ROOT
            / parent_run_id
            / mode
        )

        failed_run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        write_json(
            failed_run_dir
            / DAILY_RUN_FILENAME,
            metadata,
        )

        restore_latest_backup(
            paths=paths,
            backup_dir=rollback_dir,
            existed=previous_latest,
        )

        print()
        print(
            f"{mode.upper()} FORECAST FAILED"
        )
        print(
            "=" * 36
        )
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()
        print(
            "The previous validated "
            f"{mode} dashboard forecast "
            "was restored."
        )
        print(
            "Failure metadata:"
        )
        print(
            failed_run_dir
            / DAILY_RUN_FILENAME
        )

        return metadata

    finally:
        if rollback_dir.exists():
            shutil.rmtree(
                rollback_dir,
                ignore_errors=True,
            )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run, validate, archive, and safely "
            "publish the NYC operational flood "
            "forecast for today and tomorrow."
        )
    )

    parser.add_argument(
        "--target-date",
        type=str,
        default=None,
        help=(
            "Optional explicit America/New_York "
            "calendar date representing the "
            "'today' product. YYYY-MM-DD. "
            "The tomorrow product uses the next day."
        ),
    )

    parser.add_argument(
        "--mode",
        choices=[
            "today",
            "tomorrow",
            "both",
        ],
        default="both",
        help=(
            "Operational product(s) to refresh. "
            "Default: both."
        ),
    )

    parser.add_argument(
        "--keep-grib",
        action="store_true",
        help=(
            "Keep APCP-only HRRR GRIB messages "
            "after decoding."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_args()

    base_today = (
        parse_base_target_date(
            args.target_date
        )
    )

    run_started_at = utc_now()

    parent_run_id = (
        base_today.isoformat()
        + "__"
        + run_started_at.strftime(
            "%Y%m%dT%H%M%SZ"
        )
    )

    FORECAST_ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "RUN NYC DAILY FLOOD FORECAST"
    )
    print(
        "============================"
    )
    print()
    print(
        "NYC calendar date:",
        base_today,
    )
    print(
        "Run ID:",
        parent_run_id,
    )
    print()

    print(
        "SCIENTIFIC CONTRACT"
    )
    print(
        "-------------------"
    )
    print(
        "Training / retraining precipitation: MRMS"
    )
    print(
        "Daily operational precipitation:     HRRR"
    )
    print(
        "Forecast geography:                  NYC 1-km grid"
    )
    print(
        "Support-sensor eligibility:          precision >= 0.25"
    )
    print(
        "                                      recall >= 0.30"
    )
    print(
        "Daily run retrains models:            NO"
    )
    print(
        "Daily run reruns clustering:          NO"
    )
    print(
        "Forecast products:                    today + tomorrow"
    )
    print()

    if args.mode == "both":
        modes = list(
            SUPPORTED_MODES
        )
    else:
        modes = [
            args.mode
        ]

    aggregate_metadata: dict = {
        "run_id": parent_run_id,
        "status": "running",
        "base_today_nyc": (
            base_today.isoformat()
        ),
        "target_timezone": (
            NYC_TIMEZONE_NAME
        ),
        "started_at": (
            run_started_at.isoformat()
        ),
        "requested_mode": (
            args.mode
        ),
        "modes": {},
    }

    for mode in modes:
        target_date = (
            target_date_for_mode(
                base_today=base_today,
                mode=mode,
            )
        )

        print()
        print(
            "#" * 72
        )
        print(
            f"BEGIN {mode.upper()} "
            f"FORECAST: {target_date}"
        )
        print(
            "#" * 72
        )

        result = run_one_mode(
            mode=mode,
            target_date=target_date,
            run_started_at=utc_now(),
            parent_run_id=parent_run_id,
            keep_grib=args.keep_grib,
        )

        aggregate_metadata[
            "modes"
        ][
            mode
        ] = result

    completed_at = utc_now()

    failed_modes = [
        mode
        for mode, result
        in aggregate_metadata[
            "modes"
        ].items()
        if result.get(
            "status"
        ) != "success"
    ]

    aggregate_metadata[
        "completed_at"
    ] = completed_at.isoformat()

    aggregate_metadata[
        "duration_seconds"
    ] = (
        completed_at
        - run_started_at
    ).total_seconds()

    if failed_modes:
        aggregate_metadata[
            "status"
        ] = "failed"

        aggregate_metadata[
            "failed_modes"
        ] = failed_modes
    else:
        aggregate_metadata[
            "status"
        ] = "success"

    write_json(
        FORECAST_ARTIFACT_DIR
        / DAILY_RUN_FILENAME,
        aggregate_metadata,
    )

    print()
    print(
        "DAILY FORECAST RUN COMPLETE"
    )
    print(
        "==========================="
    )
    print(
        "Status:",
        aggregate_metadata[
            "status"
        ],
    )

    for mode, result in (
        aggregate_metadata[
            "modes"
        ].items()
    ):
        print(
            f"{mode}:",
            result.get(
                "status"
            ),
        )

    print()
    print(
        "Run metadata:"
    )
    print(
        FORECAST_ARTIFACT_DIR
        / DAILY_RUN_FILENAME
    )

    if failed_modes:
        raise RuntimeError(
            "Daily forecast failed for mode(s): "
            + ", ".join(
                failed_modes
            )
        )


if __name__ == "__main__":
    main()
