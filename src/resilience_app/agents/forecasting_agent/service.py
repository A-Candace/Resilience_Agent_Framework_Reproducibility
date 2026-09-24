

from __future__ import annotations

from collections.abc import Iterable
import json
import os
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from mlops.flood.shared.model_interface import predict_gcn_snapshot
from resilience_app.core.shared import compute_forecast_heatmap, load_flood_grid
from resilience_app.agents.forecasting_agent.schemas import SensorForecastInput

NYC_TIMEZONE = ZoneInfo("America/New_York")

FORECAST_ROOT = Path(
    os.getenv(
        "GRID_FLOOD_FORECAST_ROOT",
        "/app/artifacts/flood/forecasting",
    )
)

LEGACY_OPERATIONAL_FORECAST_PATH = Path(
    os.getenv(
        "GRID_FLOOD_FORECAST_PATH",
        str(FORECAST_ROOT / "grid_flood_forecast.parquet"),
    )
)

SUPPORTED_OPERATIONAL_MODES = {"today", "tomorrow"}

FORECAST_FILENAME = "grid_flood_forecast.parquet"
FORECAST_SUMMARY_FILENAME = "grid_flood_forecast_summary.json"
PRECIP_SUMMARY_FILENAME = "grid_precipitation_forecast_summary.json"


DAYPART_HOURS = {
    "overnight": range(0, 6),
    "morning": range(6, 12),
    "afternoon": range(12, 18),
    "evening": range(18, 22),
    "night": range(22, 24),
}


# ---------------------------------------------------------------------
# Legacy forecasting services
# ---------------------------------------------------------------------

def forecast_heatmap_service(forecast_inches: float):
    """Historical rainfall-threshold heatmap retained for compatibility."""
    frame = load_flood_grid()
    return compute_forecast_heatmap(frame, float(forecast_inches))


def forecast_sensor_snapshot_service(
    snapshot: Iterable[SensorForecastInput | dict],
) -> pd.DataFrame:
    """Direct GCN sensor inference retained for compatibility."""

    records: list[dict] = []

    for item in snapshot:
        if isinstance(item, SensorForecastInput):
            records.append(
                {
                    "deployment_id": item.deployment_id,
                    "precip_current_hour_mm": item.precip_current_hour_mm,
                    "precip_previous_6h_mm": item.precip_previous_6h_mm,
                    "daily_total_precip_mm": item.daily_total_precip_mm,
                }
            )
        elif isinstance(item, dict):
            records.append(dict(item))
        else:
            raise TypeError(
                "Each sensor snapshot item must be a "
                "SensorForecastInput or dict."
            )

    return predict_gcn_snapshot(pd.DataFrame(records))


# ---------------------------------------------------------------------
# Operational product resolution
# ---------------------------------------------------------------------

def _normalize_mode(mode: str) -> str:
    value = str(mode).strip().lower()

    if value not in SUPPORTED_OPERATIONAL_MODES:
        raise ValueError(
            "mode must be one of: "
            + ", ".join(sorted(SUPPORTED_OPERATIONAL_MODES))
        )

    return value


def _mode_directory(mode: str) -> Path:
    return FORECAST_ROOT / _normalize_mode(mode)


def _forecast_path(mode: str) -> Path:
    return _mode_directory(mode) / FORECAST_FILENAME


def _summary_path(mode: str) -> Path:
    return _mode_directory(mode) / FORECAST_SUMMARY_FILENAME


def _precip_summary_path(mode: str) -> Path:
    return _mode_directory(mode) / PRECIP_SUMMARY_FILENAME


def _read_json_if_present(path: Path) -> dict:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


def _load_operational_forecast(mode: str) -> pd.DataFrame:
    """Load one authoritative mode-specific operational 1-km forecast."""

    mode = _normalize_mode(mode)
    path = _forecast_path(mode)

    if not path.exists():
        raise FileNotFoundError(
            f"Operational {mode!r} flood forecast was not found at "
            f"{path}. Run the {mode} forecasting pipeline first."
        )

    frame = pd.read_parquet(path)

    required = {
        "grid_id",
        "forecast_hour",
        "support_sensor_id",
        "support_sensor_model",
        "predicted_flood_event",
    }

    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"Operational {mode!r} forecast is missing required columns: "
            + ", ".join(sorted(missing))
        )

    frame = frame.copy()
    frame["grid_id"] = frame["grid_id"].astype(str)
    frame["forecast_hour"] = pd.to_datetime(
        frame["forecast_hour"],
        utc=True,
        errors="coerce",
    )

    if frame["forecast_hour"].isna().any():
        raise ValueError(
            f"Operational {mode!r} forecast contains invalid forecast_hour values."
        )

    duplicate_count = frame.duplicated(["grid_id", "forecast_hour"]).sum()

    if duplicate_count:
        raise ValueError(
            f"Operational {mode!r} forecast contains "
            f"{duplicate_count} duplicate grid/hour rows."
        )

    frame["forecast_mode"] = mode

    return frame


# ---------------------------------------------------------------------
# Serialization / ranking helpers
# ---------------------------------------------------------------------

def _safe_value(value: Any) -> Any:
    """Convert pandas/numpy values into MCP-safe Python values."""

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    if isinstance(value, pd.Timestamp):
        return value.isoformat()

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError, AttributeError):
            pass

    return value


def _records(frame: pd.DataFrame) -> list[dict]:
    result: list[dict] = []

    for record in frame.to_dict(orient="records"):
        result.append(
            {
                str(key): _safe_value(value)
                for key, value in record.items()
            }
        )

    return result


def _risk_score(frame: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=frame.index, dtype=float)

    model_series = (
        frame["support_sensor_model"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    if "logistic_event_probability" in frame.columns:
        logistic = pd.to_numeric(
            frame["logistic_event_probability"],
            errors="coerce",
        ).fillna(0.0)
        mask = model_series.eq("logistic")
        score.loc[mask] = logistic.loc[mask]

    if "gcn_predicted_minutes_above_1inch" in frame.columns:
        gcn = pd.to_numeric(
            frame["gcn_predicted_minutes_above_1inch"],
            errors="coerce",
        ).fillna(0.0)
        mask = model_series.eq("gcn")
        score.loc[mask] = gcn.loc[mask]

    return score


def _forecast_hours(frame: pd.DataFrame) -> list[pd.Timestamp]:
    return [
        pd.Timestamp(value)
        for value in sorted(frame["forecast_hour"].dropna().unique())
    ]


def _local_iso(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None

    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")

    return timestamp.tz_convert(NYC_TIMEZONE).isoformat()


def _parse_forecast_hour(value: str) -> pd.Timestamp:
    requested = pd.to_datetime(value, utc=True, errors="coerce")

    if pd.isna(requested):
        raise ValueError(
            "forecast_hour must be a valid ISO timestamp. "
            "Timezone offsets are accepted."
        )

    return pd.Timestamp(requested)


def _filter_forecast_hour(
    frame: pd.DataFrame,
    forecast_hour: str | None,
) -> pd.DataFrame:
    if forecast_hour is None:
        return frame

    requested = _parse_forecast_hour(forecast_hour)
    result = frame[frame["forecast_hour"] == requested].copy()

    if result.empty:
        available = _forecast_hours(frame)

        if available:
            raise ValueError(
                f"No forecast exists for {requested.isoformat()}. "
                f"Available range is {available[0].isoformat()} through "
                f"{available[-1].isoformat()} "
                f"({_local_iso(available[0])} through "
                f"{_local_iso(available[-1])} NYC time)."
            )

        raise ValueError("The selected forecast product has no forecast hours.")

    return result


# ---------------------------------------------------------------------
# Public operational services
# ---------------------------------------------------------------------

def get_forecast_summary_service(
    mode: str = "tomorrow",
) -> dict:
    mode = _normalize_mode(mode)
    frame = _load_operational_forecast(mode)
    forecast_hours = _forecast_hours(frame)
    event_mask = frame["predicted_flood_event"].astype(bool)

    start = forecast_hours[0] if forecast_hours else None
    end = forecast_hours[-1] if forecast_hours else None

    operational_summary = _read_json_if_present(_summary_path(mode))
    precip_summary = _read_json_if_present(_precip_summary_path(mode))

    requested_hours = (
        precip_summary.get("requested_target_hour_count")
        or operational_summary.get("requested_target_hour_count")
        or 24
    )

    published_hours = int(frame["forecast_hour"].nunique())

    truncated = bool(
        precip_summary.get(
            "forecast_horizon_truncated",
            operational_summary.get(
                "forecast_horizon_truncated",
                published_hours < int(requested_hours),
            ),
        )
    )

    summary = {
        "forecast_available": True,
        "forecast_mode": mode,
        "authoritative_operational_product": True,
        "forecast_path": str(_forecast_path(mode)),
        "legacy_root_forecast_used": False,
        "rows": int(len(frame)),
        "grid_cells": int(frame["grid_id"].nunique()),
        "forecast_hours": published_hours,
        "requested_forecast_hours": int(requested_hours),
        "forecast_horizon_truncated": truncated,
        "support_sensors": int(frame["support_sensor_id"].nunique()),
        "predicted_flood_rows": int(event_mask.sum()),
        "predicted_flood_grids": int(
            frame.loc[event_mask, "grid_id"].nunique()
        ),
        "forecast_start_utc": start.isoformat() if start is not None else None,
        "forecast_end_utc": end.isoformat() if end is not None else None,
        "forecast_start_nyc": _local_iso(start),
        "forecast_available_through_nyc": _local_iso(end),
        "precipitation_feature_contract": {
            "precip_current_hour_mm": "HRRR precipitation at forecast hour H",
            "precip_previous_6h_mm": "sum of HRRR precipitation for H-6 through H-1",
            "daily_total_precip_mm": (
                "legacy column name; operationally centered rolling "
                "24-hour HRRR precipitation context H-12 through H+11"
            ),
        },
    }

    if "target_date_nyc" in frame.columns:
        values = frame["target_date_nyc"].dropna().astype(str).unique()
        summary["target_date_nyc"] = values[0] if len(values) else None
    elif start is not None:
        summary["target_date_nyc"] = str(
            start.tz_convert(NYC_TIMEZONE).date()
        )

    if "target_date_utc" in frame.columns:
        values = frame["target_date_utc"].dropna().astype(str).unique()
        summary["target_date_utc"] = values[0] if len(values) else None

    if "hrrr_cycle_id" in frame.columns:
        values = (
            frame["hrrr_cycle_id"]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        summary["hrrr_cycles"] = sorted(values)

    for generated_column in ("forecast_generated_at", "generated_at"):
        if generated_column in frame.columns:
            generated = pd.to_datetime(
                frame[generated_column],
                utc=True,
                errors="coerce",
            ).dropna()

            if not generated.empty:
                summary["generated_at"] = generated.max().isoformat()
                break

    if "support_sensor_model" in frame.columns:
        summary["rows_by_model"] = {
            str(key): int(value)
            for key, value in (
                frame["support_sensor_model"]
                .value_counts()
                .to_dict()
                .items()
            )
        }

    if precip_summary:
        for key in (
            "first_missing_context_time_utc",
            "available_through_nyc",
        ):
            if key in precip_summary:
                summary[key] = precip_summary[key]

    return summary


def get_available_forecast_hours_service(
    mode: str = "tomorrow",
) -> list[dict]:
    frame = _load_operational_forecast(mode)

    return [
        {
            "forecast_mode": _normalize_mode(mode),
            "hour_index": index,
            "forecast_hour_utc": hour.isoformat(),
            "forecast_hour_nyc": _local_iso(hour),
        }
        for index, hour in enumerate(_forecast_hours(frame))
    ]


def get_grid_forecast_service(
    grid_id: str,
    mode: str = "tomorrow",
) -> list[dict]:
    frame = _load_operational_forecast(mode)

    result = frame[
        frame["grid_id"] == str(grid_id)
    ].sort_values("forecast_hour")

    if result.empty:
        raise ValueError(
            f"Grid ID {grid_id!r} was not found in the "
            f"{_normalize_mode(mode)!r} operational forecast."
        )

    return _records(result)


def get_hourly_forecast_service(
    forecast_hour: str,
    mode: str = "tomorrow",
) -> list[dict]:
    frame = _load_operational_forecast(mode)
    result = _filter_forecast_hour(frame, forecast_hour).copy()
    result["_risk_score"] = _risk_score(result)

    result = result.sort_values(
        ["predicted_flood_event", "_risk_score"],
        ascending=[False, False],
    ).drop(columns="_risk_score")

    return _records(result)


def get_highest_risk_grids_service(
    limit: int = 20,
    mode: str = "tomorrow",
    forecast_hour: str | None = None,
) -> list[dict]:
    if limit < 1:
        raise ValueError("limit must be at least 1.")

    limit = min(int(limit), 500)

    frame = _load_operational_forecast(mode).copy()
    frame = _filter_forecast_hour(frame, forecast_hour)
    frame["_risk_score"] = _risk_score(frame)

    result = (
        frame.sort_values(
            ["predicted_flood_event", "_risk_score"],
            ascending=[False, False],
        )
        .head(limit)
        .drop(columns="_risk_score")
    )

    return _records(result)


def get_cluster_forecast_service(
    cluster_number: int,
    mode: str = "tomorrow",
    forecast_hour: str | None = None,
) -> list[dict]:
    frame = _load_operational_forecast(mode)
    frame = _filter_forecast_hour(frame, forecast_hour)

    if "cluster_number" not in frame.columns:
        raise ValueError(
            "cluster_number is not present in the operational forecast."
        )

    cluster_values = pd.to_numeric(
        frame["cluster_number"],
        errors="coerce",
    )

    result = frame[
        cluster_values == int(cluster_number)
    ].sort_values(["forecast_hour", "grid_id"])

    if result.empty:
        raise ValueError(
            f"Cluster {cluster_number} was not found in "
            f"the {_normalize_mode(mode)!r} forecast."
        )

    return _records(result)


def get_flood_event_forecast_service(
    mode: str = "tomorrow",
    forecast_hour: str | None = None,
) -> list[dict]:
    frame = _load_operational_forecast(mode)
    frame = _filter_forecast_hour(frame, forecast_hour)

    result = frame[
        frame["predicted_flood_event"].astype(bool)
    ].copy()

    result["_risk_score"] = _risk_score(result)

    result = result.sort_values(
        ["forecast_hour", "_risk_score", "grid_id"],
        ascending=[True, False, True],
    ).drop(columns="_risk_score")

    return _records(result)



def get_daypart_forecast_summary_service(
    mode: str = "tomorrow",
    daypart: str = "afternoon",
    top_n: int = 20,
) -> dict:
    """
    Return a compact NYC-local daypart forecast summary.

    This is designed for conversational requests such as
    "Where is flooding expected tomorrow afternoon?" without sending
    thousands of full 837-grid/hour records back to Bedrock.
    """

    mode = _normalize_mode(mode)
    daypart = str(daypart).strip().lower()

    if daypart not in DAYPART_HOURS:
        raise ValueError(
            "daypart must be one of: "
            + ", ".join(DAYPART_HOURS)
        )

    top_n = max(1, min(int(top_n), 50))

    frame = _load_operational_forecast(mode).copy()

    frame["forecast_hour_nyc"] = (
        frame["forecast_hour"]
        .dt.tz_convert(NYC_TIMEZONE)
    )

    allowed_hours = set(DAYPART_HOURS[daypart])

    period = frame[
        frame["forecast_hour_nyc"].dt.hour.isin(allowed_hours)
    ].copy()

    requested_daypart_hour_count = len(allowed_hours)

    available_hour_values = [
        pd.Timestamp(value)
        for value in sorted(
            period["forecast_hour"].dropna().unique()
        )
    ]

    available_daypart_hour_count = len(
        available_hour_values
    )

    overall_summary = get_forecast_summary_service(mode)

    target_date_nyc = overall_summary.get(
        "target_date_nyc"
    )

    response = {
        "forecast_mode": mode,
        "target_date_nyc": target_date_nyc,
        "requested_daypart": daypart,
        "requested_daypart_hour_count": (
            requested_daypart_hour_count
        ),
        "available_daypart_hour_count": (
            available_daypart_hour_count
        ),
        "daypart_coverage_complete": (
            available_daypart_hour_count
            == requested_daypart_hour_count
        ),
        "forecast_horizon_truncated": (
            overall_summary.get(
                "forecast_horizon_truncated",
                False,
            )
        ),
        "forecast_available_through_nyc": (
            overall_summary.get(
                "forecast_available_through_nyc"
            )
        ),
        "forecast_available_for_requested_daypart": (
            available_daypart_hour_count > 0
        ),
        "available_hours": [
            {
                "forecast_hour_utc": hour.isoformat(),
                "forecast_hour_nyc": _local_iso(hour),
            }
            for hour in available_hour_values
        ],
        "predicted_flood_grid_hours": 0,
        "unique_predicted_flood_grids": 0,
        "hourly_predicted_flood_grids": [],
        "top_risk_grid_hours": [],
    }

    if period.empty:
        response["interpretation"] = (
            "No operational forecast hours are currently "
            "published for the requested NYC-local daypart. "
            "This is unavailable forecast coverage, not a "
            "zero-flood prediction."
        )
        return response

    event_mask = (
        period["predicted_flood_event"]
        .astype(bool)
    )

    response["predicted_flood_grid_hours"] = int(
        event_mask.sum()
    )

    response["unique_predicted_flood_grids"] = int(
        period.loc[
            event_mask,
            "grid_id",
        ].nunique()
    )

    hourly = (
        period
        .groupby(
            "forecast_hour",
            as_index=False,
        )
        .agg(
            predicted_flood_grids=(
                "predicted_flood_event",
                "sum",
            )
        )
        .sort_values("forecast_hour")
    )

    response["hourly_predicted_flood_grids"] = [
        {
            "forecast_hour_utc": (
                pd.Timestamp(row.forecast_hour)
                .isoformat()
            ),
            "forecast_hour_nyc": _local_iso(
                pd.Timestamp(row.forecast_hour)
            ),
            "predicted_flood_grids": int(
                row.predicted_flood_grids
            ),
        }
        for row in hourly.itertuples(index=False)
    ]

    ranked = period.copy()
    ranked["_risk_score"] = _risk_score(ranked)

    ranked = (
        ranked
        .sort_values(
            [
                "predicted_flood_event",
                "_risk_score",
                "forecast_hour",
                "grid_id",
            ],
            ascending=[
                False,
                False,
                True,
                True,
            ],
        )
        .head(top_n)
        .copy()
    )

    compact_columns = [
        "grid_id",
        "grid_lat",
        "grid_lon",
        "forecast_hour",
        "forecast_hour_nyc",
        "predicted_flood_event",
        "support_sensor_model",
        "logistic_event_probability",
        "gcn_predicted_minutes_above_1inch",
        "native_prediction_type",
        "native_prediction_value",
        "classification_threshold",
        "precip_current_hour_mm",
        "precip_previous_6h_mm",
        "daily_total_precip_mm",
        "is_imputed_grid",
        "support_sensor_id",
    ]

    compact_columns = [
        column
        for column in compact_columns
        if column in ranked.columns
    ]

    response["top_risk_grid_hours"] = _records(
        ranked[compact_columns]
    )

    if (
        available_daypart_hour_count
        < requested_daypart_hour_count
    ):
        response["interpretation"] = (
            "Only part of the requested NYC-local daypart "
            "is currently covered by the operational forecast. "
            "Do not interpret unavailable later hours as "
            "zero flood risk."
        )
    else:
        response["interpretation"] = (
            "The complete requested NYC-local daypart is "
            "covered by the current operational forecast."
        )

    return response



def get_legacy_forecast_status_service() -> dict:
    return {
        "legacy_methodology_preserved": True,
        "authoritative_for_today_or_tomorrow": False,
        "legacy_root_forecast_path": str(
            LEGACY_OPERATIONAL_FORECAST_PATH
        ),
        "legacy_root_forecast_exists": (
            LEGACY_OPERATIONAL_FORECAST_PATH.exists()
        ),
        "legacy_threshold_heatmap_available": True,
        "legacy_sensor_snapshot_available": True,
    }
