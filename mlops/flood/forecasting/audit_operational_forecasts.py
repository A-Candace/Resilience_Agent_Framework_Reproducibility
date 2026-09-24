from __future__ import annotations

import argparse
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


NYC_TZ = ZoneInfo("America/New_York")

FORECAST_ROOT = Path("artifacts/flood/forecasting")

MODES = ("today", "tomorrow")

REQUIRED_COLUMNS = {
    "grid_id",
    "forecast_hour",
    "grid_lat",
    "grid_lon",
    "precip_current_hour_mm",
    "precip_previous_6h_mm",
    "daily_total_precip_mm",
    "support_sensor_model",
    "predicted_flood_event",
    "logistic_event_probability",
    "gcn_predicted_minutes_above_1inch",
}


def load_forecast(
    root: Path,
    mode: str,
) -> pd.DataFrame:
    path = root / mode / "grid_flood_forecast.parquet"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {mode!r} forecast artifact: {path}"
        )

    df = pd.read_parquet(path).copy()

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise RuntimeError(
            f"{mode!r} forecast is missing required columns: "
            f"{sorted(missing)}"
        )

    df["forecast_hour"] = pd.to_datetime(
        df["forecast_hour"],
        utc=True,
    )

    df["forecast_hour_nyc"] = (
        df["forecast_hour"]
        .dt.tz_convert(NYC_TZ)
    )

    df["forecast_date_nyc"] = (
        df["forecast_hour_nyc"]
        .dt.date
        .astype(str)
    )

    df["forecast_hour_nyc_label"] = (
        df["forecast_hour_nyc"]
        .dt.strftime("%Y-%m-%d %H:%M %Z")
    )

    df["predicted_flood_event"] = (
        df["predicted_flood_event"]
        .fillna(False)
        .astype(bool)
    )

    return df


def audit_structure(
    df: pd.DataFrame,
    mode: str,
) -> dict:
    duplicates = int(
        df.duplicated(
            ["grid_id", "forecast_hour"]
        ).sum()
    )

    dates = sorted(
        df["forecast_date_nyc"]
        .dropna()
        .unique()
        .tolist()
    )

    return {
        "mode": mode,
        "rows": int(len(df)),
        "grid_cells": int(df["grid_id"].nunique()),
        "forecast_hours": int(df["forecast_hour"].nunique()),
        "duplicate_grid_hours": duplicates,
        "nyc_target_dates": ",".join(dates),
        "first_hour_utc": str(df["forecast_hour"].min()),
        "last_hour_utc": str(df["forecast_hour"].max()),
        "first_hour_nyc": str(df["forecast_hour_nyc"].min()),
        "last_hour_nyc": str(df["forecast_hour_nyc"].max()),
        "flood_grid_hours": int(
            df["predicted_flood_event"].sum()
        ),
        "flood_grids": int(
            df.loc[
                df["predicted_flood_event"],
                "grid_id",
            ].nunique()
        ),
    }


def numeric_summary(
    df: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    columns = [
        "precip_current_hour_mm",
        "precip_previous_6h_mm",
        "daily_total_precip_mm",
        "logistic_event_probability",
        "gcn_predicted_minutes_above_1inch",
    ]

    rows = []

    for column in columns:
        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        rows.append(
            {
                "mode": mode,
                "metric": column,
                "count": int(values.count()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "p90": float(values.quantile(0.90)),
                "p95": float(values.quantile(0.95)),
                "p99": float(values.quantile(0.99)),
                "max": float(values.max()),
            }
        )

    return pd.DataFrame(rows)


def hourly_summary(
    df: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    work = df.copy()

    work["logistic_probability"] = pd.to_numeric(
        work["logistic_event_probability"],
        errors="coerce",
    )

    work["gcn_minutes"] = pd.to_numeric(
        work["gcn_predicted_minutes_above_1inch"],
        errors="coerce",
    )

    work["current_precip"] = pd.to_numeric(
        work["precip_current_hour_mm"],
        errors="coerce",
    )

    work["previous_6h_precip"] = pd.to_numeric(
        work["precip_previous_6h_mm"],
        errors="coerce",
    )

    work["daily_precip"] = pd.to_numeric(
        work["daily_total_precip_mm"],
        errors="coerce",
    )

    hourly = (
        work
        .groupby(
            [
                "forecast_hour",
                "forecast_hour_nyc_label",
            ],
            as_index=False,
        )
        .agg(
            grid_cells=("grid_id", "nunique"),
            flood_grids=(
                "predicted_flood_event",
                "sum",
            ),
            current_precip_mean_mm=(
                "current_precip",
                "mean",
            ),
            current_precip_max_mm=(
                "current_precip",
                "max",
            ),
            previous_6h_mean_mm=(
                "previous_6h_precip",
                "mean",
            ),
            previous_6h_max_mm=(
                "previous_6h_precip",
                "max",
            ),
            daily_total_mean_mm=(
                "daily_precip",
                "mean",
            ),
            daily_total_max_mm=(
                "daily_precip",
                "max",
            ),
            logistic_probability_mean=(
                "logistic_probability",
                "mean",
            ),
            logistic_probability_max=(
                "logistic_probability",
                "max",
            ),
            gcn_minutes_mean=(
                "gcn_minutes",
                "mean",
            ),
            gcn_minutes_max=(
                "gcn_minutes",
                "max",
            ),
        )
        .sort_values("forecast_hour")
        .reset_index(drop=True)
    )

    hourly.insert(0, "mode", mode)

    hourly["flood_grid_fraction"] = (
        hourly["flood_grids"]
        / hourly["grid_cells"]
    )

    return hourly


def routing_summary(
    df: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    routing = (
        df.groupby(
            "support_sensor_model",
            dropna=False,
        )
        .agg(
            rows=("grid_id", "size"),
            unique_grids=("grid_id", "nunique"),
            flood_grid_hours=(
                "predicted_flood_event",
                "sum",
            ),
        )
        .reset_index()
    )

    routing.insert(0, "mode", mode)

    return routing


def top_precipitation_rows(
    df: pd.DataFrame,
    mode: str,
    limit: int,
) -> pd.DataFrame:
    columns = [
        "grid_id",
        "forecast_hour",
        "forecast_hour_nyc_label",
        "grid_lat",
        "grid_lon",
        "precip_current_hour_mm",
        "precip_previous_6h_mm",
        "daily_total_precip_mm",
        "support_sensor_model",
        "predicted_flood_event",
        "logistic_event_probability",
        "gcn_predicted_minutes_above_1inch",
    ]

    result = (
        df.sort_values(
            [
                "precip_current_hour_mm",
                "precip_previous_6h_mm",
            ],
            ascending=False,
        )
        .head(limit)
        [columns]
        .copy()
    )

    result.insert(0, "mode", mode)

    return result


def top_model_rows(
    df: pd.DataFrame,
    mode: str,
    limit: int,
) -> pd.DataFrame:
    logistic = df.loc[
        df["support_sensor_model"]
        .astype(str)
        .str.lower()
        .eq("logistic")
    ].copy()

    logistic = logistic.sort_values(
        "logistic_event_probability",
        ascending=False,
    ).head(limit)

    logistic["native_score"] = (
        logistic["logistic_event_probability"]
    )
    logistic["native_score_name"] = (
        "logistic_event_probability"
    )

    gcn = df.loc[
        df["support_sensor_model"]
        .astype(str)
        .str.lower()
        .eq("gcn")
    ].copy()

    gcn = gcn.sort_values(
        "gcn_predicted_minutes_above_1inch",
        ascending=False,
    ).head(limit)

    gcn["native_score"] = (
        gcn["gcn_predicted_minutes_above_1inch"]
    )
    gcn["native_score_name"] = (
        "gcn_predicted_minutes_above_1inch"
    )

    result = pd.concat(
        [logistic, gcn],
        ignore_index=True,
    )

    columns = [
        "grid_id",
        "forecast_hour",
        "forecast_hour_nyc_label",
        "grid_lat",
        "grid_lon",
        "support_sensor_model",
        "native_score_name",
        "native_score",
        "predicted_flood_event",
        "precip_current_hour_mm",
        "precip_previous_6h_mm",
        "daily_total_precip_mm",
    ]

    result = result[columns]
    result.insert(0, "mode", mode)

    return result


def precipitation_event_relationship(
    df: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    work = df.copy()

    current = pd.to_numeric(
        work["precip_current_hour_mm"],
        errors="coerce",
    ).fillna(0.0)

    previous = pd.to_numeric(
        work["precip_previous_6h_mm"],
        errors="coerce",
    ).fillna(0.0)

    daily = pd.to_numeric(
        work["daily_total_precip_mm"],
        errors="coerce",
    ).fillna(0.0)

    work["rain_category"] = pd.cut(
        current,
        bins=[
            -np.inf,
            0.0,
            0.1,
            1.0,
            5.0,
            10.0,
            25.0,
            np.inf,
        ],
        labels=[
            "0 mm",
            ">0-0.1 mm",
            ">0.1-1 mm",
            ">1-5 mm",
            ">5-10 mm",
            ">10-25 mm",
            ">25 mm",
        ],
    )

    grouped = (
        work
        .groupby(
            "rain_category",
            observed=True,
        )
        .agg(
            grid_hours=("grid_id", "size"),
            flood_grid_hours=(
                "predicted_flood_event",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped["flood_fraction"] = (
        grouped["flood_grid_hours"]
        / grouped["grid_hours"]
    )

    grouped.insert(0, "mode", mode)

    # Additional diagnostic values repeated for convenient inspection.
    grouped["max_current_precip_mm"] = float(current.max())
    grouped["max_previous_6h_mm"] = float(previous.max())
    grouped["max_daily_total_mm"] = float(daily.max())

    return grouped


def print_mode_report(
    mode: str,
    structure: dict,
    hourly: pd.DataFrame,
) -> None:
    print()
    print("=" * 72)
    print(f"{mode.upper()} FORECAST AUDIT")
    print("=" * 72)

    print()
    print("STRUCTURE")
    print("---------")

    for key, value in structure.items():
        print(f"{key:28s}: {value}")

    print()
    print("HOURLY WEATHER + FLOOD SIGNAL")
    print("-----------------------------")

    display_columns = [
        "forecast_hour_nyc_label",
        "flood_grids",
        "flood_grid_fraction",
        "current_precip_mean_mm",
        "current_precip_max_mm",
        "previous_6h_max_mm",
        "daily_total_max_mm",
        "logistic_probability_max",
        "gcn_minutes_max",
    ]

    print(
        hourly[display_columns].to_string(
            index=False,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit persisted NYC today/tomorrow operational "
            "flood forecasts."
        )
    )

    parser.add_argument(
        "--root",
        default=str(FORECAST_ROOT),
        help="Operational forecasting artifact root.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Audit output directory. Defaults to "
            "<root>/audit."
        ),
    )

    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Number of highest-value rows to retain.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "audit"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    structures = []
    numeric_frames = []
    hourly_frames = []
    routing_frames = []
    precipitation_frames = []
    top_precip_frames = []
    top_model_frames = []

    loaded: dict[str, pd.DataFrame] = {}

    for mode in MODES:
        df = load_forecast(
            root,
            mode,
        )

        loaded[mode] = df

        structure = audit_structure(
            df,
            mode,
        )

        numeric = numeric_summary(
            df,
            mode,
        )

        hourly = hourly_summary(
            df,
            mode,
        )

        routing = routing_summary(
            df,
            mode,
        )

        precipitation_relationship = (
            precipitation_event_relationship(
                df,
                mode,
            )
        )

        top_precip = top_precipitation_rows(
            df,
            mode,
            args.top,
        )

        top_models = top_model_rows(
            df,
            mode,
            args.top,
        )

        structures.append(structure)
        numeric_frames.append(numeric)
        hourly_frames.append(hourly)
        routing_frames.append(routing)
        precipitation_frames.append(
            precipitation_relationship
        )
        top_precip_frames.append(top_precip)
        top_model_frames.append(top_models)

        print_mode_report(
            mode,
            structure,
            hourly,
        )

    structure_df = pd.DataFrame(structures)

    numeric_df = pd.concat(
        numeric_frames,
        ignore_index=True,
    )

    hourly_df = pd.concat(
        hourly_frames,
        ignore_index=True,
    )

    routing_df = pd.concat(
        routing_frames,
        ignore_index=True,
    )

    precip_relationship_df = pd.concat(
        precipitation_frames,
        ignore_index=True,
    )

    top_precip_df = pd.concat(
        top_precip_frames,
        ignore_index=True,
    )

    top_models_df = pd.concat(
        top_model_frames,
        ignore_index=True,
    )

    output_files = {
        "structure": output_dir / "forecast_structure_audit.csv",
        "numeric": output_dir / "forecast_numeric_summary.csv",
        "hourly": output_dir / "forecast_hourly_audit.csv",
        "routing": output_dir / "forecast_model_routing.csv",
        "precip_relationship": (
            output_dir
            / "forecast_precipitation_event_relationship.csv"
        ),
        "top_precip": (
            output_dir
            / "forecast_top_precipitation_rows.csv"
        ),
        "top_models": (
            output_dir
            / "forecast_top_model_rows.csv"
        ),
    }

    structure_df.to_csv(
        output_files["structure"],
        index=False,
    )

    numeric_df.to_csv(
        output_files["numeric"],
        index=False,
    )

    hourly_df.to_csv(
        output_files["hourly"],
        index=False,
    )

    routing_df.to_csv(
        output_files["routing"],
        index=False,
    )

    precip_relationship_df.to_csv(
        output_files["precip_relationship"],
        index=False,
    )

    top_precip_df.to_csv(
        output_files["top_precip"],
        index=False,
    )

    top_models_df.to_csv(
        output_files["top_models"],
        index=False,
    )

    print()
    print("=" * 72)
    print("TODAY VS TOMORROW")
    print("=" * 72)
    print()

    comparison_columns = [
        "mode",
        "rows",
        "grid_cells",
        "forecast_hours",
        "duplicate_grid_hours",
        "nyc_target_dates",
        "flood_grid_hours",
        "flood_grids",
    ]

    print(
        structure_df[
            comparison_columns
        ].to_string(index=False)
    )

    print()
    print("SAVED AUDIT ARTIFACTS")
    print("---------------------")

    for path in output_files.values():
        print(path)

    print()
    print("FORECAST AUDIT COMPLETE")


if __name__ == "__main__":
    main()