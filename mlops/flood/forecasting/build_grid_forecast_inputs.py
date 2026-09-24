"""
Build target-grid-local flood forecast inputs.

This module joins:

    1. the latest grid imputation/support reference; and
    2. day-ahead precipitation predictors for every 1-km grid cell.

IMPORTANT CONTRACT
------------------

For every target 1-km grid:

    model identity = support sensor
    rainfall       = target grid rainfall

For example:

    target grid:
        40.505_-74.255

    support sensor:
        happily-green-lion

    GCN/logistic model identity:
        happily-green-lion

    precipitation predictors:
        rainfall forecast at 40.505_-74.255

The support sensor's rainfall is NOT copied to the target grid.

These operational imputed-grid records are NEVER added to model training
history. Retraining continues to use real observations from all physical
sensors.

Inputs
------

Spatial support reference:

    artifacts/flood/spatial/
        grid_imputation_reference.parquet

Grid-local precipitation forecast:

    data/processed/forecasting/
        grid_precipitation_forecast.parquet

Required weather columns:

    grid_id
    forecast_hour
    precip_current_hour_mm
    precip_previous_6h_mm
    daily_total_precip_mm

There must be exactly one row for every grid_id / forecast_hour pair.

Outputs
-------

artifacts/flood/forecasting/

    grid_forecast_inputs.csv
    grid_forecast_inputs.parquet
    grid_forecast_inputs_summary.json

The next module performs GCN/logistic inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

DEFAULT_IMPUTATION_REFERENCE = (
    Path("artifacts")
    / "flood"
    / "spatial"
    / "grid_imputation_reference.parquet"
)

DEFAULT_GRID_WEATHER = (
    Path("data")
    / "processed"
    / "forecasting"
    / "grid_precipitation_forecast.parquet"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "flood"
    / "forecasting"
)


# ============================================================
# COLUMNS
# ============================================================

GRID_ID = "grid_id"

FORECAST_HOUR = "forecast_hour"

PRECIP_CURRENT = (
    "precip_current_hour_mm"
)

PRECIP_PREVIOUS_6H = (
    "precip_previous_6h_mm"
)

DAILY_TOTAL_PRECIP = (
    "daily_total_precip_mm"
)

MODEL_FEATURES = [
    PRECIP_CURRENT,
    PRECIP_PREVIOUS_6H,
    DAILY_TOTAL_PRECIP,
]


# ============================================================
# REQUIRED CONTRACTS
# ============================================================

REQUIRED_SPATIAL_COLUMNS = {
    GRID_ID,
    "grid_i",
    "grid_j",
    "grid_lat",
    "grid_lon",
    "cluster_number",
    "has_eligible_sensor_in_grid",
    "imputation_required",
    "imputation_method",
    "support_scope",
    "support_sensor_id",
    "support_sensor_model",
    "support_sensor_f1",
    "support_sensor_precision",
    "support_sensor_recall",
    "support_sensor_lat",
    "support_sensor_lon",
    "support_sensor_grid_id",
    "support_sensor_cluster_number",
    "support_sensor_distance_km",
    "same_cluster_support",
}

REQUIRED_WEATHER_COLUMNS = {
    GRID_ID,
    FORECAST_HOUR,
    *MODEL_FEATURES,
}


# ============================================================
# LOADERS
# ============================================================


def load_table(
    path: Path,
    *,
    description: str,
) -> pd.DataFrame:
    """
    Load CSV or Parquet.

    When a requested Parquet is absent, try a same-name CSV.
    """

    actual_path = path

    if not actual_path.exists():

        if actual_path.suffix.lower() in {
            ".parquet",
            ".pq",
        }:

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
                f"{description} not found: "
                f"{path}"
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

        result = pd.read_parquet(
            actual_path
        )

    elif suffix == ".csv":

        result = pd.read_csv(
            actual_path
        )

    else:

        raise ValueError(
            f"{description} must be CSV or Parquet."
        )

    if result.empty:

        raise ValueError(
            f"{description} is empty."
        )

    return result


# ============================================================
# SPATIAL REFERENCE
# ============================================================


def validate_spatial_reference(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    result = dataframe.copy()

    missing = (
        REQUIRED_SPATIAL_COLUMNS
        - set(
            result.columns
        )
    )

    if missing:

        raise ValueError(
            "Grid imputation reference is missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    result[
        GRID_ID
    ] = (
        result[
            GRID_ID
        ]
        .astype(str)
        .str.strip()
    )

    if result[
        GRID_ID
    ].duplicated().any():

        raise ValueError(
            "Grid imputation reference contains "
            "duplicate grid_id values."
        )

    result[
        "support_sensor_id"
    ] = (
        result[
            "support_sensor_id"
        ]
        .astype(str)
        .str.strip()
    )

    result[
        "support_sensor_model"
    ] = (
        result[
            "support_sensor_model"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    invalid_model = (
        ~result[
            "support_sensor_model"
        ]
        .isin(
            [
                "gcn",
                "logistic",
            ]
        )
    )

    if invalid_model.any():

        invalid_values = (
            result.loc[
                invalid_model,
                "support_sensor_model",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Unsupported support model(s): "
            + ", ".join(
                invalid_values
            )
        )

    numeric_columns = [
        "grid_lat",
        "grid_lon",
        "cluster_number",
        "support_sensor_f1",
        "support_sensor_precision",
        "support_sensor_recall",
        "support_sensor_lat",
        "support_sensor_lon",
        "support_sensor_cluster_number",
        "support_sensor_distance_km",
    ]

    for column in numeric_columns:

        result[
            column
        ] = pd.to_numeric(
            result[
                column
            ],
            errors="coerce",
        )

    invalid_numeric = (
        result[
            numeric_columns
        ]
        .isna()
        .any(
            axis=1
        )
    )

    if invalid_numeric.any():

        raise ValueError(
            "Grid imputation reference contains "
            f"{int(invalid_numeric.sum()):,} rows with "
            "invalid spatial/support values."
        )

    return result


# ============================================================
# WEATHER INPUT
# ============================================================


def validate_weather(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:

    result = dataframe.copy()

    missing = (
        REQUIRED_WEATHER_COLUMNS
        - set(
            result.columns
        )
    )

    if missing:

        raise ValueError(
            "Grid precipitation forecast is missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    result[
        GRID_ID
    ] = (
        result[
            GRID_ID
        ]
        .astype(str)
        .str.strip()
    )

    result[
        FORECAST_HOUR
    ] = pd.to_datetime(
        result[
            FORECAST_HOUR
        ],
        utc=True,
        errors="coerce",
    )

    if result[
        FORECAST_HOUR
    ].isna().any():

        raise ValueError(
            "Grid weather contains invalid forecast_hour values."
        )

    for column in MODEL_FEATURES:

        result[
            column
        ] = pd.to_numeric(
            result[
                column
            ],
            errors="coerce",
        )

    invalid_features = (
        result[
            MODEL_FEATURES
        ]
        .isna()
        .any(
            axis=1
        )
    )

    if invalid_features.any():

        raise ValueError(
            "Grid weather contains "
            f"{int(invalid_features.sum()):,} rows with "
            "missing/non-numeric precipitation predictors."
        )

    negative = (
        result[
            MODEL_FEATURES
        ]
        < 0
    ).any(
        axis=1
    )

    if negative.any():

        raise ValueError(
            "Grid weather contains negative precipitation values."
        )

    duplicate = result.duplicated(
        [
            GRID_ID,
            FORECAST_HOUR,
        ]
    )

    if duplicate.any():

        raise ValueError(
            "Grid weather contains duplicate "
            "grid_id / forecast_hour rows."
        )

    return (
        result
        .sort_values(
            [
                FORECAST_HOUR,
                GRID_ID,
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# COVERAGE
# ============================================================


def audit_grid_coverage(
    *,
    spatial: pd.DataFrame,
    weather: pd.DataFrame,
) -> None:
    """
    Require all 837 current spatial grid cells for every forecast hour.
    """

    expected_ids = set(
        spatial[
            GRID_ID
        ]
    )

    expected_count = len(
        expected_ids
    )

    forecast_hours = (
        weather[
            FORECAST_HOUR
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    if not forecast_hours:

        raise ValueError(
            "Weather input contains no forecast hours."
        )

    failures: list[str] = []

    for hour in forecast_hours:

        subset = weather.loc[
            weather[
                FORECAST_HOUR
            ]
            == hour
        ]

        observed_ids = set(
            subset[
                GRID_ID
            ]
        )

        missing = (
            expected_ids
            - observed_ids
        )

        extra = (
            observed_ids
            - expected_ids
        )

        if (
            len(
                subset
            )
            != expected_count
            or missing
            or extra
        ):

            failures.append(
                (
                    f"{hour}: "
                    f"rows={len(subset)}, "
                    f"missing={len(missing)}, "
                    f"extra={len(extra)}"
                )
            )

    if failures:

        raise ValueError(
            "Weather grid coverage failed:\n"
            + "\n".join(
                failures[
                    :20
                ]
            )
        )


# ============================================================
# BUILD
# ============================================================


def build_grid_forecast_inputs(
    *,
    spatial: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:

    audit_grid_coverage(
        spatial=spatial,
        weather=weather,
    )

    # --------------------------------------------------------
    # Prevent target-grid coordinate collisions during merge.
    #
    # Spatial reference is authoritative for:
    #     grid_lat
    #     grid_lon
    #
    # Weather retains its target-grid coordinates temporarily
    # only so we can verify that the HRRR preparation stage used
    # the same grid geography.
    # --------------------------------------------------------

    weather_for_join = (
        weather
        .rename(
            columns={
                "grid_lat":
                    "weather_grid_lat",
                "grid_lon":
                    "weather_grid_lon",
            }
        )
        .copy()
    )

    result = weather_for_join.merge(
        spatial,
        on=GRID_ID,
        how="left",
        validate="many_to_one",
    )

    if len(
        result
    ) != len(
        weather
    ):

        raise RuntimeError(
            "Spatial/weather join changed row count."
        )

    missing_support = (
        result[
            "support_sensor_id"
        ]
        .isna()
    )

    if missing_support.any():

        raise RuntimeError(
            "Spatial/weather join created "
            f"{int(missing_support.sum()):,} rows "
            "without support sensors."
        )

    # --------------------------------------------------------
    # Verify weather and spatial stages used the same target
    # grid centroid coordinates.
    # --------------------------------------------------------

    latitude_difference = (
        (
            result[
                "weather_grid_lat"
            ]
            -
            result[
                "grid_lat"
            ]
        )
        .abs()
    )

    longitude_difference = (
        (
            result[
                "weather_grid_lon"
            ]
            -
            result[
                "grid_lon"
            ]
        )
        .abs()
    )

    coordinate_tolerance = (
        1e-6
    )

    coordinate_mismatch = (
        (
            latitude_difference
            > coordinate_tolerance
        )
        |
        (
            longitude_difference
            > coordinate_tolerance
        )
    )

    if coordinate_mismatch.any():

        bad = (
            result.loc[
                coordinate_mismatch,
                [
                    GRID_ID,
                    "grid_lat",
                    "grid_lon",
                    "weather_grid_lat",
                    "weather_grid_lon",
                ],
            ]
            .head(
                20
            )
        )

        raise RuntimeError(
            "Weather and spatial references disagree on "
            "target-grid coordinates.\n"
            + bad.to_string(
                index=False
            )
        )

    # The check passed, so remove duplicate weather copies.
    # The authoritative target-grid coordinates remain:
    #
    #     grid_lat
    #     grid_lon
    #
    # HRRR forecast-point coordinates remain separately as:
    #
    #     hrrr_lat
    #     hrrr_lon
    #     hrrr_distance_km
    # --------------------------------------------------------

    result = result.drop(
        columns=[
            "weather_grid_lat",
            "weather_grid_lon",
        ]
    )

    # --------------------------------------------------------
    # Explicit scientific provenance
    # --------------------------------------------------------

    result[
        "weather_source_scope"
    ] = (
        "target_grid"
    )

    result[
        "model_identity_scope"
    ] = (
        "support_sensor"
    )

    result[
        "model_deployment_id"
    ] = (
        result[
            "support_sensor_id"
        ]
        .astype(str)
    )

    result[
        "is_imputed_grid"
    ] = (
        result[
            "imputation_required"
        ]
        .astype(bool)
    )

    # --------------------------------------------------------
    # Stable operational key
    # --------------------------------------------------------

    result[
        "grid_forecast_key"
    ] = (
        result[
            GRID_ID
        ]
        .astype(str)
        + "|"
        + result[
            FORECAST_HOUR
        ]
        .dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    )

    if result[
        "grid_forecast_key"
    ].duplicated().any():

        raise RuntimeError(
            "Generated grid_forecast_key values are not unique."
        )

    # --------------------------------------------------------
    # Preferred downstream contract
    # --------------------------------------------------------

    preferred_columns = [
        "grid_forecast_key",

        GRID_ID,
        FORECAST_HOUR,

        "grid_i",
        "grid_j",
        "grid_lat",
        "grid_lon",
        "cluster_number",

        PRECIP_CURRENT,
        PRECIP_PREVIOUS_6H,
        DAILY_TOTAL_PRECIP,

        # HRRR provenance
        "target_date_utc",
        "forecast_initialization",
        "hrrr_cycle_id",
        "hrrr_lat",
        "hrrr_lon",
        "hrrr_distance_km",
        "hrrr_source",
        "weather_product",
        "hrrr_apcp_description",
        "generated_at",

        # Support sensor / selected model
        "support_sensor_id",
        "support_sensor_model",

        "support_sensor_f1",
        "support_sensor_precision",
        "support_sensor_recall",

        "support_sensor_lat",
        "support_sensor_lon",

        "support_sensor_grid_id",
        "support_sensor_cluster_number",
        "support_sensor_distance_km",

        # Spatial/imputation provenance
        "has_eligible_sensor_in_grid",

        "imputation_required",
        "is_imputed_grid",

        "imputation_method",
        "support_scope",
        "same_cluster_support",

        # Scientific serving contract
        "weather_source_scope",
        "model_identity_scope",
        "model_deployment_id",
    ]

    # Some optional provenance fields may evolve over time.
    # Only place available columns into the final table.
    preferred_columns = [
        column
        for column in preferred_columns
        if column in result.columns
    ]

    extra_columns = [
        column
        for column in result.columns
        if column not in preferred_columns
    ]

    result = result[
        [
            *preferred_columns,
            *extra_columns,
        ]
    ]

    return (
        result
        .sort_values(
            [
                FORECAST_HOUR,
                "grid_i",
                "grid_j",
                GRID_ID,
            ]
        )
        .reset_index(
            drop=True
        )
    )

# ============================================================
# SUMMARY
# ============================================================


def build_summary(
    dataframe: pd.DataFrame,
) -> dict:

    hours = (
        dataframe[
            FORECAST_HOUR
        ]
        .drop_duplicates()
        .sort_values()
    )

    model_counts = (
        dataframe[
            "support_sensor_model"
        ]
        .value_counts()
    )

    unique_support = (
        dataframe[
            [
                "support_sensor_id",
                "support_sensor_model",
            ]
        ]
        .drop_duplicates()
    )

    support_by_model = (
        unique_support.groupby(
            "support_sensor_model"
        )[
            "support_sensor_id"
        ]
        .count()
    )

    return {
        "rows":
            int(
                len(
                    dataframe
                )
            ),

        "grid_cells":
            int(
                dataframe[
                    GRID_ID
                ]
                .nunique()
            ),

        "forecast_hours":
            int(
                len(
                    hours
                )
            ),

        "forecast_start":
            (
                hours.min().isoformat()
                if not hours.empty
                else None
            ),

        "forecast_end":
            (
                hours.max().isoformat()
                if not hours.empty
                else None
            ),

        "direct_grid_rows":
            int(
                (
                    ~dataframe[
                        "is_imputed_grid"
                    ]
                ).sum()
            ),

        "imputed_grid_rows":
            int(
                dataframe[
                    "is_imputed_grid"
                ].sum()
            ),

        "unique_support_sensors":
            int(
                dataframe[
                    "support_sensor_id"
                ]
                .nunique()
            ),

        "rows_by_model":
            {
                str(
                    model
                ):
                    int(
                        count
                    )
                for model, count
                in model_counts.items()
            },

        "unique_support_sensors_by_model":
            {
                str(
                    model
                ):
                    int(
                        count
                    )
                for model, count
                in support_by_model.items()
            },

        "model_features":
            MODEL_FEATURES,

        "weather_contract":
            (
                "precipitation belongs to target grid; "
                "model identity belongs to support sensor"
            ),

        "training_contract":
            (
                "grid forecast inputs are operational only "
                "and are never used as synthetic retraining observations"
            ),
    }


# ============================================================
# SAVE
# ============================================================


def save_outputs(
    *,
    dataframe: pd.DataFrame,
    summary: dict,
    output_dir: Path,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "grid_forecast_inputs.csv"
    )

    parquet_path = (
        output_dir
        / "grid_forecast_inputs.parquet"
    )

    summary_path = (
        output_dir
        / "grid_forecast_inputs_summary.json"
    )

    dataframe.to_csv(
        csv_path,
        index=False,
    )

    dataframe.to_parquet(
        parquet_path,
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
        "SAVED GRID FORECAST INPUTS"
    )

    print(
        "--------------------------"
    )

    print(
        csv_path
    )

    print(
        parquet_path
    )

    print(
        summary_path
    )


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Join target-grid-local day-ahead precipitation "
            "to the current grid support-sensor reference."
        )
    )

    parser.add_argument(
        "--imputation-reference",
        type=Path,
        default=DEFAULT_IMPUTATION_REFERENCE,
    )

    parser.add_argument(
        "--weather",
        type=Path,
        default=DEFAULT_GRID_WEATHER,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    args = parse_args()

    print(
        "BUILD NYC 1-KM GRID FORECAST INPUTS"
    )

    print(
        "==================================="
    )

    print()

    print(
        "FORECAST CONTRACT"
    )

    print(
        "-----------------"
    )

    print(
        "Model identity:  support sensor"
    )

    print(
        "Rainfall inputs: target 1-km grid"
    )

    print(
        "Training usage:  operational only"
    )

    print()

    spatial = (
        validate_spatial_reference(
            load_table(
                args.imputation_reference,
                description=(
                    "Grid imputation reference"
                ),
            )
        )
    )

    weather = (
        validate_weather(
            load_table(
                args.weather,
                description=(
                    "Grid precipitation forecast"
                ),
            )
        )
    )

    print(
        "INPUT AUDIT"
    )

    print(
        "-----------"
    )

    print(
        f"Spatial grid cells: "
        f"{len(spatial):,}"
    )

    print(
        f"Weather rows:       "
        f"{len(weather):,}"
    )

    print(
        f"Weather grid cells: "
        f"{weather[GRID_ID].nunique():,}"
    )

    print(
        f"Forecast hours:     "
        f"{weather[FORECAST_HOUR].nunique():,}"
    )

    result = (
        build_grid_forecast_inputs(
            spatial=spatial,
            weather=weather,
        )
    )

    summary = (
        build_summary(
            result
        )
    )

    print()

    print(
        "GRID FORECAST INPUT RESULTS"
    )

    print(
        "---------------------------"
    )

    print(
        f"Rows:                   "
        f"{summary['rows']:,}"
    )

    print(
        f"Grid cells:             "
        f"{summary['grid_cells']:,}"
    )

    print(
        f"Forecast hours:         "
        f"{summary['forecast_hours']:,}"
    )

    print(
        f"Direct-grid rows:       "
        f"{summary['direct_grid_rows']:,}"
    )

    print(
        f"Imputed-grid rows:      "
        f"{summary['imputed_grid_rows']:,}"
    )

    print(
        f"Unique support sensors: "
        f"{summary['unique_support_sensors']:,}"
    )

    print()

    print(
        "ROWS BY SELECTED MODEL"
    )

    print(
        "----------------------"
    )

    print(
        result[
            "support_sensor_model"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "TARGET-GRID PRECIPITATION RANGE"
    )

    print(
        "-------------------------------"
    )

    print(
        result[
            MODEL_FEATURES
        ]
        .describe()
        .to_string()
    )

    save_outputs(
        dataframe=result,
        summary=summary,
        output_dir=args.output_dir,
    )

    print()

    print(
        "GRID FORECAST INPUT BUILD COMPLETE"
    )

    print(
        "Inputs are ready for selected-model inference."
    )


if __name__ == "__main__":
    main()