"""
Build the 1-km grid imputation-support reference for flood forecasting.

This module does NOT produce a weather forecast or flood prediction.

Its purpose is to precompute which currently eligible FloodNet sensor should
support each NYC 1-km grid cell during the daily forecasting process.

The eligible sensor population is dynamic and is expected to be refreshed
periodically after GCN/logistic retraining and model-selection evaluation.

Current selection rules
-----------------------

For each 1-km grid cell:

1. If the grid contains one or more currently eligible sensors:

       use the eligible sensor in that grid with the highest selected F1.

2. If the grid contains no eligible sensor:

       identify the grid's cluster_number and use the currently eligible
       sensor in that cluster with the highest selected F1.

3. If a future weekly refresh produces a cluster with no eligible sensors:

       fall back to the geographically nearest currently eligible sensor,
       regardless of cluster.

Tie-breakers for F1-based selection:

       higher selected_recall
       higher selected_precision
       deployment_id alphabetical order

Geographic distance is retained as provenance and for fallback logic.
It does NOT override the highest-F1 rule inside a grid or cluster.

Inputs
------

Canonical grid / cluster table:

    data/raw/geospatial/grid_clusters.parquet

Authoritative 1-km grid GeoJSON:

    data/raw/geospatial/311_flooding_grid_1km.geojson

Eligible sensor spatial mapping:

    artifacts/flood/spatial/
        eligible_sensor_grid_cluster_map.parquet

Grid-level primary sensors:

    artifacts/flood/spatial/
        grid_primary_sensor_registry.parquet

Cluster-level primary sensors:

    artifacts/flood/spatial/
        cluster_primary_sensor_registry.parquet

Outputs
-------

artifacts/flood/spatial/

    grid_imputation_reference.csv
    grid_imputation_reference.parquet
    grid_imputation_reference.geojson
    grid_imputation_reference_summary.json

The primary downstream artifact is:

    grid_imputation_reference.parquet

It is intended to be consumed by the daily forecast/imputation service.

The reference contains one row per 1-km grid cell.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


# ============================================================
# DEFAULT INPUTS
# ============================================================

DEFAULT_GRID_CLUSTERS_PATH = (
    Path("data")
    / "raw"
    / "geospatial"
    / "grid_clusters.parquet"
)

DEFAULT_GRID_GEOJSON_PATH = (
    Path("data")
    / "raw"
    / "geospatial"
    / "311_flooding_grid_1km.geojson"
)

DEFAULT_SENSOR_MAP_PATH = (
    Path("artifacts")
    / "flood"
    / "spatial"
    / "eligible_sensor_grid_cluster_map.parquet"
)

DEFAULT_GRID_PRIMARY_PATH = (
    Path("artifacts")
    / "flood"
    / "spatial"
    / "grid_primary_sensor_registry.parquet"
)

DEFAULT_CLUSTER_PRIMARY_PATH = (
    Path("artifacts")
    / "flood"
    / "spatial"
    / "cluster_primary_sensor_registry.parquet"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "flood"
    / "spatial"
)


# ============================================================
# OUTPUT FILENAMES
# ============================================================

OUTPUT_CSV = (
    "grid_imputation_reference.csv"
)

OUTPUT_PARQUET = (
    "grid_imputation_reference.parquet"
)

OUTPUT_GEOJSON = (
    "grid_imputation_reference.geojson"
)

OUTPUT_SUMMARY = (
    "grid_imputation_reference_summary.json"
)


# ============================================================
# COLUMN CONTRACT
# ============================================================

GRID_ID = "grid_id"

GRID_I = "grid_i"

GRID_J = "grid_j"

GRID_LAT = "grid_lat"

GRID_LON = "grid_lon"

CLUSTER_NUMBER = "cluster_number"

SENSOR_ID = "deployment_id"

SENSOR_LAT = "sensor_lat"

SENSOR_LON = "sensor_lon"

SELECTED_MODEL = "selected_model"

SELECTED_F1 = "selected_f1"

SELECTED_PRECISION = "selected_precision"

SELECTED_RECALL = "selected_recall"


# ============================================================
# IMPUTATION METHOD LABELS
# ============================================================

METHOD_DIRECT_GRID_SENSOR = (
    "direct_grid_primary_sensor"
)

METHOD_CLUSTER_PRIMARY_SENSOR = (
    "same_cluster_primary_sensor"
)

METHOD_CROSS_CLUSTER_FALLBACK = (
    "nearest_sensor_cross_cluster_fallback"
)


# ============================================================
# DISTANCE
# ============================================================


def haversine_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate great-circle distance in kilometers.
    """

    earth_radius_km = (
        6371.0088
    )

    phi1 = math.radians(
        float(
            lat1
        )
    )

    phi2 = math.radians(
        float(
            lat2
        )
    )

    delta_phi = math.radians(
        float(
            lat2
        )
        - float(
            lat1
        )
    )

    delta_lambda = math.radians(
        float(
            lon2
        )
        - float(
            lon1
        )
    )

    a = (
        math.sin(
            delta_phi
            / 2.0
        )
        ** 2
        + math.cos(
            phi1
        )
        * math.cos(
            phi2
        )
        * math.sin(
            delta_lambda
            / 2.0
        )
        ** 2
    )

    c = (
        2.0
        * math.atan2(
            math.sqrt(
                a
            ),
            math.sqrt(
                1.0
                - a
            ),
        )
    )

    return (
        earth_radius_km
        * c
    )


# ============================================================
# GENERIC TABLE LOADER
# ============================================================


def load_table(
    path: Path,
    *,
    description: str,
) -> pd.DataFrame:
    """
    Load CSV or Parquet.

    If the requested Parquet is unavailable, automatically try a CSV with
    the same stem.
    """

    actual_path = (
        path
    )

    if not actual_path.exists():

        if (
            actual_path.suffix
            .lower()
            in {
                ".parquet",
                ".pq",
            }
        ):
            fallback = (
                actual_path
                .with_suffix(
                    ".csv"
                )
            )

            if fallback.exists():
                actual_path = (
                    fallback
                )

        if not actual_path.exists():
            raise FileNotFoundError(
                f"{description} not found: {path}"
            )

    suffix = (
        actual_path.suffix
        .lower()
    )

    if suffix in {
        ".parquet",
        ".pq",
    }:
        dataframe = pd.read_parquet(
            actual_path
        )

    elif suffix == ".csv":
        dataframe = pd.read_csv(
            actual_path
        )

    else:
        raise ValueError(
            f"{description} must be CSV or Parquet: "
            f"{actual_path}"
        )

    if dataframe.empty:
        raise ValueError(
            f"{description} is empty."
        )

    return dataframe


# ============================================================
# GRID REFERENCE
# ============================================================


def normalize_grid_clusters(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize the clustering-table schema.
    """

    result = (
        dataframe
        .copy()
    )

    cluster_column = None

    for candidate in [
        "cluster_number",
        "Cluster_Number",
        "clusters",
        "cluster",
    ]:
        if candidate in (
            result.columns
        ):
            cluster_column = (
                candidate
            )
            break

    if cluster_column is None:
        raise ValueError(
            "Grid cluster table has no recognized "
            "cluster-number column."
        )

    if (
        cluster_column
        != CLUSTER_NUMBER
    ):
        result = result.rename(
            columns={
                cluster_column:
                    CLUSTER_NUMBER
            }
        )

    required = {
        GRID_ID,
        GRID_I,
        GRID_J,
        GRID_LAT,
        GRID_LON,
        CLUSTER_NUMBER,
    }

    missing = (
        required
        - set(
            result.columns
        )
    )

    if missing:
        raise ValueError(
            "Grid cluster table is missing required columns: "
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

    for column in [
        GRID_LAT,
        GRID_LON,
        CLUSTER_NUMBER,
    ]:
        result[
            column
        ] = pd.to_numeric(
            result[
                column
            ],
            errors="coerce",
        )

    invalid = (
        result[
            [
                GRID_LAT,
                GRID_LON,
                CLUSTER_NUMBER,
            ]
        ]
        .isna()
        .any(
            axis=1
        )
    )

    if invalid.any():
        raise ValueError(
            "Grid cluster table contains "
            f"{int(invalid.sum()):,} rows with invalid "
            "coordinates or cluster assignments."
        )

    result[
        CLUSTER_NUMBER
    ] = result[
        CLUSTER_NUMBER
    ].astype(
        int
    )

    if result[
        GRID_ID
    ].duplicated().any():
        raise ValueError(
            "Grid cluster table contains duplicate grid_id values."
        )

    return (
        result
        .sort_values(
            [
                GRID_I,
                GRID_J,
                GRID_ID,
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# SENSOR REGISTRY NORMALIZATION
# ============================================================


def validate_sensor_registry(
    dataframe: pd.DataFrame,
    *,
    description: str,
) -> pd.DataFrame:
    result = (
        dataframe
        .copy()
    )

    required = {
        SENSOR_ID,
        SENSOR_LAT,
        SENSOR_LON,
        SELECTED_MODEL,
        SELECTED_F1,
        SELECTED_PRECISION,
        SELECTED_RECALL,
        GRID_ID,
        CLUSTER_NUMBER,
    }

    missing = (
        required
        - set(
            result.columns
        )
    )

    if missing:
        raise ValueError(
            f"{description} is missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    result[
        SENSOR_ID
    ] = (
        result[
            SENSOR_ID
        ]
        .astype(str)
        .str.strip()
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

    for column in [
        SENSOR_LAT,
        SENSOR_LON,
        SELECTED_F1,
        SELECTED_PRECISION,
        SELECTED_RECALL,
        CLUSTER_NUMBER,
    ]:
        result[
            column
        ] = pd.to_numeric(
            result[
                column
            ],
            errors="coerce",
        )

    invalid = (
        result[
            [
                SENSOR_LAT,
                SENSOR_LON,
                SELECTED_F1,
                SELECTED_PRECISION,
                SELECTED_RECALL,
                CLUSTER_NUMBER,
            ]
        ]
        .isna()
        .any(
            axis=1
        )
    )

    if invalid.any():
        raise ValueError(
            f"{description} contains "
            f"{int(invalid.sum()):,} rows with invalid "
            "sensor/model/spatial values."
        )

    result[
        CLUSTER_NUMBER
    ] = result[
        CLUSTER_NUMBER
    ].astype(
        int
    )

    return result


# ============================================================
# FALLBACK SENSOR
# ============================================================


def find_nearest_sensor(
    *,
    grid_lat: float,
    grid_lon: float,
    sensors: pd.DataFrame,
) -> pd.Series:
    """
    Find the nearest currently eligible sensor across all clusters.

    Distance is primary only for this exceptional cross-cluster fallback.

    Tie-breakers:
        higher selected_f1
        higher selected_recall
        higher selected_precision
        deployment_id
    """

    if sensors.empty:
        raise RuntimeError(
            "Cannot perform fallback because there are no "
            "currently eligible sensors."
        )

    candidates = (
        sensors
        .copy()
    )

    candidates[
        "_distance_km"
    ] = candidates.apply(
        lambda row: (
            haversine_km(
                grid_lat,
                grid_lon,
                float(
                    row[
                        SENSOR_LAT
                    ]
                ),
                float(
                    row[
                        SENSOR_LON
                    ]
                ),
            )
        ),
        axis=1,
    )

    candidates = (
        candidates
        .sort_values(
            [
                "_distance_km",
                SELECTED_F1,
                SELECTED_RECALL,
                SELECTED_PRECISION,
                SENSOR_ID,
            ],
            ascending=[
                True,
                False,
                False,
                False,
                True,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    return candidates.iloc[
        0
    ]


# ============================================================
# SUPPORT RECORD
# ============================================================


def support_record(
    *,
    grid: pd.Series,
    sensor: pd.Series,
    method: str,
    source_scope: str,
) -> dict:
    distance = (
        haversine_km(
            float(
                grid[
                    GRID_LAT
                ]
            ),
            float(
                grid[
                    GRID_LON
                ]
            ),
            float(
                sensor[
                    SENSOR_LAT
                ]
            ),
            float(
                sensor[
                    SENSOR_LON
                ]
            ),
        )
    )

    return {
        GRID_ID:
            str(
                grid[
                    GRID_ID
                ]
            ),

        GRID_I:
            grid[
                GRID_I
            ],

        GRID_J:
            grid[
                GRID_J
            ],

        GRID_LAT:
            float(
                grid[
                    GRID_LAT
                ]
            ),

        GRID_LON:
            float(
                grid[
                    GRID_LON
                ]
            ),

        CLUSTER_NUMBER:
            int(
                grid[
                    CLUSTER_NUMBER
                ]
            ),

        "has_eligible_sensor_in_grid":
            bool(
                source_scope
                == "grid"
            ),

        "imputation_required":
            bool(
                source_scope
                != "grid"
            ),

        "imputation_method":
            method,

        "support_scope":
            source_scope,

        "support_sensor_id":
            str(
                sensor[
                    SENSOR_ID
                ]
            ),

        "support_sensor_model":
            str(
                sensor[
                    SELECTED_MODEL
                ]
            ),

        "support_sensor_f1":
            float(
                sensor[
                    SELECTED_F1
                ]
            ),

        "support_sensor_precision":
            float(
                sensor[
                    SELECTED_PRECISION
                ]
            ),

        "support_sensor_recall":
            float(
                sensor[
                    SELECTED_RECALL
                ]
            ),

        "support_sensor_lat":
            float(
                sensor[
                    SENSOR_LAT
                ]
            ),

        "support_sensor_lon":
            float(
                sensor[
                    SENSOR_LON
                ]
            ),

        "support_sensor_grid_id":
            str(
                sensor[
                    GRID_ID
                ]
            ),

        "support_sensor_cluster_number":
            int(
                sensor[
                    CLUSTER_NUMBER
                ]
            ),

        "support_sensor_distance_km":
            float(
                distance
            ),

        "same_cluster_support":
            bool(
                int(
                    sensor[
                        CLUSTER_NUMBER
                    ]
                )
                == int(
                    grid[
                        CLUSTER_NUMBER
                    ]
                )
            ),
    }


# ============================================================
# BUILD REFERENCE
# ============================================================


def build_imputation_reference(
    *,
    grid: pd.DataFrame,
    all_sensors: pd.DataFrame,
    grid_primary: pd.DataFrame,
    cluster_primary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce one support-sensor assignment per 1-km grid cell.
    """

    grid_primary_lookup = {
        str(
            row[
                GRID_ID
            ]
        ):
            row
        for _,
        row
        in grid_primary.iterrows()
    }

    cluster_primary_lookup = {
        int(
            row[
                CLUSTER_NUMBER
            ]
        ):
            row
        for _,
        row
        in cluster_primary.iterrows()
    }

    rows: list[
        dict
    ] = []

    total = len(
        grid
    )

    print()
    print(
        "BUILD GRID SUPPORT REFERENCE"
    )

    print(
        "----------------------------"
    )

    print(
        f"Grid cells: {total:,}"
    )

    for index, (
        _,
        grid_row,
    ) in enumerate(
        grid.iterrows(),
        start=1,
    ):

        if (
            index == 1
            or index % 100 == 0
            or index == total
        ):
            print(
                f"Processing grid "
                f"{index:,}/{total:,}"
            )

        grid_id = str(
            grid_row[
                GRID_ID
            ]
        )

        cluster_number = int(
            grid_row[
                CLUSTER_NUMBER
            ]
        )

        # ----------------------------------------------------
        # Case 1:
        # Grid physically contains one or more eligible sensors.
        #
        # grid_primary already contains the highest-F1 sensor.
        # ----------------------------------------------------

        grid_sensor = (
            grid_primary_lookup
            .get(
                grid_id
            )
        )

        if grid_sensor is not None:

            rows.append(
                support_record(
                    grid=grid_row,
                    sensor=grid_sensor,
                    method=METHOD_DIRECT_GRID_SENSOR,
                    source_scope="grid",
                )
            )

            continue

        # ----------------------------------------------------
        # Case 2:
        # No eligible sensor in this grid.
        #
        # Use the highest-F1 eligible sensor in the SAME cluster.
        # ----------------------------------------------------

        cluster_sensor = (
            cluster_primary_lookup
            .get(
                cluster_number
            )
        )

        if cluster_sensor is not None:

            rows.append(
                support_record(
                    grid=grid_row,
                    sensor=cluster_sensor,
                    method=METHOD_CLUSTER_PRIMARY_SENSOR,
                    source_scope="cluster",
                )
            )

            continue

        # ----------------------------------------------------
        # Case 3:
        # Future-proof fallback.
        #
        # This can happen if a later adaptive sensor refresh produces
        # zero eligible sensors for a particular cluster.
        #
        # Use geographically nearest eligible sensor.
        # ----------------------------------------------------

        fallback_sensor = (
            find_nearest_sensor(
                grid_lat=float(
                    grid_row[
                        GRID_LAT
                    ]
                ),
                grid_lon=float(
                    grid_row[
                        GRID_LON
                    ]
                ),
                sensors=all_sensors,
            )
        )

        rows.append(
            support_record(
                grid=grid_row,
                sensor=fallback_sensor,
                method=METHOD_CROSS_CLUSTER_FALLBACK,
                source_scope="cross_cluster_fallback",
            )
        )

    result = pd.DataFrame(
        rows
    )

    if len(
        result
    ) != len(
        grid
    ):
        raise RuntimeError(
            "Imputation reference row count does not match "
            "grid-cell count."
        )

    if result[
        GRID_ID
    ].duplicated().any():
        raise RuntimeError(
            "Imputation reference contains duplicate grid_id values."
        )

    return (
        result
        .sort_values(
            [
                GRID_I,
                GRID_J,
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
    reference: pd.DataFrame,
) -> dict:
    method_counts = (
        reference[
            "imputation_method"
        ]
        .value_counts()
    )

    cluster_distribution = (
        reference[
            CLUSTER_NUMBER
        ]
        .value_counts()
        .sort_index()
    )

    imputed = reference.loc[
        reference[
            "imputation_required"
        ]
    ]

    return {
        "grid_cells_total":
            int(
                len(
                    reference
                )
            ),

        "grid_cells_direct_sensor":
            int(
                (
                    ~reference[
                        "imputation_required"
                    ]
                ).sum()
            ),

        "grid_cells_imputed":
            int(
                reference[
                    "imputation_required"
                ].sum()
            ),

        "method_counts":
            {
                str(
                    method
                ):
                    int(
                        count
                    )
                for method, count
                in method_counts.items()
            },

        "grid_cells_by_cluster":
            {
                str(
                    int(
                        cluster
                    )
                ):
                    int(
                        count
                    )
                for cluster, count
                in cluster_distribution.items()
            },

        "clusters_using_cross_cluster_fallback":
            sorted(
                int(
                    value
                )
                for value
                in reference.loc[
                    reference[
                        "imputation_method"
                    ]
                    == METHOD_CROSS_CLUSTER_FALLBACK,
                    CLUSTER_NUMBER,
                ]
                .unique()
            ),

        "support_sensor_count":
            int(
                reference[
                    "support_sensor_id"
                ]
                .nunique()
            ),

        "mean_imputed_support_distance_km":
            (
                float(
                    imputed[
                        "support_sensor_distance_km"
                    ].mean()
                )
                if not imputed.empty
                else None
            ),

        "median_imputed_support_distance_km":
            (
                float(
                    imputed[
                        "support_sensor_distance_km"
                    ].median()
                )
                if not imputed.empty
                else None
            ),

        "max_imputed_support_distance_km":
            (
                float(
                    imputed[
                        "support_sensor_distance_km"
                    ].max()
                )
                if not imputed.empty
                else None
            ),

        "selection_rule":
            (
                "direct grid: highest selected F1 in grid; "
                "unsensored grid: highest selected F1 in same cluster; "
                "fallback: nearest eligible sensor across clusters"
            ),
    }


# ============================================================
# GEOJSON ENRICHMENT
# ============================================================


def enrich_geojson(
    *,
    geojson_path: Path,
    reference: pd.DataFrame,
) -> dict:
    if not geojson_path.exists():
        raise FileNotFoundError(
            "Authoritative grid GeoJSON not found: "
            f"{geojson_path}"
        )

    with geojson_path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        geojson = json.load(
            handle
        )

    lookup = (
        reference
        .set_index(
            GRID_ID
        )
        .to_dict(
            orient="index"
        )
    )

    feature_count = 0

    for feature in geojson.get(
        "features",
        []
    ):

        properties = (
            feature.setdefault(
                "properties",
                {}
            )
        )

        grid_id = str(
            properties.get(
                GRID_ID
            )
        ).strip()

        row = lookup.get(
            grid_id
        )

        if row is None:
            raise ValueError(
                "GeoJSON grid has no imputation-reference row: "
                f"{grid_id}"
            )

        properties[
            CLUSTER_NUMBER
        ] = int(
            row[
                CLUSTER_NUMBER
            ]
        )

        properties[
            "imputation_required"
        ] = bool(
            row[
                "imputation_required"
            ]
        )

        properties[
            "imputation_method"
        ] = str(
            row[
                "imputation_method"
            ]
        )

        properties[
            "support_sensor_id"
        ] = str(
            row[
                "support_sensor_id"
            ]
        )

        properties[
            "support_sensor_model"
        ] = str(
            row[
                "support_sensor_model"
            ]
        )

        properties[
            "support_sensor_f1"
        ] = float(
            row[
                "support_sensor_f1"
            ]
        )

        properties[
            "support_sensor_distance_km"
        ] = float(
            row[
                "support_sensor_distance_km"
            ]
        )

        feature_count += 1

    if feature_count != len(
        reference
    ):
        raise ValueError(
            "GeoJSON feature count does not match "
            "imputation-reference row count."
        )

    return geojson


# ============================================================
# SAVE OUTPUTS
# ============================================================


def save_outputs(
    *,
    reference: pd.DataFrame,
    enriched_geojson: dict,
    summary: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / OUTPUT_CSV
    )

    parquet_path = (
        output_dir
        / OUTPUT_PARQUET
    )

    geojson_path = (
        output_dir
        / OUTPUT_GEOJSON
    )

    summary_path = (
        output_dir
        / OUTPUT_SUMMARY
    )

    reference.to_csv(
        csv_path,
        index=False,
    )

    reference.to_parquet(
        parquet_path,
        index=False,
    )

    with geojson_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            enriched_geojson,
            handle,
            ensure_ascii=False,
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
        "SAVED GRID IMPUTATION REFERENCE"
    )

    print(
        "-------------------------------"
    )

    print(
        csv_path
    )

    print(
        parquet_path
    )

    print(
        geojson_path
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
            "Build one primary flood-support sensor assignment "
            "for every NYC 1-km grid cell."
        )
    )

    parser.add_argument(
        "--grid-clusters",
        type=Path,
        default=DEFAULT_GRID_CLUSTERS_PATH,
    )

    parser.add_argument(
        "--grid-geojson",
        type=Path,
        default=DEFAULT_GRID_GEOJSON_PATH,
    )

    parser.add_argument(
        "--sensor-map",
        type=Path,
        default=DEFAULT_SENSOR_MAP_PATH,
    )

    parser.add_argument(
        "--grid-primary",
        type=Path,
        default=DEFAULT_GRID_PRIMARY_PATH,
    )

    parser.add_argument(
        "--cluster-primary",
        type=Path,
        default=DEFAULT_CLUSTER_PRIMARY_PATH,
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
        "BUILD NYC 1-KM GRID IMPUTATION REFERENCE"
    )

    print(
        "========================================"
    )

    print()

    print(
        "Support hierarchy:"
    )

    print(
        "    1. highest-F1 eligible sensor physically in grid"
    )

    print(
        "    2. highest-F1 eligible sensor in same cluster"
    )

    print(
        "    3. nearest eligible sensor across clusters"
    )

    print()

    print(
        "Distance does NOT override F1 inside a grid or cluster."
    )

    # --------------------------------------------------------
    # Inputs
    # --------------------------------------------------------

    grid_clusters = load_table(
        args.grid_clusters,
        description="Grid clusters",
    )

    grid = normalize_grid_clusters(
        grid_clusters
    )

    all_sensors = (
        validate_sensor_registry(
            load_table(
                args.sensor_map,
                description="Eligible sensor grid/cluster map",
            ),
            description="Eligible sensor grid/cluster map",
        )
    )

    grid_primary = (
        validate_sensor_registry(
            load_table(
                args.grid_primary,
                description="Grid primary sensor registry",
            ),
            description="Grid primary sensor registry",
        )
    )

    cluster_primary = (
        validate_sensor_registry(
            load_table(
                args.cluster_primary,
                description="Cluster primary sensor registry",
            ),
            description="Cluster primary sensor registry",
        )
    )

    print(
        "INPUT AUDIT"
    )

    print(
        "-----------"
    )

    print(
        f"Grid cells:                  "
        f"{len(grid):,}"
    )

    print(
        f"Eligible sensors:            "
        f"{len(all_sensors):,}"
    )

    print(
        f"Grid primary sensors:        "
        f"{len(grid_primary):,}"
    )

    print(
        f"Cluster primary sensors:     "
        f"{len(cluster_primary):,}"
    )

    print(
        f"Clusters represented:        "
        f"{sorted(grid[CLUSTER_NUMBER].unique().tolist())}"
    )

    # --------------------------------------------------------
    # Validate primary registries
    # --------------------------------------------------------

    if grid_primary[
        GRID_ID
    ].duplicated().any():
        raise ValueError(
            "Grid primary registry contains multiple primary "
            "sensors for the same grid."
        )

    if cluster_primary[
        CLUSTER_NUMBER
    ].duplicated().any():
        raise ValueError(
            "Cluster primary registry contains multiple primary "
            "sensors for the same cluster."
        )

    # --------------------------------------------------------
    # Build
    # --------------------------------------------------------

    reference = (
        build_imputation_reference(
            grid=grid,
            all_sensors=all_sensors,
            grid_primary=grid_primary,
            cluster_primary=cluster_primary,
        )
    )

    summary = (
        build_summary(
            reference
        )
    )

    enriched_geojson = (
        enrich_geojson(
            geojson_path=args.grid_geojson,
            reference=reference,
        )
    )

    # --------------------------------------------------------
    # Console audit
    # --------------------------------------------------------

    print()
    print(
        "GRID IMPUTATION RESULTS"
    )

    print(
        "-----------------------"
    )

    print(
        f"Total grid cells:          "
        f"{summary['grid_cells_total']:,}"
    )

    print(
        f"Direct sensor grids:       "
        f"{summary['grid_cells_direct_sensor']:,}"
    )

    print(
        f"Imputed grid cells:        "
        f"{summary['grid_cells_imputed']:,}"
    )

    print()

    print(
        "IMPUTATION METHODS"
    )

    print(
        "------------------"
    )

    print(
        reference[
            "imputation_method"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "GRID CELLS BY CLUSTER"
    )

    print(
        "---------------------"
    )

    print(
        reference[
            CLUSTER_NUMBER
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print()

    print(
        "UNIQUE SUPPORT SENSORS"
    )

    print(
        "----------------------"
    )

    print(
        reference[
            "support_sensor_id"
        ]
        .nunique()
    )

    print()

    print(
        "SUPPORT DISTANCE FOR IMPUTED CELLS"
    )

    print(
        "----------------------------------"
    )

    imputed = reference.loc[
        reference[
            "imputation_required"
        ]
    ]

    if not imputed.empty:

        print(
            imputed[
                "support_sensor_distance_km"
            ]
            .describe()
            .to_string()
        )

    print()

    print(
        "PRIMARY CLUSTER SUPPORT USED"
    )

    print(
        "----------------------------"
    )

    cluster_support = (
        reference.loc[
            reference[
                "imputation_method"
            ]
            == METHOD_CLUSTER_PRIMARY_SENSOR,
            [
                CLUSTER_NUMBER,
                "support_sensor_id",
                "support_sensor_model",
                "support_sensor_f1",
            ],
        ]
        .drop_duplicates()
        .sort_values(
            CLUSTER_NUMBER
        )
    )

    print(
        cluster_support.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_outputs(
        reference=reference,
        enriched_geojson=enriched_geojson,
        summary=summary,
        output_dir=args.output_dir,
    )

    print()

    print(
        "GRID IMPUTATION REFERENCE COMPLETE"
    )

    print(
        "All 1-km grid cells now have an authoritative "
        "support-sensor assignment."
    )


if __name__ == "__main__":
    main()