"""
Map currently eligible flood sensors to the canonical NYC 1-km grid.

This module connects the adaptive model-selection layer to the spatial
imputation layer.

Inputs
------

Latest eligible sensor registry:

    artifacts/flood/model_selection/
        eligible_sensor_model_registry_with_coordinates.parquet

Authoritative NYC 1-km grid:

    data/raw/geospatial/
        311_flooding_grid_1km.geojson

Existing five-cluster assignments:

    data/raw/geospatial/
        grid_clusters.parquet

The sensor registry is dynamic. Its number of sensors may change after each
weekly model refresh.

Outputs
-------

artifacts/flood/spatial/

    eligible_sensor_grid_cluster_map.csv
    eligible_sensor_grid_cluster_map.parquet

    grid_primary_sensor_registry.csv
    grid_primary_sensor_registry.parquet

    cluster_primary_sensor_registry.csv
    cluster_primary_sensor_registry.parquet

    sensor_grid_mapping_summary.json

Selection rules
---------------

1. Every currently eligible sensor is assigned to a 1-km grid cell.

2. The sensor inherits that grid cell's cluster_number.

3. If multiple eligible sensors occupy the same 1-km grid cell:

       highest selected_f1 wins.

   Tie-breakers:

       higher selected_recall
       higher selected_precision
       deployment_id alphabetical order

4. A highest-F1 representative sensor is also identified for each cluster.

   This cluster-level representative is NOT intended to permanently replace
   all other eligible sensors. It provides a deterministic primary support
   sensor for downstream cluster-based imputation.

5. All eligible sensors are retained in the complete mapping artifact.

The output therefore preserves:

    - every eligible sensor;
    - its selected GCN/logistic model;
    - its quality metrics;
    - its grid cell;
    - its spatial cluster;
    - its ranking within its grid;
    - its ranking within its cluster.

No forecasting or imputation occurs in this module.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


# ============================================================
# DEFAULT PATHS
# ============================================================

DEFAULT_SENSOR_REGISTRY_PATH = (
    Path("artifacts")
    / "flood"
    / "model_selection"
    / "eligible_sensor_model_registry_with_coordinates.parquet"
)

DEFAULT_GRID_GEOJSON_PATH = (
    Path("data")
    / "raw"
    / "geospatial"
    / "311_flooding_grid_1km.geojson"
)

DEFAULT_GRID_CLUSTERS_PATH = (
    Path("data")
    / "raw"
    / "geospatial"
    / "grid_clusters.parquet"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "flood"
    / "spatial"
)


# ============================================================
# OUTPUT FILENAMES
# ============================================================

FULL_MAP_CSV = (
    "eligible_sensor_grid_cluster_map.csv"
)

FULL_MAP_PARQUET = (
    "eligible_sensor_grid_cluster_map.parquet"
)

GRID_PRIMARY_CSV = (
    "grid_primary_sensor_registry.csv"
)

GRID_PRIMARY_PARQUET = (
    "grid_primary_sensor_registry.parquet"
)

CLUSTER_PRIMARY_CSV = (
    "cluster_primary_sensor_registry.csv"
)

CLUSTER_PRIMARY_PARQUET = (
    "cluster_primary_sensor_registry.parquet"
)

SUMMARY_JSON = (
    "sensor_grid_mapping_summary.json"
)


# ============================================================
# CONTRACT
# ============================================================

SENSOR_ID = "deployment_id"

SENSOR_LAT = "sensor_lat"

SENSOR_LON = "sensor_lon"

SELECTED_MODEL = "selected_model"

SELECTED_F1 = "selected_f1"

SELECTED_PRECISION = "selected_precision"

SELECTED_RECALL = "selected_recall"

GRID_ID = "grid_id"

GRID_LAT = "grid_lat"

GRID_LON = "grid_lon"

CLUSTER_NUMBER = "cluster_number"


# ============================================================
# REQUIRED SENSOR COLUMNS
# ============================================================

REQUIRED_SENSOR_COLUMNS = {
    SENSOR_ID,
    SENSOR_LAT,
    SENSOR_LON,
    SELECTED_MODEL,
    SELECTED_F1,
    SELECTED_PRECISION,
    SELECTED_RECALL,
}


# ============================================================
# REQUIRED GRID CLUSTER COLUMNS
# ============================================================

REQUIRED_CLUSTER_COLUMNS = {
    GRID_ID,
}


# ============================================================
# GEOMETRY HELPERS
# ============================================================


def point_on_segment(
    x: float,
    y: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    tolerance: float = 1e-12,
) -> bool:
    """
    Return True when point (x, y) lies on a line segment.

    Used so sensors located exactly on grid boundaries are handled
    deterministically rather than being accidentally excluded.
    """

    cross = (
        (y - y1)
        * (x2 - x1)
        - (x - x1)
        * (y2 - y1)
    )

    if abs(cross) > tolerance:
        return False

    dot = (
        (x - x1)
        * (x - x2)
        + (y - y1)
        * (y - y2)
    )

    return (
        dot
        <= tolerance
    )


def point_in_ring(
    lon: float,
    lat: float,
    ring: list,
) -> bool:
    """
    Ray-casting point-in-polygon test.

    Boundary points are treated as inside.
    """

    if len(ring) < 4:
        return False

    inside = False

    j = (
        len(ring)
        - 1
    )

    for i in range(
        len(ring)
    ):
        xi = float(
            ring[i][0]
        )

        yi = float(
            ring[i][1]
        )

        xj = float(
            ring[j][0]
        )

        yj = float(
            ring[j][1]
        )

        if point_on_segment(
            lon,
            lat,
            xi,
            yi,
            xj,
            yj,
        ):
            return True

        intersects = (
            (
                yi > lat
            )
            != (
                yj > lat
            )
        )

        if intersects:
            denominator = (
                yj - yi
            )

            if abs(
                denominator
            ) < 1e-15:
                denominator = (
                    1e-15
                )

            intersection_lon = (
                (
                    xj - xi
                )
                * (
                    lat - yi
                )
                / denominator
                + xi
            )

            if lon < intersection_lon:
                inside = (
                    not inside
                )

        j = i

    return inside


def point_in_geometry(
    lon: float,
    lat: float,
    geometry: dict,
) -> bool:
    """
    Support GeoJSON Polygon and MultiPolygon geometry.
    """

    geometry_type = geometry.get(
        "type"
    )

    coordinates = geometry.get(
        "coordinates"
    )

    if not coordinates:
        return False

    if geometry_type == "Polygon":

        exterior = coordinates[
            0
        ]

        if not point_in_ring(
            lon,
            lat,
            exterior,
        ):
            return False

        # Respect holes if present.
        for hole in coordinates[
            1:
        ]:
            if point_in_ring(
                lon,
                lat,
                hole,
            ):
                return False

        return True

    if geometry_type == "MultiPolygon":

        for polygon in coordinates:

            exterior = polygon[
                0
            ]

            if not point_in_ring(
                lon,
                lat,
                exterior,
            ):
                continue

            inside_hole = any(
                point_in_ring(
                    lon,
                    lat,
                    hole,
                )
                for hole
                in polygon[
                    1:
                ]
            )

            if not inside_hole:
                return True

        return False

    raise ValueError(
        "Unsupported GeoJSON geometry type: "
        f"{geometry_type}"
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
    Great-circle distance in kilometers.
    """

    earth_radius_km = (
        6371.0088
    )

    phi1 = math.radians(
        lat1
    )

    phi2 = math.radians(
        lat2
    )

    delta_phi = math.radians(
        lat2
        - lat1
    )

    delta_lambda = math.radians(
        lon2
        - lon1
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
# INPUT LOADERS
# ============================================================


def load_sensor_registry(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            "Eligible sensor registry not found: "
            f"{path}"
        )

    suffix = (
        path.suffix
        .lower()
    )

    if suffix in {
        ".parquet",
        ".pq",
    }:
        sensors = pd.read_parquet(
            path
        )

    elif suffix == ".csv":
        sensors = pd.read_csv(
            path
        )

    else:
        raise ValueError(
            "Sensor registry must be CSV or Parquet."
        )

    missing = (
        REQUIRED_SENSOR_COLUMNS
        - set(
            sensors.columns
        )
    )

    if missing:
        raise ValueError(
            "Eligible sensor registry is missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    if sensors.empty:
        raise ValueError(
            "Eligible sensor registry contains no sensors."
        )

    sensors = (
        sensors
        .copy()
    )

    sensors[
        SENSOR_ID
    ] = (
        sensors[
            SENSOR_ID
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
    ]:
        sensors[
            column
        ] = pd.to_numeric(
            sensors[
                column
            ],
            errors="coerce",
        )

    required_numeric = [
        SENSOR_LAT,
        SENSOR_LON,
        SELECTED_F1,
        SELECTED_PRECISION,
        SELECTED_RECALL,
    ]

    invalid = (
        sensors[
            required_numeric
        ]
        .isna()
        .any(
            axis=1
        )
    )

    if invalid.any():
        raise ValueError(
            "Eligible sensor registry contains "
            f"{int(invalid.sum()):,} sensors with missing "
            "coordinates or selected-model metrics."
        )

    if sensors[
        SENSOR_ID
    ].duplicated().any():

        duplicates = (
            sensors.loc[
                sensors[
                    SENSOR_ID
                ].duplicated(
                    keep=False
                ),
                SENSOR_ID,
            ]
            .drop_duplicates()
            .head(
                20
            )
            .tolist()
        )

        raise ValueError(
            "Eligible sensor registry contains duplicate "
            "deployment_id values: "
            + ", ".join(
                duplicates
            )
        )

    return sensors


def load_grid_geojson(
    path: Path,
) -> tuple[
    dict,
    pd.DataFrame,
]:
    if not path.exists():
        raise FileNotFoundError(
            "1-km GeoJSON not found: "
            f"{path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        geojson = json.load(
            handle
        )

    features = geojson.get(
        "features",
        []
    )

    if not features:
        raise ValueError(
            "1-km GeoJSON contains no features."
        )

    rows: list[
        dict
    ] = []

    for index, feature in enumerate(
        features
    ):
        properties = (
            feature.get(
                "properties",
                {}
            )
            or {}
        )

        grid_id = properties.get(
            GRID_ID
        )

        if grid_id is None:
            raise ValueError(
                f"GeoJSON feature {index} has no grid_id."
            )

        geometry = feature.get(
            "geometry"
        )

        if not geometry:
            raise ValueError(
                f"GeoJSON feature {index} has no geometry."
            )

        rows.append(
            {
                GRID_ID:
                    str(
                        grid_id
                    ).strip(),
                "grid_i":
                    properties.get(
                        "grid_i"
                    ),
                "grid_j":
                    properties.get(
                        "grid_j"
                    ),
                GRID_LAT:
                    pd.to_numeric(
                        properties.get(
                            GRID_LAT
                        ),
                        errors="coerce",
                    ),
                GRID_LON:
                    pd.to_numeric(
                        properties.get(
                            GRID_LON
                        ),
                        errors="coerce",
                    ),
                "geometry":
                    geometry,
            }
        )

    grid = pd.DataFrame(
        rows
    )

    if grid[
        GRID_ID
    ].duplicated().any():
        raise ValueError(
            "GeoJSON contains duplicate grid_id values."
        )

    if (
        grid[
            GRID_LAT
        ].isna().any()
        or grid[
            GRID_LON
        ].isna().any()
    ):
        raise ValueError(
            "GeoJSON contains missing grid_lat/grid_lon values."
        )

    return (
        geojson,
        grid,
    )


def load_grid_clusters(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():

        csv_fallback = (
            path.with_suffix(
                ".csv"
            )
        )

        if csv_fallback.exists():
            path = (
                csv_fallback
            )

        else:
            raise FileNotFoundError(
                "Grid cluster table not found: "
                f"{path}"
            )

    if (
        path.suffix
        .lower()
        in {
            ".parquet",
            ".pq",
        }
    ):
        clusters = pd.read_parquet(
            path
        )

    elif (
        path.suffix
        .lower()
        == ".csv"
    ):
        clusters = pd.read_csv(
            path
        )

    else:
        raise ValueError(
            "Grid clusters must be CSV or Parquet."
        )

    if GRID_ID not in (
        clusters.columns
    ):
        raise ValueError(
            "Grid cluster table does not contain grid_id."
        )

    cluster_column = None

    for candidate in [
        "cluster_number",
        "Cluster_Number",
        "clusters",
    ]:
        if candidate in (
            clusters.columns
        ):
            cluster_column = (
                candidate
            )
            break

    if cluster_column is None:
        raise ValueError(
            "Grid cluster table does not contain a recognized "
            "cluster-number column."
        )

    result = (
        clusters
        .copy()
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
        CLUSTER_NUMBER
    ] = pd.to_numeric(
        result[
            CLUSTER_NUMBER
        ],
        errors="coerce",
    )

    if result[
        CLUSTER_NUMBER
    ].isna().any():
        raise ValueError(
            "Grid cluster table contains missing cluster assignments."
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

    return result


# ============================================================
# GRID + CLUSTER REFERENCE
# ============================================================


def build_grid_reference(
    grid: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    cluster_columns = [
        column
        for column in clusters.columns
        if column not in {
            "grid_i",
            "grid_j",
            GRID_LAT,
            GRID_LON,
        }
    ]

    cluster_subset = (
        clusters[
            cluster_columns
        ]
        .copy()
    )

    result = grid.merge(
        cluster_subset,
        on=GRID_ID,
        how="left",
        validate="one_to_one",
    )

    missing_cluster = (
        result[
            CLUSTER_NUMBER
        ]
        .isna()
    )

    if missing_cluster.any():

        missing_ids = (
            result.loc[
                missing_cluster,
                GRID_ID,
            ]
            .head(
                30
            )
            .tolist()
        )

        raise ValueError(
            "Some GeoJSON grid cells do not have cluster assignments: "
            + ", ".join(
                missing_ids
            )
        )

    extra_clusters = (
        clusters.loc[
            ~clusters[
                GRID_ID
            ].isin(
                grid[
                    GRID_ID
                ]
            ),
            GRID_ID,
        ]
    )

    if not extra_clusters.empty:
        raise ValueError(
            "Grid cluster table contains IDs that are not present "
            "in the GeoJSON."
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
# SENSOR → GRID ASSIGNMENT
# ============================================================


def assign_sensor_to_grid(
    sensor_lat: float,
    sensor_lon: float,
    grid_reference: pd.DataFrame,
) -> dict:
    """
    Assign one sensor to a grid polygon.

    Primary method:
        point-in-polygon

    Boundary condition:
        If multiple polygons contain the point because it falls exactly on
        a shared boundary, choose the grid whose stored grid centroid is
        geographically closest.

    Fallback:
        If no polygon contains the point, assign the nearest grid centroid.

    The fallback is explicitly recorded in assignment_method.
    """

    matches: list[
        dict
    ] = []

    for row in grid_reference.itertuples(
        index=False
    ):
        geometry = getattr(
            row,
            "geometry"
        )

        if point_in_geometry(
            sensor_lon,
            sensor_lat,
            geometry,
        ):
            matches.append(
                {
                    "grid_id":
                        getattr(
                            row,
                            GRID_ID
                        ),
                    "grid_i":
                        getattr(
                            row,
                            "grid_i"
                        ),
                    "grid_j":
                        getattr(
                            row,
                            "grid_j"
                        ),
                    "grid_lat":
                        getattr(
                            row,
                            GRID_LAT
                        ),
                    "grid_lon":
                        getattr(
                            row,
                            GRID_LON
                        ),
                    "cluster_number":
                        getattr(
                            row,
                            CLUSTER_NUMBER
                        ),
                }
            )

    if matches:

        for match in matches:
            match[
                "sensor_to_grid_centroid_km"
            ] = haversine_km(
                sensor_lat,
                sensor_lon,
                float(
                    match[
                        "grid_lat"
                    ]
                ),
                float(
                    match[
                        "grid_lon"
                    ]
                ),
            )

        matches = sorted(
            matches,
            key=lambda item: (
                item[
                    "sensor_to_grid_centroid_km"
                ],
                item[
                    GRID_ID
                ],
            ),
        )

        winner = (
            matches[
                0
            ]
        )

        winner[
            "assignment_method"
        ] = (
            "point_in_polygon"
            if len(
                matches
            ) == 1
            else "boundary_point_nearest_centroid"
        )

        winner[
            "candidate_grid_count"
        ] = len(
            matches
        )

        return winner

    # --------------------------------------------------------
    # Fallback: nearest centroid
    # --------------------------------------------------------

    candidates: list[
        dict
    ] = []

    for row in grid_reference.itertuples(
        index=False
    ):

        distance = (
            haversine_km(
                sensor_lat,
                sensor_lon,
                float(
                    getattr(
                        row,
                        GRID_LAT
                    )
                ),
                float(
                    getattr(
                        row,
                        GRID_LON
                    )
                ),
            )
        )

        candidates.append(
            {
                "grid_id":
                    getattr(
                        row,
                        GRID_ID
                    ),
                "grid_i":
                    getattr(
                        row,
                        "grid_i"
                    ),
                "grid_j":
                    getattr(
                        row,
                        "grid_j"
                    ),
                "grid_lat":
                    getattr(
                        row,
                        GRID_LAT
                    ),
                "grid_lon":
                    getattr(
                        row,
                        GRID_LON
                    ),
                "cluster_number":
                    getattr(
                        row,
                        CLUSTER_NUMBER
                    ),
                "sensor_to_grid_centroid_km":
                    distance,
            }
        )

    candidates = sorted(
        candidates,
        key=lambda item: (
            item[
                "sensor_to_grid_centroid_km"
            ],
            item[
                GRID_ID
            ],
        ),
    )

    winner = candidates[
        0
    ]

    winner[
        "assignment_method"
    ] = (
        "nearest_grid_centroid_fallback"
    )

    winner[
        "candidate_grid_count"
    ] = 0

    return winner


def map_sensors_to_grid(
    sensors: pd.DataFrame,
    grid_reference: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict
    ] = []

    total = len(
        sensors
    )

    print()
    print(
        "SENSOR → GRID ASSIGNMENT"
    )

    print(
        "------------------------"
    )

    print(
        f"Eligible sensors: {total:,}"
    )

    for index, sensor in enumerate(
        sensors.itertuples(
            index=False
        ),
        start=1,
    ):

        if (
            index == 1
            or index % 25 == 0
            or index == total
        ):
            print(
                f"Mapping sensor "
                f"{index:,}/{total:,}"
            )

        sensor_lat = float(
            getattr(
                sensor,
                SENSOR_LAT
            )
        )

        sensor_lon = float(
            getattr(
                sensor,
                SENSOR_LON
            )
        )

        assignment = (
            assign_sensor_to_grid(
                sensor_lat,
                sensor_lon,
                grid_reference,
            )
        )

        sensor_record = (
            sensor._asdict()
        )

        sensor_record.update(
            assignment
        )

        rows.append(
            sensor_record
        )

    result = pd.DataFrame(
        rows
    )

    return result


# ============================================================
# SUPPORT RANKING
# ============================================================


def add_support_ranking(
    mapped: pd.DataFrame,
) -> pd.DataFrame:
    """
    Rank eligible sensors within each grid and within each cluster.

    Highest F1 wins.

    Tie-break:
        recall
        precision
        deployment_id alphabetical
    """

    result = (
        mapped
        .copy()
    )

    ranking_order = [
        SELECTED_F1,
        SELECTED_RECALL,
        SELECTED_PRECISION,
        SENSOR_ID,
    ]

    ranking_ascending = [
        False,
        False,
        False,
        True,
    ]

    # --------------------------------------------------------
    # Grid-level ranking
    # --------------------------------------------------------

    result = (
        result
        .sort_values(
            [
                GRID_ID,
                *ranking_order,
            ],
            ascending=[
                True,
                *ranking_ascending,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    result[
        "support_rank_in_grid"
    ] = (
        result.groupby(
            GRID_ID
        )
        .cumcount()
        + 1
    )

    result[
        "is_primary_support_in_grid"
    ] = (
        result[
            "support_rank_in_grid"
        ]
        == 1
    )

    result[
        "eligible_sensor_count_in_grid"
    ] = (
        result.groupby(
            GRID_ID
        )[
            SENSOR_ID
        ]
        .transform(
            "count"
        )
    )

    # --------------------------------------------------------
    # Cluster-level ranking
    # --------------------------------------------------------

    result = (
        result
        .sort_values(
            [
                CLUSTER_NUMBER,
                *ranking_order,
            ],
            ascending=[
                True,
                *ranking_ascending,
            ],
        )
        .reset_index(
            drop=True
        )
    )

    result[
        "support_rank_in_cluster"
    ] = (
        result.groupby(
            CLUSTER_NUMBER
        )
        .cumcount()
        + 1
    )

    result[
        "is_primary_support_in_cluster"
    ] = (
        result[
            "support_rank_in_cluster"
        ]
        == 1
    )

    result[
        "eligible_sensor_count_in_cluster"
    ] = (
        result.groupby(
            CLUSTER_NUMBER
        )[
            SENSOR_ID
        ]
        .transform(
            "count"
        )
    )

    return (
        result
        .sort_values(
            SENSOR_ID
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PRIMARY REGISTRIES
# ============================================================


def build_grid_primary_registry(
    mapped: pd.DataFrame,
) -> pd.DataFrame:
    return (
        mapped.loc[
            mapped[
                "is_primary_support_in_grid"
            ]
        ]
        .copy()
        .sort_values(
            GRID_ID
        )
        .reset_index(
            drop=True
        )
    )


def build_cluster_primary_registry(
    mapped: pd.DataFrame,
) -> pd.DataFrame:
    return (
        mapped.loc[
            mapped[
                "is_primary_support_in_cluster"
            ]
        ]
        .copy()
        .sort_values(
            CLUSTER_NUMBER
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# SUMMARY
# ============================================================


def build_summary(
    mapped: pd.DataFrame,
    grid_reference: pd.DataFrame,
) -> dict:
    primary_grid = (
        mapped.loc[
            mapped[
                "is_primary_support_in_grid"
            ]
        ]
    )

    cluster_counts = (
        mapped.groupby(
            CLUSTER_NUMBER
        )[
            SENSOR_ID
        ]
        .count()
        .sort_index()
    )

    grids_with_sensor = int(
        mapped[
            GRID_ID
        ].nunique()
    )

    total_grids = int(
        grid_reference[
            GRID_ID
        ].nunique()
    )

    return {
        "eligible_sensor_count":
            int(
                len(
                    mapped
                )
            ),

        "total_grid_cells":
            total_grids,

        "grid_cells_with_eligible_sensor":
            grids_with_sensor,

        "grid_cells_without_eligible_sensor":
            int(
                total_grids
                - grids_with_sensor
            ),

        "grid_cells_with_multiple_eligible_sensors":
            int(
                (
                    primary_grid[
                        "eligible_sensor_count_in_grid"
                    ]
                    > 1
                )
                .sum()
            ),

        "clusters_present":
            sorted(
                int(
                    value
                )
                for value
                in mapped[
                    CLUSTER_NUMBER
                ].unique()
            ),

        "eligible_sensors_by_cluster":
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
                in cluster_counts.items()
            },

        "assignment_methods":
            {
                str(
                    key
                ):
                    int(
                        value
                    )
                for key, value
                in (
                    mapped[
                        "assignment_method"
                    ]
                    .value_counts()
                    .items()
                )
            },

        "selection_rule":
            (
                "highest selected_f1; "
                "tie-break selected_recall, "
                "selected_precision, deployment_id"
            ),
    }


# ============================================================
# SAVE
# ============================================================


def save_dataframe(
    dataframe: pd.DataFrame,
    csv_path: Path,
    parquet_path: Path,
) -> None:
    dataframe.to_csv(
        csv_path,
        index=False,
    )

    dataframe.to_parquet(
        parquet_path,
        index=False,
    )


def save_outputs(
    *,
    mapped: pd.DataFrame,
    grid_primary: pd.DataFrame,
    cluster_primary: pd.DataFrame,
    summary: dict,
    output_dir: Path,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_dataframe(
        mapped,
        output_dir
        / FULL_MAP_CSV,
        output_dir
        / FULL_MAP_PARQUET,
    )

    save_dataframe(
        grid_primary,
        output_dir
        / GRID_PRIMARY_CSV,
        output_dir
        / GRID_PRIMARY_PARQUET,
    )

    save_dataframe(
        cluster_primary,
        output_dir
        / CLUSTER_PRIMARY_CSV,
        output_dir
        / CLUSTER_PRIMARY_PARQUET,
    )

    summary_path = (
        output_dir
        / SUMMARY_JSON
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
        "SAVED SPATIAL SENSOR REGISTRIES"
    )

    print(
        "-------------------------------"
    )

    print(
        output_dir
        / FULL_MAP_PARQUET
    )

    print(
        output_dir
        / GRID_PRIMARY_PARQUET
    )

    print(
        output_dir
        / CLUSTER_PRIMARY_PARQUET
    )

    print(
        summary_path
    )


# ============================================================
# MAIN
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map the latest eligible flood sensors to the "
            "canonical NYC 1-km grid and cluster reference."
        )
    )

    parser.add_argument(
        "--sensors",
        type=Path,
        default=DEFAULT_SENSOR_REGISTRY_PATH,
    )

    parser.add_argument(
        "--grid",
        type=Path,
        default=DEFAULT_GRID_GEOJSON_PATH,
    )

    parser.add_argument(
        "--clusters",
        type=Path,
        default=DEFAULT_GRID_CLUSTERS_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(
        "MAP ELIGIBLE FLOOD SENSORS TO 1-KM GRID"
    )

    print(
        "======================================="
    )

    print(
        f"Sensor registry: {args.sensors}"
    )

    print(
        f"Grid GeoJSON:    {args.grid}"
    )

    print(
        f"Grid clusters:   {args.clusters}"
    )

    print()

    # --------------------------------------------------------
    # Load inputs
    # --------------------------------------------------------

    sensors = (
        load_sensor_registry(
            args.sensors
        )
    )

    _, grid = (
        load_grid_geojson(
            args.grid
        )
    )

    clusters = (
        load_grid_clusters(
            args.clusters
        )
    )

    grid_reference = (
        build_grid_reference(
            grid,
            clusters,
        )
    )

    print(
        "INPUT AUDIT"
    )

    print(
        "-----------"
    )

    print(
        f"Eligible sensors: "
        f"{len(sensors):,}"
    )

    print(
        f"Grid cells:       "
        f"{len(grid_reference):,}"
    )

    print(
        f"Clusters:         "
        f"{sorted(grid_reference[CLUSTER_NUMBER].unique().tolist())}"
    )

    # --------------------------------------------------------
    # Sensor → spatial assignment
    # --------------------------------------------------------

    mapped = (
        map_sensors_to_grid(
            sensors,
            grid_reference,
        )
    )

    # --------------------------------------------------------
    # Rank supports
    # --------------------------------------------------------

    mapped = (
        add_support_ranking(
            mapped
        )
    )

    grid_primary = (
        build_grid_primary_registry(
            mapped
        )
    )

    cluster_primary = (
        build_cluster_primary_registry(
            mapped
        )
    )

    summary = (
        build_summary(
            mapped,
            grid_reference,
        )
    )

    # --------------------------------------------------------
    # Console audit
    # --------------------------------------------------------

    print()
    print(
        "SPATIAL MAPPING RESULTS"
    )

    print(
        "-----------------------"
    )

    print(
        f"Eligible sensors mapped: "
        f"{len(mapped):,}"
    )

    print(
        f"Grid cells with ≥1 eligible sensor: "
        f"{mapped[GRID_ID].nunique():,}"
    )

    print(
        f"Grid cells without eligible sensor: "
        f"{len(grid_reference) - mapped[GRID_ID].nunique():,}"
    )

    print(
        f"Grid cells with multiple eligible sensors: "
        f"{summary['grid_cells_with_multiple_eligible_sensors']:,}"
    )

    print()
    print(
        "SENSORS BY CLUSTER"
    )

    print(
        "------------------"
    )

    cluster_distribution = (
        mapped[
            CLUSTER_NUMBER
        ]
        .value_counts()
        .sort_index()
    )

    print(
        cluster_distribution.to_string()
    )

    print()
    print(
        "ASSIGNMENT METHODS"
    )

    print(
        "------------------"
    )

    print(
        mapped[
            "assignment_method"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "PRIMARY SENSOR BY CLUSTER"
    )

    print(
        "-------------------------"
    )

    display_columns = [
        CLUSTER_NUMBER,
        SENSOR_ID,
        SELECTED_MODEL,
        SELECTED_F1,
        SELECTED_PRECISION,
        SELECTED_RECALL,
        GRID_ID,
    ]

    print(
        cluster_primary[
            display_columns
        ]
        .to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_outputs(
        mapped=mapped,
        grid_primary=grid_primary,
        cluster_primary=cluster_primary,
        summary=summary,
        output_dir=args.output_dir,
    )

    print()
    print(
        "SENSOR → GRID → CLUSTER MAPPING COMPLETE"
    )


if __name__ == "__main__":
    main()