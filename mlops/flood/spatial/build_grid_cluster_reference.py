"""
Build the canonical NYC 1-km flood-grid + cluster reference.

This module does NOT re-run clustering.

It takes:
    1. the authoritative 1-km GeoJSON grid; and
    2. the existing clustering output containing Cluster_Number / clusters

and produces:
    - grid_clusters.csv
    - grid_clusters.parquet
    - canonical_grid_cluster_reference.csv
    - canonical_grid_cluster_reference.parquet
    - canonical_grid_cluster_reference.geojson

The canonical reference is intended to be static/versioned and reused by the
weekly sensor-selection refresh and the daily forecasting/imputation pipeline.

Example
-------
python -m mlops.flood.spatial.build_grid_cluster_reference ^
    --grid data/raw/geospatial/311_flooding_grid_1km.geojson ^
    --clusters data/raw/geospatial/Get_Clusters_with_Clusters.xlsx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_GRID_PATH = (
    Path("data")
    / "raw"
    / "geospatial"
    / "311_flooding_grid_1km.geojson"
)

DEFAULT_CLUSTER_PATH = (
    Path("data")
    / "raw"
    / "geospatial"
    / "Get_Clusters_with_Clusters.xlsx"
)

DEFAULT_OUTPUT_DIR = (
    Path("artifacts")
    / "flood"
    / "spatial"
)

GRID_CLUSTERS_CSV = "grid_clusters.csv"
GRID_CLUSTERS_PARQUET = "grid_clusters.parquet"

CANONICAL_CSV = "canonical_grid_cluster_reference.csv"
CANONICAL_PARQUET = "canonical_grid_cluster_reference.parquet"
CANONICAL_GEOJSON = "canonical_grid_cluster_reference.geojson"

EXPECTED_GRID_COUNT = 837
EXPECTED_CLUSTERS = {1, 2, 3, 4, 5}


GRID_ID_ALIASES = (
    "grid_id",
    "gridid",
    "grid id",
)

GRID_LAT_ALIASES = (
    "grid_lat",
    "grid latitude",
    "latitude",
    "lat",
)

GRID_LON_ALIASES = (
    "grid_lon",
    "grid longitude",
    "longitude",
    "lon",
    "long",
)

CLUSTER_ALIASES = (
    "cluster_number",
    "cluster number",
    "clusters",
    "cluster",
    "Cluster_Number",
)

OPTIONAL_CLUSTER_COLUMNS = {
    "slope_mean": (
        "Slope_mean",
        "slope_mean",
        "mean slope",
    ),
    "elevation_mean": (
        "Elevation_Mean",
        "Elevation_mean",
        "elevation_mean",
        "mean elevation",
    ),
    "mean_ifld_risks": (
        "Mean IFLD_RISKS",
        "mean_IFLD_RISKS",
        "Mean_IFLD_RISKS",
        "mean_ifld_risks",
    ),
    "sum_shape_area": (
        "Sum Shape_Area",
        "sum_shape_area",
        "Sum_Shape_Area",
    ),
    "summarized_area_squarefeet": (
        "Summarized Area in SQUAREFEET",
        "sum_Area_SQUAREFEET",
        "summarized_area_squarefeet",
    ),
    "polygon_count": (
        "Count of Polygons",
        "Polygon_Count",
        "count_of_polygons",
        "polygon_count",
    ),
}


def _normalized_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
    )


def _find_column(
    columns: Iterable[str],
    aliases: Iterable[str],
) -> str | None:
    lookup = {
        _normalized_name(column): column
        for column in columns
    }

    for alias in aliases:
        found = lookup.get(
            _normalized_name(alias)
        )
        if found is not None:
            return found

    return None


def _grid_id_from_coordinates(
    lat: pd.Series,
    lon: pd.Series,
) -> pd.Series:
    lat_num = pd.to_numeric(
        lat,
        errors="coerce",
    )

    lon_num = pd.to_numeric(
        lon,
        errors="coerce",
    )

    if lat_num.isna().any() or lon_num.isna().any():
        raise ValueError(
            "Cannot derive grid_id because grid latitude/longitude "
            "contains missing or non-numeric values."
        )

    return pd.Series(
        [
            f"{latitude:.3f}_{longitude:.3f}"
            for latitude, longitude
            in zip(
                lat_num,
                lon_num,
            )
        ],
        index=lat.index,
        dtype="string",
    )


def load_grid_geojson(
    path: Path,
) -> tuple[dict, pd.DataFrame]:
    if not path.exists():
        raise FileNotFoundError(
            f"1-km grid GeoJSON not found: {path}"
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
        [],
    )

    if not features:
        raise ValueError(
            "Grid GeoJSON contains no features."
        )

    rows: list[dict] = []

    for index, feature in enumerate(
        features
    ):
        properties = (
            feature.get(
                "properties",
                {},
            )
            or {}
        )

        geometry = feature.get(
            "geometry"
        )

        if not geometry:
            raise ValueError(
                f"Grid feature {index} has no geometry."
            )

        grid_id = properties.get(
            "grid_id"
        )

        grid_lat = properties.get(
            "grid_lat"
        )

        grid_lon = properties.get(
            "grid_lon"
        )

        if grid_id is None:
            if (
                grid_lat is None
                or grid_lon is None
            ):
                raise ValueError(
                    f"Grid feature {index} has neither grid_id "
                    "nor grid_lat/grid_lon."
                )

            grid_id = (
                f"{float(grid_lat):.3f}_"
                f"{float(grid_lon):.3f}"
            )

        rows.append(
            {
                "grid_id":
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
                "grid_lat":
                    pd.to_numeric(
                        grid_lat,
                        errors="coerce",
                    ),
                "grid_lon":
                    pd.to_numeric(
                        grid_lon,
                        errors="coerce",
                    ),
                "n_sf":
                    properties.get(
                        "n_sf"
                    ),
                "n_cb":
                    properties.get(
                        "n_cb"
                    ),
                "n_total":
                    properties.get(
                        "n_total"
                    ),
                "_feature_index":
                    index,
            }
        )

    grid = pd.DataFrame(
        rows
    )

    if grid["grid_id"].duplicated().any():
        duplicates = (
            grid.loc[
                grid["grid_id"].duplicated(
                    keep=False
                ),
                "grid_id",
            ]
            .drop_duplicates()
            .head(20)
            .tolist()
        )

        raise ValueError(
            "Grid GeoJSON contains duplicate grid_id values: "
            + ", ".join(
                duplicates
            )
        )

    if (
        grid["grid_lat"].isna().any()
        or grid["grid_lon"].isna().any()
    ):
        raise ValueError(
            "Grid GeoJSON contains missing grid_lat/grid_lon."
        )

    print(
        "GRID GEOJSON"
    )
    print(
        "------------"
    )
    print(
        f"Features: {len(grid):,}"
    )
    print(
        f"Unique grid_id: "
        f"{grid['grid_id'].nunique():,}"
    )

    if (
        EXPECTED_GRID_COUNT
        and len(grid)
        != EXPECTED_GRID_COUNT
    ):
        print(
            "WARNING: expected "
            f"{EXPECTED_GRID_COUNT:,} grid cells "
            f"but found {len(grid):,}."
        )

    return (
        geojson,
        grid,
    )


def read_cluster_source(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Cluster source not found: {path}"
        )

    suffix = (
        path.suffix
        .lower()
    )

    if suffix in {
        ".xlsx",
        ".xls",
    }:
        return pd.read_excel(
            path
        )

    if suffix == ".csv":
        return pd.read_csv(
            path
        )

    if suffix in {
        ".parquet",
        ".pq",
    }:
        return pd.read_parquet(
            path
        )

    raise ValueError(
        "Unsupported cluster source. "
        "Use .xlsx, .xls, .csv, or .parquet."
    )


def normalize_cluster_table(
    source: pd.DataFrame,
) -> pd.DataFrame:
    if source.empty:
        raise ValueError(
            "Cluster source is empty."
        )

    cluster_column = _find_column(
        source.columns,
        CLUSTER_ALIASES,
    )

    if cluster_column is None:
        raise ValueError(
            "Could not find cluster assignment column. "
            "Expected one of: "
            + ", ".join(
                CLUSTER_ALIASES
            )
        )

    grid_id_column = _find_column(
        source.columns,
        GRID_ID_ALIASES,
    )

    result = pd.DataFrame(
        index=source.index
    )

    if grid_id_column is not None:
        result["grid_id"] = (
            source[
                grid_id_column
            ]
            .astype(str)
            .str.strip()
        )

    else:
        lat_column = _find_column(
            source.columns,
            GRID_LAT_ALIASES,
        )

        lon_column = _find_column(
            source.columns,
            GRID_LON_ALIASES,
        )

        if (
            lat_column is None
            or lon_column is None
        ):
            raise ValueError(
                "Cluster source does not contain grid_id and "
                "grid_id cannot be derived because grid latitude/"
                "longitude columns were not found."
            )

        result["grid_id"] = (
            _grid_id_from_coordinates(
                source[
                    lat_column
                ],
                source[
                    lon_column
                ],
            )
        )

    result[
        "cluster_number"
    ] = pd.to_numeric(
        source[
            cluster_column
        ],
        errors="coerce",
    )

    if result[
        "cluster_number"
    ].isna().any():
        raise ValueError(
            "Cluster assignment contains missing or non-numeric values."
        )

    result[
        "cluster_number"
    ] = result[
        "cluster_number"
    ].astype(
        int
    )

    for canonical_name, aliases in (
        OPTIONAL_CLUSTER_COLUMNS.items()
    ):
        source_column = (
            _find_column(
                source.columns,
                aliases,
            )
        )

        if source_column is None:
            continue

        result[
            canonical_name
        ] = pd.to_numeric(
            source[
                source_column
            ],
            errors="coerce",
        )

    if result[
        "grid_id"
    ].duplicated().any():
        duplicates = (
            result.loc[
                result[
                    "grid_id"
                ].duplicated(
                    keep=False
                ),
                "grid_id",
            ]
            .drop_duplicates()
            .head(20)
            .tolist()
        )

        raise ValueError(
            "Cluster source contains duplicate grid_id values: "
            + ", ".join(
                duplicates
            )
        )

    observed_clusters = set(
        result[
            "cluster_number"
        ].unique()
    )

    invalid_clusters = (
        observed_clusters
        - EXPECTED_CLUSTERS
    )

    if invalid_clusters:
        raise ValueError(
            "Unexpected cluster values found: "
            + ", ".join(
                str(value)
                for value
                in sorted(
                    invalid_clusters
                )
            )
        )

    print()
    print(
        "CLUSTER SOURCE"
    )
    print(
        "--------------"
    )
    print(
        f"Rows: {len(result):,}"
    )
    print(
        f"Unique grid_id: "
        f"{result['grid_id'].nunique():,}"
    )
    print(
        "Cluster counts:"
    )
    print(
        result[
            "cluster_number"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    return (
        result
        .sort_values(
            "grid_id"
        )
        .reset_index(
            drop=True
        )
    )


def validate_grid_cluster_join(
    grid: pd.DataFrame,
    clusters: pd.DataFrame,
) -> pd.DataFrame:
    joined = grid.merge(
        clusters,
        on="grid_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    missing_cluster = (
        joined[
            "_merge"
        ]
        != "both"
    )

    extra_cluster_ids = (
        clusters.loc[
            ~clusters[
                "grid_id"
            ].isin(
                grid[
                    "grid_id"
                ]
            ),
            "grid_id",
        ]
        .tolist()
    )

    print()
    print(
        "GRID / CLUSTER JOIN AUDIT"
    )
    print(
        "-------------------------"
    )
    print(
        f"Grid cells: "
        f"{len(grid):,}"
    )
    print(
        f"Matched clusters: "
        f"{int((~missing_cluster).sum()):,}"
    )
    print(
        f"Grid cells missing cluster: "
        f"{int(missing_cluster.sum()):,}"
    )
    print(
        f"Cluster IDs absent from grid: "
        f"{len(extra_cluster_ids):,}"
    )

    if missing_cluster.any():
        missing_ids = (
            joined.loc[
                missing_cluster,
                "grid_id",
            ]
            .head(30)
            .tolist()
        )

        raise ValueError(
            "Not every 1-km grid cell has a cluster assignment. "
            "First missing grid IDs: "
            + ", ".join(
                missing_ids
            )
        )

    if extra_cluster_ids:
        raise ValueError(
            "Cluster source contains grid IDs that are not present "
            "in the authoritative GeoJSON. First extra IDs: "
            + ", ".join(
                extra_cluster_ids[
                    :30
                ]
            )
        )

    joined = (
        joined
        .drop(
            columns=[
                "_merge",
                "_feature_index",
            ],
            errors="ignore",
        )
        .sort_values(
            [
                "grid_i",
                "grid_j",
                "grid_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return joined


def build_enriched_geojson(
    original_geojson: dict,
    canonical: pd.DataFrame,
) -> dict:
    output = json.loads(
        json.dumps(
            original_geojson
        )
    )

    lookup = (
        canonical
        .set_index(
            "grid_id"
        )
        .to_dict(
            orient="index"
        )
    )

    for feature in output.get(
        "features",
        [],
    ):
        properties = (
            feature.setdefault(
                "properties",
                {},
            )
        )

        grid_id = str(
            properties.get(
                "grid_id"
            )
        ).strip()

        row = lookup.get(
            grid_id
        )

        if row is None:
            raise ValueError(
                "Could not enrich GeoJSON feature with "
                f"grid_id={grid_id!r}."
            )

        properties[
            "cluster_number"
        ] = int(
            row[
                "cluster_number"
            ]
        )

        for optional_column in (
            OPTIONAL_CLUSTER_COLUMNS
        ):
            if optional_column not in row:
                continue

            value = row[
                optional_column
            ]

            if pd.isna(
                value
            ):
                properties[
                    optional_column
                ] = None
            else:
                properties[
                    optional_column
                ] = float(
                    value
                )

    return output


def save_outputs(
    *,
    output_dir: Path,
    clusters: pd.DataFrame,
    canonical: pd.DataFrame,
    enriched_geojson: dict,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    grid_clusters_csv = (
        output_dir
        / GRID_CLUSTERS_CSV
    )

    grid_clusters_parquet = (
        output_dir
        / GRID_CLUSTERS_PARQUET
    )

    canonical_csv = (
        output_dir
        / CANONICAL_CSV
    )

    canonical_parquet = (
        output_dir
        / CANONICAL_PARQUET
    )

    canonical_geojson = (
        output_dir
        / CANONICAL_GEOJSON
    )

    clusters.to_csv(
        grid_clusters_csv,
        index=False,
    )

    clusters.to_parquet(
        grid_clusters_parquet,
        index=False,
    )

    canonical.to_csv(
        canonical_csv,
        index=False,
    )

    canonical.to_parquet(
        canonical_parquet,
        index=False,
    )

    with canonical_geojson.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            enriched_geojson,
            handle,
            ensure_ascii=False,
        )

    print()
    print(
        "SAVED SPATIAL REFERENCE"
    )
    print(
        "-----------------------"
    )
    print(
        grid_clusters_csv
    )
    print(
        grid_clusters_parquet
    )
    print(
        canonical_csv
    )
    print(
        canonical_parquet
    )
    print(
        canonical_geojson
    )


def build_reference(
    *,
    grid_path: Path,
    cluster_path: Path,
    output_dir: Path,
) -> pd.DataFrame:
    geojson, grid = (
        load_grid_geojson(
            grid_path
        )
    )

    source = (
        read_cluster_source(
            cluster_path
        )
    )

    clusters = (
        normalize_cluster_table(
            source
        )
    )

    canonical = (
        validate_grid_cluster_join(
            grid,
            clusters,
        )
    )

    enriched_geojson = (
        build_enriched_geojson(
            geojson,
            canonical,
        )
    )

    save_outputs(
        output_dir=output_dir,
        clusters=clusters,
        canonical=canonical,
        enriched_geojson=enriched_geojson,
    )

    print()
    print(
        "GRID CLUSTER REFERENCE COMPLETE"
    )
    print(
        "==============================="
    )
    print(
        f"Canonical grid cells: "
        f"{len(canonical):,}"
    )
    print(
        f"Clusters: "
        f"{sorted(canonical['cluster_number'].unique().tolist())}"
    )

    return canonical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the canonical NYC 1-km flood-grid + "
            "existing-cluster reference."
        )
    )

    parser.add_argument(
        "--grid",
        type=Path,
        default=DEFAULT_GRID_PATH,
        help=(
            "Path to the authoritative 1-km GeoJSON."
        ),
    )

    parser.add_argument(
        "--clusters",
        type=Path,
        default=DEFAULT_CLUSTER_PATH,
        help=(
            "Path to clustering output (.xlsx/.csv/.parquet)."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Directory for normalized cluster and canonical "
            "spatial reference artifacts."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(
        "BUILD NYC 1-KM GRID CLUSTER REFERENCE"
    )
    print(
        "====================================="
    )
    print(
        f"Grid:     {args.grid}"
    )
    print(
        f"Clusters: {args.clusters}"
    )
    print(
        f"Output:   {args.output_dir}"
    )
    print()
    print(
        "Existing cluster assignments will be preserved. "
        "KMeans will NOT be re-run."
    )
    print()

    build_reference(
        grid_path=args.grid,
        cluster_path=args.clusters,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()