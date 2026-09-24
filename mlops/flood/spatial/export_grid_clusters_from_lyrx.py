"""
Export the existing ArcGIS 1-km clustering result into tabular files.

This script DOES NOT rerun clustering.

It reads the ArcGIS .lyrx file, resolves its underlying File Geodatabase
feature class, extracts the existing grid/cluster assignments, and saves:

    data/raw/geospatial/grid_clusters.csv
    data/raw/geospatial/grid_clusters.parquet

Run this script using the ArcGIS Pro Python environment because it requires
arcpy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_LYRX_PATH = (
    Path("data")
    / "raw"
    / "geospatial"
    / "infld_ftprn_SummarizeWithin.lyrx"
)

DEFAULT_OUTPUT_DIR = (
    Path("data")
    / "raw"
    / "geospatial"
)


# ------------------------------------------------------------
# Fields we want to preserve from the existing cluster product.
# ------------------------------------------------------------

DESIRED_FIELDS = [
    "grid_id",
    "grid_i",
    "grid_j",
    "grid_lat",
    "grid_lon",
    "mean_IFLD_RISKS",
    "sum_shape_area",
    "sum_Area_SQUAREFEET",
    "Polygon_Count",
    "clusters",
]


def load_layer_definition(
    lyrx_path: Path,
) -> dict:
    if not lyrx_path.exists():
        raise FileNotFoundError(
            f"Layer file not found: {lyrx_path}"
        )

    with lyrx_path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        document = json.load(
            handle
        )

    definitions = document.get(
        "layerDefinitions",
        []
    )

    if not definitions:
        raise ValueError(
            "No layerDefinitions were found in the .lyrx file."
        )

    layer = definitions[
        0
    ]

    feature_table = layer.get(
        "featureTable",
        {}
    )

    connection = feature_table.get(
        "dataConnection",
        {}
    )

    if not connection:
        raise ValueError(
            "The .lyrx does not contain a featureTable dataConnection."
        )

    return connection


def parse_database_path(
    workspace_connection_string: str,
) -> str:
    """
    ArcGIS stores the workspace as something like:

        DATABASE=.\\ArcGIS\\Projects\\CommunityBoards\\CommunityBoards.gdb

    Return only the path after DATABASE=.
    """

    prefix = "DATABASE="

    value = (
        workspace_connection_string
        .strip()
    )

    if not value.upper().startswith(
        prefix
    ):
        raise ValueError(
            "Unexpected workspaceConnectionString: "
            f"{workspace_connection_string}"
        )

    return value[
        len(
            prefix
        ):
    ]


def resolve_feature_class(
    lyrx_path: Path,
    connection: dict,
) -> Path:
    workspace_string = connection.get(
        "workspaceConnectionString"
    )

    dataset = connection.get(
        "dataset"
    )

    if not workspace_string:
        raise ValueError(
            "No workspaceConnectionString found in .lyrx."
        )

    if not dataset:
        raise ValueError(
            "No dataset name found in .lyrx."
        )

    database_string = (
        parse_database_path(
            workspace_string
        )
    )

    database_path = Path(
        database_string
    )

    # If ArcGIS stored an absolute path, use it directly.
    if database_path.is_absolute():
        geodatabase = (
            database_path
        )

    else:
        # First try relative to the layer file itself.
        candidate = (
            lyrx_path.parent
            / database_path
        ).resolve()

        if candidate.exists():
            geodatabase = candidate

        else:
            # Then try relative to the current working directory.
            candidate = (
                Path.cwd()
                / database_path
            ).resolve()

            geodatabase = candidate

    feature_class = (
        geodatabase
        / dataset
    )

    return feature_class


def export_cluster_table(
    *,
    feature_class: Path,
    output_dir: Path,
) -> pd.DataFrame:
    try:
        import arcpy
    except ImportError as exc:
        raise RuntimeError(
            "arcpy is required. Run this script from the "
            "ArcGIS Pro Python environment."
        ) from exc

    feature_class_string = str(
        feature_class
    )

    print(
        "ARCGIS CLUSTER EXPORT"
    )

    print(
        "====================="
    )

    print(
        "Feature class:"
    )

    print(
        feature_class_string
    )

    if not arcpy.Exists(
        feature_class_string
    ):
        raise FileNotFoundError(
            "ArcGIS could not find the feature class:\n"
            f"{feature_class_string}\n\n"
            "If the geodatabase has moved, supply the feature "
            "class explicitly using --feature-class."
        )

    available_fields = {
        field.name
        for field
        in arcpy.ListFields(
            feature_class_string
        )
    }

    print()

    print(
        "AVAILABLE RELEVANT FIELDS"
    )

    print(
        "-------------------------"
    )

    for field in DESIRED_FIELDS:
        print(
            f"{field}: "
            f"{field in available_fields}"
        )

    required = {
        "grid_id",
        "clusters",
    }

    missing_required = (
        required
        - available_fields
    )

    if missing_required:
        raise ValueError(
            "Underlying ArcGIS feature class is missing required fields: "
            + ", ".join(
                sorted(
                    missing_required
                )
            )
        )

    fields_to_read = [
        field
        for field
        in DESIRED_FIELDS
        if field
        in available_fields
    ]

    rows: list[
        dict
    ] = []

    with arcpy.da.SearchCursor(
        feature_class_string,
        fields_to_read,
    ) as cursor:

        for values in cursor:
            rows.append(
                dict(
                    zip(
                        fields_to_read,
                        values,
                    )
                )
            )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        raise RuntimeError(
            "No records were returned from the ArcGIS feature class."
        )

    # --------------------------------------------------------
    # Normalize names for the MLOps spatial contract.
    # --------------------------------------------------------

    rename_map = {
        "clusters":
            "cluster_number",

        "mean_IFLD_RISKS":
            "mean_ifld_risks",

        "sum_shape_area":
            "sum_shape_area",

        "sum_Area_SQUAREFEET":
            "summarized_area_squarefeet",

        "Polygon_Count":
            "polygon_count",
    }

    result = result.rename(
        columns=rename_map
    )

    result[
        "grid_id"
    ] = (
        result[
            "grid_id"
        ]
        .astype(str)
        .str.strip()
    )

    result[
        "cluster_number"
    ] = pd.to_numeric(
        result[
            "cluster_number"
        ],
        errors="coerce",
    )

    if result[
        "cluster_number"
    ].isna().any():

        missing_count = int(
            result[
                "cluster_number"
            ]
            .isna()
            .sum()
        )

        raise ValueError(
            f"{missing_count:,} rows have missing cluster assignments."
        )

    result[
        "cluster_number"
    ] = result[
        "cluster_number"
    ].astype(
        int
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
            .head(
                20
            )
            .tolist()
        )

        raise ValueError(
            "Duplicate grid_id values found: "
            + ", ".join(
                duplicates
            )
        )

    # --------------------------------------------------------
    # Actual cluster audit.
    # --------------------------------------------------------

    actual_clusters = sorted(
        result[
            "cluster_number"
        ]
        .unique()
        .tolist()
    )

    print()

    print(
        "CLUSTER AUDIT"
    )

    print(
        "-------------"
    )

    print(
        f"Rows: "
        f"{len(result):,}"
    )

    print(
        f"Unique grid IDs: "
        f"{result['grid_id'].nunique():,}"
    )

    print(
        f"Actual cluster numbers: "
        f"{actual_clusters}"
    )

    print()

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

    # --------------------------------------------------------
    # Save.
    # --------------------------------------------------------

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / "grid_clusters.csv"
    )

    parquet_path = (
        output_dir
        / "grid_clusters.parquet"
    )

    result.to_csv(
        csv_path,
        index=False,
    )

    result.to_parquet(
        parquet_path,
        index=False,
    )

    print()

    print(
        "SAVED"
    )

    print(
        "-----"
    )

    print(
        csv_path
    )

    print(
        parquet_path
    )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lyrx",
        type=Path,
        default=DEFAULT_LYRX_PATH,
    )

    parser.add_argument(
        "--feature-class",
        type=Path,
        default=None,
        help=(
            "Optional direct path to the ArcGIS feature class. "
            "Use this if the .lyrx's stored geodatabase path "
            "no longer matches your machine."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.feature_class is None:

        connection = (
            load_layer_definition(
                args.lyrx
            )
        )

        print(
            "LYRX DATA CONNECTION"
        )

        print(
            "===================="
        )

        print(
            "Workspace:"
        )

        print(
            connection.get(
                "workspaceConnectionString"
            )
        )

        print(
            "Dataset:"
        )

        print(
            connection.get(
                "dataset"
            )
        )

        print()

        feature_class = (
            resolve_feature_class(
                args.lyrx,
                connection,
            )
        )

    else:

        feature_class = (
            args.feature_class
        )

    export_cluster_table(
        feature_class=feature_class,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()