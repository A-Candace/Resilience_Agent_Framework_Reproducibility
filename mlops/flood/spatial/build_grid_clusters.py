"""
Build five spatial clusters for the authoritative NYC 1-km flood grid.

This module intentionally reruns the KMeans clustering from the raw grid
attribute workbook because the original ArcGIS feature class is unavailable.

Clustering features
-------------------
    Slope_mean
    Elevation_Mean
    Mean IFLD_RISKS
    Summarized Area in SQUAREFEET
    Count of Polygons

Excluded redundant feature
---------------------------
    Sum Shape_Area

Method
------
    1. Read the grid attribute workbook.
    2. Convert clustering fields to numeric.
    3. Fill missing values with the corresponding column mean.
    4. Standardize with StandardScaler.
    5. Run KMeans(n_clusters=5, random_state=42, n_init=10).
    6. Add Cluster_Number = zero-based label + 1.
    7. Save normalized cluster artifacts for downstream spatial joining.

The output grid_id values are intended to join directly to the authoritative
311_flooding_grid_1km.geojson file.

Run
---
python -m mlops.flood.spatial.build_grid_clusters
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT_PATH = (
    Path("data")
    / "raw"
    / "geospatial"
    / "Get_Clusters.xlsx"
)

DEFAULT_OUTPUT_DIR = (
    Path("data")
    / "raw"
    / "geospatial"
)

ARTIFACT_OUTPUT_DIR = (
    Path("artifacts")
    / "flood"
    / "spatial"
)

FEATURES = [
    "Slope_mean",
    "Elevation_Mean",
    "Mean IFLD_RISKS",
    "Summarized Area in SQUAREFEET",
    "Count of Polygons",
]

PRESERVED_COLUMNS = [
    "grid_id",
    "grid_i",
    "grid_j",
    "grid_lat",
    "grid_lon",
]

N_CLUSTERS = 5
RANDOM_STATE = 42
N_INIT = 10

NULL_STRINGS = {
    "",
    "<null>",
    "null",
    "none",
    "nan",
    "n/a",
    "#n/a",
}

_MAIN_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

_REL_NS = {
    "pkg": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def _column_index(cell_reference: str) -> int:
    match = re.match(r"([A-Z]+)", cell_reference.upper())
    if not match:
        raise ValueError(
            f"Invalid Excel cell reference: {cell_reference}"
        )

    letters = match.group(1)

    result = 0
    for char in letters:
        result = (
            result * 26
            + ord(char)
            - ord("A")
            + 1
        )

    return result - 1


def _read_shared_strings(
    archive: ZipFile,
) -> list[str]:
    try:
        raw = archive.read(
            "xl/sharedStrings.xml"
        )
    except KeyError:
        return []

    root = ET.fromstring(
        raw
    )

    strings = []

    for item in root.findall(
        "main:si",
        _MAIN_NS,
    ):
        parts = [
            node.text or ""
            for node in item.findall(
                ".//main:t",
                _MAIN_NS,
            )
        ]

        strings.append(
            "".join(
                parts
            )
        )

    return strings


def _first_worksheet_path(
    archive: ZipFile,
) -> str:
    workbook = ET.fromstring(
        archive.read(
            "xl/workbook.xml"
        )
    )

    first_sheet = workbook.find(
        "main:sheets/main:sheet",
        _MAIN_NS,
    )

    if first_sheet is None:
        raise ValueError(
            "Workbook contains no worksheets."
        )

    relationship_id = first_sheet.attrib.get(
        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    )

    if not relationship_id:
        raise ValueError(
            "Could not resolve first worksheet relationship."
        )

    relationships = ET.fromstring(
        archive.read(
            "xl/_rels/workbook.xml.rels"
        )
    )

    target = None

    for relation in relationships.findall(
        "pkg:Relationship",
        _REL_NS,
    ):
        if relation.attrib.get(
            "Id"
        ) == relationship_id:
            target = relation.attrib.get(
                "Target"
            )
            break

    if not target:
        raise ValueError(
            "Could not resolve first worksheet path."
        )

    if target.startswith(
        "/"
    ):
        return target.lstrip(
            "/"
        )

    return (
        Path("xl")
        / target
    ).as_posix()


def _cell_value(
    cell: ET.Element,
    shared_strings: list[str],
) -> object:
    cell_type = cell.attrib.get(
        "t"
    )

    if cell_type == "inlineStr":
        text_parts = [
            node.text or ""
            for node in cell.findall(
                ".//main:t",
                _MAIN_NS,
            )
        ]
        return "".join(
            text_parts
        )

    value_node = cell.find(
        "main:v",
        _MAIN_NS,
    )

    if value_node is None:
        return None

    raw = value_node.text

    if raw is None:
        return None

    if cell_type == "s":
        index = int(
            raw
        )
        return shared_strings[
            index
        ]

    if cell_type in {
        "str",
        "e",
    }:
        return raw

    if cell_type == "b":
        return raw == "1"

    try:
        numeric = float(
            raw
        )

        if numeric.is_integer():
            return int(
                numeric
            )

        return numeric

    except ValueError:
        return raw


def read_xlsx_first_sheet(
    path: Path,
) -> list[list[object]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Clustering workbook not found: {path}"
        )

    with ZipFile(
        path
    ) as archive:
        shared_strings = (
            _read_shared_strings(
                archive
            )
        )

        worksheet_path = (
            _first_worksheet_path(
                archive
            )
        )

        root = ET.fromstring(
            archive.read(
                worksheet_path
            )
        )

        rows: list[
            list[object]
        ] = []

        max_width = 0

        for row_node in root.findall(
            ".//main:sheetData/main:row",
            _MAIN_NS,
        ):
            values: dict[
                int,
                object,
            ] = {}

            for cell in row_node.findall(
                "main:c",
                _MAIN_NS,
            ):
                reference = cell.attrib.get(
                    "r"
                )

                if not reference:
                    continue

                column = (
                    _column_index(
                        reference
                    )
                )

                values[
                    column
                ] = _cell_value(
                    cell,
                    shared_strings,
                )

                max_width = max(
                    max_width,
                    column + 1,
                )

            output_row = [
                None
            ] * max_width

            for column, value in values.items():
                if column >= len(
                    output_row
                ):
                    output_row.extend(
                        [
                            None
                        ]
                        * (
                            column
                            - len(
                                output_row
                            )
                            + 1
                        )
                    )

                output_row[
                    column
                ] = value

            rows.append(
                output_row
            )

        for row in rows:
            if len(
                row
            ) < max_width:
                row.extend(
                    [
                        None
                    ]
                    * (
                        max_width
                        - len(
                            row
                        )
                    )
                )

        return rows


def numeric(
    value: object,
) -> float:
    if value is None:
        return math.nan

    if isinstance(
        value,
        str,
    ):
        stripped = (
            value
            .strip()
        )

        if (
            stripped.lower()
            in NULL_STRINGS
        ):
            return math.nan

        try:
            return float(
                stripped
            )

        except ValueError:
            return math.nan

    try:
        return float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return math.nan


def build_clusters(
    input_path: Path,
) -> tuple[
    list[dict],
    StandardScaler,
    KMeans,
    dict,
]:
    sheet = read_xlsx_first_sheet(
        input_path
    )

    if len(
        sheet
    ) < 2:
        raise ValueError(
            "Clustering workbook does not contain data rows."
        )

    headers = [
        str(
            value
        ).strip()
        if value is not None
        else ""
        for value in sheet[
            0
        ]
    ]

    column_index = {
        name: index
        for index, name
        in enumerate(
            headers
        )
    }

    required_columns = (
        PRESERVED_COLUMNS
        + FEATURES
    )

    missing = [
        column
        for column in required_columns
        if column not in column_index
    ]

    if missing:
        raise ValueError(
            "Clustering workbook is missing required columns: "
            + ", ".join(
                missing
            )
        )

    source_rows = sheet[
        1:
    ]

    feature_matrix = np.array(
        [
            [
                numeric(
                    row[
                        column_index[
                            feature
                        ]
                    ]
                )
                for feature in FEATURES
            ]
            for row in source_rows
        ],
        dtype=float,
    )

    missing_counts = {
        feature:
            int(
                np.isnan(
                    feature_matrix[
                        :,
                        index,
                    ]
                ).sum()
            )
        for index, feature
        in enumerate(
            FEATURES
        )
    }

    feature_means = np.nanmean(
        feature_matrix,
        axis=0,
    )

    if np.isnan(
        feature_means
    ).any():
        raise ValueError(
            "At least one clustering feature contains no valid values."
        )

    imputed = np.where(
        np.isnan(
            feature_matrix
        ),
        feature_means,
        feature_matrix,
    )

    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        imputed
    )

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        n_init=N_INIT,
    )

    labels = (
        kmeans.fit_predict(
            scaled
        )
        + 1
    )

    output_rows: list[
        dict
    ] = []

    for source_row, label in zip(
        source_rows,
        labels,
    ):
        output_rows.append(
            {
                "grid_id":
                    str(
                        source_row[
                            column_index[
                                "grid_id"
                            ]
                        ]
                    ).strip(),
                "grid_i":
                    int(
                        numeric(
                            source_row[
                                column_index[
                                    "grid_i"
                                ]
                            ]
                        )
                    ),
                "grid_j":
                    int(
                        numeric(
                            source_row[
                                column_index[
                                    "grid_j"
                                ]
                            ]
                        )
                    ),
                "grid_lat":
                    numeric(
                        source_row[
                            column_index[
                                "grid_lat"
                            ]
                        ]
                    ),
                "grid_lon":
                    numeric(
                        source_row[
                            column_index[
                                "grid_lon"
                            ]
                        ]
                    ),
                "Slope_mean":
                    numeric(
                        source_row[
                            column_index[
                                "Slope_mean"
                            ]
                        ]
                    ),
                "Elevation_Mean":
                    numeric(
                        source_row[
                            column_index[
                                "Elevation_Mean"
                            ]
                        ]
                    ),
                "Mean IFLD_RISKS":
                    numeric(
                        source_row[
                            column_index[
                                "Mean IFLD_RISKS"
                            ]
                        ]
                    ),
                "Summarized Area in SQUAREFEET":
                    numeric(
                        source_row[
                            column_index[
                                "Summarized Area in SQUAREFEET"
                            ]
                        ]
                    ),
                "Count of Polygons":
                    numeric(
                        source_row[
                            column_index[
                                "Count of Polygons"
                            ]
                        ]
                    ),
                "Cluster_Number":
                    int(
                        label
                    ),
            }
        )

    grid_ids = [
        row[
            "grid_id"
        ]
        for row in output_rows
    ]

    if len(
        set(
            grid_ids
        )
    ) != len(
        grid_ids
    ):
        raise ValueError(
            "Duplicate grid_id values were found in the clustering workbook."
        )

    unique_labels, label_counts = (
        np.unique(
            labels,
            return_counts=True,
        )
    )

    metadata = {
        "n_grid_cells":
            len(
                output_rows
            ),
        "n_clusters":
            N_CLUSTERS,
        "random_state":
            RANDOM_STATE,
        "n_init":
            N_INIT,
        "features":
            FEATURES,
        "excluded_redundant_feature":
            "Sum Shape_Area",
        "missing_value_strategy":
            "column mean before standardization",
        "missing_counts":
            missing_counts,
        "imputation_means":
            {
                feature:
                    float(
                        feature_means[
                            index
                        ]
                    )
                for index, feature
                in enumerate(
                    FEATURES
                )
            },
        "cluster_counts":
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
                in zip(
                    unique_labels,
                    label_counts,
                )
            },
    }

    return (
        output_rows,
        scaler,
        kmeans,
        metadata,
    )


def save_outputs(
    *,
    rows: list[dict],
    scaler: StandardScaler,
    kmeans: KMeans,
    metadata: dict,
    output_dir: Path,
    artifact_dir: Path,
) -> None:
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    artifact_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "grid_id",
        "grid_i",
        "grid_j",
        "grid_lat",
        "grid_lon",
        *FEATURES,
        "Cluster_Number",
    ]

    csv_path = (
        output_dir
        / "grid_clusters.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    parquet_path = (
        output_dir
        / "grid_clusters.parquet"
    )

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        arrays = {}

        for field in fieldnames:
            values = [
                row[
                    field
                ]
                for row in rows
            ]

            if field == "grid_id":
                arrays[
                    field
                ] = pa.array(
                    values,
                    type=pa.string(),
                )

            elif field in {
                "grid_i",
                "grid_j",
                "Cluster_Number",
            }:
                arrays[
                    field
                ] = pa.array(
                    values,
                    type=pa.int64(),
                )

            else:
                arrays[
                    field
                ] = pa.array(
                    values,
                    type=pa.float64(),
                )

        pq.write_table(
            pa.table(
                arrays
            ),
            parquet_path,
        )

    except ImportError:
        print(
            "WARNING: pyarrow is unavailable; "
            "grid_clusters.parquet was not written."
        )

    joblib.dump(
        scaler,
        artifact_dir
        / "cluster_feature_scaler.pkl",
    )

    joblib.dump(
        kmeans,
        artifact_dir
        / "grid_kmeans_5.pkl",
    )

    (
        artifact_dir
        / "clustering_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
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

    if parquet_path.exists():
        print(
            parquet_path
        )

    print(
        artifact_dir
        / "cluster_feature_scaler.pkl"
    )

    print(
        artifact_dir
        / "grid_kmeans_5.pkl"
    )

    print(
        artifact_dir
        / "clustering_metadata.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=ARTIFACT_OUTPUT_DIR,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(
        "BUILD NYC 1-KM GRID CLUSTERS"
    )
    print(
        "============================"
    )
    print(
        f"Input: {args.input}"
    )
    print()
    print(
        "Clustering features:"
    )

    for feature in FEATURES:
        print(
            f"    {feature}"
        )

    print()
    print(
        "Excluded redundant feature:"
    )
    print(
        "    Sum Shape_Area"
    )
    print()

    (
        rows,
        scaler,
        kmeans,
        metadata,
    ) = build_clusters(
        args.input
    )

    print(
        "MISSING VALUES IMPUTED"
    )
    print(
        "----------------------"
    )

    for feature, count in (
        metadata[
            "missing_counts"
        ].items()
    ):
        print(
            f"{feature}: {count}"
        )

    print()
    print(
        "CLUSTER DISTRIBUTION"
    )
    print(
        "--------------------"
    )

    cluster_counts = metadata[
        "cluster_counts"
    ]

    for cluster_number in sorted(
        cluster_counts,
        key=int,
    ):
        print(
            f"Cluster {cluster_number}: "
            f"{cluster_counts[cluster_number]} grids"
        )

    print(
        "--------------------"
    )
    print(
        f"Total: "
        f"{sum(cluster_counts.values())} grids"
    )

    save_outputs(
        rows=rows,
        scaler=scaler,
        kmeans=kmeans,
        metadata=metadata,
        output_dir=args.output_dir,
        artifact_dir=args.artifact_dir,
    )

    print()
    print(
        "GRID CLUSTERING COMPLETE"
    )


if __name__ == "__main__":
    main()