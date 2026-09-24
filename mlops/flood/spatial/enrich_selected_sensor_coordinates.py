"""
Attach coordinates to the freshly selected eligible-sensor registry.

Coordinate provenance
---------------------

Priority 1:
    lifecycle-aware canonical FloodNet -> MRMS mapping

Priority 2:
    freshly trained GCN node table for historically valid sensors
    that are absent from the current lifecycle mapping

Inputs
------
artifacts/flood/model_selection/
    eligible_sensor_model_registry.parquet

S3 canonical sensor mapping
---------------------------
s3://<DATA_S3_BUCKET>/mlops/flood/reference/
    sensor_mrms_grid_map.parquet

Historical fallback
-------------------
artifacts/flood/gcn/
    node_table.parquet

Output
------
artifacts/flood/model_selection/
    eligible_sensor_model_registry_with_coordinates.parquet

This stage deliberately does not perform model selection or NYC
1-km grid assignment. It creates the stable boundary between the
fresh model-selection population and downstream spatial routing.

The canonical lifecycle mapping always has priority when a sensor
exists in both coordinate sources. The GCN node table is used only
for selected sensors that are historically valid but absent from
the current lifecycle mapping.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlops.flood.ingestion.sensor_grid_map import (
    SENSOR_ID,
    SENSOR_LAT,
    SENSOR_LON,
    load_existing_mapping,
)


# ============================================================
# PATHS
# ============================================================

ARTIFACT_ROOT = (
    Path("artifacts")
    / "flood"
)

MODEL_SELECTION_DIR = (
    ARTIFACT_ROOT
    / "model_selection"
)

GCN_DIR = (
    ARTIFACT_ROOT
    / "gcn"
)

INPUT_PATH = (
    MODEL_SELECTION_DIR
    / "eligible_sensor_model_registry.parquet"
)

OUTPUT_PATH = (
    MODEL_SELECTION_DIR
    / "eligible_sensor_model_registry_with_coordinates.parquet"
)

GCN_NODE_TABLE_PATH = (
    GCN_DIR
    / "node_table.parquet"
)


# ============================================================
# CONTRACT
# ============================================================

COORDINATE_SOURCE_COLUMN = (
    "coordinate_source"
)

CANONICAL_COORDINATE_SOURCE = (
    "canonical_sensor_mapping"
)

GCN_COORDINATE_SOURCE = (
    "gcn_node_table"
)


# ============================================================
# HELPERS
# ============================================================

def normalize_sensor_ids(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize deployment_id values without modifying
    the caller's DataFrame.
    """

    result = dataframe.copy()

    if SENSOR_ID not in result.columns:
        raise ValueError(
            f"DataFrame is missing required "
            f"sensor ID column {SENSOR_ID!r}."
        )

    result[SENSOR_ID] = (
        result[SENSOR_ID]
        .astype(str)
        .str.strip()
    )

    return result


def validate_unique_sensor_ids(
    dataframe: pd.DataFrame,
    label: str,
) -> None:
    """
    Require one row per deployment_id.
    """

    duplicate_mask = (
        dataframe[SENSOR_ID]
        .duplicated(
            keep=False
        )
    )

    if not duplicate_mask.any():
        return

    duplicates = (
        dataframe.loc[
            duplicate_mask,
            SENSOR_ID,
        ]
        .astype(str)
        .unique()
        .tolist()
    )

    raise ValueError(
        f"{label} contains duplicate "
        f"deployment IDs: "
        + ", ".join(
            sorted(
                duplicates
            )
        )
    )


def validate_coordinate_columns(
    dataframe: pd.DataFrame,
    label: str,
) -> None:
    """
    Require deployment_id, sensor_lat, and sensor_lon.
    """

    required = {
        SENSOR_ID,
        SENSOR_LAT,
        SENSOR_LON,
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise ValueError(
            f"{label} is missing required "
            "coordinate columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )


def clean_coordinate_table(
    dataframe: pd.DataFrame,
    coordinate_source: str,
) -> pd.DataFrame:
    """
    Normalize a coordinate source into one row per sensor.
    """

    validate_coordinate_columns(
        dataframe,
        coordinate_source,
    )

    result = (
        dataframe[
            [
                SENSOR_ID,
                SENSOR_LAT,
                SENSOR_LON,
            ]
        ]
        .copy()
    )

    result = normalize_sensor_ids(
        result
    )

    result[SENSOR_LAT] = (
        pd.to_numeric(
            result[SENSOR_LAT],
            errors="coerce",
        )
    )

    result[SENSOR_LON] = (
        pd.to_numeric(
            result[SENSOR_LON],
            errors="coerce",
        )
    )

    result = (
        result
        .dropna(
            subset=[
                SENSOR_ID,
                SENSOR_LAT,
                SENSOR_LON,
            ]
        )
        .drop_duplicates(
            subset=[
                SENSOR_ID
            ],
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    result[
        COORDINATE_SOURCE_COLUMN
    ] = coordinate_source

    return result


# ============================================================
# SELECTED SENSOR REGISTRY
# ============================================================

def load_selected_sensor_registry() -> pd.DataFrame:
    """
    Load the freshly generated eligible model-selection registry.
    """

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "Fresh eligible sensor registry "
            "not found: "
            f"{INPUT_PATH}"
        )

    selected = pd.read_parquet(
        INPUT_PATH
    )

    if selected.empty:
        raise RuntimeError(
            "Eligible sensor registry is empty."
        )

    selected = normalize_sensor_ids(
        selected
    )

    validate_unique_sensor_ids(
        selected,
        "Eligible sensor registry",
    )

    return selected


# ============================================================
# COORDINATE SOURCES
# ============================================================

def load_canonical_coordinates() -> pd.DataFrame:
    """
    Load the lifecycle-aware canonical FloodNet coordinate
    reference from S3.
    """

    print(
        "Loading canonical FloodNet sensor mapping "
        "from S3..."
    )

    mapping = load_existing_mapping()

    if mapping.empty:
        raise RuntimeError(
            "Canonical FloodNet sensor mapping is empty."
        )

    coordinates = clean_coordinate_table(
        mapping,
        CANONICAL_COORDINATE_SOURCE,
    )

    print(
        f"Canonical mapped sensors: "
        f"{len(coordinates):,}"
    )

    return coordinates


def load_gcn_node_coordinates() -> pd.DataFrame:
    """
    Load historically validated coordinates from the
    freshly trained GCN node table.

    This is a fallback source only.
    """

    if not GCN_NODE_TABLE_PATH.exists():
        print(
            "GCN node table not found; "
            "historical coordinate fallback unavailable."
        )

        return pd.DataFrame(
            columns=[
                SENSOR_ID,
                SENSOR_LAT,
                SENSOR_LON,
                COORDINATE_SOURCE_COLUMN,
            ]
        )

    node_table = pd.read_parquet(
        GCN_NODE_TABLE_PATH
    )

    if node_table.empty:
        print(
            "GCN node table is empty; "
            "historical coordinate fallback unavailable."
        )

        return pd.DataFrame(
            columns=[
                SENSOR_ID,
                SENSOR_LAT,
                SENSOR_LON,
                COORDINATE_SOURCE_COLUMN,
            ]
        )

    coordinates = clean_coordinate_table(
        node_table,
        GCN_COORDINATE_SOURCE,
    )

    print(
        f"GCN node-table sensors: "
        f"{len(coordinates):,}"
    )

    return coordinates


def build_coordinate_reference() -> pd.DataFrame:
    """
    Build the deployment coordinate reference.

    Priority:
        1. canonical lifecycle mapping
        2. GCN node table fallback

    The canonical mapping always wins when a sensor exists in
    both sources.
    """

    canonical = (
        load_canonical_coordinates()
    )

    gcn_fallback = (
        load_gcn_node_coordinates()
    )

    canonical_ids = set(
        canonical[SENSOR_ID]
        .astype(str)
    )

    if not gcn_fallback.empty:
        gcn_fallback = (
            gcn_fallback.loc[
                ~gcn_fallback[
                    SENSOR_ID
                ].isin(
                    canonical_ids
                )
            ]
            .copy()
        )

    coordinate_reference = pd.concat(
        [
            canonical,
            gcn_fallback,
        ],
        ignore_index=True,
    )

    coordinate_reference = (
        coordinate_reference
        .drop_duplicates(
            subset=[
                SENSOR_ID
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    validate_unique_sensor_ids(
        coordinate_reference,
        "Coordinate reference",
    )

    return coordinate_reference


# ============================================================
# CROSS-SOURCE COORDINATE AUDIT
# ============================================================

def audit_coordinate_consistency(
    canonical: pd.DataFrame,
    gcn_coordinates: pd.DataFrame,
) -> None:
    """
    Audit coordinate agreement for sensors represented
    in both sources.

    This does not block on tiny floating-point differences.
    Large differences indicate a data-governance problem.
    """

    if (
        canonical.empty
        or gcn_coordinates.empty
    ):
        return

    comparison = canonical[
        [
            SENSOR_ID,
            SENSOR_LAT,
            SENSOR_LON,
        ]
    ].merge(
        gcn_coordinates[
            [
                SENSOR_ID,
                SENSOR_LAT,
                SENSOR_LON,
            ]
        ],
        on=SENSOR_ID,
        how="inner",
        suffixes=(
            "_canonical",
            "_gcn",
        ),
    )

    if comparison.empty:
        return

    lat_difference = (
        comparison[
            f"{SENSOR_LAT}_canonical"
        ]
        - comparison[
            f"{SENSOR_LAT}_gcn"
        ]
    ).abs()

    lon_difference = (
        comparison[
            f"{SENSOR_LON}_canonical"
        ]
        - comparison[
            f"{SENSOR_LON}_gcn"
        ]
    ).abs()

    # Approximately 0.001 degree is around 100 m.
    tolerance = 0.001

    mismatch_mask = (
        (lat_difference > tolerance)
        | (lon_difference > tolerance)
    )

    mismatch_count = int(
        mismatch_mask.sum()
    )

    print()
    print(
        "CROSS-SOURCE COORDINATE AUDIT"
    )
    print(
        "-----------------------------"
    )

    print(
        f"Sensors present in both sources: "
        f"{len(comparison):,}"
    )

    print(
        f"Coordinate mismatches > "
        f"{tolerance} degrees: "
        f"{mismatch_count:,}"
    )

    if mismatch_count:
        mismatch_ids = (
            comparison.loc[
                mismatch_mask,
                SENSOR_ID,
            ]
            .astype(str)
            .tolist()
        )

        raise RuntimeError(
            "Canonical mapping and GCN node table "
            "contain materially different coordinates "
            "for sensors: "
            + ", ".join(
                mismatch_ids
            )
        )


# ============================================================
# BUILD
# ============================================================

def build_coordinate_enriched_registry() -> pd.DataFrame:
    """
    Attach deployment coordinates to every freshly selected
    eligible sensor.
    """

    print(
        "ENRICH ELIGIBLE SENSOR REGISTRY "
        "WITH COORDINATES"
    )

    print(
        "============================================"
        "===="
    )

    print()

    print(
        f"Selected-sensor registry: "
        f"{INPUT_PATH}"
    )

    selected = (
        load_selected_sensor_registry()
    )

    print(
        f"Eligible sensors: "
        f"{len(selected):,}"
    )

    print()

    canonical = (
        load_canonical_coordinates()
    )

    gcn_coordinates = (
        load_gcn_node_coordinates()
    )

    audit_coordinate_consistency(
        canonical,
        gcn_coordinates,
    )

    canonical_ids = set(
        canonical[SENSOR_ID]
        .astype(str)
    )

    historical_fallback = (
        gcn_coordinates.loc[
            ~gcn_coordinates[
                SENSOR_ID
            ].isin(
                canonical_ids
            )
        ]
        .copy()
    )

    coordinates = pd.concat(
        [
            canonical,
            historical_fallback,
        ],
        ignore_index=True,
    )

    coordinates = (
        coordinates
        .drop_duplicates(
            subset=[
                SENSOR_ID
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    print()

    print(
        f"Combined coordinate reference sensors: "
        f"{len(coordinates):,}"
    )

    print(
        f"Historical fallback sensors available: "
        f"{len(historical_fallback):,}"
    )

    enriched = selected.merge(
        coordinates,
        on=SENSOR_ID,
        how="left",
        validate="one_to_one",
    )

    missing_coordinate_mask = (
        enriched[
            [
                SENSOR_LAT,
                SENSOR_LON,
            ]
        ]
        .isna()
        .any(
            axis=1
        )
    )

    missing_count = int(
        missing_coordinate_mask.sum()
    )

    matched_count = (
        len(selected)
        - missing_count
    )

    print()

    print(
        "COORDINATE JOIN AUDIT"
    )

    print(
        "---------------------"
    )

    print(
        f"Selected sensors: "
        f"{len(selected):,}"
    )

    print(
        f"Coordinate matches: "
        f"{matched_count:,}"
    )

    print(
        f"Missing coordinates: "
        f"{missing_count:,}"
    )

    if missing_count:
        missing_ids = (
            enriched.loc[
                missing_coordinate_mask,
                SENSOR_ID,
            ]
            .astype(str)
            .tolist()
        )

        raise RuntimeError(
            "Selected eligible sensors are missing "
            "coordinates from both the canonical "
            "lifecycle mapping and the GCN node table: "
            + ", ".join(
                missing_ids
            )
        )

    if len(enriched) != len(selected):
        raise RuntimeError(
            "Coordinate enrichment changed the "
            "eligible-sensor row count."
        )

    print()

    print(
        "COORDINATE SOURCES"
    )

    print(
        "------------------"
    )

    source_counts = (
        enriched[
            COORDINATE_SOURCE_COLUMN
        ]
        .value_counts(
            dropna=False
        )
    )

    print(
        source_counts.to_string()
    )

    fallback_selected = (
        enriched.loc[
            enriched[
                COORDINATE_SOURCE_COLUMN
            ]
            == GCN_COORDINATE_SOURCE,
            SENSOR_ID,
        ]
        .astype(str)
        .tolist()
    )

    if fallback_selected:
        print()

        print(
            "Selected sensors using historical "
            "GCN coordinate fallback:"
        )

        for sensor_id in fallback_selected:
            print(
                f"    {sensor_id}"
            )

    return enriched


# ============================================================
# SAVE
# ============================================================

def save_registry(
    dataframe: pd.DataFrame,
) -> None:
    """
    Persist the coordinate-enriched selected-sensor registry.
    """

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print()

    print(
        "SAVED COORDINATE-ENRICHED REGISTRY"
    )

    print(
        "----------------------------------"
    )

    print(
        OUTPUT_PATH
    )

    print(
        f"Rows: {len(dataframe):,}"
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

def validate_saved_registry() -> None:
    """
    Re-open the persisted artifact and validate the deployment
    boundary before downstream spatial mapping is allowed.
    """

    if not OUTPUT_PATH.exists():
        raise RuntimeError(
            "Coordinate-enriched registry was not saved."
        )

    dataframe = pd.read_parquet(
        OUTPUT_PATH
    )

    if dataframe.empty:
        raise RuntimeError(
            "Saved coordinate-enriched registry is empty."
        )

    required = {
        SENSOR_ID,
        SENSOR_LAT,
        SENSOR_LON,
        COORDINATE_SOURCE_COLUMN,
        "selected_model",
        "selected_precision",
        "selected_recall",
        "selected_f1",
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Saved coordinate-enriched registry is "
            "missing required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    null_coordinate_mask = (
        dataframe[
            [
                SENSOR_LAT,
                SENSOR_LON,
            ]
        ]
        .isna()
        .any(
            axis=1
        )
    )

    if null_coordinate_mask.any():
        raise RuntimeError(
            "Saved coordinate-enriched registry "
            "contains null coordinates."
        )

    validate_unique_sensor_ids(
        dataframe,
        "Saved coordinate-enriched registry",
    )

    print()

    print(
        "SAVED REGISTRY VALIDATION"
    )

    print(
        "-------------------------"
    )

    print(
        f"Rows: "
        f"{len(dataframe):,}"
    )

    print(
        f"Unique sensors: "
        f"{dataframe[SENSOR_ID].nunique():,}"
    )

    print(
        "Null coordinates: 0"
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    enriched = (
        build_coordinate_enriched_registry()
    )

    save_registry(
        enriched
    )

    validate_saved_registry()

    print()

    print(
        "ELIGIBLE SENSOR COORDINATE "
        "ENRICHMENT COMPLETE"
    )


if __name__ == "__main__":
    main()