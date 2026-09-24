"""
Build and refresh the canonical FloodNet -> MRMS sensor grid mapping.

The canonical mapping is persistent and lifecycle-aware.

Behavior
--------
Existing deployment:
    Preserve its existing MRMS assignment.

New deployment:
    Assign it to the nearest canonical MRMS cell.

Decommissioned deployment:
    Preserve the mapping and retain date_down.

Deployment no longer returned by the current API:
    Preserve the existing mapping rather than deleting history.

This mapping is therefore the union of sensors encountered over time,
not merely a snapshot of today's active FloodNet network.
"""

from __future__ import annotations

import io
import os

import boto3
import pandas as pd

from mlops.flood.ingestion.floodnet import (
    get_sensor_inventory,
)

from mlops.flood.ingestion.sensor_grid_map import (
    SENSOR_ID,
    SENSOR_LAT,
    SENSOR_LON,
    MRMS_LAT,
    MRMS_LON,
    build_sensor_grid_mapping,
    load_existing_mapping,
    mapping_summary,
    save_mapping_to_s3,
    unique_required_mrms_cells,
)


# ============================================================
# CONFIGURATION
# ============================================================

AWS_REGION = os.getenv(
    "AWS_REGION",
    "us-east-1",
)

S3_BUCKET = os.getenv(
    "FLOOD_DATA_BUCKET",
    "nyc-resilience-data",
)

MRMS_GRID_KEY = os.getenv(
    "FLOOD_MRMS_GRID_KEY",
    "mlops/flood/reference/mrms_nyc_grid_points.csv",
)

EXPECTED_MRMS_GRID_POINTS = 3250


# ============================================================
# LIFECYCLE COLUMNS
# ============================================================

DATE_DEPLOYED = "date_deployed"

DATE_DOWN = "date_down"

IS_ACTIVE = "is_active"

IN_CURRENT_INVENTORY = "in_current_inventory"

LIFECYCLE_STATUS = "lifecycle_status"

MAPPING_UPDATED_AT = "mapping_updated_at"


# ============================================================
# LOAD CANONICAL MRMS GRID
# ============================================================

def load_mrms_grid_from_s3() -> pd.DataFrame:
    """
    Load the canonical 3,250-point NYC MRMS grid.
    """

    print(
        "Loading canonical MRMS grid geometry from S3..."
    )

    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION,
    )

    response = s3.get_object(
        Bucket=S3_BUCKET,
        Key=MRMS_GRID_KEY,
    )

    raw = response["Body"].read()

    grid = pd.read_csv(
        io.BytesIO(raw)
    )

    required = {
        MRMS_LAT,
        MRMS_LON,
    }

    missing = (
        required
        - set(grid.columns)
    )

    if missing:
        raise ValueError(
            "MRMS grid reference is missing required columns: "
            + ", ".join(sorted(missing))
        )

    grid = grid[
        [
            MRMS_LAT,
            MRMS_LON,
        ]
    ].copy()

    grid[MRMS_LAT] = pd.to_numeric(
        grid[MRMS_LAT],
        errors="coerce",
    )

    grid[MRMS_LON] = pd.to_numeric(
        grid[MRMS_LON],
        errors="coerce",
    )

    grid = (
        grid.dropna(
            subset=[
                MRMS_LAT,
                MRMS_LON,
            ]
        )
        .drop_duplicates()
        .sort_values(
            [
                MRMS_LAT,
                MRMS_LON,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    if grid.empty:
        raise RuntimeError(
            "Canonical MRMS grid loaded from S3 is empty."
        )

    return grid


# ============================================================
# VALIDATE MRMS GRID
# ============================================================

def validate_mrms_grid(
    grid: pd.DataFrame,
) -> None:
    """
    Validate the canonical NYC MRMS reference grid.
    """

    if len(grid) != EXPECTED_MRMS_GRID_POINTS:
        raise ValueError(
            "Unexpected MRMS grid size. "
            f"Expected {EXPECTED_MRMS_GRID_POINTS:,} "
            f"but found {len(grid):,}."
        )

    if grid[
        [
            MRMS_LAT,
            MRMS_LON,
        ]
    ].duplicated().any():

        raise ValueError(
            "Duplicate MRMS coordinates detected."
        )

    if not grid[MRMS_LAT].between(
        40.0,
        41.5,
    ).all():

        raise ValueError(
            "MRMS latitude values fall outside expected bounds."
        )

    if not grid[MRMS_LON].between(
        -75.0,
        -73.0,
    ).all():

        raise ValueError(
            "MRMS longitude values fall outside expected bounds."
        )


# ============================================================
# NORMALIZE CURRENT INVENTORY
# ============================================================

def normalize_inventory(
    sensors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize FloodNet lifecycle metadata.
    """

    sensors = sensors.copy()

    sensors[SENSOR_ID] = (
        sensors[SENSOR_ID]
        .astype(str)
    )

    for column in [
        DATE_DEPLOYED,
        DATE_DOWN,
    ]:

        if column not in sensors.columns:
            sensors[column] = pd.NaT

        sensors[column] = pd.to_datetime(
            sensors[column],
            utc=True,
            errors="coerce",
        )

    sensors[IN_CURRENT_INVENTORY] = True

    sensors[IS_ACTIVE] = (
        sensors[DATE_DOWN]
        .isna()
    )

    sensors[LIFECYCLE_STATUS] = (
        sensors[IS_ACTIVE]
        .map(
            {
                True: "active",
                False: "decommissioned",
            }
        )
    )

    return sensors


# ============================================================
# ENRICH CURRENT MAPPINGS
# ============================================================

def enrich_current_mapping(
    mapping: pd.DataFrame,
    sensors: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach current FloodNet metadata to mapped sensors.
    """

    metadata_columns = [
        SENSOR_ID,
        DATE_DEPLOYED,
        DATE_DOWN,
        IS_ACTIVE,
        IN_CURRENT_INVENTORY,
        LIFECYCLE_STATUS,
    ]

    for optional in [
        "name",
        "deploy_type",
    ]:

        if optional in sensors.columns:
            metadata_columns.append(
                optional
            )

    metadata = sensors[
        metadata_columns
    ].drop_duplicates(
        subset=[
            SENSOR_ID,
        ],
        keep="last",
    )

    result = mapping.merge(
        metadata,
        on=SENSOR_ID,
        how="left",
        validate="one_to_one",
    )

    return result


# ============================================================
# PRESERVE HISTORICAL-ONLY MAPPINGS
# ============================================================

def preserve_historical_mappings(
    current_mapping: pd.DataFrame,
    existing_mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Preserve mapping rows no longer present in today's inventory.

    We never delete an existing deployment solely because the
    current FloodNet API does not return it.
    """

    if existing_mapping.empty:
        return current_mapping

    existing = existing_mapping.copy()

    existing[SENSOR_ID] = (
        existing[SENSOR_ID]
        .astype(str)
    )

    current_ids = set(
        current_mapping[SENSOR_ID]
        .astype(str)
    )

    historical_only = existing[
        ~existing[SENSOR_ID].isin(
            current_ids
        )
    ].copy()

    if historical_only.empty:
        return current_mapping

    print(
        "Historical mappings absent from current inventory: "
        f"{len(historical_only):,}"
    )

    # --------------------------------------------------------
    # Lifecycle metadata for old mapping schemas
    # --------------------------------------------------------

    if DATE_DEPLOYED not in historical_only.columns:
        historical_only[DATE_DEPLOYED] = pd.NaT

    if DATE_DOWN not in historical_only.columns:
        historical_only[DATE_DOWN] = pd.NaT

    historical_only[DATE_DEPLOYED] = pd.to_datetime(
        historical_only[DATE_DEPLOYED],
        utc=True,
        errors="coerce",
    )

    historical_only[DATE_DOWN] = pd.to_datetime(
        historical_only[DATE_DOWN],
        utc=True,
        errors="coerce",
    )

    historical_only[
        IN_CURRENT_INVENTORY
    ] = False

    # A known date_down is an explicit decommissioning signal.
    # If the old row has no date_down and is absent from today's
    # inventory, we preserve it but do not pretend we know why.
    historical_only[IS_ACTIVE] = pd.Series(
        pd.NA,
        index=historical_only.index,
        dtype="boolean",
    )

    has_date_down = (
        historical_only[DATE_DOWN]
        .notna()
    )

    historical_only.loc[
        has_date_down,
        IS_ACTIVE,
    ] = False

    historical_only[
        LIFECYCLE_STATUS
    ] = "not_in_current_inventory"

    historical_only.loc[
        has_date_down,
        LIFECYCLE_STATUS,
    ] = "decommissioned"

    # --------------------------------------------------------
    # Align schemas before concatenation
    # --------------------------------------------------------

    all_columns = list(
        dict.fromkeys(
            list(current_mapping.columns)
            +
            list(historical_only.columns)
        )
    )

    current_aligned = (
        current_mapping.reindex(
            columns=all_columns
        )
    )

    historical_aligned = (
        historical_only.reindex(
            columns=all_columns
        )
    )

    result = pd.concat(
        [
            current_aligned,
            historical_aligned,
        ],
        ignore_index=True,
    )

    return result


# ============================================================
# FINALIZE CANONICAL MAPPING
# ============================================================

def finalize_mapping(
    mapping: pd.DataFrame,
) -> pd.DataFrame:
    """
    Final cleanup and lifecycle validation.
    """

    result = mapping.copy()

    result[SENSOR_ID] = (
        result[SENSOR_ID]
        .astype(str)
    )

    for column in [
        DATE_DEPLOYED,
        DATE_DOWN,
    ]:

        if column not in result.columns:
            result[column] = pd.NaT

        result[column] = pd.to_datetime(
            result[column],
            utc=True,
            errors="coerce",
        )

    if IN_CURRENT_INVENTORY not in result.columns:
        result[IN_CURRENT_INVENTORY] = False

    if IS_ACTIVE not in result.columns:
        result[IS_ACTIVE] = pd.NA

    if LIFECYCLE_STATUS not in result.columns:
        result[LIFECYCLE_STATUS] = "unknown"

    result[MAPPING_UPDATED_AT] = (
        pd.Timestamp.now(
            tz="UTC"
        )
    )

    duplicates = (
        result.duplicated(
            subset=[
                SENSOR_ID,
            ]
        )
        .sum()
    )

    if duplicates:
        raise ValueError(
            f"Canonical mapping contains "
            f"{duplicates:,} duplicate deployment IDs."
        )

    result = (
        result.sort_values(
            SENSOR_ID
        )
        .reset_index(
            drop=True
        )
    )

    return result


# ============================================================
# ACTIVE-AT-HOUR HELPER
# ============================================================

def active_at_hour(
    mapping: pd.DataFrame,
    hour,
) -> pd.DataFrame:
    """
    Return deployments active at a requested UTC feature hour.

    A deployment is active when:

        date_deployed <= hour

    and either:

        date_down is null

    or:

        hour < date_down

    Rows without a known deployment start are excluded rather than
    guessed.
    """

    timestamp = pd.to_datetime(
        hour,
        utc=True,
        errors="raise",
    )

    result = mapping.copy()

    result[DATE_DEPLOYED] = pd.to_datetime(
        result[DATE_DEPLOYED],
        utc=True,
        errors="coerce",
    )

    result[DATE_DOWN] = pd.to_datetime(
        result[DATE_DOWN],
        utc=True,
        errors="coerce",
    )

    active = (
        result[DATE_DEPLOYED].notna()
        &
        (
            result[DATE_DEPLOYED]
            <= timestamp
        )
        &
        (
            result[DATE_DOWN].isna()
            |
            (
                timestamp
                < result[DATE_DOWN]
            )
        )
    )

    return (
        result.loc[
            active
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )


# ============================================================
# PRINT SUMMARY
# ============================================================

def print_mapping_summary(
    mapping: pd.DataFrame,
) -> None:
    """
    Print canonical mapping/lifecycle statistics.
    """

    summary = mapping_summary(
        mapping
    )

    print(
        "\n=== FloodNet -> MRMS Mapping ==="
    )

    print(
        f"Canonical sensors: "
        f"{summary['sensors']:,}"
    )

    print(
        "Unique MRMS cells needed: "
        f"{summary['unique_mrms_cells']:,}"
    )

    print(
        "Reused existing assignments: "
        f"{summary['existing_assignments']:,}"
    )

    print(
        "New nearest-cell assignments: "
        f"{summary['new_assignments']:,}"
    )

    current_count = int(
        mapping[
            IN_CURRENT_INVENTORY
        ]
        .fillna(False)
        .sum()
    )

    active_count = int(
        (
            mapping[
                LIFECYCLE_STATUS
            ]
            == "active"
        ).sum()
    )

    decommissioned_count = int(
        (
            mapping[
                LIFECYCLE_STATUS
            ]
            == "decommissioned"
        ).sum()
    )

    historical_only_count = int(
        (
            mapping[
                LIFECYCLE_STATUS
            ]
            == "not_in_current_inventory"
        ).sum()
    )

    print(
        "\n=== Sensor Lifecycle ==="
    )

    print(
        f"In current inventory: "
        f"{current_count:,}"
    )

    print(
        f"Active: "
        f"{active_count:,}"
    )

    print(
        f"Decommissioned: "
        f"{decommissioned_count:,}"
    )

    print(
        "Historical/not in current inventory: "
        f"{historical_only_count:,}"
    )

    mean_distance = summary.get(
        "mean_distance_km"
    )

    max_distance = summary.get(
        "max_distance_km"
    )

    if mean_distance is not None:
        print(
            "Mean sensor -> MRMS distance: "
            f"{mean_distance:.3f} km"
        )

    if max_distance is not None:
        print(
            "Max sensor -> MRMS distance: "
            f"{max_distance:.3f} km"
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Refresh and persist the lifecycle-aware canonical mapping.
    """

    print(
        "Refreshing canonical FloodNet -> MRMS mapping..."
    )

    # --------------------------------------------------------
    # Current FloodNet inventory
    # --------------------------------------------------------

    print(
        "\nRetrieving current FloodNet inventory..."
    )

    sensors = normalize_inventory(
        get_sensor_inventory()
    )

    print(
        "Current inventory rows: "
        f"{len(sensors):,}"
    )

    print(
        "Currently active: "
        f"{sensors[IS_ACTIVE].sum():,}"
    )

    print(
        "Known decommissioned: "
        f"{(~sensors[IS_ACTIVE]).sum():,}"
    )

    # --------------------------------------------------------
    # MRMS grid
    # --------------------------------------------------------

    print(
        "\nLoading MRMS grid..."
    )

    grid_uri = (
        f"s3://{S3_BUCKET}/"
        f"{MRMS_GRID_KEY}"
    )

    print(
        grid_uri
    )

    mrms_grid = (
        load_mrms_grid_from_s3()
    )

    validate_mrms_grid(
        mrms_grid
    )

    print(
        "MRMS grid points: "
        f"{len(mrms_grid):,}"
    )

    # --------------------------------------------------------
    # Existing canonical mapping
    # --------------------------------------------------------

    print(
        "\nLoading existing canonical mapping..."
    )

    existing_mapping = (
        load_existing_mapping()
    )

    print(
        "Existing mapping rows: "
        f"{len(existing_mapping):,}"
    )

    # --------------------------------------------------------
    # Current assignments
    # --------------------------------------------------------

    current_mapping = (
        build_sensor_grid_mapping(
            sensors=sensors,
            mrms_grid=mrms_grid,
            existing_mapping=existing_mapping,
        )
    )

    current_mapping = enrich_current_mapping(
        current_mapping,
        sensors,
    )

    # --------------------------------------------------------
    # Preserve older deployments
    # --------------------------------------------------------

    canonical_mapping = (
        preserve_historical_mappings(
            current_mapping,
            existing_mapping,
        )
    )

    canonical_mapping = finalize_mapping(
        canonical_mapping
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    print_mapping_summary(
        canonical_mapping
    )

    required_cells = (
        unique_required_mrms_cells(
            canonical_mapping
        )
    )

    print(
        "\n=== MRMS Extraction Reduction ==="
    )

    print(
        f"Full MRMS NYC grid: "
        f"{len(mrms_grid):,}"
    )

    print(
        f"Canonical required cells: "
        f"{len(required_cells):,}"
    )

    reduction = (
        1.0
        -
        (
            len(required_cells)
            / len(mrms_grid)
        )
    ) * 100.0

    print(
        f"Grid reduction: "
        f"{reduction:.1f}%"
    )

    # --------------------------------------------------------
    # Example historical active set
    # --------------------------------------------------------

    example_hour = pd.Timestamp(
        "2026-03-13 07:00:00",
        tz="UTC",
    )

    example_active = active_at_hour(
        canonical_mapping,
        example_hour,
    )

    print(
        "\nActive deployments at "
        f"{example_hour}: "
        f"{len(example_active):,}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print(
        "\nSaving lifecycle-aware canonical mapping..."
    )

    uri = save_mapping_to_s3(
        canonical_mapping
    )

    print(
        "Saved:"
    )

    print(
        uri
    )

    print(
        "\nSample:"
    )

    sample_columns = [
        SENSOR_ID,
        SENSOR_LAT,
        SENSOR_LON,
        MRMS_LAT,
        MRMS_LON,
        DATE_DEPLOYED,
        DATE_DOWN,
        IS_ACTIVE,
        IN_CURRENT_INVENTORY,
        LIFECYCLE_STATUS,
        "mapping_source",
    ]

    sample_columns = [
        column
        for column in sample_columns
        if column in canonical_mapping.columns
    ]

    print(
        canonical_mapping[
            sample_columns
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nLIFECYCLE-AWARE SENSOR MAPPING COMPLETE"
    )


if __name__ == "__main__":
    main()