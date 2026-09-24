"""
MLflow observability wrapper for NYC flood spatial/imputation governance.

Purpose
-------
This module preserves the validated spatial/imputation pipeline in:

    mlops.flood.spatial.build_grid_imputation_reference

and adds MLflow tracking for the operational spatial support layer.

It records:

- total NYC 1-km grids;
- direct vs. imputed grids;
- operational support-sensor counts;
- support-sensor model routing;
- grid counts supported by GCN vs. Logistic;
- support distance statistics;
- support precision / recall / F1 statistics;
- imputation-method counts;
- same-cluster vs. cross-cluster support;
- missing assignments;
- duplicate grid IDs;
- model-assignment consistency;
- complete spatial artifacts for administrator audit.

Modes
-----

Log already-generated validated spatial artifacts without rebuilding them:

    python -m mlops.flood.spatial.spatial_imputation_mlflow --log-existing

Rebuild the validated spatial/imputation reference and then log it:

    python -m mlops.flood.spatial.spatial_imputation_mlflow

Scientific / serving contract
-----------------------------
This wrapper does NOT alter spatial routing logic.

The operational contract remains:

    model identity = support sensor
    rainfall       = target grid rainfall

Operational imputed-grid records are not training observations.
Retraining continues to use real physical sensor observations.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from mlops.flood.spatial import build_grid_imputation_reference as spatial_builder


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MLFLOW_EXPERIMENT = os.getenv(
    "FLOOD_SPATIAL_MLFLOW_EXPERIMENT",
    "nyc-resilience-flood-spatial-imputation",
)

MLFLOW_RUN_NAME = os.getenv(
    "FLOOD_SPATIAL_MLFLOW_RUN_NAME",
    "grid-spatial-imputation-governance",
)


# ============================================================
# PATHS
# ============================================================

SPATIAL_OUTPUT_DIR = Path(
    getattr(
        spatial_builder,
        "OUTPUT_DIR",
        Path("artifacts") / "flood" / "spatial",
    )
)

GRID_IMPUTATION_REFERENCE_PATH = Path(
    getattr(
        spatial_builder,
        "GRID_IMPUTATION_REFERENCE_PARQUET_PATH",
        getattr(
            spatial_builder,
            "GRID_IMPUTATION_REFERENCE_PATH",
            SPATIAL_OUTPUT_DIR / "grid_imputation_reference.parquet",
        ),
    )
)

GRID_IMPUTATION_REFERENCE_CSV_PATH = (
    SPATIAL_OUTPUT_DIR
    / "grid_imputation_reference.csv"
)

GRID_IMPUTATION_SUMMARY_PATH = (
    SPATIAL_OUTPUT_DIR
    / "grid_imputation_reference_summary.json"
)

GRID_PRIMARY_SENSOR_REGISTRY_PATH = (
    SPATIAL_OUTPUT_DIR
    / "grid_primary_sensor_registry.parquet"
)

CLUSTER_PRIMARY_SENSOR_REGISTRY_PATH = (
    SPATIAL_OUTPUT_DIR
    / "cluster_primary_sensor_registry.parquet"
)

ELIGIBLE_SENSOR_GRID_CLUSTER_MAP_PATH = (
    SPATIAL_OUTPUT_DIR
    / "eligible_sensor_grid_cluster_map.parquet"
)

MODEL_SELECTION_REGISTRY_PATH = Path(
    os.getenv(
        "FLOOD_ELIGIBLE_MODEL_REGISTRY",
        "artifacts/flood/model_selection/"
        "eligible_sensor_model_registry.csv",
    )
)

AUDIT_DIR = (
    SPATIAL_OUTPUT_DIR
    / "mlflow_audit"
)

SUPPORT_SENSOR_AUDIT_PATH = (
    AUDIT_DIR
    / "operational_support_sensor_audit.csv"
)

IMPUTED_GRID_AUDIT_PATH = (
    AUDIT_DIR
    / "imputed_grid_audit.csv"
)

DIRECT_GRID_AUDIT_PATH = (
    AUDIT_DIR
    / "direct_grid_audit.csv"
)

SPATIAL_GOVERNANCE_SUMMARY_PATH = (
    AUDIT_DIR
    / "spatial_governance_summary.json"
)


# ============================================================
# REQUIRED CONTRACT
# ============================================================

REQUIRED_COLUMNS = {
    "grid_id",
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


# ============================================================
# HELPERS
# ============================================================

def _git_value(
    *args: str,
) -> str:
    try:
        result = subprocess.run(
            [
                "git",
                *args,
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except Exception:
        return "unknown"


def _safe_float(
    value: Any,
) -> float | None:
    try:
        numeric = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(
        numeric
    ):
        return None

    return numeric


def _numeric_series(
    dataframe: pd.DataFrame,
    column: str,
) -> pd.Series:
    return pd.to_numeric(
        dataframe[
            column
        ],
        errors="coerce",
    )


def _log_metrics(
    metrics: dict[str, Any],
) -> None:
    clean: dict[str, float] = {}

    for name, value in metrics.items():
        numeric = _safe_float(
            value
        )

        if numeric is None:
            continue

        clean[
            name
        ] = numeric

    if clean:
        mlflow.log_metrics(
            clean
        )


def validate_existing_artifacts() -> None:
    required_paths = [
        GRID_IMPUTATION_REFERENCE_PATH,
        MODEL_SELECTION_REGISTRY_PATH,
    ]

    missing = [
        path
        for path in required_paths
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required spatial-governance artifact(s) are missing:\n"
            + "\n".join(
                f"  - {path}"
                for path in missing
            )
        )


# ============================================================
# LOADERS
# ============================================================

def load_spatial_reference() -> pd.DataFrame:
    dataframe = pd.read_parquet(
        GRID_IMPUTATION_REFERENCE_PATH
    )

    missing = (
        REQUIRED_COLUMNS
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise ValueError(
            "grid_imputation_reference.parquet is missing "
            "required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    dataframe = (
        dataframe
        .copy()
        .reset_index(
            drop=True
        )
    )

    dataframe[
        "grid_id"
    ] = (
        dataframe[
            "grid_id"
        ]
        .astype(str)
        .str.strip()
    )

    dataframe[
        "support_sensor_id"
    ] = (
        dataframe[
            "support_sensor_id"
        ]
        .astype(str)
        .str.strip()
    )

    dataframe[
        "support_sensor_model"
    ] = (
        dataframe[
            "support_sensor_model"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    dataframe[
        "imputation_required"
    ] = (
        dataframe[
            "imputation_required"
        ]
        .fillna(False)
        .astype(bool)
    )

    dataframe[
        "same_cluster_support"
    ] = (
        dataframe[
            "same_cluster_support"
        ]
        .fillna(False)
        .astype(bool)
    )

    return dataframe


def load_model_registry() -> pd.DataFrame:
    dataframe = pd.read_csv(
        MODEL_SELECTION_REGISTRY_PATH
    )

    required = {
        "deployment_id",
        "selected_model",
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise ValueError(
            "Eligible model registry is missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    dataframe = dataframe.copy()

    dataframe[
        "deployment_id"
    ] = (
        dataframe[
            "deployment_id"
        ]
        .astype(str)
        .str.strip()
    )

    dataframe[
        "selected_model"
    ] = (
        dataframe[
            "selected_model"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    return dataframe


# ============================================================
# VALIDATION
# ============================================================

def validate_spatial_contract(
    spatial: pd.DataFrame,
    registry: pd.DataFrame,
) -> dict[str, int]:
    duplicate_grid_count = int(
        spatial[
            "grid_id"
        ].duplicated().sum()
    )

    missing_support_sensor_count = int(
        (
            spatial[
                "support_sensor_id"
            ]
            .isna()
            | spatial[
                "support_sensor_id"
            ]
            .astype(str)
            .str.strip()
            .isin(
                [
                    "",
                    "nan",
                    "none",
                ]
            )
        ).sum()
    )

    missing_support_model_count = int(
        (
            spatial[
                "support_sensor_model"
            ]
            .isna()
            | spatial[
                "support_sensor_model"
            ]
            .astype(str)
            .str.strip()
            .isin(
                [
                    "",
                    "nan",
                    "none",
                ]
            )
        ).sum()
    )

    registry_lookup = (
        registry[
            [
                "deployment_id",
                "selected_model",
            ]
        ]
        .drop_duplicates(
            subset=[
                "deployment_id"
            ]
        )
        .set_index(
            "deployment_id"
        )[
            "selected_model"
        ]
        .to_dict()
    )

    spatial = spatial.copy()

    spatial[
        "registry_selected_model"
    ] = (
        spatial[
            "support_sensor_id"
        ].map(
            registry_lookup
        )
    )

    support_sensor_not_in_registry_count = int(
        spatial[
            "registry_selected_model"
        ].isna().sum()
    )

    model_assignment_mismatch_count = int(
        (
            spatial[
                "registry_selected_model"
            ].notna()
            & (
                spatial[
                    "registry_selected_model"
                ]
                != spatial[
                    "support_sensor_model"
                ]
            )
        ).sum()
    )

    if duplicate_grid_count:
        raise RuntimeError(
            "Spatial reference contains duplicate grid IDs: "
            f"{duplicate_grid_count}"
        )

    if missing_support_sensor_count:
        raise RuntimeError(
            "Spatial reference contains grids without support sensors: "
            f"{missing_support_sensor_count}"
        )

    if missing_support_model_count:
        raise RuntimeError(
            "Spatial reference contains grids without support models: "
            f"{missing_support_model_count}"
        )

    if support_sensor_not_in_registry_count:
        raise RuntimeError(
            "Spatial reference contains support sensors outside the "
            "eligible model registry: "
            f"{support_sensor_not_in_registry_count} grid row(s)"
        )

    if model_assignment_mismatch_count:
        raise RuntimeError(
            "Spatial reference model assignment disagrees with the "
            "eligible model registry for "
            f"{model_assignment_mismatch_count} grid row(s)"
        )

    return {
        "duplicate_grid_id_count":
            duplicate_grid_count,

        "missing_support_sensor_count":
            missing_support_sensor_count,

        "missing_support_model_count":
            missing_support_model_count,

        "support_sensor_not_in_registry_count":
            support_sensor_not_in_registry_count,

        "model_assignment_mismatch_count":
            model_assignment_mismatch_count,
    }


# ============================================================
# AUDIT TABLES
# ============================================================

def build_audit_artifacts(
    spatial: pd.DataFrame,
    registry: pd.DataFrame,
) -> None:
    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    support_columns = [
        "support_sensor_id",
        "support_sensor_model",
        "support_sensor_f1",
        "support_sensor_precision",
        "support_sensor_recall",
        "support_sensor_lat",
        "support_sensor_lon",
        "support_sensor_grid_id",
        "support_sensor_cluster_number",
    ]

    support_sensor_audit = (
        spatial[
            support_columns
        ]
        .drop_duplicates(
            subset=[
                "support_sensor_id"
            ]
        )
        .sort_values(
            "support_sensor_id"
        )
        .reset_index(
            drop=True
        )
    )

    registry_columns = [
        column
        for column in [
            "deployment_id",
            "selected_model",
            "selected_precision",
            "selected_recall",
            "selected_f1",
        ]
        if column in registry.columns
    ]

    registry_subset = (
        registry[
            registry_columns
        ]
        .drop_duplicates(
            subset=[
                "deployment_id"
            ]
        )
        .rename(
            columns={
                "deployment_id":
                    "support_sensor_id",
                "selected_model":
                    "registry_selected_model",
                "selected_precision":
                    "registry_selected_precision",
                "selected_recall":
                    "registry_selected_recall",
                "selected_f1":
                    "registry_selected_f1",
            }
        )
    )

    support_sensor_audit = (
        support_sensor_audit
        .merge(
            registry_subset,
            on="support_sensor_id",
            how="left",
            validate="one_to_one",
        )
    )

    support_sensor_audit[
        "model_assignment_matches_registry"
    ] = (
        support_sensor_audit[
            "support_sensor_model"
        ]
        == support_sensor_audit[
            "registry_selected_model"
        ]
    )

    support_sensor_audit.to_csv(
        SUPPORT_SENSOR_AUDIT_PATH,
        index=False,
    )

    preferred_grid_columns = [
        "grid_id",
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
    ]

    grid_audit = (
        spatial[
            preferred_grid_columns
        ]
        .copy()
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

    grid_audit.loc[
        grid_audit[
            "imputation_required"
        ]
    ].to_csv(
        IMPUTED_GRID_AUDIT_PATH,
        index=False,
    )

    grid_audit.loc[
        ~grid_audit[
            "imputation_required"
        ]
    ].to_csv(
        DIRECT_GRID_AUDIT_PATH,
        index=False,
    )


# ============================================================
# METRICS
# ============================================================

def build_spatial_metrics(
    spatial: pd.DataFrame,
    validation: dict[str, int],
) -> dict[str, float]:
    direct_mask = (
        ~spatial[
            "imputation_required"
        ]
    )

    imputed_mask = (
        spatial[
            "imputation_required"
        ]
    )

    support_sensors = (
        spatial[
            [
                "support_sensor_id",
                "support_sensor_model",
            ]
        ]
        .drop_duplicates(
            subset=[
                "support_sensor_id"
            ]
        )
    )

    support_model_counts = (
        support_sensors[
            "support_sensor_model"
        ]
        .value_counts()
    )

    grid_model_counts = (
        spatial[
            "support_sensor_model"
        ]
        .value_counts()
    )

    imputation_method_counts = (
        spatial.loc[
            imputed_mask,
            "imputation_method",
        ]
        .astype(str)
        .value_counts()
    )

    support_scope_counts = (
        spatial[
            "support_scope"
        ]
        .astype(str)
        .value_counts()
    )

    distance = _numeric_series(
        spatial,
        "support_sensor_distance_km",
    )

    support_precision = _numeric_series(
        spatial,
        "support_sensor_precision",
    )

    support_recall = _numeric_series(
        spatial,
        "support_sensor_recall",
    )

    support_f1 = _numeric_series(
        spatial,
        "support_sensor_f1",
    )

    metrics: dict[str, float] = {
        "operational_grid_count":
            float(
                spatial[
                    "grid_id"
                ].nunique()
            ),

        "spatial_reference_row_count":
            float(
                len(
                    spatial
                )
            ),

        "direct_grid_count":
            float(
                direct_mask.sum()
            ),

        "imputed_grid_count":
            float(
                imputed_mask.sum()
            ),

        "direct_grid_fraction":
            float(
                direct_mask.mean()
            ),

        "imputed_grid_fraction":
            float(
                imputed_mask.mean()
            ),

        "operational_support_sensor_count":
            float(
                support_sensors[
                    "support_sensor_id"
                ].nunique()
            ),

        "operational_support_gcn_count":
            float(
                support_model_counts.get(
                    "gcn",
                    0,
                )
            ),

        "operational_support_logistic_count":
            float(
                support_model_counts.get(
                    "logistic",
                    0,
                )
            ),

        "gcn_supported_grid_count":
            float(
                grid_model_counts.get(
                    "gcn",
                    0,
                )
            ),

        "logistic_supported_grid_count":
            float(
                grid_model_counts.get(
                    "logistic",
                    0,
                )
            ),

        "same_cluster_support_grid_count":
            float(
                spatial[
                    "same_cluster_support"
                ].sum()
            ),

        "cross_cluster_support_grid_count":
            float(
                (
                    ~spatial[
                        "same_cluster_support"
                    ]
                ).sum()
            ),

        "eligible_sensor_in_grid_count":
            float(
                spatial[
                    "has_eligible_sensor_in_grid"
                ]
                .fillna(False)
                .astype(bool)
                .sum()
            ),
    }

    for name, value in validation.items():
        metrics[
            name
        ] = float(
            value
        )

    if distance.notna().any():
        metrics.update(
            {
                "support_distance_km_mean":
                    float(
                        distance.mean()
                    ),

                "support_distance_km_median":
                    float(
                        distance.median()
                    ),

                "support_distance_km_max":
                    float(
                        distance.max()
                    ),

                "support_distance_km_p95":
                    float(
                        distance.quantile(
                            0.95
                        )
                    ),
            }
        )

    if support_precision.notna().any():
        metrics[
            "grid_weighted_support_precision_mean"
        ] = float(
            support_precision.mean()
        )

    if support_recall.notna().any():
        metrics[
            "grid_weighted_support_recall_mean"
        ] = float(
            support_recall.mean()
        )

    if support_f1.notna().any():
        metrics[
            "grid_weighted_support_f1_mean"
        ] = float(
            support_f1.mean()
        )

    # Sensor-weighted performance avoids repeated grids giving a support
    # sensor disproportionate weight merely because it serves more grids.
    sensor_performance = (
        spatial[
            [
                "support_sensor_id",
                "support_sensor_precision",
                "support_sensor_recall",
                "support_sensor_f1",
            ]
        ]
        .drop_duplicates(
            subset=[
                "support_sensor_id"
            ]
        )
    )

    sensor_precision = _numeric_series(
        sensor_performance,
        "support_sensor_precision",
    )

    sensor_recall = _numeric_series(
        sensor_performance,
        "support_sensor_recall",
    )

    sensor_f1 = _numeric_series(
        sensor_performance,
        "support_sensor_f1",
    )

    if sensor_precision.notna().any():
        metrics[
            "operational_support_sensor_precision_mean"
        ] = float(
            sensor_precision.mean()
        )

    if sensor_recall.notna().any():
        metrics[
            "operational_support_sensor_recall_mean"
        ] = float(
            sensor_recall.mean()
        )

    if sensor_f1.notna().any():
        metrics[
            "operational_support_sensor_f1_mean"
        ] = float(
            sensor_f1.mean()
        )

    # Stable metric names for imputation methods and support scopes.
    for method, count in imputation_method_counts.items():
        safe_name = (
            str(
                method
            )
            .strip()
            .lower()
            .replace(
                " ",
                "_",
            )
            .replace(
                "-",
                "_",
            )
        )

        metrics[
            f"imputation_method_{safe_name}_count"
        ] = float(
            count
        )

    for scope, count in support_scope_counts.items():
        safe_name = (
            str(
                scope
            )
            .strip()
            .lower()
            .replace(
                " ",
                "_",
            )
            .replace(
                "-",
                "_",
            )
        )

        metrics[
            f"support_scope_{safe_name}_grid_count"
        ] = float(
            count
        )

    return metrics


# ============================================================
# SUMMARY ARTIFACT
# ============================================================

def write_governance_summary(
    metrics: dict[str, float],
) -> None:
    AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    payload = {
        "contract": {
            "model_identity":
                "support sensor",
            "rainfall":
                "target grid rainfall",
            "training_usage":
                (
                    "operational imputed-grid records are not "
                    "synthetic retraining observations"
                ),
        },
        "spatial_reference":
            str(
                GRID_IMPUTATION_REFERENCE_PATH
            ),
        "eligible_model_registry":
            str(
                MODEL_SELECTION_REGISTRY_PATH
            ),
        "metrics":
            metrics,
    }

    with SPATIAL_GOVERNANCE_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            indent=2,
            sort_keys=True,
        )


# ============================================================
# MLFLOW METADATA
# ============================================================

def run_params() -> dict[str, str]:
    return {
        "spatial_builder_module":
            "mlops.flood.spatial.build_grid_imputation_reference",

        "spatial_reference_path":
            str(
                GRID_IMPUTATION_REFERENCE_PATH
            ),

        "eligible_model_registry_path":
            str(
                MODEL_SELECTION_REGISTRY_PATH
            ),

        "operational_prediction_grid":
            "NYC 1-km",

        "model_identity_scope":
            "support_sensor",

        "weather_source_scope":
            "target_grid",
    }


def run_tags(
    *,
    log_existing: bool,
) -> dict[str, str]:
    return {
        "pipeline":
            "nyc_resilience_flood",

        "stage":
            "spatial_imputation",

        "logging_mode":
            (
                "existing-artifacts"
                if log_existing
                else "rebuild-and-log"
            ),

        "scientific_spatial_logic_modified":
            "false",

        "training_contract":
            (
                "real_physical_sensor_observations_only"
            ),

        "git_commit":
            _git_value(
                "rev-parse",
                "HEAD",
            ),

        "git_branch":
            _git_value(
                "rev-parse",
                "--abbrev-ref",
                "HEAD",
            ),

        "spatial_builder_git_blob":
            _git_value(
                "hash-object",
                (
                    "mlops/flood/spatial/"
                    "build_grid_imputation_reference.py"
                ),
            ),
    }


# ============================================================
# ARTIFACT LOGGING
# ============================================================

def log_spatial_artifacts() -> None:
    if SPATIAL_OUTPUT_DIR.exists():
        mlflow.log_artifacts(
            str(
                SPATIAL_OUTPUT_DIR
            ),
            artifact_path="spatial",
        )


# ============================================================
# RUN VALIDATED BUILDER
# ============================================================

def run_validated_spatial_builder() -> None:
    if not hasattr(
        spatial_builder,
        "main",
    ):
        raise AttributeError(
            "build_grid_imputation_reference.py does not expose main(). "
            "Do not change its spatial methodology just for MLflow; "
            "update this thin wrapper to call its existing public "
            "execution function."
        )

    spatial_builder.main()


# ============================================================
# EXECUTE
# ============================================================

def execute(
    *,
    log_existing: bool,
) -> str:
    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    if log_existing:
        print(
            "Logging existing validated spatial/imputation artifacts; "
            "spatial routing will NOT rebuild."
        )

    else:
        print(
            "Running validated spatial/imputation builder..."
        )

        run_validated_spatial_builder()

    validate_existing_artifacts()

    spatial = (
        load_spatial_reference()
    )

    registry = (
        load_model_registry()
    )

    validation = (
        validate_spatial_contract(
            spatial,
            registry,
        )
    )

    build_audit_artifacts(
        spatial,
        registry,
    )

    metrics = (
        build_spatial_metrics(
            spatial,
            validation,
        )
    )

    write_governance_summary(
        metrics
    )

    with mlflow.start_run(
        run_name=MLFLOW_RUN_NAME
    ) as run:

        mlflow.log_params(
            run_params()
        )

        mlflow.set_tags(
            run_tags(
                log_existing=log_existing
            )
        )

        _log_metrics(
            metrics
        )

        log_spatial_artifacts()

        run_id = (
            run.info.run_id
        )

    print()
    print(
        "MLFLOW SPATIAL/IMPUTATION RUN COMPLETE"
    )
    print(
        "======================================"
    )
    print(
        f"Tracking URI: {MLFLOW_TRACKING_URI}"
    )
    print(
        f"Experiment: {MLFLOW_EXPERIMENT}"
    )
    print(
        f"Run ID: {run_id}"
    )

    print()
    print(
        "SPATIAL GOVERNANCE AUDIT"
    )
    print(
        "========================"
    )

    audit_names = [
        "operational_grid_count",
        "direct_grid_count",
        "imputed_grid_count",
        "operational_support_sensor_count",
        "operational_support_gcn_count",
        "operational_support_logistic_count",
        "gcn_supported_grid_count",
        "logistic_supported_grid_count",
        "same_cluster_support_grid_count",
        "cross_cluster_support_grid_count",
        "eligible_sensor_in_grid_count",
        "support_distance_km_mean",
        "support_distance_km_median",
        "support_distance_km_p95",
        "support_distance_km_max",
        "operational_support_sensor_precision_mean",
        "operational_support_sensor_recall_mean",
        "operational_support_sensor_f1_mean",
        "grid_weighted_support_precision_mean",
        "grid_weighted_support_recall_mean",
        "grid_weighted_support_f1_mean",
        "duplicate_grid_id_count",
        "missing_support_sensor_count",
        "missing_support_model_count",
        "support_sensor_not_in_registry_count",
        "model_assignment_mismatch_count",
    ]

    for name in audit_names:

        if name not in metrics:
            continue

        print(
            f"{name}: "
            f"{metrics[name]}"
        )

    print()
    print(
        "AUDIT ARTIFACTS"
    )
    print(
        "==============="
    )
    print(
        SUPPORT_SENSOR_AUDIT_PATH
    )
    print(
        DIRECT_GRID_AUDIT_PATH
    )
    print(
        IMPUTED_GRID_AUDIT_PATH
    )
    print(
        SPATIAL_GOVERNANCE_SUMMARY_PATH
    )

    return run_id


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or log the validated NYC flood spatial/imputation "
            "layer with MLflow governance."
        )
    )

    parser.add_argument(
        "--log-existing",
        action="store_true",
        help=(
            "Do not rebuild the spatial/imputation reference. "
            "Audit and log the existing validated artifacts."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    execute(
        log_existing=args.log_existing
    )


if __name__ == "__main__":
    main()
