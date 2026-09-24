"""
MLflow observability wrapper for NYC flood per-sensor model selection.

Purpose
-------
This module preserves the validated scientific model-selection pipeline in
`mlops.flood.training.select_sensor_models` and adds MLflow observability for:

1. per-sensor model-selection results;
2. eligible sensor/model registry;
3. matched-test evaluation metrics;
4. the authoritative operational spatial support population;
5. direct vs. imputed grid support;
6. eligible sensors that are not currently used by the spatial layer.

Important
---------
`--log-existing` DOES NOT rerun scientific model selection. It logs the
existing validated artifacts.

Without `--log-existing`, this wrapper reruns
`mlops.flood.training.select_sensor_models.main()` first and then logs the
newly generated artifacts.

This module does not change model-selection criteria, thresholds, metrics,
or spatial routing logic.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from mlops.flood.training import select_sensor_models as selection


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MLFLOW_EXPERIMENT = os.getenv(
    "FLOOD_MODEL_SELECTION_MLFLOW_EXPERIMENT",
    "nyc-resilience-flood-model-selection",
)

MLFLOW_RUN_NAME = os.getenv(
    "FLOOD_MODEL_SELECTION_MLFLOW_RUN_NAME",
    "per-sensor-model-selection",
)


# ============================================================
# MODEL-SELECTION ARTIFACT PATHS
# ============================================================

OUTPUT_DIR = Path(
    getattr(
        selection,
        "OUTPUT_DIR",
        Path("artifacts") / "flood" / "model_selection",
    )
)

SENSOR_SELECTION_PATH = Path(
    getattr(
        selection,
        "SENSOR_SELECTION_PATH",
        getattr(
            selection,
            "SENSOR_MODEL_SELECTION_PATH",
            OUTPUT_DIR / "sensor_model_selection.csv",
        ),
    )
)

ELIGIBLE_REGISTRY_PATH = Path(
    getattr(
        selection,
        "ELIGIBLE_SENSOR_REGISTRY_PATH",
        getattr(
            selection,
            "ELIGIBLE_SENSOR_MODEL_REGISTRY_PATH",
            OUTPUT_DIR / "eligible_sensor_model_registry.csv",
        ),
    )
)

SUMMARY_PATH = Path(
    getattr(
        selection,
        "SUMMARY_PATH",
        getattr(
            selection,
            "MODEL_SELECTION_SUMMARY_PATH",
            OUTPUT_DIR / "model_selection_summary.csv",
        ),
    )
)

MATCHED_TEST_PATH = Path(
    getattr(
        selection,
        "MATCHED_TEST_PATH",
        getattr(
            selection,
            "MATCHED_TEST_PREDICTIONS_PATH",
            OUTPUT_DIR / "matched_test_predictions.parquet",
        ),
    )
)


# ============================================================
# OPERATIONAL SPATIAL GOVERNANCE
# ============================================================

SPATIAL_IMPUTATION_REFERENCE_PATH = Path(
    os.getenv(
        "FLOOD_SPATIAL_IMPUTATION_REFERENCE",
        "artifacts/flood/spatial/grid_imputation_reference.parquet",
    )
)

MLFLOW_AUDIT_DIR = OUTPUT_DIR / "mlflow_audit"

OPERATIONAL_SUPPORT_REGISTRY_PATH = (
    MLFLOW_AUDIT_DIR
    / "operational_support_sensor_registry.csv"
)

ELIGIBLE_NOT_OPERATIONAL_PATH = (
    MLFLOW_AUDIT_DIR
    / "eligible_not_currently_operational.csv"
)

OPERATIONAL_GRID_AUDIT_PATH = (
    MLFLOW_AUDIT_DIR
    / "operational_grid_support_audit.csv"
)


# ============================================================
# HELPERS
# ============================================================

def _first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    for column in candidates:
        if column in dataframe.columns:
            return column
    return None


def _numeric_mean(
    dataframe: pd.DataFrame,
    candidates: list[str],
) -> float | None:
    column = _first_existing_column(
        dataframe,
        candidates,
    )

    if column is None:
        return None

    values = pd.to_numeric(
        dataframe[column],
        errors="coerce",
    )

    value = values.mean()

    if pd.isna(value):
        return None

    return float(value)


def _safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(result):
        return None

    return result


def log_numeric_metrics(
    metrics: dict[str, Any],
) -> None:
    clean: dict[str, float] = {}

    for name, value in metrics.items():
        numeric = _safe_float(value)

        if numeric is None:
            continue

        clean[name] = numeric

    if clean:
        mlflow.log_metrics(clean)


def validate_existing_artifacts() -> None:
    required = [
        SENSOR_SELECTION_PATH,
        ELIGIBLE_REGISTRY_PATH,
        SUMMARY_PATH,
        MATCHED_TEST_PATH,
        SPATIAL_IMPUTATION_REFERENCE_PATH,
    ]

    missing = [
        path
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required validated artifact(s) are missing:\n"
            + "\n".join(
                f"  - {path}"
                for path in missing
            )
        )


# ============================================================
# MODEL-SELECTION METRICS
# ============================================================

def load_selection_metrics() -> dict[str, float]:
    dataframe = pd.read_csv(
        SENSOR_SELECTION_PATH
    )

    if dataframe.empty:
        raise ValueError(
            "sensor_model_selection.csv is empty."
        )

    metrics: dict[str, float] = {
        "sensor_count": float(
            len(dataframe)
        ),
    }

    selected_model_column = (
        _first_existing_column(
            dataframe,
            [
                "selected_model",
                "support_sensor_model",
                "model",
            ],
        )
    )

    if selected_model_column is not None:
        selected_models = (
            dataframe[
                selected_model_column
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        metrics[
            "selected_gcn_sensor_count"
        ] = float(
            (
                selected_models
                == "gcn"
            ).sum()
        )

        metrics[
            "selected_logistic_sensor_count"
        ] = float(
            (
                selected_models
                == "logistic"
            ).sum()
        )

    for metric_name, candidates in {
        "selected_precision_sensor_mean": [
            "selected_precision",
            "precision",
        ],
        "selected_recall_sensor_mean": [
            "selected_recall",
            "recall",
        ],
        "selected_f1_sensor_mean": [
            "selected_f1",
            "f1",
            "f1_score",
        ],
    }.items():
        value = _numeric_mean(
            dataframe,
            candidates,
        )

        if value is not None:
            metrics[
                metric_name
            ] = value

    return metrics


def load_registry_metrics() -> dict[str, float]:
    dataframe = pd.read_csv(
        ELIGIBLE_REGISTRY_PATH
    )

    if dataframe.empty:
        raise ValueError(
            "eligible_sensor_model_registry.csv is empty."
        )

    selected_model_column = (
        _first_existing_column(
            dataframe,
            [
                "selected_model",
                "support_sensor_model",
                "model",
            ],
        )
    )

    metrics: dict[str, float] = {
        "eligible_registry_sensor_count":
            float(
                len(dataframe)
            ),
    }

    if selected_model_column is not None:
        selected_models = (
            dataframe[
                selected_model_column
            ]
            .astype(str)
            .str.strip()
            .str.lower()
        )

        gcn_count = float(
            (
                selected_models
                == "gcn"
            ).sum()
        )

        logistic_count = float(
            (
                selected_models
                == "logistic"
            ).sum()
        )

        metrics[
            "gcn_eligible_sensor_count"
        ] = gcn_count

        metrics[
            "logistic_eligible_sensor_count"
        ] = logistic_count

        metrics[
            "eligible_registry_gcn_selected"
        ] = gcn_count

        metrics[
            "eligible_registry_logistic_selected"
        ] = logistic_count

    return metrics


def load_summary_metrics() -> dict[str, float]:
    if not SUMMARY_PATH.exists():
        return {}

    dataframe = pd.read_csv(
        SUMMARY_PATH
    )

    if dataframe.empty:
        return {}

    metrics: dict[str, float] = {}

    # Preserve numeric summary values where the artifact is already in
    # metric/value form.
    if {
        "metric",
        "value",
    }.issubset(
        dataframe.columns
    ):
        for _, row in dataframe.iterrows():
            name = str(
                row[
                    "metric"
                ]
            ).strip()

            value = _safe_float(
                row[
                    "value"
                ]
            )

            if name and value is not None:
                metrics[
                    f"selection_summary_{name}"
                ] = value

    return metrics


# ============================================================
# MATCHED-TEST METRICS
# ============================================================

def load_matched_test_metrics() -> dict[str, float]:
    dataframe = pd.read_parquet(
        MATCHED_TEST_PATH
    )

    if dataframe.empty:
        return {
            "matched_test_row_count": 0.0,
        }

    metrics: dict[str, float] = {
        "matched_test_row_count":
            float(
                len(dataframe)
            ),
    }

    sensor_column = (
        _first_existing_column(
            dataframe,
            [
                "deployment_id",
                "sensor_id",
                "support_sensor_id",
            ],
        )
    )

    if sensor_column is not None:
        metrics[
            "matched_test_sensor_count"
        ] = float(
            dataframe[
                sensor_column
            ].astype(str).nunique()
        )

    # Log means for known metric columns if they are present.
    for metric_name, candidates in {
        "matched_test_precision_mean": [
            "selected_precision",
            "precision",
        ],
        "matched_test_recall_mean": [
            "selected_recall",
            "recall",
        ],
        "matched_test_f1_mean": [
            "selected_f1",
            "f1",
            "f1_score",
        ],
    }.items():
        value = _numeric_mean(
            dataframe,
            candidates,
        )

        if value is not None:
            metrics[
                metric_name
            ] = value

    return metrics


# ============================================================
# OPERATIONAL SPATIAL AUDIT
# ============================================================

def build_operational_spatial_audit() -> dict[str, float]:
    """
    Compare the model-eligible sensor registry with the authoritative
    NYC operational grid-support reference.

    This creates a transparent governance distinction between:

        1. sensors evaluated by the scientific model-selection pipeline;
        2. sensors eligible for a model;
        3. eligible sensors currently used as operational grid support;
        4. eligible sensors not currently needed by the spatial layer.

    No scientific model-selection output is changed.
    """

    if not ELIGIBLE_REGISTRY_PATH.exists():
        raise FileNotFoundError(
            "Eligible sensor registry is missing: "
            f"{ELIGIBLE_REGISTRY_PATH}"
        )

    if not SPATIAL_IMPUTATION_REFERENCE_PATH.exists():
        raise FileNotFoundError(
            "Operational spatial imputation reference is missing: "
            f"{SPATIAL_IMPUTATION_REFERENCE_PATH}"
        )

    eligible = pd.read_csv(
        ELIGIBLE_REGISTRY_PATH
    )

    spatial = pd.read_parquet(
        SPATIAL_IMPUTATION_REFERENCE_PATH
    )

    required_eligible = {
        "deployment_id",
        "selected_model",
        "selected_precision",
        "selected_recall",
        "selected_f1",
    }

    missing_eligible = (
        required_eligible
        - set(
            eligible.columns
        )
    )

    if missing_eligible:
        raise ValueError(
            "Eligible sensor registry is missing columns: "
            + ", ".join(
                sorted(
                    missing_eligible
                )
            )
        )

    required_spatial = {
        "grid_id",
        "support_sensor_id",
        "support_sensor_model",
        "imputation_required",
    }

    missing_spatial = (
        required_spatial
        - set(
            spatial.columns
        )
    )

    if missing_spatial:
        raise ValueError(
            "Grid imputation reference is missing columns: "
            + ", ".join(
                sorted(
                    missing_spatial
                )
            )
        )

    eligible = eligible.copy()
    spatial = spatial.copy()

    eligible[
        "deployment_id"
    ] = (
        eligible[
            "deployment_id"
        ]
        .astype(str)
        .str.strip()
    )

    eligible[
        "selected_model"
    ] = (
        eligible[
            "selected_model"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    spatial[
        "support_sensor_id"
    ] = (
        spatial[
            "support_sensor_id"
        ]
        .astype(str)
        .str.strip()
    )

    spatial[
        "support_sensor_model"
    ] = (
        spatial[
            "support_sensor_model"
        ]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Current operational support-sensor population
    # --------------------------------------------------------

    support_columns = [
        "support_sensor_id",
        "support_sensor_model",
    ]

    optional_support_columns = [
        "support_sensor_f1",
        "support_sensor_precision",
        "support_sensor_recall",
        "support_sensor_lat",
        "support_sensor_lon",
        "support_sensor_grid_id",
        "support_sensor_cluster_number",
    ]

    for column in optional_support_columns:
        if column in spatial.columns:
            support_columns.append(
                column
            )

    operational_support = (
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

    # --------------------------------------------------------
    # Assert production support sensors came from eligible registry
    # --------------------------------------------------------

    eligible_ids = set(
        eligible[
            "deployment_id"
        ]
    )

    operational_ids = set(
        operational_support[
            "support_sensor_id"
        ]
    )

    operational_not_eligible = (
        operational_ids
        - eligible_ids
    )

    if operational_not_eligible:
        raise RuntimeError(
            "Operational support sensors were found outside the "
            "eligible model registry: "
            + ", ".join(
                sorted(
                    operational_not_eligible
                )
            )
        )

    # --------------------------------------------------------
    # Assert model routing agrees between registry and spatial layer
    # --------------------------------------------------------

    registry_model_lookup = (
        eligible[
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

    operational_support[
        "registry_selected_model"
    ] = (
        operational_support[
            "support_sensor_id"
        ].map(
            registry_model_lookup
        )
    )

    operational_support[
        "model_assignment_matches_registry"
    ] = (
        operational_support[
            "support_sensor_model"
        ]
        == operational_support[
            "registry_selected_model"
        ]
    )

    model_mismatches = (
        operational_support.loc[
            ~operational_support[
                "model_assignment_matches_registry"
            ]
        ]
    )

    if not model_mismatches.empty:
        examples = (
            model_mismatches[
                [
                    "support_sensor_id",
                    "support_sensor_model",
                    "registry_selected_model",
                ]
            ]
            .head(10)
            .to_dict(
                orient="records"
            )
        )

        raise RuntimeError(
            "Operational support-sensor model assignments do not "
            "match the eligible registry. Examples: "
            f"{examples}"
        )

    # --------------------------------------------------------
    # Eligible sensors not currently used by grid support
    # --------------------------------------------------------

    unused_ids = (
        eligible_ids
        - operational_ids
    )

    eligible_not_operational = (
        eligible.loc[
            eligible[
                "deployment_id"
            ].isin(
                unused_ids
            )
        ]
        .copy()
        .sort_values(
            "deployment_id"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Complete grid-level audit
    # --------------------------------------------------------

    audit_columns = [
        "grid_id",
        "support_sensor_id",
        "support_sensor_model",
        "imputation_required",
    ]

    optional_grid_columns = [
        "grid_lat",
        "grid_lon",
        "cluster_number",
        "has_eligible_sensor_in_grid",
        "imputation_method",
        "support_scope",
        "support_sensor_f1",
        "support_sensor_precision",
        "support_sensor_recall",
        "support_sensor_distance_km",
        "same_cluster_support",
    ]

    for column in optional_grid_columns:
        if column in spatial.columns:
            audit_columns.append(
                column
            )

    operational_grid_audit = (
        spatial[
            audit_columns
        ]
        .copy()
        .sort_values(
            "grid_id"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Persist administrator/audit artifacts
    # --------------------------------------------------------

    MLFLOW_AUDIT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    operational_support.to_csv(
        OPERATIONAL_SUPPORT_REGISTRY_PATH,
        index=False,
    )

    eligible_not_operational.to_csv(
        ELIGIBLE_NOT_OPERATIONAL_PATH,
        index=False,
    )

    operational_grid_audit.to_csv(
        OPERATIONAL_GRID_AUDIT_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------

    operational_models = (
        operational_support[
            "support_sensor_model"
        ]
        .value_counts()
    )

    eligible_models = (
        eligible[
            "selected_model"
        ]
        .value_counts()
    )

    imputed_mask = (
        spatial[
            "imputation_required"
        ]
        .fillna(False)
        .astype(bool)
    )

    metrics = {
        "eligible_registry_sensor_count":
            float(
                len(
                    eligible
                )
            ),

        "eligible_registry_gcn_selected":
            float(
                eligible_models.get(
                    "gcn",
                    0,
                )
            ),

        "eligible_registry_logistic_selected":
            float(
                eligible_models.get(
                    "logistic",
                    0,
                )
            ),

        "operational_support_sensor_count":
            float(
                len(
                    operational_support
                )
            ),

        "operational_support_gcn_count":
            float(
                operational_models.get(
                    "gcn",
                    0,
                )
            ),

        "operational_support_logistic_count":
            float(
                operational_models.get(
                    "logistic",
                    0,
                )
            ),

        "eligible_not_currently_operational_count":
            float(
                len(
                    eligible_not_operational
                )
            ),

        "operational_grid_count":
            float(
                spatial[
                    "grid_id"
                ].nunique()
            ),

        "direct_grid_count":
            float(
                (
                    ~imputed_mask
                ).sum()
            ),

        "imputed_grid_count":
            float(
                imputed_mask.sum()
            ),

        "operational_model_assignment_mismatch_count":
            float(
                len(
                    model_mismatches
                )
            ),
    }

    model_grid_counts = (
        spatial[
            "support_sensor_model"
        ]
        .value_counts()
    )

    metrics[
        "gcn_supported_grid_count"
    ] = float(
        model_grid_counts.get(
            "gcn",
            0,
        )
    )

    metrics[
        "logistic_supported_grid_count"
    ] = float(
        model_grid_counts.get(
            "logistic",
            0,
        )
    )

    return metrics


# ============================================================
# MLFLOW METADATA
# ============================================================

def run_params() -> dict[str, str]:
    return {
        "selection_module":
            "mlops.flood.training.select_sensor_models",
        "selection_output_dir":
            str(
                OUTPUT_DIR
            ),
        "spatial_reference":
            str(
                SPATIAL_IMPUTATION_REFERENCE_PATH
            ),
        "governance_population":
            "authoritative_spatial_imputation_reference",
    }


def run_tags(
    *,
    log_existing: bool,
) -> dict[str, str]:
    return {
        "pipeline":
            "nyc_resilience_flood",
        "stage":
            "per_sensor_model_selection",
        "model_families":
            "logistic,gcn",
        "log_existing":
            str(
                log_existing
            ).lower(),
        "scientific_selection_modified":
            "false",
        "operational_spatial_audit":
            "true",
        "audit_contract":
            "evaluation->eligibility->operational_support->grid_support",
    }


# ============================================================
# ARTIFACT LOGGING
# ============================================================

def log_selection_artifacts() -> None:
    # Log the validated model-selection directory as generated by the
    # scientific pipeline. The audit directory created by this wrapper
    # is included as well.
    if OUTPUT_DIR.exists():
        mlflow.log_artifacts(
            str(
                OUTPUT_DIR
            ),
            artifact_path="model_selection",
        )


# ============================================================
# RUN
# ============================================================

def execute(
    *,
    log_existing: bool,
) -> str:
    """
    Run one model-selection MLflow audit.

    --log-existing:
        do not rerun model selection; log existing validated artifacts.

    default:
        rerun the validated selection pipeline first, then audit/log it.
    """

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    if log_existing:
        print(
            "Logging existing validated model-selection artifacts; "
            "selection will NOT rerun."
        )
    else:
        print(
            "Running validated per-sensor model selection..."
        )

        selection.main()

    validate_existing_artifacts()

    selection_metrics = (
        load_selection_metrics()
    )

    registry_metrics = (
        load_registry_metrics()
    )

    summary_metrics = (
        load_summary_metrics()
    )

    matched_metrics = (
        load_matched_test_metrics()
    )

    operational_metrics = (
        build_operational_spatial_audit()
    )

    all_metrics = {
        **selection_metrics,
        **registry_metrics,
        **summary_metrics,
        **matched_metrics,
        **operational_metrics,
    }

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

        log_numeric_metrics(
            all_metrics
        )

        log_selection_artifacts()

        run_id = (
            run.info.run_id
        )

    print()
    print(
        "MLFLOW MODEL-SELECTION RUN COMPLETE"
    )
    print(
        "==================================="
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
        "MODEL GOVERNANCE AUDIT"
    )
    print(
        "======================"
    )

    audit_names = [
        "sensor_count",
        "gcn_eligible_sensor_count",
        "logistic_eligible_sensor_count",
        "eligible_registry_sensor_count",
        "eligible_registry_gcn_selected",
        "eligible_registry_logistic_selected",
        "operational_support_sensor_count",
        "operational_support_gcn_count",
        "operational_support_logistic_count",
        "eligible_not_currently_operational_count",
        "operational_grid_count",
        "direct_grid_count",
        "imputed_grid_count",
        "gcn_supported_grid_count",
        "logistic_supported_grid_count",
        "operational_model_assignment_mismatch_count",
        "selected_precision_sensor_mean",
        "selected_recall_sensor_mean",
        "selected_f1_sensor_mean",
        "matched_test_row_count",
        "matched_test_sensor_count",
        "matched_test_precision_mean",
        "matched_test_recall_mean",
        "matched_test_f1_mean",
    ]

    for name in audit_names:

        if name not in all_metrics:
            continue

        print(
            f"{name}: "
            f"{all_metrics[name]}"
        )

    print()
    print(
        "AUDIT ARTIFACTS"
    )
    print(
        "==============="
    )
    print(
        OPERATIONAL_SUPPORT_REGISTRY_PATH
    )
    print(
        ELIGIBLE_NOT_OPERATIONAL_PATH
    )
    print(
        OPERATIONAL_GRID_AUDIT_PATH
    )

    return run_id


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run or log NYC flood per-sensor model selection "
            "with MLflow governance and operational spatial auditing."
        )
    )

    parser.add_argument(
        "--log-existing",
        action="store_true",
        help=(
            "Do not rerun model selection. Log and audit the "
            "existing validated model-selection and spatial artifacts."
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
