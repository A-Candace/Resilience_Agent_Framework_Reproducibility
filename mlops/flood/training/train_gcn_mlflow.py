"""
Thin MLflow wrapper around the validated flood GCN training pipeline.

This module does NOT change:
- canonical data
- dirty-day filtering
- graph construction
- chronological splitting
- scaling
- GCN architecture
- loss function
- training behavior
- event evaluation

It only adds experiment tracking around the validated train_gcn.py
implementation.

Modes
-----
Log the already-generated local artifacts without retraining:

    python -m mlops.flood.training.train_gcn_mlflow --log-existing

Run training and log the resulting run:

    python -m mlops.flood.training.train_gcn_mlflow
"""

from __future__ import annotations

import argparse
import os
import subprocess
from dataclasses import asdict
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from mlops.flood.training import train_gcn as gcn


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MLFLOW_EXPERIMENT = os.getenv(
    "FLOOD_MLFLOW_EXPERIMENT",
    "nyc-resilience-flood-gcn",
)

MLFLOW_RUN_NAME = os.getenv(
    "FLOOD_MLFLOW_RUN_NAME",
    "precipitation-only-gcn",
)

os.environ.setdefault(
    "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING",
    "false",
)


# ============================================================
# GIT METADATA
# ============================================================

def git_value(*args: str) -> str:
    """
    Run a small read-only git command.

    Returns 'unknown' rather than failing the training/logging run
    when Git metadata is unavailable.
    """

    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except Exception:
        return "unknown"


def current_git_commit() -> str:
    return git_value(
        "rev-parse",
        "HEAD",
    )


def current_git_branch() -> str:
    return git_value(
        "rev-parse",
        "--abbrev-ref",
        "HEAD",
    )


def train_script_hash() -> str:
    return git_value(
        "hash-object",
        "mlops/flood/training/train_gcn.py",
    )


# ============================================================
# ARTIFACT VALIDATION
# ============================================================

EXPECTED_ARTIFACTS = [
    gcn.MODEL_PATH,
    gcn.SCALERS_PATH,
    gcn.NODE_TABLE_PATH,
    gcn.DIRTY_DIAGNOSTICS_PATH,
    gcn.TEST_PREDICTIONS_PATH,
    gcn.METRICS_PATH,
    gcn.SENSOR_METRICS_PATH,
    gcn.EVENT_METRICS_PATH,
    gcn.TRAINING_HISTORY_PATH,
]


def validate_existing_artifacts() -> None:
    """
    Confirm that the validated training run produced all expected
    local artifacts before attempting MLflow logging.
    """

    missing = [
        path
        for path in EXPECTED_ARTIFACTS
        if not Path(path).exists()
    ]

    if missing:
        lines = "\n".join(
            f"    {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "Required GCN artifacts are missing:\n"
            f"{lines}\n\n"
            "Run the GCN training pipeline first."
        )


# ============================================================
# LOAD SAVED METRICS
# ============================================================

def load_pooled_metrics() -> dict[str, float]:
    """
    Load pooled test regression metrics.
    """

    df = pd.read_csv(
        gcn.METRICS_PATH
    )

    required = {
        "metric",
        "value",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "metrics.csv is missing columns: "
            + ", ".join(sorted(missing))
        )

    result = {}

    for row in df.itertuples(
        index=False
    ):
        result[
            str(row.metric)
        ] = float(row.value)

    return result


def load_training_metrics() -> tuple[
    pd.DataFrame,
    dict[str, float],
]:
    """
    Load per-epoch training history and derive run-level metrics.
    """

    history = pd.read_csv(
        gcn.TRAINING_HISTORY_PATH
    )

    required = {
        "epoch",
        "train_loss",
        "validation_loss",
    }

    missing = required - set(
        history.columns
    )

    if missing:
        raise ValueError(
            "training_history.csv is missing columns: "
            + ", ".join(sorted(missing))
        )

    history["epoch"] = pd.to_numeric(
        history["epoch"],
        errors="raise",
    ).astype(int)

    history["train_loss"] = pd.to_numeric(
        history["train_loss"],
        errors="raise",
    )

    history["validation_loss"] = pd.to_numeric(
        history["validation_loss"],
        errors="raise",
    )

    summary = {
        "epochs_completed": float(
            len(history)
        ),
        "best_validation_loss": float(
            history[
                "validation_loss"
            ].min()
        ),
        "final_train_loss": float(
            history[
                "train_loss"
            ].iloc[-1]
        ),
        "final_validation_loss": float(
            history[
                "validation_loss"
            ].iloc[-1]
        ),
    }

    return history, summary


def load_sensor_regression_metrics() -> dict[str, float]:
    """
    Derive aggregate diagnostics from per-sensor regression metrics.
    """

    df = pd.read_csv(
        gcn.SENSOR_METRICS_PATH
    )

    if df.empty:
        return {
            "sensor_r2_count": 0.0,
        }

    r2 = pd.to_numeric(
        df["r2"],
        errors="coerce",
    )

    rmse = pd.to_numeric(
        df["rmse_minutes"],
        errors="coerce",
    )

    mae = pd.to_numeric(
        df["mae_minutes"],
        errors="coerce",
    )

    return {
        "sensor_r2_count": float(
            r2.notna().sum()
        ),
        "sensor_mean_r2": float(
            r2.mean()
        ),
        "sensor_median_r2": float(
            r2.median()
        ),
        "sensors_r2_above_0p50": float(
            (r2 > 0.50).sum()
        ),
        "sensor_mean_rmse_minutes": float(
            rmse.mean()
        ),
        "sensor_mean_mae_minutes": float(
            mae.mean()
        ),
    }


def load_event_metrics() -> dict[str, float]:
    """
    Compute pooled/micro and sensor-mean/macro event metrics from
    the saved per-sensor ±24-hour event table.

    These calculations use the already-corrected 1-inch event
    evaluation generated by train_gcn.py.
    """

    df = pd.read_csv(
        gcn.EVENT_METRICS_PATH
    )

    required = {
        "actual_flood_events",
        "predicted_flood_events",
        "captured_actual_events",
        "missed_actual_events",
        "matched_predictions",
        "false_alarm_predictions",
        "precision",
        "recall",
        "f1",
    }

    missing = required - set(
        df.columns
    )

    if missing:
        raise ValueError(
            "sensor_24h_event_metrics.csv is missing columns: "
            + ", ".join(sorted(missing))
        )

    total_actual = int(
        pd.to_numeric(
            df["actual_flood_events"],
            errors="coerce",
        ).fillna(0).sum()
    )

    total_predicted = int(
        pd.to_numeric(
            df["predicted_flood_events"],
            errors="coerce",
        ).fillna(0).sum()
    )

    total_captured = int(
        pd.to_numeric(
            df["captured_actual_events"],
            errors="coerce",
        ).fillna(0).sum()
    )

    total_matched = int(
        pd.to_numeric(
            df["matched_predictions"],
            errors="coerce",
        ).fillna(0).sum()
    )

    total_false_alarms = int(
        pd.to_numeric(
            df["false_alarm_predictions"],
            errors="coerce",
        ).fillna(0).sum()
    )

    micro_precision = (
        total_matched
        / max(total_predicted, 1)
    )

    micro_recall = (
        total_captured
        / max(total_actual, 1)
    )

    if (
        micro_precision
        + micro_recall
        > 0
    ):
        micro_f1 = (
            2.0
            * micro_precision
            * micro_recall
            / (
                micro_precision
                + micro_recall
            )
        )
    else:
        micro_f1 = 0.0

    precision = pd.to_numeric(
        df["precision"],
        errors="coerce",
    )

    recall = pd.to_numeric(
        df["recall"],
        errors="coerce",
    )

    f1 = pd.to_numeric(
        df["f1"],
        errors="coerce",
    )

    return {
        "event_actual_total": float(
            total_actual
        ),
        "event_predicted_total": float(
            total_predicted
        ),
        "event_captured_total": float(
            total_captured
        ),
        "event_matched_predictions_total": float(
            total_matched
        ),
        "event_false_alarms_total": float(
            total_false_alarms
        ),

        # Pooled/micro metrics.
        "event_precision_micro": float(
            micro_precision
        ),
        "event_recall_micro": float(
            micro_recall
        ),
        "event_f1_micro": float(
            micro_f1
        ),

        # Mean of individual sensor metrics.
        "event_precision_sensor_mean": float(
            precision.mean()
        ),
        "event_recall_sensor_mean": float(
            recall.mean()
        ),
        "event_f1_sensor_mean": float(
            f1.mean()
        ),
    }


# ============================================================
# TEST-DATA METADATA
# ============================================================

def load_test_metadata() -> dict[str, object]:
    """
    Record the realized test prediction range and sensor coverage.
    """

    df = pd.read_parquet(
        gcn.TEST_PREDICTIONS_PATH,
        columns=[
            gcn.TIME_COLUMN,
            gcn.SENSOR_ID,
        ],
    )

    df[gcn.TIME_COLUMN] = pd.to_datetime(
        df[gcn.TIME_COLUMN],
        utc=True,
        errors="coerce",
    )

    return {
        "test_start": str(
            df[gcn.TIME_COLUMN].min()
        ),
        "test_end": str(
            df[gcn.TIME_COLUMN].max()
        ),
        "test_sensors": int(
            df[gcn.SENSOR_ID].nunique()
        ),
        "test_node_hours": int(
            len(df)
        ),
    }


# ============================================================
# MLFLOW PARAMS / TAGS
# ============================================================

def training_params() -> dict[str, object]:
    """
    Build immutable run parameters describing the validated GCN.
    """

    config = asdict(
        gcn.DEFAULT_GCN_CONFIG
    )

    params = {
        **config,

        "features": ",".join(
            gcn.FEATURES
        ),

        "feature_count": len(
            gcn.FEATURES
        ),

        "training_target": (
            gcn.TARGET_COLUMN
        ),

        "canonical_source_target": (
            gcn.SOURCE_TARGET_COLUMN
        ),

        "depth_threshold_mm": (
            gcn.DEPTH_THRESHOLD_MM
        ),

        "depth_threshold_inches": 1.0,

        "event_prediction_threshold_minutes": (
            gcn.PREDICTED_DURATION_THRESHOLD_MINUTES
        ),

        "event_match_window_hours": (
            gcn.MATCH_WINDOW_HOURS
        ),

        "event_zero_tolerance_minutes": (
            gcn.EVENT_ZERO_TOLERANCE_MINUTES
        ),

        "historical_cutoff": str(
            gcn.HISTORICAL_CUTOFF
        ),

        "forward_start": str(
            gcn.FORWARD_START
        ),

        "historical_canonical_prefix": (
            gcn.HISTORICAL_CANONICAL_PREFIX
        ),

        "forward_canonical_prefix": (
            gcn.FORWARD_CANONICAL_PREFIX
        ),

        "historical_partition_count": len(
            gcn.list_parquet_keys(
                gcn.HISTORICAL_CANONICAL_PREFIX
            )
        ),

        "forward_partition_count": len(
            gcn.list_parquet_keys(
                gcn.FORWARD_CANONICAL_PREFIX
            )
        ),
    }

    return params


def run_tags(
    *,
    log_existing: bool,
) -> dict[str, str]:
    """
    Metadata useful for reproducibility and model governance.
    """

    return {
        "pipeline": "flood-gcn",
        "model_family": "graph-convolutional-network",
        "predictor_family": "precipitation-only",
        "scientific_threshold": (
            "1 inch / 25.4 mm"
        ),
        "canonical_history_policy": (
            "immutable-historical-plus-append-only-forward"
        ),
        "git_commit": current_git_commit(),
        "git_branch": current_git_branch(),
        "train_gcn_git_blob": (
            train_script_hash()
        ),
        "logging_mode": (
            "existing-artifacts"
            if log_existing
            else "train-and-log"
        ),
    }


# ============================================================
# LOGGING
# ============================================================

def log_epoch_metrics(
    history: pd.DataFrame,
) -> None:
    """
    Preserve the full learning curve in MLflow.
    """

    for row in history.itertuples(
        index=False
    ):
        step = int(
            row.epoch
        )

        mlflow.log_metric(
            "train_loss",
            float(row.train_loss),
            step=step,
        )

        mlflow.log_metric(
            "validation_loss",
            float(
                row.validation_loss
            ),
            step=step,
        )


def log_saved_run() -> None:
    """
    Log all metrics and artifacts generated by train_gcn.py.
    """

    validate_existing_artifacts()

    pooled = load_pooled_metrics()

    (
        history,
        training_summary,
    ) = load_training_metrics()

    sensor_summary = (
        load_sensor_regression_metrics()
    )

    event_summary = (
        load_event_metrics()
    )

    test_metadata = (
        load_test_metadata()
    )

    # Epoch curves.
    log_epoch_metrics(
        history
    )

    # Final scalar metrics.
    scalar_metrics = {
        **pooled,
        **training_summary,
        **sensor_summary,
        **event_summary,
    }

    for name, value in scalar_metrics.items():

        if value is None:
            continue

        if not np.isfinite(
            float(value)
        ):
            continue

        mlflow.log_metric(
            name,
            float(value),
        )

    # Test-period realization goes in tags because timestamps are
    # strings and do not belong in MLflow numeric metrics.
    mlflow.set_tags(
        {
            "test_start": (
                test_metadata[
                    "test_start"
                ]
            ),
            "test_end": (
                test_metadata[
                    "test_end"
                ]
            ),
            "test_sensors": str(
                test_metadata[
                    "test_sensors"
                ]
            ),
            "test_node_hours": str(
                test_metadata[
                    "test_node_hours"
                ]
            ),
        }
    )

    # Log every local output exactly as generated by the validated
    # scientific pipeline.
    mlflow.log_artifacts(
        str(
            gcn.GCN_OUTPUT_DIR
        ),
        artifact_path="gcn",
    )


# ============================================================
# RUN
# ============================================================

def execute(
    *,
    log_existing: bool,
) -> str:
    """
    Start one MLflow run.

    When log_existing=True, no training is performed. This is the
    safest first smoke test because it logs the already-validated run.
    """

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    with mlflow.start_run(
        run_name=MLFLOW_RUN_NAME
    ) as run:

        mlflow.log_params(
            training_params()
        )

        mlflow.set_tags(
            run_tags(
                log_existing=log_existing
            )
        )

        if log_existing:

            print(
                "Logging existing validated GCN artifacts; "
                "training will NOT run."
            )

        else:

            print(
                "Running validated GCN training..."
            )

            gcn.main()

        log_saved_run()

        run_id = (
            run.info.run_id
        )

        print(
            "\nMLFLOW RUN COMPLETE"
        )

        print(
            "==================="
        )

        print(
            f"Tracking URI: "
            f"{MLFLOW_TRACKING_URI}"
        )

        print(
            f"Experiment: "
            f"{MLFLOW_EXPERIMENT}"
        )

        print(
            f"Run ID: "
            f"{run_id}"
        )

        return run_id


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run or log the validated NYC flood GCN with MLflow."
        )
    )

    parser.add_argument(
        "--log-existing",
        action="store_true",
        help=(
            "Do not retrain. Log the existing validated artifacts "
            "under artifacts/flood/gcn."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    execute(
        log_existing=args.log_existing
    )


if __name__ == "__main__":
    main()