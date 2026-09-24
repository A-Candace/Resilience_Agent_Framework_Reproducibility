"""
Thin MLflow wrapper around the validated flood Logistic training pipeline.

This module does NOT change:
- canonical data
- dirty-day filtering
- chronological splitting
- scaling
- LogisticRegression configuration
- classification threshold
- event evaluation
- per-sensor evaluation

It only adds MLflow experiment tracking around the validated
train_logistic.py implementation.

Modes
-----

Log already-generated Logistic artifacts without retraining:

    python -m mlops.flood.training.train_logistic_mlflow --log-existing

Train Logistic and log the resulting run:

    python -m mlops.flood.training.train_logistic_mlflow
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from mlops.flood.training import train_logistic as logistic


# ============================================================
# MLFLOW CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

MLFLOW_EXPERIMENT = os.getenv(
    "FLOOD_LOGISTIC_MLFLOW_EXPERIMENT",
    "nyc-resilience-flood-logistic",
)

MLFLOW_RUN_NAME = os.getenv(
    "FLOOD_LOGISTIC_MLFLOW_RUN_NAME",
    "precipitation-only-logistic",
)

os.environ.setdefault(
    "MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING",
    "false",
)


# ============================================================
# GIT METADATA
# ============================================================

def git_value(
    *args: str,
) -> str:
    """Run a read-only Git command without making logging fail."""

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
        "mlops/flood/training/train_logistic.py",
    )


# ============================================================
# ARTIFACT VALIDATION
# ============================================================

EXPECTED_ARTIFACTS = [
    logistic.MODEL_PATH,
    logistic.SCALER_PATH,
    logistic.TEST_PREDICTIONS_PATH,
    logistic.VALIDATION_PREDICTIONS_PATH,
    logistic.OVERALL_METRICS_PATH,
    logistic.SENSOR_METRICS_PATH,
    logistic.METADATA_PATH,
]


def validate_existing_artifacts() -> None:
    """Require all validated Logistic outputs before MLflow logging."""

    missing = [
        Path(path)
        for path in EXPECTED_ARTIFACTS
        if not Path(path).exists()
    ]

    if missing:
        lines = "\n".join(
            f"    {path}"
            for path in missing
        )

        raise FileNotFoundError(
            "Required Logistic artifacts are missing:\n"
            f"{lines}\n\n"
            "Run the Logistic training pipeline first."
        )


# ============================================================
# LOAD SAVED METRICS
# ============================================================

def load_overall_metrics() -> dict[str, float]:
    """Load held-out Logistic test metrics."""

    dataframe = pd.read_csv(
        logistic.OVERALL_METRICS_PATH
    )

    required = {
        "metric",
        "value",
    }

    missing = (
        required
        - set(
            dataframe.columns
        )
    )

    if missing:
        raise ValueError(
            "Logistic metrics.csv is missing columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    result: dict[str, float] = {}

    for row in dataframe.itertuples(
        index=False
    ):
        try:
            result[
                str(
                    row.metric
                )
            ] = float(
                row.value
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

    return result


def load_sensor_summary() -> dict[str, float]:
    """Derive administrator-level summaries from per-sensor metrics."""

    dataframe = pd.read_csv(
        logistic.SENSOR_METRICS_PATH
    )

    if dataframe.empty:
        return {
            "sensor_count": 0.0,
        }

    summary: dict[str, float] = {
        "sensor_count": float(
            len(
                dataframe
            )
        ),
    }

    metric_columns = [
        "precision",
        "recall",
        "f1",
        "accuracy",
        "roc_auc",
        "average_precision",
        "actual_events",
        "predicted_events",
        "true_positive",
        "false_positive",
        "true_negative",
        "false_negative",
    ]

    for column in metric_columns:

        if column not in dataframe.columns:
            continue

        values = pd.to_numeric(
            dataframe[
                column
            ],
            errors="coerce",
        )

        finite = values.dropna()

        if finite.empty:
            continue

        summary[
            f"sensor_{column}_mean"
        ] = float(
            finite.mean()
        )

        summary[
            f"sensor_{column}_median"
        ] = float(
            finite.median()
        )

    if "precision" in dataframe.columns:

        precision = pd.to_numeric(
            dataframe[
                "precision"
            ],
            errors="coerce",
        )

        summary[
            "sensors_meeting_min_precision_0p25"
        ] = float(
            (
                precision >= 0.25
            ).sum()
        )

    if "recall" in dataframe.columns:

        recall = pd.to_numeric(
            dataframe[
                "recall"
            ],
            errors="coerce",
        )

        summary[
            "sensors_meeting_min_recall_0p30"
        ] = float(
            (
                recall >= 0.30
            ).sum()
        )

    if (
        "precision" in dataframe.columns
        and "recall" in dataframe.columns
    ):
        precision = pd.to_numeric(
            dataframe[
                "precision"
            ],
            errors="coerce",
        )

        recall = pd.to_numeric(
            dataframe[
                "recall"
            ],
            errors="coerce",
        )

        summary[
            "sensors_meeting_selection_eligibility"
        ] = float(
            (
                (precision >= 0.25)
                & (recall >= 0.30)
            ).sum()
        )

    return summary


# ============================================================
# TEST-DATA METADATA
# ============================================================

def load_test_metadata() -> dict[str, object]:
    """Record held-out Logistic test-period coverage."""

    dataframe = pd.read_parquet(
        logistic.TEST_PREDICTIONS_PATH,
        columns=[
            logistic.TIME_COLUMN,
            logistic.SENSOR_ID,
        ],
    )

    dataframe[
        logistic.TIME_COLUMN
    ] = pd.to_datetime(
        dataframe[
            logistic.TIME_COLUMN
        ],
        utc=True,
        errors="coerce",
    )

    return {
        "test_start": str(
            dataframe[
                logistic.TIME_COLUMN
            ].min()
        ),
        "test_end": str(
            dataframe[
                logistic.TIME_COLUMN
            ].max()
        ),
        "test_sensors": int(
            dataframe[
                logistic.SENSOR_ID
            ].nunique()
        ),
        "test_sensor_hours": int(
            len(
                dataframe
            )
        ),
    }


# ============================================================
# SAVED TRAINING METADATA
# ============================================================

def load_saved_metadata() -> dict:
    """Load metadata generated by train_logistic.py."""

    with Path(
        logistic.METADATA_PATH
    ).open(
        "r",
        encoding="utf-8",
    ) as handle:

        value = json.load(
            handle
        )

    if not isinstance(
        value,
        dict,
    ):
        raise ValueError(
            "Logistic metadata.json must contain an object."
        )

    return value


# ============================================================
# PARAMETERS AND TAGS
# ============================================================

def training_params() -> dict[str, object]:
    """Build immutable parameters describing the Logistic model."""

    metadata = load_saved_metadata()

    return {
        "features": ",".join(
            logistic.FEATURES
        ),
        "feature_count": len(
            logistic.FEATURES
        ),
        "continuous_target": (
            logistic.TARGET_COLUMN
        ),
        "binary_target": (
            "actual_event"
        ),
        "classification_threshold": (
            logistic.CLASSIFICATION_THRESHOLD
        ),
        "event_zero_tolerance_minutes": (
            logistic.EVENT_ZERO_TOLERANCE_MINUTES
        ),
        "max_iter": metadata.get(
            "max_iter",
            2000,
        ),
        "class_weight": metadata.get(
            "class_weight",
            "balanced",
        ),
        "train_fraction": (
            logistic.TRAIN_FRACTION
        ),
        "validation_fraction": (
            logistic.VALIDATION_FRACTION
        ),
        "test_fraction": (
            1.0
            - logistic.TRAIN_FRACTION
            - logistic.VALIDATION_FRACTION
        ),
    }


def run_tags(
    *,
    log_existing: bool,
) -> dict[str, str]:
    """Build reproducibility and governance metadata."""

    metadata = load_saved_metadata()

    return {
        "pipeline":
            "flood-logistic",

        "model_family":
            "logistic-regression",

        "predictor_family":
            "precipitation-only",

        "scientific_threshold":
            "1 inch / 25.4 mm",

        "canonical_history_policy":
            "immutable-historical-plus-append-only-forward",

        "git_commit":
            current_git_commit(),

        "git_branch":
            current_git_branch(),

        "train_logistic_git_blob":
            train_script_hash(),

        "logging_mode":
            (
                "existing-artifacts"
                if log_existing
                else "train-and-log"
            ),

        "train_end_hour":
            str(
                metadata.get(
                    "train_end_hour",
                    "unknown",
                )
            ),

        "validation_end_hour":
            str(
                metadata.get(
                    "validation_end_hour",
                    "unknown",
                )
            ),
    }


# ============================================================
# LOG SAVED RUN
# ============================================================

def log_saved_run() -> None:
    """Log metrics and artifacts generated by train_logistic.py."""

    validate_existing_artifacts()

    overall_metrics = (
        load_overall_metrics()
    )

    sensor_summary = (
        load_sensor_summary()
    )

    test_metadata = (
        load_test_metadata()
    )

    scalar_metrics = {
        **overall_metrics,
        **sensor_summary,
    }

    for name, value in (
        scalar_metrics.items()
    ):

        if value is None:
            continue

        try:
            numeric_value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            continue

        if not np.isfinite(
            numeric_value
        ):
            continue

        mlflow.log_metric(
            name,
            numeric_value,
        )

    mlflow.set_tags(
        {
            "test_start":
                test_metadata[
                    "test_start"
                ],

            "test_end":
                test_metadata[
                    "test_end"
                ],

            "test_sensors":
                str(
                    test_metadata[
                        "test_sensors"
                    ]
                ),

            "test_sensor_hours":
                str(
                    test_metadata[
                        "test_sensor_hours"
                    ]
                ),
        }
    )

    mlflow.log_artifacts(
        str(
            logistic.LOGISTIC_OUTPUT_DIR
        ),
        artifact_path="logistic",
    )


# ============================================================
# EXECUTE
# ============================================================

def execute(
    *,
    log_existing: bool,
) -> str:
    """Start one Logistic MLflow run."""

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    with mlflow.start_run(
        run_name=MLFLOW_RUN_NAME
    ) as run:

        if log_existing:

            print(
                "Logging existing validated Logistic artifacts; "
                "training will NOT run."
            )

        else:

            print(
                "Running validated Logistic training..."
            )

            logistic.main()

        # Parameters and tags are logged after training so metadata.json
        # definitely reflects the current run.
        mlflow.log_params(
            training_params()
        )

        mlflow.set_tags(
            run_tags(
                log_existing=log_existing
            )
        )

        log_saved_run()

        run_id = (
            run.info.run_id
        )

        print(
            "\nMLFLOW LOGISTIC RUN COMPLETE"
        )

        print(
            "============================"
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
            "Run or log the validated NYC flood "
            "Logistic model with MLflow."
        )
    )

    parser.add_argument(
        "--log-existing",
        action="store_true",
        help=(
            "Do not retrain. Log the existing validated artifacts "
            "under artifacts/flood/logistic."
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