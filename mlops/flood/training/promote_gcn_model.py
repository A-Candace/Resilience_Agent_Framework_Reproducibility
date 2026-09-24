"""
Promote a registered NYC flood GCN model version to a stable MLflow alias.

This is an explicit release step. It does NOT train a model, register a new
version, or modify canonical data.

Typical first promotion:

    python -m mlops.flood.training.promote_gcn_model \
        --version 1 \
        --alias champion \
        --approve

Dry-run validation only:

    python -m mlops.flood.training.promote_gcn_model \
        --version 1 \
        --alias champion

Optional numerical gates can be supplied explicitly, for example:

    --min-r2 -0.05
    --max-rmse 10
    --min-event-f1 0.20

No numerical quality threshold is invented by default. The script always
requires core provenance metadata and a successful registered-model reload
smoke test. The alias is changed only when --approve is supplied.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from mlops.flood.training import train_gcn as gcn


# ============================================================
# CONFIGURATION
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)

REGISTERED_MODEL_NAME = os.getenv(
    "FLOOD_GCN_REGISTERED_MODEL_NAME",
    "nyc-resilience-flood-gcn",
)

DEFAULT_ALIAS = os.getenv(
    "FLOOD_GCN_PRODUCTION_ALIAS",
    "champion",
)

PREDICTION_COLUMN = "predicted_minutes_above_1inch"

REQUIRED_VERSION_TAGS = {
    "pipeline": "flood-gcn",
    "model_family": "graph-convolutional-network",
    "predictor_family": "precipitation-only",
    "scientific_threshold": "1 inch / 25.4 mm",
    "training_target": gcn.TARGET_COLUMN,
    "canonical_source_target": gcn.SOURCE_TARGET_COLUMN,
    "source_run_id": None,
    "git_commit": None,
    "test_start": None,
    "test_end": None,
}


# ============================================================
# HELPERS
# ============================================================

def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def require_version_metadata(
    model_version,
) -> None:
    """
    Require the provenance and scientific-definition tags that were attached
    during registration.
    """

    tags = model_version.tags or {}

    problems: list[str] = []

    for key, expected_value in REQUIRED_VERSION_TAGS.items():

        value = tags.get(key)

        if value is None or str(value).strip() == "":
            problems.append(
                f"missing required tag: {key}"
            )
            continue

        if (
            expected_value is not None
            and str(value) != str(expected_value)
        ):
            problems.append(
                f"{key}={value!r}; expected {expected_value!r}"
            )

    if problems:
        raise ValueError(
            "Model version failed required metadata validation:\n"
            + "\n".join(
                f"    - {problem}"
                for problem in problems
            )
        )


def get_run_metrics(
    client: MlflowClient,
    model_version,
) -> dict[str, float]:
    """
    Retrieve the source training-run metrics used by optional quality gates.
    """

    run_id = model_version.run_id

    if not run_id:
        raise ValueError(
            "Registered model version has no source run_id."
        )

    run = client.get_run(
        run_id
    )

    metrics = {
        key: float(value)
        for key, value
        in run.data.metrics.items()
    }

    return metrics


def require_metric(
    metrics: dict[str, float],
    key: str,
) -> float:
    """
    Require a finite metric only when an explicit numerical gate needs it.
    """

    if key not in metrics:
        raise ValueError(
            f"Required MLflow run metric is not logged: {key}"
        )

    value = float(
        metrics[key]
    )

    if not np.isfinite(value):
        raise ValueError(
            f"Required MLflow run metric is not finite: {key}={value}"
        )

    return value


# ============================================================
# OPTIONAL QUALITY GATES
# ============================================================

def evaluate_quality_gates(
    metrics: dict[str, float],
    *,
    min_r2: float | None,
    max_rmse: float | None,
    max_mae: float | None,
    min_event_precision: float | None,
    min_event_recall: float | None,
    min_event_f1: float | None,
) -> list[str]:
    """
    Evaluate only thresholds that the operator explicitly supplied.

    The project has not established approved numerical promotion thresholds
    yet, so this script does not silently invent them.
    """

    failures: list[str] = []

    if min_r2 is not None:
        value = require_metric(
            metrics,
            "r2",
        )
        if value < min_r2:
            failures.append(
                f"r2={value:.6f} < required {min_r2:.6f}"
            )

    if max_rmse is not None:
        value = require_metric(
            metrics,
            "rmse_minutes",
        )
        if value > max_rmse:
            failures.append(
                f"rmse_minutes={value:.6f} > allowed {max_rmse:.6f}"
            )

    if max_mae is not None:
        value = require_metric(
            metrics,
            "mae_minutes",
        )
        if value > max_mae:
            failures.append(
                f"mae_minutes={value:.6f} > allowed {max_mae:.6f}"
            )

    if min_event_precision is not None:
        value = require_metric(
            metrics,
            "event_precision_micro",
        )
        if value < min_event_precision:
            failures.append(
                "event_precision_micro="
                f"{value:.6f} < required {min_event_precision:.6f}"
            )

    if min_event_recall is not None:
        value = require_metric(
            metrics,
            "event_recall_micro",
        )
        if value < min_event_recall:
            failures.append(
                "event_recall_micro="
                f"{value:.6f} < required {min_event_recall:.6f}"
            )

    if min_event_f1 is not None:
        value = require_metric(
            metrics,
            "event_f1_micro",
        )
        if value < min_event_f1:
            failures.append(
                f"event_f1_micro={value:.6f} < required {min_event_f1:.6f}"
            )

    return failures


# ============================================================
# REGISTERED-MODEL SMOKE TEST
# ============================================================

def build_smoke_test_input() -> pd.DataFrame:
    """
    Build a small valid known-sensor hourly snapshot from the saved node table.

    The values are deliberately synthetic. This checks serving integrity, not
    predictive accuracy.
    """

    node_table = pd.read_parquet(
        gcn.NODE_TABLE_PATH
    )

    deployment_ids = (
        node_table[
            gcn.SENSOR_ID
        ]
        .astype(str)
        .head(3)
        .tolist()
    )

    if not deployment_ids:
        raise ValueError(
            "Saved node table has no deployment IDs for smoke testing."
        )

    return pd.DataFrame(
        {
            gcn.SENSOR_ID: deployment_ids,
            "precip_current_hour_mm": (
                [0.0, 1.0, 5.0][
                    : len(deployment_ids)
                ]
            ),
            "precip_previous_6h_mm": (
                [0.0, 2.0, 10.0][
                    : len(deployment_ids)
                ]
            ),
            "daily_total_precip_mm": (
                [0.0, 3.0, 15.0][
                    : len(deployment_ids)
                ]
            ),
        }
    )


def smoke_test_model_version(
    version: str,
) -> pd.DataFrame:
    """
    Reload the exact registered version through MLflow and score a valid
    snapshot before promotion.
    """

    model_uri = (
        f"models:/{REGISTERED_MODEL_NAME}/{version}"
    )

    print(
        "\nREGISTERED VERSION SMOKE TEST"
    )
    print(
        "============================="
    )
    print(
        f"Loading: {model_uri}"
    )

    loaded = mlflow.pyfunc.load_model(
        model_uri
    )

    input_frame = (
        build_smoke_test_input()
    )

    output = loaded.predict(
        input_frame
    )

    if not isinstance(
        output,
        pd.DataFrame,
    ):
        raise RuntimeError(
            "Registered GCN did not return a pandas DataFrame."
        )

    required_columns = {
        gcn.SENSOR_ID,
        PREDICTION_COLUMN,
        "predicted_event",
    }

    missing = (
        required_columns
        - set(output.columns)
    )

    if missing:
        raise RuntimeError(
            "Registered GCN output is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    values = pd.to_numeric(
        output[
            PREDICTION_COLUMN
        ],
        errors="coerce",
    )

    if values.isna().any():
        raise RuntimeError(
            "Registered GCN produced NaN predictions."
        )

    if not np.isfinite(
        values.to_numpy()
    ).all():
        raise RuntimeError(
            "Registered GCN produced non-finite predictions."
        )

    if (
        (values < 0.0).any()
        or (values > 60.0).any()
    ):
        raise RuntimeError(
            "Registered GCN produced predictions outside 0-60 minutes."
        )

    print(
        output.to_string(
            index=False
        )
    )

    print(
        "\nREGISTERED VERSION SMOKE TEST PASSED"
    )

    return output


# ============================================================
# ALIAS / PROMOTION
# ============================================================

def current_alias_version(
    client: MlflowClient,
    alias: str,
):
    """
    Return the current aliased version, or None when the alias is not assigned.
    """

    try:
        return client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            alias,
        )
    except Exception:
        return None


def promote(
    *,
    version: str,
    alias: str,
    approve: bool,
    min_r2: float | None,
    max_rmse: float | None,
    max_mae: float | None,
    min_event_precision: float | None,
    min_event_recall: float | None,
    min_event_f1: float | None,
) -> None:

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    client = MlflowClient()

    candidate = (
        client.get_model_version(
            REGISTERED_MODEL_NAME,
            str(version),
        )
    )

    print(
        "FLOOD GCN PROMOTION CHECK"
    )
    print(
        "========================="
    )
    print(
        f"Tracking URI:     {MLFLOW_TRACKING_URI}"
    )
    print(
        f"Registered model: {REGISTERED_MODEL_NAME}"
    )
    print(
        f"Candidate version:{candidate.version}"
    )
    print(
        f"Source run:       {candidate.run_id}"
    )
    print(
        f"Target alias:     {alias}"
    )

    # --------------------------------------------------------
    # Scientific/provenance gate
    # --------------------------------------------------------

    require_version_metadata(
        candidate
    )

    print(
        "\nMetadata/provenance check: PASSED"
    )

    # --------------------------------------------------------
    # Optional numerical gates
    # --------------------------------------------------------

    metrics = get_run_metrics(
        client,
        candidate,
    )

    failures = evaluate_quality_gates(
        metrics,
        min_r2=min_r2,
        max_rmse=max_rmse,
        max_mae=max_mae,
        min_event_precision=min_event_precision,
        min_event_recall=min_event_recall,
        min_event_f1=min_event_f1,
    )

    requested_gate_count = sum(
        value is not None
        for value in [
            min_r2,
            max_rmse,
            max_mae,
            min_event_precision,
            min_event_recall,
            min_event_f1,
        ]
    )

    if requested_gate_count == 0:
        print(
            "Numerical quality gates: none configured "
            "(no project thresholds have been assumed)"
        )
    else:
        print(
            f"Numerical quality gates requested: {requested_gate_count}"
        )

    if failures:
        print(
            "\nPROMOTION BLOCKED"
        )
        print(
            "-----------------"
        )
        for failure in failures:
            print(
                f"  - {failure}"
            )
        raise SystemExit(2)

    if requested_gate_count:
        print(
            "Numerical quality gates: PASSED"
        )

    # --------------------------------------------------------
    # Serving/reload gate
    # --------------------------------------------------------

    smoke_test_model_version(
        str(candidate.version)
    )

    # --------------------------------------------------------
    # Show relevant metrics
    # --------------------------------------------------------

    print(
        "\nSOURCE RUN METRICS"
    )
    print(
        "------------------"
    )

    for key in [
        "r2",
        "rmse_minutes",
        "mae_minutes",
        "event_precision_micro",
        "event_recall_micro",
        "event_f1_micro",
        "event_precision_sensor_mean",
        "event_recall_sensor_mean",
        "event_f1_sensor_mean",
    ]:
        if key in metrics:
            print(
                f"{key}: {metrics[key]:.6f}"
            )

    existing = current_alias_version(
        client,
        alias,
    )

    if existing is None:
        print(
            f"\nCurrent @{alias}: not assigned"
        )
    else:
        print(
            f"\nCurrent @{alias}: version {existing.version}"
        )

    if not approve:
        print(
            "\nDRY RUN COMPLETE"
        )
        print(
            "================"
        )
        print(
            "All configured checks passed."
        )
        print(
            "No alias was changed because --approve was not supplied."
        )
        print(
            "\nTo promote this exact version, run again with:"
        )
        print(
            f"    --version {candidate.version} --alias {alias} --approve"
        )
        return

    # --------------------------------------------------------
    # Promotion mutation
    # --------------------------------------------------------

    promoted_at = utc_now_iso()

    if (
        existing is not None
        and str(existing.version)
        != str(candidate.version)
    ):
        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=str(existing.version),
            key="promotion_status",
            value="superseded",
        )

        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=str(existing.version),
            key="superseded_by_version",
            value=str(candidate.version),
        )

        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=str(existing.version),
            key="superseded_at",
            value=promoted_at,
        )

    client.set_registered_model_alias(
        REGISTERED_MODEL_NAME,
        alias,
        str(candidate.version),
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=str(candidate.version),
        key="promotion_status",
        value=alias,
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=str(candidate.version),
        key="promoted_at",
        value=promoted_at,
    )

    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=str(candidate.version),
        key="promotion_validation",
        value="metadata+registered_reload_smoke_test",
    )

    # Verify the alias after mutation.
    aliased = (
        client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            alias,
        )
    )

    if (
        str(aliased.version)
        != str(candidate.version)
    ):
        raise RuntimeError(
            "Alias verification failed after promotion."
        )

    print(
        "\nPROMOTION COMPLETE"
    )
    print(
        "=================="
    )
    print(
        f"{REGISTERED_MODEL_NAME}@{alias} -> version {candidate.version}"
    )
    print(
        f"Serving URI: models:/{REGISTERED_MODEL_NAME}@{alias}"
    )

    if existing is not None:
        if (
            str(existing.version)
            != str(candidate.version)
        ):
            print(
                f"Previous @{alias}: version {existing.version}"
            )
    else:
        print(
            "Previous alias: none"
        )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Validate and explicitly promote a registered NYC flood GCN model "
            "version to a stable MLflow alias."
        )
    )

    parser.add_argument(
        "--version",
        required=True,
        help="Registered model version to validate/promote.",
    )

    parser.add_argument(
        "--alias",
        default=DEFAULT_ALIAS,
        help=f"Stable MLflow model alias. Default: {DEFAULT_ALIAS}",
    )

    parser.add_argument(
        "--approve",
        action="store_true",
        help=(
            "Actually move the alias. Without this flag the command is a dry run."
        ),
    )

    # Optional project-defined numerical gates.
    parser.add_argument(
        "--min-r2",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-rmse",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--max-mae",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-event-precision",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-event-recall",
        type=float,
        default=None,
    )
    parser.add_argument(
        "--min-event-f1",
        type=float,
        default=None,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    promote(
        version=args.version,
        alias=args.alias,
        approve=args.approve,
        min_r2=args.min_r2,
        max_rmse=args.max_rmse,
        max_mae=args.max_mae,
        min_event_precision=args.min_event_precision,
        min_event_recall=args.min_event_recall,
        min_event_f1=args.min_event_f1,
    )


if __name__ == "__main__":
    main()