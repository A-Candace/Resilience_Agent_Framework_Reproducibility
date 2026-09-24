"""
Weekly NYC flood retraining and governance orchestrator.

This module coordinates the validated flood MLOps components without
duplicating their scientific logic.

Production flow
---------------

1. Ingest append-only forward FloodNet observations.
2. Ingest append-only forward MRMS precipitation features.
3. Refresh append-only forward FloodNet response data.
4. Refresh append-only forward canonical training data.
5. Train + log GCN to MLflow.
6. Train + log Logistic regression to MLflow.
7. Re-run per-sensor model selection.
8. Refresh the lifecycle-aware FloodNet -> MRMS sensor mapping.
9. Enrich freshly selected sensors with deployment coordinates.
10. Re-map eligible sensors to the authoritative NYC 1-km grid.
11. Rebuild the grid imputation/support reference.
12. Log model-selection governance to MLflow.
13. Log spatial/imputation governance to MLflow.

A timestamped retraining manifest and individual stage logs are persisted
for every cycle.

Important contracts
-------------------

- Historical canonical monthly partitions remain immutable.
- New observations belong in append-only forward partitions.
- Training uses real physical sensor observations.
- Operational imputed-grid records are NEVER fed back into training.
- Model identity for an operational grid comes from its support sensor.
- Rainfall predictors remain local to the target grid.
- A failed stage stops all downstream stages.
- This runner does not silently promote a failed or partial refresh.

The scheduled workflow should invoke this module once per weekly cycle:

    python -m mlops.flood.pipeline.weekly_retraining

Use --dry-run to inspect the exact stage sequence without executing it.
"""

from __future__ import annotations

from uuid import uuid4
import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ============================================================
# PATHS / CONFIGURATION
# ============================================================

ARTIFACT_ROOT = Path(
    os.getenv(
        "FLOOD_ARTIFACT_ROOT",
        "artifacts/flood",
    )
)

RETRAINING_ROOT = Path(
    os.getenv(
        "FLOOD_RETRAINING_ROOT",
        str(
            ARTIFACT_ROOT
            / "retraining"
        ),
    )
)

MODEL_SELECTION_DIR = (
    ARTIFACT_ROOT
    / "model_selection"
)

SPATIAL_DIR = (
    ARTIFACT_ROOT
    / "spatial"
)

GCN_DIR = (
    ARTIFACT_ROOT
    / "gcn"
)

LOGISTIC_DIR = (
    ARTIFACT_ROOT
    / "logistic"
)

GCN_MODEL_PATH = (
    GCN_DIR
    / "duration_response_gcn_precipitation_only.pt"
)

GCN_TEST_PREDICTIONS_PATH = (
    GCN_DIR
    / "test_predictions.parquet"
)

LOGISTIC_MODEL_PATH = (
    LOGISTIC_DIR
    / "precipitation_only_logistic.pkl"
)

LOGISTIC_TEST_PREDICTIONS_PATH = (
    LOGISTIC_DIR
    / "test_predictions.parquet"
)

ELIGIBLE_REGISTRY_PATH = (
    MODEL_SELECTION_DIR
    / "eligible_sensor_model_registry.csv"
)

ELIGIBLE_REGISTRY_WITH_COORDINATES_PATH = (
    MODEL_SELECTION_DIR
    / "eligible_sensor_model_registry_with_coordinates.parquet"
)

SENSOR_SELECTION_PATH = (
    MODEL_SELECTION_DIR
    / "sensor_model_selection.csv"
)

GRID_PRIMARY_SENSOR_REGISTRY_PATH = (
    SPATIAL_DIR
    / "grid_primary_sensor_registry.parquet"
)

CLUSTER_PRIMARY_SENSOR_REGISTRY_PATH = (
    SPATIAL_DIR
    / "cluster_primary_sensor_registry.parquet"
)

ELIGIBLE_SENSOR_GRID_CLUSTER_MAP_PATH = (
    SPATIAL_DIR
    / "eligible_sensor_grid_cluster_map.parquet"
)

GRID_IMPUTATION_REFERENCE_PATH = (
    SPATIAL_DIR
    / "grid_imputation_reference.parquet"
)

SPATIAL_GOVERNANCE_SUMMARY_PATH = (
    SPATIAL_DIR
    / "mlflow_audit"
    / "spatial_governance_summary.json"
)

MODEL_SELECTION_GOVERNANCE_AUDIT_PATH = (
    MODEL_SELECTION_DIR
    / "mlflow_audit"
    / "operational_grid_support_audit.csv"
)

DEFAULT_MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000",
)


# ============================================================
# STAGE CONTRACT
# ============================================================

@dataclass(frozen=True)
class Stage:
    name: str
    module: str
    arguments: tuple[str, ...] = ()
    expected_outputs: tuple[Path, ...] = ()
    description: str = ""


STAGES: tuple[Stage, ...] = (

    # ========================================================
    # 1. SOURCE DATA INGESTION
    # ========================================================

    Stage(
        name="ingest_floodnet",
        module=(
            "mlops.flood.ingestion."
            "backfill_floodnet"
        ),
        description=(
            "Ingest append-only raw FloodNet depth observations "
            "for completed forward weeks. Existing S3 partitions "
            "are preserved and skipped."
        ),
    ),

    Stage(
        name="ingest_mrms",
        module=(
            "mlops.flood.ingestion."
            "backfill_mrms"
        ),
        description=(
            "Ingest append-only NOAA MRMS precipitation features "
            "for completed forward weeks. Existing S3 partitions "
            "are preserved and skipped."
        ),
    ),

    # ========================================================
    # 2. FORWARD TRAINING DATA CONSTRUCTION
    # ========================================================

    Stage(
        name="forward_response_refresh",
        module=(
            "mlops.flood.training."
            "build_forward_response"
        ),
        description=(
            "Build append-only hourly FloodNet response "
            "partitions from the refreshed raw FloodNet layer."
        ),
    ),

    Stage(
        name="forward_canonical_refresh",
        module=(
            "mlops.flood.training."
            "build_forward_canonical"
        ),
        description=(
            "Build append-only forward canonical training "
            "partitions from FloodNet response and MRMS data."
        ),
    ),

    # ========================================================
    # 3. MODEL RETRAINING + MLFLOW
    # ========================================================

    Stage(
        name="train_gcn_mlflow",
        module=(
            "mlops.flood.training."
            "train_gcn_mlflow"
        ),
        expected_outputs=(
            GCN_MODEL_PATH,
            GCN_TEST_PREDICTIONS_PATH,
        ),
        description=(
            "Retrain the validated GCN using immutable historical "
            "plus append-only forward canonical observations and "
            "create a fresh MLflow run."
        ),
    ),

    Stage(
        name="train_logistic_mlflow",
        module=(
            "mlops.flood.training."
            "train_logistic_mlflow"
        ),
        expected_outputs=(
            LOGISTIC_MODEL_PATH,
            LOGISTIC_TEST_PREDICTIONS_PATH,
        ),
        description=(
            "Retrain Logistic regression using the same canonical "
            "training contract and create a fresh MLflow run."
        ),
    ),

    # ========================================================
    # 4. PER-SENSOR MODEL SELECTION
    # ========================================================

    Stage(
        name="select_sensor_models",
        module=(
            "mlops.flood.training."
            "select_sensor_models"
        ),
        expected_outputs=(
            SENSOR_SELECTION_PATH,
            ELIGIBLE_REGISTRY_PATH,
        ),
        description=(
            "Re-evaluate GCN versus Logistic per physical sensor "
            "and rebuild the eligible sensor/model registry."
        ),
    ),

    # ========================================================
    # 5. SENSOR REFERENCE / COORDINATE REFRESH
    # ========================================================

    Stage(
        name="refresh_sensor_grid_map",
        module=(
            "mlops.flood.ingestion."
            "build_sensor_grid_map"
        ),
        description=(
            "Refresh the lifecycle-aware canonical FloodNet "
            "sensor-to-MRMS mapping before spatial deployment."
        ),
    ),

    Stage(
        name="enrich_selected_sensor_coordinates",
        module=(
            "mlops.flood.spatial."
            "enrich_selected_sensor_coordinates"
        ),
        expected_outputs=(
            ELIGIBLE_REGISTRY_WITH_COORDINATES_PATH,
        ),
        description=(
            "Attach canonical or validated historical coordinates "
            "to every freshly selected eligible sensor."
        ),
    ),

    # ========================================================
    # 6. SPATIAL OPERATIONAL ROUTING
    # ========================================================

    Stage(
        name="map_eligible_sensors_to_grid",
        module=(
            "mlops.flood.spatial."
            "map_eligible_sensors_to_grid"
        ),
        expected_outputs=(
            ELIGIBLE_SENSOR_GRID_CLUSTER_MAP_PATH,
            GRID_PRIMARY_SENSOR_REGISTRY_PATH,
            CLUSTER_PRIMARY_SENSOR_REGISTRY_PATH,
        ),
        description=(
            "Map the refreshed eligible sensor registry to NYC "
            "1-km grids and spatial clusters."
        ),
    ),

    Stage(
        name="build_grid_imputation_reference",
        module=(
            "mlops.flood.spatial."
            "build_grid_imputation_reference"
        ),
        expected_outputs=(
            GRID_IMPUTATION_REFERENCE_PATH,
        ),
        description=(
            "Rebuild the authoritative NYC 1-km operational "
            "support and imputation reference."
        ),
    ),

    # ========================================================
    # 7. GOVERNANCE / MLFLOW AUDITING
    # ========================================================

    Stage(
        name="model_selection_mlflow_governance",
        module=(
            "mlops.flood.training."
            "select_sensor_models_mlflow"
        ),
        arguments=(
            "--log-existing",
        ),
        expected_outputs=(
            MODEL_SELECTION_GOVERNANCE_AUDIT_PATH,
        ),
        description=(
            "Log refreshed model-selection governance after the "
            "sensor registry and spatial layer are available."
        ),
    ),

    Stage(
        name="spatial_imputation_mlflow_governance",
        module=(
            "mlops.flood.spatial."
            "spatial_imputation_mlflow"
        ),
        arguments=(
            "--log-existing",
        ),
        expected_outputs=(
            GRID_IMPUTATION_REFERENCE_PATH,
            SPATIAL_GOVERNANCE_SUMMARY_PATH,
        ),
        description=(
            "Log refreshed operational spatial/imputation "
            "governance to MLflow."
        ),
    ),
)

# ============================================================
# TIME / GIT HELPERS
# ============================================================

def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def iso_utc(
    value: datetime | None = None,
) -> str:
    value = (
        value
        if value is not None
        else utc_now()
    )

    return value.isoformat()


def cycle_id() -> str:
    """
    Return a collision-resistant UTC cycle identifier.

    A short UUID suffix prevents collisions when multiple pipeline
    invocations begin within the same operating-system clock tick.
    """

    timestamp = utc_now().strftime(
        "%Y%m%dT%H%M%SZ"
    )

    unique_suffix = uuid4().hex[:8]

    return (
        f"{timestamp}_{unique_suffix}"
    )


def git_value(
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


# ============================================================
# MANIFEST HELPERS
# ============================================================

def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            default=str,
        ),
        encoding="utf-8",
    )

    temporary.replace(
        path
    )


def initial_manifest(
    *,
    run_id: str,
    run_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    return {
        "cycle_id":
            run_id,

        "status":
            (
                "dry_run"
                if dry_run
                else "running"
            ),

        "started_at_utc":
            iso_utc(),

        "finished_at_utc":
            None,

        "python":
            sys.version,

        "python_executable":
            sys.executable,

        "working_directory":
            str(
                Path.cwd()
            ),

        "run_directory":
            str(
                run_dir
            ),

        "mlflow_tracking_uri":
            DEFAULT_MLFLOW_TRACKING_URI,

        "data_contract": {
            "historical_canonical":
                "immutable",

            "forward_canonical":
                "append-only",

            "training_observations":
                "real physical FloodNet sensors",

            "operational_imputed_grids_added_to_training":
                False,

            "operational_model_identity":
                "support sensor",

            "operational_rainfall":
                "target-grid-local",
        },

        "git": {
            "commit":
                git_value(
                    "rev-parse",
                    "HEAD",
                ),

            "branch":
                git_value(
                    "rev-parse",
                    "--abbrev-ref",
                    "HEAD",
                ),

            "status_porcelain":
                git_value(
                    "status",
                    "--porcelain",
                ),
        },

        "stages": [],
    }


def update_manifest(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> None:
    atomic_write_json(
        manifest_path,
        manifest,
    )


# ============================================================
# COMMAND EXECUTION
# ============================================================

def command_for_stage(
    stage: Stage,
) -> list[str]:
    return [
        sys.executable,
        "-m",
        stage.module,
        *stage.arguments,
    ]


def verify_outputs(
    stage: Stage,
) -> list[str]:
    missing = [
        str(
            path
        )
        for path in stage.expected_outputs
        if not path.exists()
    ]

    return missing


def run_stage(
    *,
    stage: Stage,
    index: int,
    total: int,
    logs_dir: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    started = utc_now()

    command = command_for_stage(
        stage
    )

    print()
    print(
        "=" * 78
    )
    print(
        f"STAGE {index}/{total}: "
        f"{stage.name}"
    )
    print(
        "=" * 78
    )
    print(
        stage.description
    )
    print()
    print(
        "COMMAND:"
    )
    print(
        " ".join(
            command
        )
    )

    log_path = (
        logs_dir
        / f"{index:02d}_{stage.name}.log"
    )

    log_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    stage_record: dict[str, Any] = {
        "name":
            stage.name,

        "module":
            stage.module,

        "arguments":
            list(
                stage.arguments
            ),

        "description":
            stage.description,

        "command":
            command,

        "started_at_utc":
            iso_utc(
                started
            ),

        "finished_at_utc":
            None,

        "duration_seconds":
            None,

        "return_code":
            None,

        "status":
            "running",

        "log_path":
            str(
                log_path
            ),

        "expected_outputs":
            [
                str(
                    path
                )
                for path
                in stage.expected_outputs
            ],

        "missing_outputs":
            [],
    }

    with log_path.open(
        "w",
        encoding="utf-8",
    ) as log_handle:

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=environment,
        )

        assert process.stdout is not None

        for line in process.stdout:
            print(
                line,
                end="",
            )

            log_handle.write(
                line
            )

            log_handle.flush()

        return_code = (
            process.wait()
        )

    finished = utc_now()

    stage_record[
        "finished_at_utc"
    ] = iso_utc(
        finished
    )

    stage_record[
        "duration_seconds"
    ] = (
        finished
        - started
    ).total_seconds()

    stage_record[
        "return_code"
    ] = int(
        return_code
    )

    if return_code != 0:
        stage_record[
            "status"
        ] = "failed"

        return stage_record

    missing = verify_outputs(
        stage
    )

    stage_record[
        "missing_outputs"
    ] = missing

    if missing:
        stage_record[
            "status"
        ] = "failed_output_validation"

        return stage_record

    stage_record[
        "status"
    ] = "passed"

    print()
    print(
        f"STAGE PASSED: {stage.name}"
    )

    return stage_record


# ============================================================
# FINAL CONTRACT VALIDATION
# ============================================================

def load_json_if_exists(
    path: Path,
) -> dict[str, Any] | None:
    if not path.exists():
        return None

    try:
        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:
        return None

    if isinstance(
        value,
        dict,
    ):
        return value

    return None


def final_output_inventory() -> dict[str, Any]:
    paths = {
        "gcn_model":
            GCN_MODEL_PATH,

        "gcn_test_predictions":
            GCN_TEST_PREDICTIONS_PATH,

        "logistic_model":
            LOGISTIC_MODEL_PATH,

        "logistic_test_predictions":
            LOGISTIC_TEST_PREDICTIONS_PATH,

        "sensor_model_selection":
            SENSOR_SELECTION_PATH,

        "eligible_sensor_registry":
            ELIGIBLE_REGISTRY_PATH,

        "eligible_sensor_registry_with_coordinates":
            ELIGIBLE_REGISTRY_WITH_COORDINATES_PATH,

        "eligible_sensor_grid_cluster_map":
            ELIGIBLE_SENSOR_GRID_CLUSTER_MAP_PATH,

        "grid_primary_sensor_registry":
            GRID_PRIMARY_SENSOR_REGISTRY_PATH,

        "cluster_primary_sensor_registry":
            CLUSTER_PRIMARY_SENSOR_REGISTRY_PATH,

        "grid_imputation_reference":
            GRID_IMPUTATION_REFERENCE_PATH,

        "spatial_governance_summary":
            SPATIAL_GOVERNANCE_SUMMARY_PATH,
    }

    return {
        name: {
            "path":
                str(
                    path
                ),

            "exists":
                path.exists(),

            "size_bytes":
                (
                    path.stat().st_size
                    if path.exists()
                    else None
                ),

            "modified_at_utc":
                (
                    datetime.fromtimestamp(
                        path.stat().st_mtime,
                        tz=timezone.utc,
                    ).isoformat()
                    if path.exists()
                    else None
                ),
        }
        for name, path
        in paths.items()
    }


# ============================================================
# OPTIONAL CYCLE SNAPSHOT
# ============================================================

def snapshot_small_governance_artifacts(
    *,
    run_dir: Path,
) -> list[str]:
    """
    Copy compact governance artifacts into the timestamped retraining
    directory.

    Heavy model/test-prediction files remain in their normal artifact
    locations and MLflow. This snapshot is meant to preserve the compact
    decision/audit state for the weekly cycle.
    """

    snapshot_dir = (
        run_dir
        / "governance_snapshot"
    )

    snapshot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidates = [
        SENSOR_SELECTION_PATH,
        ELIGIBLE_REGISTRY_PATH,
        ELIGIBLE_REGISTRY_WITH_COORDINATES_PATH,
        MODEL_SELECTION_DIR
        / "model_selection_summary.csv",
        SPATIAL_DIR
        / "sensor_grid_mapping_summary.json",
        SPATIAL_DIR
        / "grid_imputation_reference_summary.json",
        SPATIAL_GOVERNANCE_SUMMARY_PATH,
        MODEL_SELECTION_DIR
        / "mlflow_audit"
        / "operational_support_sensor_registry.csv",
        MODEL_SELECTION_DIR
        / "mlflow_audit"
        / "eligible_not_currently_operational.csv",
        SPATIAL_DIR
        / "mlflow_audit"
        / "operational_support_sensor_audit.csv",
    ]

    copied: list[str] = []

    for source in candidates:

        if not source.exists():
            continue

        destination = (
            snapshot_dir
            / source.name
        )

        shutil.copy2(
            source,
            destination,
        )

        copied.append(
            str(
                destination
            )
        )

    return copied


# ============================================================
# DRY RUN
# ============================================================

def print_dry_run() -> None:
    print(
        "NYC FLOOD WEEKLY RETRAINING DRY RUN"
    )
    print(
        "==================================="
    )
    print()
    print(
        f"MLflow tracking URI: "
        f"{DEFAULT_MLFLOW_TRACKING_URI}"
    )
    print()
    print(
        "Historical canonical data: immutable"
    )
    print(
        "Forward canonical data:    append-only"
    )
    print(
        "Training observations:     physical sensors only"
    )
    print()

    for index, stage in enumerate(
        STAGES,
        start=1,
    ):
        print(
            f"{index:02d}. "
            f"{stage.name}"
        )

        print(
            "    "
            + " ".join(
                command_for_stage(
                    stage
                )
            )
        )

        if stage.description:
            print(
                f"    {stage.description}"
            )


# ============================================================
# EXECUTION
# ============================================================

def execute(
    *,
    dry_run: bool,
) -> int:
    run_id = cycle_id()

    run_dir = (
        RETRAINING_ROOT
        / run_id
    )

    logs_dir = (
        run_dir
        / "logs"
    )

    manifest_path = (
        run_dir
        / "retraining_manifest.json"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    manifest = initial_manifest(
        run_id=run_id,
        run_dir=run_dir,
        dry_run=dry_run,
    )

    update_manifest(
        manifest_path,
        manifest,
    )

    if dry_run:
        print_dry_run()

        manifest[
            "finished_at_utc"
        ] = iso_utc()

        manifest[
            "status"
        ] = "dry_run_complete"

        update_manifest(
            manifest_path,
            manifest,
        )

        print()
        print(
            f"Dry-run manifest: {manifest_path}"
        )

        return 0

    environment = os.environ.copy()

    # The individual wrappers already read this variable. Keeping it in
    # one shared environment ensures all four governance experiments are
    # written to the same MLflow server.
    environment[
        "MLFLOW_TRACKING_URI"
    ] = DEFAULT_MLFLOW_TRACKING_URI

    print(
        "NYC FLOOD WEEKLY RETRAINING CYCLE"
    )
    print(
        "================================="
    )
    print(
        f"Cycle ID: {run_id}"
    )
    print(
        f"MLflow:   {DEFAULT_MLFLOW_TRACKING_URI}"
    )
    print(
        f"Manifest: {manifest_path}"
    )

    try:
        for index, stage in enumerate(
            STAGES,
            start=1,
        ):
            record = run_stage(
                stage=stage,
                index=index,
                total=len(
                    STAGES
                ),
                logs_dir=logs_dir,
                environment=environment,
            )

            manifest[
                "stages"
            ].append(
                record
            )

            update_manifest(
                manifest_path,
                manifest,
            )

            if record[
                "status"
            ] != "passed":
                manifest[
                    "status"
                ] = "failed"

                manifest[
                    "failed_stage"
                ] = stage.name

                manifest[
                    "finished_at_utc"
                ] = iso_utc()

                manifest[
                    "final_outputs"
                ] = (
                    final_output_inventory()
                )

                update_manifest(
                    manifest_path,
                    manifest,
                )

                print()
                print(
                    "WEEKLY RETRAINING FAILED"
                )
                print(
                    "========================"
                )
                print(
                    f"Failed stage: "
                    f"{stage.name}"
                )
                print(
                    f"Stage log: "
                    f"{record['log_path']}"
                )
                print(
                    f"Manifest: "
                    f"{manifest_path}"
                )

                return 1

        manifest[
            "final_outputs"
        ] = (
            final_output_inventory()
        )

        manifest[
            "governance_snapshot"
        ] = (
            snapshot_small_governance_artifacts(
                run_dir=run_dir
            )
        )

        manifest[
            "status"
        ] = "passed"

        manifest[
            "finished_at_utc"
        ] = iso_utc()

        update_manifest(
            manifest_path,
            manifest,
        )

        print()
        print(
            "WEEKLY RETRAINING COMPLETE"
        )
        print(
            "=========================="
        )
        print(
            f"Cycle ID: {run_id}"
        )
        print(
            f"Manifest: {manifest_path}"
        )
        print()
        print(
            "All required stages passed."
        )
        print(
            "Fresh MLflow runs were created for GCN and Logistic "
            "training, model-selection governance, and "
            "spatial/imputation governance."
        )

        return 0

    except KeyboardInterrupt:
        manifest[
            "status"
        ] = "interrupted"

        manifest[
            "finished_at_utc"
        ] = iso_utc()

        update_manifest(
            manifest_path,
            manifest,
        )

        raise

    except Exception as exc:
        manifest[
            "status"
        ] = "failed_unhandled_exception"

        manifest[
            "finished_at_utc"
        ] = iso_utc()

        manifest[
            "error"
        ] = {
            "type":
                type(
                    exc
                ).__name__,

            "message":
                str(
                    exc
                ),

            "traceback":
                traceback.format_exc(),
        }

        manifest[
            "final_outputs"
        ] = (
            final_output_inventory()
        )

        update_manifest(
            manifest_path,
            manifest,
        )

        print()
        print(
            "WEEKLY RETRAINING FAILED WITH "
            "UNHANDLED EXCEPTION"
        )
        print(
            "================================"
        )
        print(
            repr(
                exc
            )
        )
        print(
            f"Manifest: {manifest_path}"
        )

        return 1


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the complete weekly NYC flood retraining, "
            "model-selection, spatial refresh, and MLflow "
            "governance cycle."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print and record the stage sequence without "
            "executing any pipeline stage."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    return_code = execute(
        dry_run=args.dry_run
    )

    raise SystemExit(
        return_code
    )


if __name__ == "__main__":
    main()
