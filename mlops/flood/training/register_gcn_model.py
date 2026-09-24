"""
Register the validated NYC flood GCN as an MLflow PyFunc model.

This script does NOT retrain the GCN and does NOT modify canonical S3 data.

It packages the already-validated artifacts from artifacts/flood/gcn:

    duration_response_gcn_precipitation_only.pt
    scalers.pkl
    node_table.parquet

as a deployable MLflow PythonModel and creates a model version under:

    nyc-resilience-flood-gcn

The serving contract is one hourly sensor snapshot. The input DataFrame must
contain one row per active sensor and these columns:

    deployment_id
    precip_current_hour_mm
    precip_previous_6h_mm
    daily_total_precip_mm

The GCN graph is reconstructed from the saved node table using the same
4-nearest-neighbor haversine methodology as training. Sensors omitted from an
input snapshot are treated as inactive. Unknown deployment IDs are rejected;
adding new sensors requires a new trained/registered model version.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.neighbors import NearestNeighbors

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

DEFAULT_RUN_ID = os.getenv(
    "FLOOD_GCN_RUN_ID",
    "a2d0fbdacb38473591216dbf4200125f",
)

MODEL_ARTIFACT_PATH = "serving_model"

PREDICTION_COLUMN = "predicted_minutes_above_1inch"
PREDICTED_EVENT_COLUMN = "predicted_event"

REQUIRED_INPUT_COLUMNS = [
    gcn.SENSOR_ID,
    *gcn.FEATURES,
]


# ============================================================
# SERVING-SIDE GCN
# ============================================================

def create_active_adjacency(
    active_mask: torch.Tensor,
    base_adjacency: torch.Tensor,
) -> torch.Tensor:
    """
    Reproduce the normalized active-node adjacency used in train_gcn.py.

    active_mask shape:
        [batch, nodes]

    base_adjacency shape:
        [nodes, nodes]
    """

    pair_mask = (
        active_mask.unsqueeze(2)
        * active_mask.unsqueeze(1)
    )

    adjacency = (
        base_adjacency.unsqueeze(0)
        * pair_mask
    )

    degree = adjacency.sum(
        dim=-1
    )

    degree_inverse_sqrt = torch.where(
        degree > 0,
        torch.rsqrt(
            degree.clamp_min(1e-8)
        ),
        torch.zeros_like(degree),
    )

    return (
        degree_inverse_sqrt.unsqueeze(2)
        * adjacency
        * degree_inverse_sqrt.unsqueeze(1)
    )


class GraphConv(nn.Module):
    """
    Same graph-convolution layer used by the validated training pipeline.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
    ):
        super().__init__()

        self.linear = nn.Linear(
            input_size,
            output_size,
        )

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:

        propagated = torch.einsum(
            "bij,bjf->bif",
            adjacency,
            x,
        )

        output = self.linear(
            propagated
        )

        return (
            output
            * active_mask.unsqueeze(-1)
        )


class ServingDepthGCN(nn.Module):
    """
    Inference equivalent of train_gcn.DepthGCN.

    Dropout is present so the state-dict architecture matches training,
    but model.eval() disables dropout during prediction.
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int,
        dropout: float,
        base_adjacency: torch.Tensor,
    ):
        super().__init__()

        self.base_adjacency = (
            base_adjacency
        )

        self.gcn1 = GraphConv(
            n_features,
            hidden_size,
        )

        self.gcn2 = GraphConv(
            hidden_size,
            hidden_size,
        )

        self.dropout = nn.Dropout(
            dropout
        )

        self.output = nn.Linear(
            hidden_size,
            1,
        )

    def forward(
        self,
        x: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:

        adjacency = (
            create_active_adjacency(
                active_mask,
                self.base_adjacency,
            )
        )

        x = torch.relu(
            self.gcn1(
                x,
                adjacency,
                active_mask,
            )
        )

        x = self.dropout(x)

        x = torch.relu(
            self.gcn2(
                x,
                adjacency,
                active_mask,
            )
        )

        x = self.dropout(x)

        prediction = (
            self.output(x)
            .squeeze(-1)
        )

        return (
            prediction
            * active_mask
        )


# ============================================================
# GRAPH RECONSTRUCTION
# ============================================================

def build_base_adjacency(
    node_table: pd.DataFrame,
    k_neighbors: int,
) -> torch.Tensor:
    """
    Reconstruct exactly the same undirected geographic KNN graph used in
    training.

    Node ordering comes from node_table.parquet, which was saved by training.
    """

    required = {
        gcn.SENSOR_ID,
        "sensor_lat",
        "sensor_lon",
    }

    missing = (
        required
        - set(node_table.columns)
    )

    if missing:
        raise ValueError(
            "Saved node table is missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    n_nodes = len(node_table)

    if n_nodes == 0:
        raise ValueError(
            "Saved node table is empty."
        )

    coordinates = np.radians(
        node_table[
            [
                "sensor_lat",
                "sensor_lon",
            ]
        ].to_numpy(
            dtype=float
        )
    )

    knn = NearestNeighbors(
        n_neighbors=min(
            k_neighbors + 1,
            n_nodes,
        ),
        metric="haversine",
    ).fit(coordinates)

    _, neighbors = (
        knn.kneighbors(
            coordinates
        )
    )

    adjacency = np.zeros(
        (
            n_nodes,
            n_nodes,
        ),
        dtype=np.float32,
    )

    for source in range(
        n_nodes
    ):
        for position in range(
            1,
            neighbors.shape[1],
        ):
            target = int(
                neighbors[
                    source,
                    position,
                ]
            )

            adjacency[
                source,
                target,
            ] = 1.0

            adjacency[
                target,
                source,
            ] = 1.0

    adjacency += np.eye(
        n_nodes,
        dtype=np.float32,
    )

    return torch.tensor(
        adjacency,
        dtype=torch.float32,
    )


# ============================================================
# PYFUNC MODEL
# ============================================================


def resolve_portable_artifact_path(
    artifact_path: str,
) -> Path:
    """
    Resolve an MLflow packaged artifact path across Windows/Linux.

    MLflow packages created on Windows may preserve backslashes in
    artifact paths. Linux treats those backslashes as literal characters
    rather than directory separators.
    """

    raw_path = str(
        artifact_path
    )

    original = Path(
        raw_path
    )

    if original.exists():
        return original

    normalized = Path(
        raw_path.replace(
            "\\",
            os.sep,
        )
    )

    if normalized.exists():
        return normalized

    raise FileNotFoundError(
        "MLflow model artifact could not be resolved. "
        f"Original path: {raw_path!r}; "
        f"normalized path: {str(normalized)!r}"
    )


class FloodGCNPythonModel(
    mlflow.pyfunc.PythonModel
):
    """
    MLflow serving wrapper for the validated precipitation-only flood GCN.

    One call represents one hourly sensor snapshot.

    Missing known sensors are treated as inactive for that snapshot.
    Unknown sensors are rejected because the registered graph is fixed to the
    node set used during training.
    """

    def load_context(
        self,
        context,
    ) -> None:

        checkpoint_path = (
            resolve_portable_artifact_path(
                context.artifacts[
                    "checkpoint"
                ]
            )
        )

        scalers_path = (
            resolve_portable_artifact_path(
                context.artifacts[
                    "scalers"
                ]
            )
        )

        node_table_path = (
            resolve_portable_artifact_path(
                context.artifacts[
                    "node_table"
                ]
            )
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        with open(
            scalers_path,
            "rb",
        ) as handle:
            scaler_payload = (
                pickle.load(handle)
            )

        node_table = pd.read_parquet(
            node_table_path
        )

        # Training saved this table in canonical graph-node order.
        node_table = (
            node_table
            .reset_index(
                drop=True
            )
        )

        self.node_table = node_table

        self.node_ids = (
            node_table[
                gcn.SENSOR_ID
            ]
            .astype(str)
            .tolist()
        )

        self.node_to_index = {
            node_id: index
            for index, node_id
            in enumerate(
                self.node_ids
            )
        }

        checkpoint_node_ids = [
            str(value)
            for value
            in checkpoint[
                "node_ids"
            ]
        ]

        if (
            checkpoint_node_ids
            != self.node_ids
        ):
            raise ValueError(
                "Checkpoint node ordering does not match "
                "saved node_table.parquet."
            )

        self.features = list(
            checkpoint.get(
                "features",
                gcn.FEATURES,
            )
        )

        self.target = str(
            checkpoint.get(
                "target",
                gcn.TARGET_COLUMN,
            )
        )

        self.depth_threshold_mm = float(
            checkpoint.get(
                "depth_threshold_mm",
                gcn.DEPTH_THRESHOLD_MM,
            )
        )

        training_config = (
            checkpoint.get(
                "training_config",
                {},
            )
        )

        hidden_size = int(
            training_config.get(
                "hidden_size",
                gcn.HIDDEN_SIZE,
            )
        )

        dropout = float(
            training_config.get(
                "dropout",
                gcn.DROPOUT,
            )
        )

        k_neighbors = int(
            training_config.get(
                "k_neighbors",
                gcn.K_NEIGHBORS,
            )
        )

        self.predicted_duration_threshold_minutes = float(
            training_config.get(
                "predicted_duration_threshold_minutes",
                gcn.PREDICTED_DURATION_THRESHOLD_MINUTES,
            )
        )

        self.feature_scaler = (
            scaler_payload[
                "feature_scaler"
            ]
        )

        self.target_scaler = (
            scaler_payload[
                "target_scaler"
            ]
        )

        scaler_features = list(
            scaler_payload.get(
                "features",
                self.features,
            )
        )

        if (
            scaler_features
            != self.features
        ):
            raise ValueError(
                "Feature order differs between checkpoint "
                "and saved feature scaler."
            )

        base_adjacency = (
            build_base_adjacency(
                node_table,
                k_neighbors,
            )
        )

        self.model = (
            ServingDepthGCN(
                n_features=len(
                    self.features
                ),
                hidden_size=hidden_size,
                dropout=dropout,
                base_adjacency=base_adjacency,
            )
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.eval()

    def _validate_input(
        self,
        model_input: pd.DataFrame,
    ) -> pd.DataFrame:

        if not isinstance(
            model_input,
            pd.DataFrame,
        ):
            raise TypeError(
                "GCN inference requires a pandas DataFrame."
            )

        missing = (
            set(
                REQUIRED_INPUT_COLUMNS
            )
            - set(
                model_input.columns
            )
        )

        if missing:
            raise ValueError(
                "GCN inference input is missing required columns: "
                + ", ".join(
                    sorted(missing)
                )
            )

        result = (
            model_input[
                REQUIRED_INPUT_COLUMNS
            ]
            .copy()
        )

        if result.empty:
            raise ValueError(
                "GCN inference input is empty."
            )

        result[
            gcn.SENSOR_ID
        ] = (
            result[
                gcn.SENSOR_ID
            ]
            .astype(str)
        )

        duplicates = (
            result[
                gcn.SENSOR_ID
            ]
            .duplicated()
        )

        if duplicates.any():

            duplicate_ids = (
                result.loc[
                    duplicates,
                    gcn.SENSOR_ID,
                ]
                .drop_duplicates()
                .tolist()
            )

            raise ValueError(
                "One inference call must contain at most one row per "
                "deployment_id. Duplicate IDs: "
                + ", ".join(
                    duplicate_ids[:20]
                )
            )

        unknown = sorted(
            set(
                result[
                    gcn.SENSOR_ID
                ]
            )
            - set(
                self.node_ids
            )
        )

        if unknown:
            raise ValueError(
                "Inference contains deployment IDs that are not present "
                "in this registered graph/model version. "
                "Retrain/register a new version before scoring new sensors. "
                "Unknown IDs: "
                + ", ".join(
                    unknown[:20]
                )
            )

        for column in self.features:

            result[column] = (
                pd.to_numeric(
                    result[column],
                    errors="coerce",
                )
            )

        invalid_features = (
            result[
                self.features
            ]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .isna()
            .any(
                axis=1
            )
        )

        if invalid_features.any():

            bad_ids = (
                result.loc[
                    invalid_features,
                    gcn.SENSOR_ID,
                ]
                .tolist()
            )

            raise ValueError(
                "All three precipitation predictors must be finite "
                "numeric values. Invalid sensor rows: "
                + ", ".join(
                    bad_ids[:20]
                )
            )

        return result

    def predict(
        self,
        context,
        model_input,
        params=None,
    ) -> pd.DataFrame:

        frame = (
            self._validate_input(
                model_input
            )
        )

        n_nodes = len(
            self.node_ids
        )

        n_features = len(
            self.features
        )

        x = np.zeros(
            (
                n_nodes,
                n_features,
            ),
            dtype=np.float32,
        )

        active_mask = np.zeros(
            n_nodes,
            dtype=np.float32,
        )

        scaled_features = (
            self.feature_scaler
            .transform(
                frame[
                    self.features
                ]
            )
            .astype(
                np.float32
            )
        )

        for row_position, deployment_id in enumerate(
            frame[
                gcn.SENSOR_ID
            ]
        ):

            node_index = (
                self.node_to_index[
                    deployment_id
                ]
            )

            x[
                node_index,
                :,
            ] = (
                scaled_features[
                    row_position,
                    :,
                ]
            )

            active_mask[
                node_index
            ] = 1.0

        x_tensor = (
            torch.tensor(
                x,
                dtype=torch.float32,
            )
            .unsqueeze(0)
        )

        active_tensor = (
            torch.tensor(
                active_mask,
                dtype=torch.float32,
            )
            .unsqueeze(0)
        )

        with torch.no_grad():

            scaled_prediction = (
                self.model(
                    x_tensor,
                    active_tensor,
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )

        active_indices = [
            self.node_to_index[
                deployment_id
            ]
            for deployment_id
            in frame[
                gcn.SENSOR_ID
            ]
        ]

        active_scaled_prediction = (
            scaled_prediction[
                active_indices
            ]
        )

        prediction_minutes = (
            self.target_scaler
            .inverse_transform(
                active_scaled_prediction
                .reshape(
                    -1,
                    1,
                )
            )
            .reshape(-1)
        )

        # Flood duration cannot be negative and cannot exceed one hour.
        prediction_minutes = np.clip(
            prediction_minutes,
            0.0,
            60.0,
        )

        output = pd.DataFrame(
            {
                gcn.SENSOR_ID: (
                    frame[
                        gcn.SENSOR_ID
                    ]
                    .to_numpy()
                ),
                PREDICTION_COLUMN: (
                    prediction_minutes
                ),
            }
        )

        output[
            PREDICTED_EVENT_COLUMN
        ] = (
            output[
                PREDICTION_COLUMN
            ]
            > self.predicted_duration_threshold_minutes
        )

        return output


# ============================================================
# LOCAL ARTIFACT VALIDATION
# ============================================================

def validate_local_artifacts() -> None:
    """
    Registration uses the validated local artifacts produced by train_gcn.py.
    """

    required_paths = [
        gcn.MODEL_PATH,
        gcn.SCALERS_PATH,
        gcn.NODE_TABLE_PATH,
    ]

    missing = [
        path
        for path in required_paths
        if not Path(path).exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required GCN serving artifacts are missing:\n"
            + "\n".join(
                f"    {path}"
                for path in missing
            )
        )


# ============================================================
# RUN / VERSION METADATA
# ============================================================

def safe_metric(
    client: MlflowClient,
    run_id: str,
    key: str,
) -> str:
    """
    Convert an MLflow metric to a model-version tag when present.
    """

    run = client.get_run(
        run_id
    )

    value = (
        run.data.metrics.get(
            key
        )
    )

    if value is None:
        return "not_logged"

    if (
        isinstance(
            value,
            float,
        )
        and not math.isfinite(value)
    ):
        return "not_finite"

    return str(value)


def version_tags(
    client: MlflowClient,
    run_id: str,
) -> dict[str, str]:

    run = client.get_run(
        run_id
    )

    run_tags = (
        run.data.tags
    )

    return {
        "pipeline": "flood-gcn",
        "model_family": "graph-convolutional-network",
        "predictor_family": "precipitation-only",
        "scientific_threshold": "1 inch / 25.4 mm",
        "training_target": gcn.TARGET_COLUMN,
        "canonical_source_target": gcn.SOURCE_TARGET_COLUMN,
        "source_run_id": run_id,
        "git_commit": run_tags.get(
            "git_commit",
            "unknown",
        ),
        "git_branch": run_tags.get(
            "git_branch",
            "unknown",
        ),
        "test_start": run_tags.get(
            "test_start",
            "unknown",
        ),
        "test_end": run_tags.get(
            "test_end",
            "unknown",
        ),
        "pooled_r2": safe_metric(
            client,
            run_id,
            "r2",
        ),
        "pooled_rmse_minutes": safe_metric(
            client,
            run_id,
            "rmse_minutes",
        ),
        "pooled_mae_minutes": safe_metric(
            client,
            run_id,
            "mae_minutes",
        ),
        "event_precision_micro": safe_metric(
            client,
            run_id,
            "event_precision_micro",
        ),
        "event_recall_micro": safe_metric(
            client,
            run_id,
            "event_recall_micro",
        ),
        "event_f1_micro": safe_metric(
            client,
            run_id,
            "event_f1_micro",
        ),
        "registration_status": "candidate",
        "promotion_status": "not_promoted",
    }


# ============================================================
# INPUT EXAMPLE / SIGNATURE
# ============================================================

def build_input_example() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Provide an explicit serving contract without using test labels.
    """

    node_table = pd.read_parquet(
        gcn.NODE_TABLE_PATH
    )

    example_ids = (
        node_table[
            gcn.SENSOR_ID
        ]
        .astype(str)
        .head(3)
        .tolist()
    )

    input_example = pd.DataFrame(
        {
            gcn.SENSOR_ID: (
                example_ids
            ),
            "precip_current_hour_mm": (
                [0.0, 1.0, 5.0][
                    : len(example_ids)
                ]
            ),
            "precip_previous_6h_mm": (
                [0.0, 2.0, 10.0][
                    : len(example_ids)
                ]
            ),
            "daily_total_precip_mm": (
                [0.0, 3.0, 15.0][
                    : len(example_ids)
                ]
            ),
        }
    )

    output_example = pd.DataFrame(
        {
            gcn.SENSOR_ID: (
                example_ids
            ),
            PREDICTION_COLUMN: (
                np.zeros(
                    len(example_ids),
                    dtype=float,
                )
            ),
            PREDICTED_EVENT_COLUMN: (
                np.zeros(
                    len(example_ids),
                    dtype=bool,
                )
            ),
        }
    )

    return (
        input_example,
        output_example,
    )


# ============================================================
# REGISTER
# ============================================================

def register_model(
    run_id: str,
) -> str:
    """
    Package a validated run as a new MLflow registered-model version.

    This intentionally creates a candidate version only. It does not set a
    production/champion alias and does not replace an existing deployed model.
    """

    validate_local_artifacts()

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    client = MlflowClient()

    # Fail early if the requested run does not exist.
    run = client.get_run(
        run_id
    )

    print(
        "REGISTERING FLOOD GCN MODEL"
    )

    print(
        "==========================="
    )

    print(
        f"Tracking URI: {MLFLOW_TRACKING_URI}"
    )

    print(
        f"Source run:   {run_id}"
    )

    print(
        f"Run name:     "
        f"{run.data.tags.get('mlflow.runName', 'unknown')}"
    )

    print(
        f"Model name:   {REGISTERED_MODEL_NAME}"
    )

    input_example, output_example = (
        build_input_example()
    )

    signature = infer_signature(
        input_example,
        output_example,
    )

    artifacts = {
        "checkpoint": str(
            Path(
                gcn.MODEL_PATH
            ).resolve()
        ),
        "scalers": str(
            Path(
                gcn.SCALERS_PATH
            ).resolve()
        ),
        "node_table": str(
            Path(
                gcn.NODE_TABLE_PATH
            ).resolve()
        ),
    }

    # Re-open the completed training run so the serving model is attached to
    # the exact run whose metrics/artifacts it represents.
    with mlflow.start_run(
        run_id=run_id
    ):

        model_info = (
            mlflow.pyfunc.log_model(
                artifact_path=MODEL_ARTIFACT_PATH,
                python_model=FloodGCNPythonModel(),
                artifacts=artifacts,
                signature=signature,
                input_example=input_example,
                registered_model_name=REGISTERED_MODEL_NAME,
                pip_requirements=[
                    "mlflow==3.14.0",
                    "numpy",
                    "pandas",
                    "pyarrow",
                    "torch",
                    "scikit-learn==1.8.0",
                ],
                metadata={
                    "scientific_threshold": "1 inch / 25.4 mm",
                    "training_target": gcn.TARGET_COLUMN,
                    "canonical_source_target": gcn.SOURCE_TARGET_COLUMN,
                    "features": gcn.FEATURES,
                    "serving_contract": (
                        "one hourly sensor snapshot; "
                        "one row per active known deployment_id"
                    ),
                },
                await_registration_for=300,
            )
        )

    print(
        "\nLogged serving model:"
    )

    print(
        model_info.model_uri
    )

    # MLflow 3.x returns the exact registered version created by log_model().
    # Use that directly rather than searching model versions by run ID.
    registered_version = (
        model_info.registered_model_version
    )

    if registered_version is None:
        raise RuntimeError(
            "MLflow logged and registered the serving model, "
            "but did not return a registered model version."
        )

    model_version = (
        client.get_model_version(
            name=REGISTERED_MODEL_NAME,
            version=str(
                registered_version
            ),
        )
    )

    tags = version_tags(
        client,
        run_id,
    )

    for key, value in tags.items():

        client.set_model_version_tag(
            name=REGISTERED_MODEL_NAME,
            version=model_version.version,
            key=key,
            value=str(value),
        )

    # Registered-model-level metadata describes the invariant serving contract.
    client.set_registered_model_tag(
        REGISTERED_MODEL_NAME,
        "model_family",
        "graph-convolutional-network",
    )

    client.set_registered_model_tag(
        REGISTERED_MODEL_NAME,
        "scientific_threshold",
        "1 inch / 25.4 mm",
    )

    client.set_registered_model_tag(
        REGISTERED_MODEL_NAME,
        "serving_contract",
        "hourly-known-sensor-snapshot",
    )

    print(
        "\nMODEL VERSION CREATED"
    )

    print(
        "====================="
    )

    print(
        f"Registered model: "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"Version:          "
        f"{model_version.version}"
    )

    print(
        f"Source run:       "
        f"{run_id}"
    )

    print(
        "Status:           "
        "candidate / not promoted"
    )

    print(
        "\nNo production/champion alias was changed."
    )

    return str(
        model_version.version
    )

# ============================================================
# RELOAD SMOKE TEST
# ============================================================

def smoke_test_registered_model(
    version: str,
) -> None:
    """
    Reload the just-registered model through MLflow and score a small valid
    snapshot. This verifies packaging, graph reconstruction, checkpoint reload,
    scaler reload, and PyFunc inference.
    """

    model_uri = (
        f"models:/{REGISTERED_MODEL_NAME}/{version}"
    )

    print(
        "\nREGISTERED MODEL RELOAD TEST"
    )

    print(
        "============================"
    )

    print(
        f"Loading: {model_uri}"
    )

    loaded = (
        mlflow.pyfunc.load_model(
            model_uri
        )
    )

    input_example, _ = (
        build_input_example()
    )

    predictions = (
        loaded.predict(
            input_example
        )
    )

    print(
        predictions.to_string(
            index=False
        )
    )

    required_output_columns = {
        gcn.SENSOR_ID,
        PREDICTION_COLUMN,
        PREDICTED_EVENT_COLUMN,
    }

    missing = (
        required_output_columns
        - set(
            predictions.columns
        )
    )

    if missing:
        raise RuntimeError(
            "Reloaded model output is missing columns: "
            + ", ".join(
                sorted(missing)
            )
        )

    values = pd.to_numeric(
        predictions[
            PREDICTION_COLUMN
        ],
        errors="coerce",
    )

    if (
        values.isna().any()
        or not np.isfinite(
            values.to_numpy()
        ).all()
    ):
        raise RuntimeError(
            "Reloaded model produced non-finite predictions."
        )

    if (
        (values < 0).any()
        or (values > 60).any()
    ):
        raise RuntimeError(
            "Reloaded model produced flood-duration predictions outside 0-60 minutes."
        )

    print(
        "\nREGISTERED MODEL RELOAD TEST PASSED"
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Register the validated NYC flood GCN as an MLflow PyFunc model."
        )
    )

    parser.add_argument(
        "--run-id",
        default=DEFAULT_RUN_ID,
        help=(
            "MLflow training run ID to package and register. "
            f"Default: {DEFAULT_RUN_ID}"
        ),
    )

    parser.add_argument(
        "--skip-smoke-test",
        action="store_true",
        help=(
            "Register the model but do not reload and score the registered "
            "version."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    version = register_model(
        args.run_id
    )

    if not args.skip_smoke_test:

        smoke_test_registered_model(
            version
        )


if __name__ == "__main__":
    main()