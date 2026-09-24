"""
Notebook-faithful GCN training pipeline for NYC flood prediction.

Training data is a logical in-memory combination of:
  1. immutable historical canonical monthly partitions through
     2026-03-13 06:00 UTC; and
  2. append-only forward canonical daily partitions beginning
     2026-03-13 07:00 UTC.

The model methodology is preserved from:
    all_sensors_gcn_part3_precipitation_only_24hours.ipynb

The upstream canonical pipeline already reconstructs the hourly flood
response, so this module does not rebuild response targets from raw sensor
readings. Historical and forward S3 data are read-only here.
"""

from __future__ import annotations

import copy
import io
import math
import os
import pickle
import random
from dataclasses import asdict, dataclass

import boto3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from mlops.flood.shared.config import FLOOD_ARTIFACT_DIR


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42


def set_random_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_random_seed()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# DATA / SCIENTIFIC CONFIGURATION
# ============================================================

SENSOR_ID = "deployment_id"
TIME_COLUMN = "hour"
MAX_DEPTH_COLUMN = "hourly_max_depth_mm"
RESPONSE_OBSERVED_COLUMN = "response_observed"
DATA_SOURCE_COLUMN = "training_data_source"

HISTORICAL_CUTOFF = pd.Timestamp("2026-03-13 06:00:00", tz="UTC")
FORWARD_START = HISTORICAL_CUTOFF + pd.Timedelta(hours=1)

DATA_BUCKET = os.getenv("DATA_S3_BUCKET", "nyc-resilience-data")
HISTORICAL_CANONICAL_PREFIX = os.getenv(
    "FLOOD_HISTORICAL_CANONICAL_PREFIX",
    "mlops/flood/processed/canonical",
)
FORWARD_CANONICAL_PREFIX = os.getenv(
    "FLOOD_FORWARD_CANONICAL_PREFIX",
    "mlops/flood/processed/forward/canonical",
)

# Scientific response threshold used by the active model.
# 1 inch = 25.4 mm.
DEPTH_THRESHOLD_MM = 25.4
BIN_MINUTES = 5
BIN_FREQUENCY = "5min"
MIN_VALID_DEPTH_BINS_PER_HOUR = 9

# The existing historical and forward canonical parquet files retain this
# legacy source-column name. They are read-only and are NOT renamed in S3.
SOURCE_TARGET_COLUMN = "minutes_above_1p5_inch"

# Correct semantic target name used inside training and in newly written
# local evaluation artifacts.
TARGET_COLUMN = "minutes_above_1inch"

FEATURES = [
    "precip_current_hour_mm",
    "precip_previous_6h_mm",
    "daily_total_precip_mm",
]

RAIN_ZERO_TOLERANCE_MM = 1e-6

K_NEIGHBORS = 4
HIDDEN_SIZE = 16
DROPOUT = 0.10
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 8
PATIENCE = 2
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
GRADIENT_CLIP_NORM = 5.0

PREDICTED_DURATION_THRESHOLD_MINUTES = 1.0
MATCH_WINDOW_HOURS = 24

# StandardScaler inverse_transform can return tiny floating-point residues
# around zero (for example 1e-15). These are scientifically zero minutes and
# must not be interpreted as flood events.
EVENT_ZERO_TOLERANCE_MINUTES = 1e-6

GCN_OUTPUT_DIR = FLOOD_ARTIFACT_DIR / "gcn"
GCN_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = GCN_OUTPUT_DIR / "duration_response_gcn_precipitation_only.pt"
SCALERS_PATH = GCN_OUTPUT_DIR / "scalers.pkl"
DIRTY_DIAGNOSTICS_PATH = GCN_OUTPUT_DIR / "dirty_sensor_day_diagnostics.parquet"
TEST_PREDICTIONS_PATH = GCN_OUTPUT_DIR / "test_predictions.parquet"
METRICS_PATH = GCN_OUTPUT_DIR / "metrics.csv"
SENSOR_METRICS_PATH = GCN_OUTPUT_DIR / "sensor_metrics.csv"
EVENT_METRICS_PATH = GCN_OUTPUT_DIR / "sensor_24h_event_metrics.csv"
TRAINING_HISTORY_PATH = GCN_OUTPUT_DIR / "training_history.csv"
NODE_TABLE_PATH = GCN_OUTPUT_DIR / "node_table.parquet"


@dataclass(frozen=True)
class GCNTrainingConfig:
    seed: int = SEED
    depth_threshold_mm: float = DEPTH_THRESHOLD_MM
    bin_minutes: int = BIN_MINUTES
    minimum_valid_depth_bins_per_hour: int = MIN_VALID_DEPTH_BINS_PER_HOUR
    k_neighbors: int = K_NEIGHBORS
    hidden_size: int = HIDDEN_SIZE
    dropout: float = DROPOUT
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    max_epochs: int = MAX_EPOCHS
    patience: int = PATIENCE
    train_fraction: float = TRAIN_FRACTION
    validation_fraction: float = VALIDATION_FRACTION
    gradient_clip_norm: float = GRADIENT_CLIP_NORM
    predicted_duration_threshold_minutes: float = PREDICTED_DURATION_THRESHOLD_MINUTES
    match_window_hours: int = MATCH_WINDOW_HOURS


DEFAULT_GCN_CONFIG = GCNTrainingConfig()

REQUIRED_TRAINING_COLUMNS = {
    SENSOR_ID,
    TIME_COLUMN,
    MAX_DEPTH_COLUMN,
    RESPONSE_OBSERVED_COLUMN,
    SOURCE_TARGET_COLUMN,
    *FEATURES,
    "sensor_lat",
    "sensor_lon",
}


# ============================================================
# S3 LOADING
# ============================================================


def s3_client():
    return boto3.client("s3")


def list_parquet_keys(prefix: str) -> list[str]:
    client = s3_client()
    paginator = client.get_paginator("list_objects_v2")
    keys: list[str] = []
    for page in paginator.paginate(
        Bucket=DATA_BUCKET,
        Prefix=prefix.rstrip("/") + "/",
    ):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/part.parquet"):
                keys.append(key)
    return sorted(keys)


def read_parquet_object(key: str) -> pd.DataFrame:
    response = s3_client().get_object(Bucket=DATA_BUCKET, Key=key)
    return pd.read_parquet(io.BytesIO(response["Body"].read()))


def normalize_training_frame(df: pd.DataFrame, *, source_name: str) -> pd.DataFrame:
    """Normalize one canonical source without modifying the S3 schema.

    Canonical parquet files physically contain SOURCE_TARGET_COLUMN, whose
    legacy name says 1.5 inches even though the active response threshold is
    1 inch (25.4 mm).  We copy that source column to the correctly named
    in-memory TARGET_COLUMN and use TARGET_COLUMN everywhere downstream.
    """

    missing = REQUIRED_TRAINING_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"{source_name} canonical dataset is missing required columns: "
            + ", ".join(sorted(missing))
        )

    result = df.copy()

    # Correct semantic alias used only in memory/local training outputs.
    result[TARGET_COLUMN] = result[SOURCE_TARGET_COLUMN]

    result[SENSOR_ID] = result[SENSOR_ID].astype(str)
    result[TIME_COLUMN] = pd.to_datetime(
        result[TIME_COLUMN],
        utc=True,
        errors="coerce",
    )

    for column in FEATURES + [
        MAX_DEPTH_COLUMN,
        TARGET_COLUMN,
        "sensor_lat",
        "sensor_lon",
    ]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    result[RESPONSE_OBSERVED_COLUMN] = (
        result[RESPONSE_OBSERVED_COLUMN]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )

    result = result.dropna(
        subset=[SENSOR_ID, TIME_COLUMN, "sensor_lat", "sensor_lon"]
    )
    result[DATA_SOURCE_COLUMN] = source_name
    return result


def load_historical_training_features() -> pd.DataFrame:
    keys = list_parquet_keys(HISTORICAL_CANONICAL_PREFIX)
    if not keys:
        raise RuntimeError(
            "No historical canonical partitions found under "
            f"s3://{DATA_BUCKET}/{HISTORICAL_CANONICAL_PREFIX}/"
        )

    print(f"Historical canonical partitions found: {len(keys):,}")
    pieces = []
    for index, key in enumerate(keys, start=1):
        if index == 1 or index == len(keys) or index % 12 == 0:
            print(f"    loading historical partition {index:,}/{len(keys):,}")
        pieces.append(read_parquet_object(key))

    historical = normalize_training_frame(
        pd.concat(pieces, ignore_index=True, sort=False),
        source_name="historical",
    )

    beyond = historical[historical[TIME_COLUMN] > HISTORICAL_CUTOFF]
    if not beyond.empty:
        raise ValueError(
            f"Historical canonical data contains {len(beyond):,} rows after "
            f"{HISTORICAL_CUTOFF}."
        )

    historical = historical[historical[TIME_COLUMN] <= HISTORICAL_CUTOFF].copy()

    duplicates = historical.duplicated([SENSOR_ID, TIME_COLUMN]).sum()
    if duplicates:
        raise ValueError(
            f"Historical canonical data contains {duplicates:,} duplicate sensor-hours."
        )

    return historical.sort_values([TIME_COLUMN, SENSOR_ID]).reset_index(drop=True)


def load_forward_training_features() -> pd.DataFrame:
    keys = list_parquet_keys(FORWARD_CANONICAL_PREFIX)
    if not keys:
        print("No forward canonical partitions found.")
        return pd.DataFrame()

    print(f"Forward canonical partitions found: {len(keys):,}")
    pieces = []
    for index, key in enumerate(keys, start=1):
        if index == 1 or index == len(keys) or index % 25 == 0:
            print(f"    loading forward partition {index:,}/{len(keys):,}")
        pieces.append(read_parquet_object(key))

    forward = normalize_training_frame(
        pd.concat(pieces, ignore_index=True, sort=False),
        source_name="forward",
    )

    overlap = forward[forward[TIME_COLUMN] <= HISTORICAL_CUTOFF]
    if not overlap.empty:
        raise ValueError(
            f"Forward canonical data contains {len(overlap):,} rows in the historical period."
        )

    if forward[TIME_COLUMN].min() < FORWARD_START:
        raise ValueError(
            f"Forward canonical data begins before expected boundary {FORWARD_START}."
        )

    duplicates = forward.duplicated([SENSOR_ID, TIME_COLUMN]).sum()
    if duplicates:
        raise ValueError(
            f"Forward canonical data contains {duplicates:,} duplicate sensor-hours."
        )

    return forward.sort_values([TIME_COLUMN, SENSOR_ID]).reset_index(drop=True)


def combine_training_sources(
    historical: pd.DataFrame,
    forward: pd.DataFrame,
) -> pd.DataFrame:
    combined = historical.copy() if forward.empty else pd.concat(
        [historical, forward], ignore_index=True, sort=False
    )

    duplicate_mask = combined.duplicated([SENSOR_ID, TIME_COLUMN], keep=False)
    if duplicate_mask.any():
        raise ValueError(
            "Historical and forward canonical sources overlap or contain duplicate "
            f"sensor-hours: {int(duplicate_mask.sum()):,} rows."
        )

    return combined.sort_values([TIME_COLUMN, SENSOR_ID]).reset_index(drop=True)


def load_training_features() -> pd.DataFrame:
    print("Loading immutable historical canonical data...")
    historical = load_historical_training_features()
    print(f"Historical rows: {len(historical):,}")
    print(f"Historical sensors: {historical[SENSOR_ID].nunique():,}")
    print(
        f"Historical range: {historical[TIME_COLUMN].min()} -> "
        f"{historical[TIME_COLUMN].max()}"
    )

    print("\nLoading append-only forward canonical data...")
    forward = load_forward_training_features()
    if not forward.empty:
        print(f"Forward rows: {len(forward):,}")
        print(f"Forward sensors: {forward[SENSOR_ID].nunique():,}")
        print(
            f"Forward range: {forward[TIME_COLUMN].min()} -> "
            f"{forward[TIME_COLUMN].max()}"
        )

    combined = combine_training_sources(historical, forward)
    print("\nLogical training view:")
    print(f"    rows: {len(combined):,}")
    print(f"    sensors: {combined[SENSOR_ID].nunique():,}")
    print(f"    range: {combined[TIME_COLUMN].min()} -> {combined[TIME_COLUMN].max()}")
    return combined


# ============================================================
# NOTEBOOK DIRTY SENSOR-DAY FILTER
# ============================================================


def remove_dirty_sensor_days(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    result = df.copy()
    result["date"] = result[TIME_COLUMN].dt.normalize()

    sensor_day_summary = (
        result.groupby([SENSOR_ID, "date"], as_index=False)
        .agg(
            daily_total_precip_mm=("daily_total_precip_mm", "max"),
            maximum_depth_that_day_mm=(MAX_DEPTH_COLUMN, "max"),
            observed_duration_hours=(TARGET_COLUMN, lambda s: int(s.notna().sum())),
        )
    )

    sensor_day_summary["zero_rain_day"] = (
        sensor_day_summary["daily_total_precip_mm"].notna()
        & (
            sensor_day_summary["daily_total_precip_mm"].abs()
            <= RAIN_ZERO_TOLERANCE_MM
        )
    )
    sensor_day_summary["positive_depth_that_day"] = (
        sensor_day_summary["maximum_depth_that_day_mm"] > 0
    )
    sensor_day_summary["remove_sensor_day"] = (
        sensor_day_summary["zero_rain_day"]
        & sensor_day_summary["positive_depth_that_day"]
    )

    days_to_remove = sensor_day_summary.loc[
        sensor_day_summary["remove_sensor_day"], [SENSOR_ID, "date"]
    ].copy()

    print("\nDIRTY SENSOR-DAY FILTER")
    print("-----------------------")
    print(f"Total sensor-days: {len(sensor_day_summary):,}")
    print(f"Dirty sensor-days: {len(days_to_remove):,}")

    result = result.merge(
        days_to_remove.assign(remove_sensor_day=True),
        on=[SENSOR_ID, "date"],
        how="left",
    )
    result["remove_sensor_day"] = (
        result["remove_sensor_day"]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    removed_rows = int(result["remove_sensor_day"].sum())

    result = (
        result.loc[~result["remove_sensor_day"]]
        .drop(columns=["remove_sensor_day"])
        .reset_index(drop=True)
    )

    print(f"Hourly rows removed: {removed_rows:,}")
    print(f"Rows retained: {len(result):,}")
    print(f"Sensors retained: {result[SENSOR_ID].nunique():,}")
    print(f"Observed targets retained: {result[TARGET_COLUMN].notna().sum():,}")

    post = (
        result.groupby([SENSOR_ID, "date"], as_index=False)
        .agg(
            daily_total_precip_mm=("daily_total_precip_mm", "max"),
            maximum_depth_that_day_mm=(MAX_DEPTH_COLUMN, "max"),
        )
    )
    remaining_problem_days = (
        post["daily_total_precip_mm"].notna()
        & (post["daily_total_precip_mm"].abs() <= RAIN_ZERO_TOLERANCE_MM)
        & (post["maximum_depth_that_day_mm"] > 0)
    )
    if remaining_problem_days.any():
        raise RuntimeError("Dirty-data filtering failed.")

    return result, sensor_day_summary


# ============================================================
# GEOGRAPHIC GRAPH
# ============================================================


def build_geographic_graph(df: pd.DataFrame):
    node_table = (
        df[[SENSOR_ID, "sensor_lat", "sensor_lon"]]
        .drop_duplicates(subset=[SENSOR_ID])
        .sort_values(SENSOR_ID)
        .reset_index(drop=True)
    )

    node_ids = node_table[SENSOR_ID].tolist()
    node_to_index = {node_id: i for i, node_id in enumerate(node_ids)}
    n_nodes = len(node_ids)
    if n_nodes == 0:
        raise RuntimeError("No graph nodes available after filtering.")

    coordinates = np.radians(node_table[["sensor_lat", "sensor_lon"]].to_numpy())
    knn = NearestNeighbors(
        n_neighbors=min(K_NEIGHBORS + 1, n_nodes),
        metric="haversine",
    ).fit(coordinates)
    _, neighbors = knn.kneighbors(coordinates)

    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    for source in range(n_nodes):
        for position in range(1, neighbors.shape[1]):
            target = int(neighbors[source, position])
            adjacency[source, target] = 1.0
            adjacency[target, source] = 1.0

    adjacency += np.eye(n_nodes, dtype=np.float32)
    base_adjacency = torch.tensor(adjacency, dtype=torch.float32, device=DEVICE)

    print("\nGEOGRAPHIC GRAPH")
    print("----------------")
    print(f"Graph nodes: {n_nodes:,}")
    print(
        "Graph edges/nonzeros including self-loops: "
        f"{int((adjacency > 0).sum()):,}"
    )

    return node_table, node_ids, node_to_index, base_adjacency


# ============================================================
# CHRONOLOGICAL SPLIT + TRAINING-ONLY SCALING
# ============================================================


def scale_training_data(df: pd.DataFrame):
    result = df.copy()
    hours = result[TIME_COLUMN].drop_duplicates().sort_values().reset_index(drop=True)
    n_hours = len(hours)
    if n_hours < 3:
        raise RuntimeError("Not enough unique hours for train/validation/test split.")

    train_end_pos = int(n_hours * TRAIN_FRACTION)
    validation_end_pos = int(n_hours * (TRAIN_FRACTION + VALIDATION_FRACTION))

    if train_end_pos <= 0 or validation_end_pos <= train_end_pos or validation_end_pos >= n_hours:
        raise RuntimeError("Chronological split positions are invalid.")

    train_end_hour = hours.iloc[train_end_pos - 1]
    validation_end_hour = hours.iloc[validation_end_pos - 1]
    training_rows = result.loc[result[TIME_COLUMN] <= train_end_hour].copy()

    feature_training = training_rows[FEATURES].replace([np.inf, -np.inf], np.nan)
    feature_fit_rows = feature_training.dropna()
    if feature_fit_rows.empty:
        raise RuntimeError("No complete training predictor rows.")

    feature_scaler = StandardScaler()
    feature_scaler.fit(feature_fit_rows)

    target_training = training_rows[[TARGET_COLUMN]].dropna()
    if target_training.empty:
        raise RuntimeError("No observed training targets.")

    target_scaler = StandardScaler()
    target_scaler.fit(target_training)

    scaled_feature_names = [f"{column}_scaled" for column in FEATURES]
    for column in scaled_feature_names:
        result[column] = np.nan

    clean_features = result[FEATURES].replace([np.inf, -np.inf], np.nan)
    complete_feature_mask = clean_features.notna().all(axis=1)
    transformed = feature_scaler.transform(
        result.loc[complete_feature_mask, FEATURES]
    ).astype(np.float32)

    for index, column in enumerate(scaled_feature_names):
        result.loc[complete_feature_mask, column] = transformed[:, index]

    result["target_scaled"] = np.nan
    target_observed = result[TARGET_COLUMN].notna()
    result.loc[target_observed, "target_scaled"] = (
        target_scaler.transform(
            result.loc[target_observed, [TARGET_COLUMN]]
        )
        .reshape(-1)
        .astype(np.float32)
    )

    print("\nCHRONOLOGICAL SPLIT")
    print("-------------------")
    print(f"Unique hours: {n_hours:,}")
    print(f"Train through: {train_end_hour}")
    print(f"Validation through: {validation_end_hour}")
    print(f"Test begins after: {validation_end_hour}")

    return (
        result,
        hours,
        train_end_pos,
        validation_end_pos,
        feature_scaler,
        target_scaler,
        scaled_feature_names,
    )


# ============================================================
# SNAPSHOT TENSORS
# ============================================================


def build_snapshot_arrays(
    df: pd.DataFrame,
    hours: pd.Series,
    node_to_index: dict[str, int],
    scaled_feature_names: list[str],
):
    hour_to_index = {hour: i for i, hour in enumerate(hours)}
    n_hours = len(hours)
    n_nodes = len(node_to_index)
    n_features = len(FEATURES)

    x = np.zeros((n_hours, n_nodes, n_features), dtype=np.float32)
    y = np.zeros((n_hours, n_nodes), dtype=np.float32)
    y_mask = np.zeros((n_hours, n_nodes), dtype=np.float32)
    active_mask = np.zeros((n_hours, n_nodes), dtype=np.float32)

    for row in df.itertuples(index=False):
        ti = hour_to_index[getattr(row, TIME_COLUMN)]
        ni = node_to_index[getattr(row, SENSOR_ID)]

        feature_values = np.array(
            [getattr(row, column) for column in scaled_feature_names],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(feature_values)):
            continue

        x[ti, ni, :] = feature_values
        active_mask[ti, ni] = 1.0

        target_value = getattr(row, "target_scaled")
        if pd.notna(target_value):
            y[ti, ni] = float(target_value)
            y_mask[ti, ni] = 1.0

    active_nodes_per_hour = active_mask.sum(axis=1)
    print("\nGRAPH SNAPSHOTS")
    print("---------------")
    print(f"X shape: {x.shape}")
    print(f"Active node-hours: {int(active_mask.sum()):,}")
    print(f"Observed target node-hours: {int(y_mask.sum()):,}")
    print(f"Median active sensors/hour: {float(np.median(active_nodes_per_hour)):.1f}")

    return x, y, y_mask, active_mask


class SnapshotDataset(Dataset):
    def __init__(self, x, y, y_mask, active_mask, start_index: int, end_index: int):
        self.x = x
        self.y = y
        self.y_mask = y_mask
        self.active_mask = active_mask
        self.indices = np.arange(start_index, end_index, dtype=int)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        idx = int(self.indices[item])
        return {
            "x": torch.tensor(self.x[idx], dtype=torch.float32),
            "y": torch.tensor(self.y[idx], dtype=torch.float32),
            "target_mask": torch.tensor(self.y_mask[idx], dtype=torch.float32),
            "active_mask": torch.tensor(self.active_mask[idx], dtype=torch.float32),
            "hour_index": idx,
        }


# ============================================================
# NOTEBOOK GCN
# ============================================================


def create_active_adjacency(active_mask: torch.Tensor, base_adjacency: torch.Tensor):
    pair_mask = active_mask.unsqueeze(2) * active_mask.unsqueeze(1)
    adjacency = base_adjacency.unsqueeze(0) * pair_mask
    degree = adjacency.sum(dim=-1)
    degree_inverse_sqrt = torch.where(
        degree > 0,
        torch.rsqrt(degree.clamp_min(1e-8)),
        torch.zeros_like(degree),
    )
    return (
        degree_inverse_sqrt.unsqueeze(2)
        * adjacency
        * degree_inverse_sqrt.unsqueeze(1)
    )


class GraphConv(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, adjacency, active_mask):
        propagated = torch.einsum("bij,bjf->bif", adjacency, x)
        output = self.linear(propagated)
        return output * active_mask.unsqueeze(-1)


class DepthGCN(nn.Module):
    def __init__(self, n_features: int, base_adjacency: torch.Tensor):
        super().__init__()
        self.base_adjacency = base_adjacency
        self.gcn1 = GraphConv(n_features, HIDDEN_SIZE)
        self.gcn2 = GraphConv(HIDDEN_SIZE, HIDDEN_SIZE)
        self.dropout = nn.Dropout(DROPOUT)
        self.output = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x, active_mask):
        adjacency = create_active_adjacency(active_mask, self.base_adjacency)
        x = torch.relu(self.gcn1(x, adjacency, active_mask))
        x = self.dropout(x)
        x = torch.relu(self.gcn2(x, adjacency, active_mask))
        x = self.dropout(x)
        prediction = self.output(x).squeeze(-1)
        return prediction * active_mask


def masked_mse(prediction, target, target_mask):
    squared_error = (prediction - target) ** 2
    return (squared_error * target_mask).sum() / target_mask.sum().clamp_min(1.0)


# ============================================================
# TRAIN
# ============================================================


def train_model(model, train_loader, validation_loader):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_state = None
    best_validation_loss = math.inf
    wait = 0
    history = []

    print("\nTRAINING")
    print("--------")

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            xb = batch["x"].to(DEVICE)
            yb = batch["y"].to(DEVICE)
            target_mask = batch["target_mask"].to(DEVICE)
            active_mask = batch["active_mask"].to(DEVICE)

            if target_mask.sum().item() == 0:
                continue

            optimizer.zero_grad()
            prediction = model(xb, active_mask)
            loss = masked_mse(prediction, yb, target_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
            optimizer.step()
            train_losses.append(float(loss.item()))

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for batch in validation_loader:
                xb = batch["x"].to(DEVICE)
                yb = batch["y"].to(DEVICE)
                target_mask = batch["target_mask"].to(DEVICE)
                active_mask = batch["active_mask"].to(DEVICE)

                if target_mask.sum().item() == 0:
                    continue

                prediction = model(xb, active_mask)
                validation_losses.append(
                    float(masked_mse(prediction, yb, target_mask).item())
                )

        if not train_losses:
            raise RuntimeError("No valid training batches contained observed targets.")
        if not validation_losses:
            raise RuntimeError("No valid validation batches contained observed targets.")

        train_loss = float(np.mean(train_losses))
        validation_loss = float(np.mean(validation_losses))
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )

        print(
            f"Epoch {epoch:02d} | train={train_loss:.5f} | "
            f"validation={validation_loss:.5f}"
        )

        if validation_loss < best_validation_loss - 1e-5:
            best_validation_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        if wait >= PATIENCE:
            print("Early stopping.")
            break

    if best_state is None:
        raise RuntimeError("No valid model state was saved.")

    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_validation_loss


# ============================================================
# QUANTITATIVE TEST EVALUATION
# ============================================================


def evaluate_test(
    model,
    test_loader,
    hours,
    node_ids,
    target_scaler,
):
    model.eval()
    records = []

    with torch.no_grad():
        for batch in test_loader:
            xb = batch["x"].to(DEVICE)
            active_mask = batch["active_mask"].to(DEVICE)
            prediction_scaled = model(xb, active_mask).cpu().numpy()

            y_scaled = batch["y"].numpy()
            target_mask = batch["target_mask"].numpy()
            hour_indices = batch["hour_index"].numpy()

            for b in range(prediction_scaled.shape[0]):
                hour = hours.iloc[int(hour_indices[b])]
                observed_nodes = np.where(target_mask[b] == 1)[0]
                if len(observed_nodes) == 0:
                    continue

                actual_minutes = target_scaler.inverse_transform(
                    y_scaled[b, observed_nodes].reshape(-1, 1)
                ).reshape(-1)
                actual_minutes = np.where(
                    np.abs(actual_minutes) <= EVENT_ZERO_TOLERANCE_MINUTES,
                    0.0,
                    actual_minutes,
                )
                predicted_minutes = target_scaler.inverse_transform(
                    prediction_scaled[b, observed_nodes].reshape(-1, 1)
                ).reshape(-1)
                predicted_minutes = np.clip(predicted_minutes, 0.0, 60.0)

                for node_index, actual_value, predicted_value in zip(
                    observed_nodes,
                    actual_minutes,
                    predicted_minutes,
                ):
                    records.append(
                        {
                            "hour": hour,
                            SENSOR_ID: node_ids[int(node_index)],
                            "actual_minutes_above_1inch": float(actual_value),
                            "predicted_minutes_above_1inch": float(predicted_value),
                        }
                    )

    test_predictions = pd.DataFrame(records)
    if test_predictions.empty:
        raise RuntimeError("No test predictions were produced.")

    actual = test_predictions["actual_minutes_above_1inch"]
    predicted = test_predictions["predicted_minutes_above_1inch"]

    test_r2 = r2_score(actual, predicted)
    test_rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
    test_mae = mean_absolute_error(actual, predicted)

    metrics = pd.DataFrame(
        {
            "metric": ["pooled_r2", "rmse_minutes", "mae_minutes", "test_node_hours"],
            "value": [test_r2, test_rmse, test_mae, len(test_predictions)],
        }
    )

    print("\nTEST METRICS")
    print("------------")
    print(f"Pooled R²: {test_r2:.4f}")
    print(f"RMSE:      {test_rmse:.4f} minutes")
    print(f"MAE:       {test_mae:.4f} minutes")
    print(f"Node-hours:{len(test_predictions):,}")

    return test_predictions, metrics


def calculate_sensor_metrics(test_predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sensor_id, group in test_predictions.groupby(SENSOR_ID):
        actual = group["actual_minutes_above_1inch"]
        predicted = group["predicted_minutes_above_1inch"]
        if len(group) < 2 or actual.nunique() < 2:
            continue
        rows.append(
            {
                SENSOR_ID: sensor_id,
                "r2": r2_score(actual, predicted),
                "rmse_minutes": float(np.sqrt(mean_squared_error(actual, predicted))),
                "mae_minutes": mean_absolute_error(actual, predicted),
                "hours": len(group),
                "hours_with_positive_duration": int(
                    (actual > EVENT_ZERO_TOLERANCE_MINUTES).sum()
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                SENSOR_ID,
                "r2",
                "rmse_minutes",
                "mae_minutes",
                "hours",
                "hours_with_positive_duration",
            ]
        )

    sensor_metrics = pd.DataFrame(rows).sort_values("r2", ascending=False).reset_index(drop=True)
    print("\nPER-SENSOR METRICS")
    print("------------------")
    print(f"Sensors with valid R²: {len(sensor_metrics):,}")
    print(f"Mean sensor R²: {sensor_metrics['r2'].mean():.4f}")
    print(f"Median sensor R²: {sensor_metrics['r2'].median():.4f}")
    print(f"Sensors with R² > 0.50: {(sensor_metrics['r2'] > 0.50).sum():,}")
    print("\nTop sensors:")
    print(sensor_metrics.head(20).to_string(index=False))
    return sensor_metrics


# ============================================================
# NOTEBOOK ±24-HOUR EVENT EVALUATION
# ============================================================


def evaluate_events_24h(test_predictions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    event_source = test_predictions[
        [
            "hour",
            SENSOR_ID,
            "actual_minutes_above_1inch",
            "predicted_minutes_above_1inch",
        ]
    ].rename(
        columns={
            "actual_minutes_above_1inch": "actual_minutes",
            "predicted_minutes_above_1inch": "predicted_minutes",
        }
    ).copy()

    event_source["hour"] = pd.to_datetime(event_source["hour"], utc=True)

    # Remove floating-point residue around zero before event classification.
    event_source["actual_minutes"] = event_source["actual_minutes"].where(
        event_source["actual_minutes"].abs() > EVENT_ZERO_TOLERANCE_MINUTES,
        0.0,
    )
    event_source["predicted_minutes"] = event_source["predicted_minutes"].where(
        event_source["predicted_minutes"].abs() > EVENT_ZERO_TOLERANCE_MINUTES,
        0.0,
    )

    # Actual event: positive 1-inch response duration, excluding numerical noise.
    event_source["actual_event"] = (
        event_source["actual_minutes"] > EVENT_ZERO_TOLERANCE_MINUTES
    ).astype(int)

    # Notebook event-detection rule: prediction must exceed 1 minute.
    event_source["predicted_event"] = (
        event_source["predicted_minutes"] > PREDICTED_DURATION_THRESHOLD_MINUTES
    ).astype(int)

    evaluation_rows = []
    match_window = pd.Timedelta(hours=MATCH_WINDOW_HOURS)

    for sensor_id, group in event_source.groupby(SENSOR_ID):
        group = group.sort_values("hour").copy()
        actual_times = group.loc[group["actual_event"] == 1, "hour"].sort_values().tolist()
        predicted_times = group.loc[group["predicted_event"] == 1, "hour"].sort_values().tolist()

        actual_matches = [
            int(any(abs(predicted_time - actual_time) <= match_window for predicted_time in predicted_times))
            for actual_time in actual_times
        ]
        prediction_matches = [
            int(any(abs(actual_time - predicted_time) <= match_window for actual_time in actual_times))
            for predicted_time in predicted_times
        ]

        true_positive_predictions = int(sum(prediction_matches))
        false_positive_predictions = len(prediction_matches) - true_positive_predictions
        captured_actual_events = int(sum(actual_matches))
        missed_actual_events = len(actual_matches) - captured_actual_events

        precision = true_positive_predictions / max(len(predicted_times), 1)
        recall = captured_actual_events / max(len(actual_times), 1)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

        evaluation_rows.append(
            {
                SENSOR_ID: sensor_id,
                "actual_flood_events": len(actual_times),
                "predicted_flood_events": len(predicted_times),
                "captured_actual_events": captured_actual_events,
                "missed_actual_events": missed_actual_events,
                "matched_predictions": true_positive_predictions,
                "false_alarm_predictions": false_positive_predictions,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    sensor_event_metrics = pd.DataFrame(evaluation_rows)
    sensor_event_metrics = sensor_event_metrics.sort_values(
        ["f1", "recall", "precision"], ascending=False
    ).reset_index(drop=True)

    total_actual = int(sensor_event_metrics["actual_flood_events"].sum())
    total_predicted = int(sensor_event_metrics["predicted_flood_events"].sum())
    total_captured = int(sensor_event_metrics["captured_actual_events"].sum())
    total_missed = int(sensor_event_metrics["missed_actual_events"].sum())
    total_matched_predictions = int(sensor_event_metrics["matched_predictions"].sum())
    total_false_alarms = int(sensor_event_metrics["false_alarm_predictions"].sum())

    overall_precision = total_matched_predictions / max(total_predicted, 1)
    overall_recall = total_captured / max(total_actual, 1)
    overall_f1 = (
        2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        if overall_precision + overall_recall > 0
        else 0.0
    )

    overall = {
        "actual_flood_events": total_actual,
        "predicted_flood_events": total_predicted,
        "captured_actual_events": total_captured,
        "missed_actual_events": total_missed,
        "matched_predictions": total_matched_predictions,
        "false_alarm_predictions": total_false_alarms,
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
    }

    print("\n±24-HOUR EVENT DETECTION")
    print("------------------------")
    print(f"Actual flood events: {total_actual:,}")
    print(f"Predicted flood events: {total_predicted:,}")
    print(f"Actual events captured: {total_captured:,}")
    print(f"Actual events missed: {total_missed:,}")
    print(f"Matched predictions: {total_matched_predictions:,}")
    print(f"False-alarm predictions: {total_false_alarms:,}")
    print(f"Precision (pooled/micro): {overall_precision:.4f}")
    print(f"Recall (pooled/micro):    {overall_recall:.4f}")
    print(f"F1 (pooled/micro):        {overall_f1:.4f}")

    macro_precision = float(sensor_event_metrics["precision"].mean())
    macro_recall = float(sensor_event_metrics["recall"].mean())
    macro_f1 = float(sensor_event_metrics["f1"].mean())

    print(f"Precision (sensor mean):  {macro_precision:.4f}")
    print(f"Recall (sensor mean):     {macro_recall:.4f}")
    print(f"F1 (sensor mean):         {macro_f1:.4f}")

    print("\nPER-SENSOR ±24-HOUR EVENT METRICS")
    print("---------------------------------")
    display_columns = [
        SENSOR_ID,
        "actual_flood_events",
        "predicted_flood_events",
        "captured_actual_events",
        "missed_actual_events",
        "matched_predictions",
        "false_alarm_predictions",
        "precision",
        "recall",
        "f1",
    ]
    print(
        sensor_event_metrics[display_columns].to_string(
            index=False,
            formatters={
                "precision": lambda value: f"{value:.4f}",
                "recall": lambda value: f"{value:.4f}",
                "f1": lambda value: f"{value:.4f}",
            },
        )
    )

    return sensor_event_metrics, overall


# ============================================================
# SAVE ARTIFACTS
# ============================================================


def save_outputs(
    model,
    node_table,
    node_ids,
    feature_scaler,
    target_scaler,
    sensor_day_summary,
    test_predictions,
    metrics,
    sensor_metrics,
    sensor_event_metrics,
    event_overall,
    training_history,
    best_validation_loss,
    train_end_hour,
    validation_end_hour,
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "node_ids": node_ids,
            "features": FEATURES,
            "target": TARGET_COLUMN,
            "source_target": SOURCE_TARGET_COLUMN,
            "depth_threshold_mm": DEPTH_THRESHOLD_MM,
            "bin_minutes": BIN_MINUTES,
            "minimum_valid_depth_bins_per_hour": MIN_VALID_DEPTH_BINS_PER_HOUR,
            "filter_rule": (
                "Remove entire sensor-day when daily_total_precip_mm == 0 and "
                "maximum sensor depth that day > 0 mm."
            ),
            "historical_cutoff": str(HISTORICAL_CUTOFF),
            "forward_start": str(FORWARD_START),
            "train_end_hour": str(train_end_hour),
            "validation_end_hour": str(validation_end_hour),
            "best_validation_loss": float(best_validation_loss),
            "training_config": asdict(DEFAULT_GCN_CONFIG),
            "event_metrics_overall": event_overall,
        },
        MODEL_PATH,
    )

    with open(SCALERS_PATH, "wb") as handle:
        pickle.dump(
            {
                "feature_scaler": feature_scaler,
                "target_scaler": target_scaler,
                "features": FEATURES,
                "target": TARGET_COLUMN,
            },
            handle,
        )

    node_table.to_parquet(NODE_TABLE_PATH, index=False)
    sensor_day_summary.to_parquet(DIRTY_DIAGNOSTICS_PATH, index=False)
    test_predictions.to_parquet(TEST_PREDICTIONS_PATH, index=False)
    metrics.to_csv(METRICS_PATH, index=False)
    sensor_metrics.to_csv(SENSOR_METRICS_PATH, index=False)
    sensor_event_metrics.to_csv(EVENT_METRICS_PATH, index=False)
    training_history.to_csv(TRAINING_HISTORY_PATH, index=False)

    print("\nSAVED OUTPUTS")
    print("-------------")
    print(GCN_OUTPUT_DIR)


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print("NOTEBOOK-FAITHFUL GCN TRAINING")
    print("==============================")
    print(f"Device: {DEVICE}")
    print(f"Depth threshold: {DEPTH_THRESHOLD_MM} mm (1 inch)")
    print(f"Canonical source target: {SOURCE_TARGET_COLUMN} [legacy column name]")
    print(f"Training target: {TARGET_COLUMN}")
    print(f"Features: {FEATURES}")

    # --------------------------------------------------------
    # Load canonical historical + forward training view
    # --------------------------------------------------------
    df = load_training_features()

    # The notebook keeps predictor rows with unobserved target values and
    # represents target availability using Y_MASK later. Do not drop those rows.
    df, sensor_day_summary = remove_dirty_sensor_days(df)

    # --------------------------------------------------------
    # Graph
    # --------------------------------------------------------
    node_table, node_ids, node_to_index, base_adjacency = build_geographic_graph(df)

    # --------------------------------------------------------
    # Chronological split and training-only scaling
    # --------------------------------------------------------
    (
        df,
        hours,
        train_end_pos,
        validation_end_pos,
        feature_scaler,
        target_scaler,
        scaled_feature_names,
    ) = scale_training_data(df)

    train_end_hour = hours.iloc[train_end_pos - 1]
    validation_end_hour = hours.iloc[validation_end_pos - 1]

    # --------------------------------------------------------
    # Hourly snapshots
    # --------------------------------------------------------
    x, y, y_mask, active_mask = build_snapshot_arrays(
        df,
        hours,
        node_to_index,
        scaled_feature_names,
    )

    train_dataset = SnapshotDataset(
        x, y, y_mask, active_mask, 0, train_end_pos
    )
    validation_dataset = SnapshotDataset(
        x, y, y_mask, active_mask, train_end_pos, validation_end_pos
    )
    test_dataset = SnapshotDataset(
        x, y, y_mask, active_mask, validation_end_pos, len(hours)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = DepthGCN(len(FEATURES), base_adjacency).to(DEVICE)

    print("\nMODEL")
    print("-----")
    print(f"Train snapshots: {len(train_dataset):,}")
    print(f"Validation snapshots: {len(validation_dataset):,}")
    print(f"Test snapshots: {len(test_dataset):,}")
    print(
        "Trainable parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    model, training_history, best_validation_loss = train_model(
        model,
        train_loader,
        validation_loader,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    test_predictions, metrics = evaluate_test(
        model,
        test_loader,
        hours,
        node_ids,
        target_scaler,
    )
    sensor_metrics = calculate_sensor_metrics(test_predictions)
    sensor_event_metrics, event_overall = evaluate_events_24h(test_predictions)

    # --------------------------------------------------------
    # Save local artifacts only
    # --------------------------------------------------------
    save_outputs(
        model=model,
        node_table=node_table,
        node_ids=node_ids,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        sensor_day_summary=sensor_day_summary,
        test_predictions=test_predictions,
        metrics=metrics,
        sensor_metrics=sensor_metrics,
        sensor_event_metrics=sensor_event_metrics,
        event_overall=event_overall,
        training_history=training_history,
        best_validation_loss=best_validation_loss,
        train_end_hour=train_end_hour,
        validation_end_hour=validation_end_hour,
    )

    print("\nGCN TRAINING COMPLETE")


if __name__ == "__main__":
    main()