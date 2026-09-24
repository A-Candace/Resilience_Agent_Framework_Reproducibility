from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import mlops.flood.training.train_gcn as gcn
import mlops.flood.training.train_logistic as logistic


# ============================================================
# EXPERIMENT CONTRACT
# ============================================================

EXPERIMENT_NAME = "static_vs_centered_rolling24"

OUTPUT_DIR = (
    Path("artifacts")
    / "flood"
    / "experiments"
    / EXPERIMENT_NAME
)

REGISTRY_PATH = (
    Path("artifacts")
    / "flood"
    / "model_selection"
    / "eligible_sensor_model_registry.csv"
)

TOP_SENSORS = [
    "visually-endless-martin",
    "apparently-darling-gecko",
    "eagerly_militant_sloth",
    "happily-green-lion",
    "quarrelsomely_vacuous_starfish",
]

ROLLING_OFFSETS = list(range(-12, 12))

PRECIP_COLUMN = "precip_current_hour_mm"
STATIC_COLUMN = "daily_total_precip_mm"

ROLLING_COLUMN = "rolling_24h_precip_mm"

SENSOR_COLUMN = gcn.SENSOR_ID
TIME_COLUMN = gcn.TIME_COLUMN
TARGET_COLUMN = gcn.TARGET_COLUMN

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed() -> None:
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    gcn.set_random_seed(RANDOM_SEED)


# ============================================================
# LOAD BASELINE REGISTRY
# ============================================================

def load_established_baselines() -> pd.DataFrame:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(
            f"Model-selection registry not found: {REGISTRY_PATH}"
        )

    registry = pd.read_csv(REGISTRY_PATH)

    required = {
        "deployment_id",
        "selected_model",
        "selected_precision",
        "selected_recall",
        "selected_f1",
    }

    missing = required - set(registry.columns)

    if missing:
        raise RuntimeError(
            f"Registry is missing required columns: {sorted(missing)}"
        )

    baseline = registry[
        registry["deployment_id"].isin(TOP_SENSORS)
    ].copy()

    missing_sensors = set(TOP_SENSORS) - set(
        baseline["deployment_id"]
    )

    if missing_sensors:
        raise RuntimeError(
            "Selected sensors missing from registry: "
            f"{sorted(missing_sensors)}"
        )

    return baseline


# ============================================================
# CENTERED 24-HOUR FEATURE
# ============================================================

def add_centered_rolling24(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Construct:

        rolling_24h(H)
          = sum precipitation from H-12 through H+11

    The window is accepted only when all 24 expected hourly
    timestamps exist for that sensor.

    Missing hours are NOT treated as zero.
    """

    required = {
        SENSOR_COLUMN,
        TIME_COLUMN,
        PRECIP_COLUMN,
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            "Cannot construct rolling-24 feature. "
            f"Missing columns: {sorted(missing)}"
        )

    result = df.copy()

    result[TIME_COLUMN] = pd.to_datetime(
        result[TIME_COLUMN],
        utc=True,
        errors="coerce",
    )

    if result[TIME_COLUMN].isna().any():
        raise RuntimeError(
            "Invalid timestamps found while constructing rolling-24."
        )

    result[PRECIP_COLUMN] = pd.to_numeric(
        result[PRECIP_COLUMN],
        errors="coerce",
    )

    result = result.sort_values(
        [SENSOR_COLUMN, TIME_COLUMN]
    ).reset_index(drop=True)

    if result.duplicated(
        [SENSOR_COLUMN, TIME_COLUMN]
    ).any():
        raise RuntimeError(
            "Duplicate sensor/hour rows exist in training data."
        )

    groups = result.groupby(
        SENSOR_COLUMN,
        sort=False,
        group_keys=False,
    )

    precip_parts = []
    valid_parts = []

    for offset in ROLLING_OFFSETS:

        # We want the value at H + offset.
        # For a chronologically sorted Series:
        # shift(-offset) gives the row located at H + offset.
        shifted_precip = groups[PRECIP_COLUMN].shift(
            -offset
        )

        shifted_time = groups[TIME_COLUMN].shift(
            -offset
        )

        expected_time = (
            result[TIME_COLUMN]
            + pd.to_timedelta(offset, unit="h")
        )

        exact_hour = shifted_time.eq(expected_time)

        precip_parts.append(
            shifted_precip.where(exact_hour)
        )

        valid_parts.append(exact_hour)

    precip_matrix = pd.concat(
        precip_parts,
        axis=1,
    )

    valid_matrix = pd.concat(
        valid_parts,
        axis=1,
    )

    complete_window = (
        valid_matrix.all(axis=1)
        & precip_matrix.notna().all(axis=1)
    )

    result[ROLLING_COLUMN] = np.nan

    result.loc[
        complete_window,
        ROLLING_COLUMN,
    ] = precip_matrix.loc[
        complete_window
    ].sum(axis=1)

    result["rolling24_complete"] = complete_window

    return result


# ============================================================
# ORIGINAL SPLIT BOUNDARIES
# ============================================================

def determine_original_boundaries(
    df: pd.DataFrame,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Derive boundaries from the ORIGINAL cleaned training frame,
    before rolling-window rows are removed.

    This prevents the rolling feature from moving the historical
    train/validation/test boundaries.
    """

    hours = (
        df[TIME_COLUMN]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    n_hours = len(hours)

    train_end_pos = int(
        n_hours * gcn.TRAIN_FRACTION
    )

    validation_end_pos = int(
        n_hours
        * (
            gcn.TRAIN_FRACTION
            + gcn.VALIDATION_FRACTION
        )
    )

    if (
        train_end_pos <= 0
        or validation_end_pos <= train_end_pos
        or validation_end_pos >= n_hours
    ):
        raise RuntimeError(
            "Original chronological split is invalid."
        )

    train_end_hour = hours.iloc[
        train_end_pos - 1
    ]

    validation_end_hour = hours.iloc[
        validation_end_pos - 1
    ]

    return (
        train_end_hour,
        validation_end_hour,
    )


# ============================================================
# MATCHED DATASET
# ============================================================

def prepare_experiment_frames():
    print("Loading canonical MRMS training features...")

    raw = gcn.load_training_features()

    print(
        f"Raw rows: {len(raw):,}"
    )

    cleaned, dirty_summary = (
        gcn.remove_dirty_sensor_days(raw)
    )

    print(
        f"Rows after dirty-day exclusion: "
        f"{len(cleaned):,}"
    )

    (
        train_end_hour,
        validation_end_hour,
    ) = determine_original_boundaries(
        cleaned
    )

    print()
    print("ORIGINAL CHRONOLOGICAL BOUNDARIES")
    print("---------------------------------")
    print(
        f"Train through:      {train_end_hour}"
    )
    print(
        f"Validation through: "
        f"{validation_end_hour}"
    )
    print(
        "Test begins after:  "
        f"{validation_end_hour}"
    )

    engineered = add_centered_rolling24(
        cleaned
    )

    complete = engineered[
        engineered["rolling24_complete"]
    ].copy()

    dropped = len(engineered) - len(complete)

    print()
    print("ROLLING-24 AVAILABILITY")
    print("-----------------------")
    print(
        f"Candidate rows: {len(engineered):,}"
    )
    print(
        f"Complete H-12:H+11 rows: "
        f"{len(complete):,}"
    )
    print(
        f"Dropped incomplete rows: {dropped:,}"
    )

    static = complete.copy()

    rolling = complete.copy()

    rolling[STATIC_COLUMN] = rolling[
        ROLLING_COLUMN
    ]

    return (
        static,
        rolling,
        train_end_hour,
        validation_end_hour,
        dirty_summary,
    )


# ============================================================
# FIXED-BOUNDARY GCN SCALING
# ============================================================

def scale_gcn_with_fixed_boundaries(
    df: pd.DataFrame,
    *,
    train_end_hour: pd.Timestamp,
    validation_end_hour: pd.Timestamp,
):
    result = df.copy()

    hours = (
        result[TIME_COLUMN]
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    training_rows = result.loc[
        result[TIME_COLUMN] <= train_end_hour
    ].copy()

    feature_training = (
        training_rows[gcn.FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if feature_training.empty:
        raise RuntimeError(
            "No complete GCN training predictor rows."
        )

    feature_scaler = StandardScaler()
    feature_scaler.fit(feature_training)

    target_training = (
        training_rows[[TARGET_COLUMN]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if target_training.empty:
        raise RuntimeError(
            "No observed GCN training targets."
        )

    target_scaler = StandardScaler()
    target_scaler.fit(target_training)

    scaled_feature_names = [
        f"{column}_scaled"
        for column in gcn.FEATURES
    ]

    for column in scaled_feature_names:
        result[column] = np.nan

    clean_features = (
        result[gcn.FEATURES]
        .replace([np.inf, -np.inf], np.nan)
    )

    complete_feature_mask = (
        clean_features.notna().all(axis=1)
    )

    transformed = feature_scaler.transform(
        result.loc[
            complete_feature_mask,
            gcn.FEATURES,
        ]
    ).astype(np.float32)

    for index, column in enumerate(
        scaled_feature_names
    ):
        result.loc[
            complete_feature_mask,
            column,
        ] = transformed[:, index]

    result["target_scaled"] = np.nan

    target_observed = (
        result[TARGET_COLUMN].notna()
    )

    result.loc[
        target_observed,
        "target_scaled",
    ] = (
        target_scaler.transform(
            result.loc[
                target_observed,
                [TARGET_COLUMN],
            ]
        )
        .reshape(-1)
        .astype(np.float32)
    )

    train_end_pos = int(
        (hours <= train_end_hour).sum()
    )

    validation_end_pos = int(
        (hours <= validation_end_hour).sum()
    )

    if train_end_pos <= 0:
        raise RuntimeError(
            "No GCN training snapshots after matching."
        )

    if validation_end_pos <= train_end_pos:
        raise RuntimeError(
            "No GCN validation snapshots after matching."
        )

    if validation_end_pos >= len(hours):
        raise RuntimeError(
            "No GCN test snapshots after matching."
        )

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
# RUN GCN VARIANT
# ============================================================

def run_gcn_variant(
    df: pd.DataFrame,
    *,
    variant: str,
    train_end_hour: pd.Timestamp,
    validation_end_hour: pd.Timestamp,
):
    print()
    print("=" * 80)
    print(
        f"GCN EXPERIMENT: {variant.upper()}"
    )
    print("=" * 80)

    set_seed()

    (
        node_table,
        node_ids,
        node_to_index,
        base_adjacency,
    ) = gcn.build_geographic_graph(df)

    (
        scaled,
        hours,
        train_end_pos,
        validation_end_pos,
        feature_scaler,
        target_scaler,
        scaled_feature_names,
    ) = scale_gcn_with_fixed_boundaries(
        df,
        train_end_hour=train_end_hour,
        validation_end_hour=validation_end_hour,
    )

    (
        x,
        y,
        y_mask,
        active_mask,
    ) = gcn.build_snapshot_arrays(
        scaled,
        hours,
        node_to_index,
        scaled_feature_names,
    )

    train_dataset = gcn.SnapshotDataset(
        x,
        y,
        y_mask,
        active_mask,
        0,
        train_end_pos,
    )

    validation_dataset = gcn.SnapshotDataset(
        x,
        y,
        y_mask,
        active_mask,
        train_end_pos,
        validation_end_pos,
    )

    test_dataset = gcn.SnapshotDataset(
        x,
        y,
        y_mask,
        active_mask,
        validation_end_pos,
        len(hours),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=gcn.BATCH_SIZE,
        shuffle=True,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=gcn.BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=gcn.BATCH_SIZE,
        shuffle=False,
    )

    model = gcn.DepthGCN(
        len(gcn.FEATURES),
        base_adjacency,
    ).to(gcn.DEVICE)

    print(
        f"Train snapshots:      {len(train_dataset):,}"
    )
    print(
        f"Validation snapshots: "
        f"{len(validation_dataset):,}"
    )
    print(
        f"Test snapshots:       {len(test_dataset):,}"
    )

    (
        model,
        training_history,
        best_validation_loss,
    ) = gcn.train_model(
        model,
        train_loader,
        validation_loader,
    )

    (
        test_predictions,
        regression_metrics,
    ) = gcn.evaluate_test(
        model,
        test_loader,
        hours,
        node_ids,
        target_scaler,
    )

    (
        sensor_event_metrics,
        event_overall,
    ) = gcn.evaluate_events_24h(
        test_predictions
    )

    return {
        "test_predictions": test_predictions,
        "sensor_event_metrics": sensor_event_metrics,
        "event_overall": event_overall,
        "regression_metrics": regression_metrics,
        "best_validation_loss": float(
            best_validation_loss
        ),
    }


# ============================================================
# LOGISTIC FIXED SPLIT
# ============================================================

def split_logistic_fixed(
    modeling: pd.DataFrame,
    *,
    train_end_hour: pd.Timestamp,
    validation_end_hour: pd.Timestamp,
):
    train = modeling[
        modeling[TIME_COLUMN]
        <= train_end_hour
    ].copy()

    validation = modeling[
        (
            modeling[TIME_COLUMN]
            > train_end_hour
        )
        &
        (
            modeling[TIME_COLUMN]
            <= validation_end_hour
        )
    ].copy()

    test = modeling[
        modeling[TIME_COLUMN]
        > validation_end_hour
    ].copy()

    if train.empty:
        raise RuntimeError(
            "Matched Logistic training set is empty."
        )

    if validation.empty:
        raise RuntimeError(
            "Matched Logistic validation set is empty."
        )

    if test.empty:
        raise RuntimeError(
            "Matched Logistic test set is empty."
        )

    return train, validation, test


def run_logistic_variant(
    df: pd.DataFrame,
    *,
    variant: str,
    train_end_hour: pd.Timestamp,
    validation_end_hour: pd.Timestamp,
):
    print()
    print("=" * 80)
    print(
        f"LOGISTIC EXPERIMENT: {variant.upper()}"
    )
    print("=" * 80)

    modeling = logistic.prepare_modeling_rows(
        df
    )

    train, validation, test = (
        split_logistic_fixed(
            modeling,
            train_end_hour=train_end_hour,
            validation_end_hour=(
                validation_end_hour
            ),
        )
    )

    scaler, model = logistic.fit_model(
        train
    )

    validation_predictions = (
        logistic.predict_split(
            validation,
            scaler=scaler,
            model=model,
            split_name="validation",
        )
    )

    test_predictions = (
        logistic.predict_split(
            test,
            scaler=scaler,
            model=model,
            split_name="test",
        )
    )

    overall = (
        logistic.calculate_binary_metrics(
            test_predictions
        )
    )

    sensor_metrics = (
        logistic.calculate_sensor_metrics(
            test_predictions
        )
    )

    return {
        "validation_predictions": (
            validation_predictions
        ),
        "test_predictions": test_predictions,
        "overall": overall,
        "sensor_metrics": sensor_metrics,
    }


# ============================================================
# STANDARDIZE SENSOR METRICS
# ============================================================

def standardize_metrics(
    df: pd.DataFrame,
    *,
    model_name: str,
    variant: str,
) -> pd.DataFrame:
    result = df.copy()

    # Allow either deployment_id or the module's sensor constant.
    if SENSOR_COLUMN not in result.columns:
        if "deployment_id" in result.columns:
            result = result.rename(
                columns={
                    "deployment_id": SENSOR_COLUMN
                }
            )
        else:
            raise RuntimeError(
                f"{model_name} sensor metrics do not "
                "contain a sensor identifier. "
                f"Columns: {result.columns.tolist()}"
            )

    required_metrics = {
        "precision",
        "recall",
        "f1",
    }

    missing = (
        required_metrics
        - set(result.columns)
    )

    if missing:
        raise RuntimeError(
            f"{model_name} sensor metrics are missing "
            f"{sorted(missing)}. "
            f"Columns: {result.columns.tolist()}"
        )

    result = result[
        result[SENSOR_COLUMN].isin(
            TOP_SENSORS
        )
    ].copy()

    result["model"] = model_name
    result["variant"] = variant

    keep = [
        SENSOR_COLUMN,
        "model",
        "variant",
    ]

    optional = [
        "n_rows",
        "test_rows",
        "actual_events",
        "predicted_events",
        "precision",
        "recall",
        "f1",
    ]

    keep.extend(
        [
            column
            for column in optional
            if column in result.columns
        ]
    )

    return result[keep]


# ============================================================
# FINAL COMPARISON
# ============================================================

def build_comparison(
    established: pd.DataFrame,
    matched_metrics: pd.DataFrame,
) -> pd.DataFrame:
    established = established.rename(
        columns={
            "deployment_id": SENSOR_COLUMN,
            "selected_model": "established_model",
            "selected_precision": (
                "established_precision"
            ),
            "selected_recall": (
                "established_recall"
            ),
            "selected_f1": (
                "established_f1"
            ),
        }
    )

    static = matched_metrics[
        matched_metrics["variant"]
        == "static"
    ].copy()

    rolling = matched_metrics[
        matched_metrics["variant"]
        == "rolling24"
    ].copy()

    static = static.rename(
        columns={
            "precision": "matched_static_precision",
            "recall": "matched_static_recall",
            "f1": "matched_static_f1",
            "actual_events": (
                "matched_static_actual_events"
            ),
            "predicted_events": (
                "matched_static_predicted_events"
            ),
            "n_rows": "matched_static_rows",
            "test_rows": "matched_static_test_rows",
        }
    )

    rolling = rolling.rename(
        columns={
            "precision": "rolling24_precision",
            "recall": "rolling24_recall",
            "f1": "rolling24_f1",
            "actual_events": (
                "rolling24_actual_events"
            ),
            "predicted_events": (
                "rolling24_predicted_events"
            ),
            "n_rows": "rolling24_rows",
            "test_rows": "rolling24_test_rows",
        }
    )

    static_drop = {
        "variant",
    }

    rolling_drop = {
        "variant",
        "model",
    }

    static = static.drop(
        columns=[
            c for c in static_drop
            if c in static.columns
        ]
    )

    rolling = rolling.drop(
        columns=[
            c for c in rolling_drop
            if c in rolling.columns
        ]
    )

    comparison = established.merge(
        static,
        on=SENSOR_COLUMN,
        how="left",
    )

    comparison = comparison.merge(
        rolling,
        on=SENSOR_COLUMN,
        how="left",
    )

    comparison["delta_precision"] = (
        comparison["rolling24_precision"]
        - comparison["matched_static_precision"]
    )

    comparison["delta_recall"] = (
        comparison["rolling24_recall"]
        - comparison["matched_static_recall"]
    )

    comparison["delta_f1"] = (
        comparison["rolling24_f1"]
        - comparison["matched_static_f1"]
    )

    comparison[
        "delta_vs_established_f1"
    ] = (
        comparison["rolling24_f1"]
        - comparison["established_f1"]
    )

    order = {
        sensor: i
        for i, sensor in enumerate(
            TOP_SENSORS
        )
    }

    comparison["_order"] = (
        comparison[SENSOR_COLUMN]
        .map(order)
    )

    comparison = (
        comparison
        .sort_values("_order")
        .drop(columns="_order")
        .reset_index(drop=True)
    )

    return comparison


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print(
        "MRMS STATIC DAILY TOTAL vs "
        "CENTERED ROLLING-24 EXPERIMENT"
    )
    print(
        "=" * 72
    )

    print()
    print("SCIENTIFIC CONTRACT")
    print("-------------------")
    print(
        "Prediction frequency: hourly"
    )
    print(
        "Training precipitation: MRMS"
    )
    print(
        "Static feature: complete historical "
        "daily_total_precip_mm"
    )
    print(
        "Experimental feature: "
        "sum(H-12 ... H+11)"
    )
    print(
        "Current-hour feature: unchanged"
    )
    print(
        "Previous-6h feature: unchanged"
    )
    print(
        "Targets: unchanged"
    )
    print(
        "Model architectures: unchanged"
    )
    print(
        "Production artifacts: NOT modified"
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    established = (
        load_established_baselines()
    )

    (
        static,
        rolling,
        train_end_hour,
        validation_end_hour,
        dirty_summary,
    ) = prepare_experiment_frames()

    # --------------------------------------------------------
    # GCN
    # --------------------------------------------------------

    static_gcn = run_gcn_variant(
        static,
        variant="static",
        train_end_hour=train_end_hour,
        validation_end_hour=validation_end_hour,
    )

    rolling_gcn = run_gcn_variant(
        rolling,
        variant="rolling24",
        train_end_hour=train_end_hour,
        validation_end_hour=validation_end_hour,
    )

    # --------------------------------------------------------
    # Logistic
    # --------------------------------------------------------

    static_logistic = run_logistic_variant(
        static,
        variant="static",
        train_end_hour=train_end_hour,
        validation_end_hour=validation_end_hour,
    )

    rolling_logistic = run_logistic_variant(
        rolling,
        variant="rolling24",
        train_end_hour=train_end_hour,
        validation_end_hour=validation_end_hour,
    )

    # --------------------------------------------------------
    # Standardize per-sensor event metrics
    # --------------------------------------------------------

    metric_frames = [
        standardize_metrics(
            static_gcn["sensor_event_metrics"],
            model_name="gcn",
            variant="static",
        ),
        standardize_metrics(
            rolling_gcn["sensor_event_metrics"],
            model_name="gcn",
            variant="rolling24",
        ),
        standardize_metrics(
            static_logistic["sensor_metrics"],
            model_name="logistic",
            variant="static",
        ),
        standardize_metrics(
            rolling_logistic["sensor_metrics"],
            model_name="logistic",
            variant="rolling24",
        ),
    ]

    all_metrics = pd.concat(
        metric_frames,
        ignore_index=True,
    )

    # Keep only each sensor's already-selected model family.
    selected_models = (
        established[
            [
                "deployment_id",
                "selected_model",
            ]
        ]
        .rename(
            columns={
                "deployment_id": SENSOR_COLUMN,
                "selected_model": "expected_model",
            }
        )
    )

    all_metrics = all_metrics.merge(
        selected_models,
        on=SENSOR_COLUMN,
        how="inner",
    )

    all_metrics = all_metrics[
        all_metrics["model"].str.lower()
        ==
        all_metrics["expected_model"].str.lower()
    ].copy()

    all_metrics = all_metrics.drop(
        columns=["expected_model"]
    )

    comparison = build_comparison(
        established,
        all_metrics,
    )

    # --------------------------------------------------------
    # Save experiment only
    # --------------------------------------------------------

    all_metrics.to_csv(
        OUTPUT_DIR
        / "matched_sensor_metrics.csv",
        index=False,
    )

    comparison.to_csv(
        OUTPUT_DIR
        / "comparison.csv",
        index=False,
    )

    static_gcn[
        "test_predictions"
    ].to_parquet(
        OUTPUT_DIR
        / "gcn_static_test_predictions.parquet",
        index=False,
    )

    rolling_gcn[
        "test_predictions"
    ].to_parquet(
        OUTPUT_DIR
        / "gcn_rolling24_test_predictions.parquet",
        index=False,
    )

    static_logistic[
        "test_predictions"
    ].to_parquet(
        OUTPUT_DIR
        / "logistic_static_test_predictions.parquet",
        index=False,
    )

    rolling_logistic[
        "test_predictions"
    ].to_parquet(
        OUTPUT_DIR
        / "logistic_rolling24_test_predictions.parquet",
        index=False,
    )

    summary = {
        "experiment": EXPERIMENT_NAME,
        "rolling_definition": (
            "sum precip_current_hour_mm "
            "from H-12 through H+11"
        ),
        "rolling_hours": 24,
        "train_end_hour": str(
            train_end_hour
        ),
        "validation_end_hour": str(
            validation_end_hour
        ),
        "matched_rows": int(
            len(static)
        ),
        "matched_sensors": int(
            static[SENSOR_COLUMN].nunique()
        ),
        "selected_evaluation_sensors": (
            TOP_SENSORS
        ),
        "production_modified": False,
    }

    with open(
        OUTPUT_DIR / "experiment_summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    display_columns = [
        SENSOR_COLUMN,
        "established_model",
        "established_precision",
        "established_recall",
        "established_f1",
        "matched_static_precision",
        "matched_static_recall",
        "matched_static_f1",
        "rolling24_precision",
        "rolling24_recall",
        "rolling24_f1",
        "delta_precision",
        "delta_recall",
        "delta_f1",
    ]

    display_columns = [
        c
        for c in display_columns
        if c in comparison.columns
    ]

    print()
    print("=" * 120)
    print("FINAL FIVE-SENSOR COMPARISON")
    print("=" * 120)

    print(
        comparison[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}",
        )
    )

    print()
    print("OUTPUTS")
    print("-------")
    print(
        OUTPUT_DIR
        / "comparison.csv"
    )
    print(
        OUTPUT_DIR
        / "matched_sensor_metrics.csv"
    )
    print(
        OUTPUT_DIR
        / "experiment_summary.json"
    )

    print()
    print(
        "EXPERIMENT COMPLETE — "
        "NO PRODUCTION MODEL WAS MODIFIED"
    )


if __name__ == "__main__":
    main()