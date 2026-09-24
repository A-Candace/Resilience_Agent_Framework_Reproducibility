"""
Run NYC 1-km operational flood inference for same-day or day-ahead products.

Operational architecture
------------------------
Training / evaluation:
    MRMS precipitation
        -> GCN + logistic training
        -> per-sensor evaluation
        -> precision / recall eligibility
        -> highest-F1 qualifying model
        -> spatial support assignment

Operational forecasting:
    HRRR precipitation at target 1-km grid
        +
    selected support sensor/model
        ->
    model inference
        ->
    NYC 1-km flood forecast

Important
---------
The two model families have different native outputs.

Logistic:
    probability of flood event

GCN:
    predicted minutes above 1-inch flood threshold

Therefore this module DOES NOT manufacture a common probability
for GCN and logistic predictions.

The common operational output is:

    predicted_flood_event

using the same model-specific thresholds used during evaluation:

    logistic:
        probability >= 0.5

    GCN:
        predicted_minutes_above_1inch > 1.0
"""

from __future__ import annotations


import os
import argparse
import argparse
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd


# ============================================================
# PATHS / FORECAST MODES
# ============================================================

FORECAST_ROOT = Path("artifacts/flood/forecasting")
SUPPORTED_FORECAST_MODES = ("today", "tomorrow")

LOGISTIC_MODEL_PATH = Path(
    "artifacts/flood/logistic/precipitation_only_logistic.pkl"
)
LOGISTIC_SCALER_PATH = Path(
    "artifacts/flood/logistic/feature_scaler.pkl"
)
LOGISTIC_METADATA_PATH = Path(
    "artifacts/flood/logistic/metadata.json"
)


def resolve_operational_paths(
    *,
    mode: str,
    inputs: str | Path | None,
    output_dir: str | Path | None,
) -> tuple[Path, Path]:
    """Resolve isolated operational input/output paths for a forecast mode."""
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in SUPPORTED_FORECAST_MODES:
        raise ValueError(
            f"Unsupported forecast mode {mode!r}. "
            f"Expected one of: {', '.join(SUPPORTED_FORECAST_MODES)}"
        )

    resolved_output_dir = (
        Path(output_dir)
        if output_dir is not None
        else FORECAST_ROOT / normalized_mode
    )
    resolved_input_path = (
        Path(inputs)
        if inputs is not None
        else resolved_output_dir / "grid_forecast_inputs.parquet"
    )
    return resolved_input_path, resolved_output_dir


def build_output_paths(output_dir: Path) -> tuple[Path, Path, Path]:
    return (
        output_dir / "grid_flood_forecast.parquet",
        output_dir / "grid_flood_forecast.csv",
        output_dir / "grid_flood_forecast_summary.json",
    )


# ============================================================
# MODEL CONTRACT
# ============================================================

FEATURES = [
    "precip_current_hour_mm",
    "precip_previous_6h_mm",
    "daily_total_precip_mm",
]

LOGISTIC_THRESHOLD = 0.5

GCN_DURATION_THRESHOLD_MINUTES = 1.0

GCN_MODEL_URI = (
    "models:/nyc-resilience-flood-gcn@champion"
)

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://mlflow:5000",
)

# ============================================================
# BASIC HELPERS
# ============================================================

def load_pickle(
    path: Path,
):
    """
    Load a pickle/joblib-compatible artifact.

    sklearn artifacts written with joblib can generally be
    loaded through joblib more reliably, so try joblib first.
    """

    try:
        import joblib

        return joblib.load(
            path
        )

    except Exception:

        with path.open(
            "rb"
        ) as file:
            return pickle.load(
                file
            )


def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    frame_name: str,
) -> None:

    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]

    if missing:

        raise RuntimeError(
            f"{frame_name} is missing required columns: "
            + ", ".join(
                missing
            )
        )


def safe_json_value(
    value,
):

    if isinstance(
        value,
        (
            np.integer,
        ),
    ):
        return int(
            value
        )

    if isinstance(
        value,
        (
            np.floating,
        ),
    ):
        return float(
            value
        )

    if isinstance(
        value,
        (
            np.bool_,
        ),
    ):
        return bool(
            value
        )

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    return value


# ============================================================
# LOAD INPUTS
# ============================================================

def load_forecast_inputs(
    forecast_input_path: Path,
) -> pd.DataFrame:

    if not forecast_input_path.exists():

        raise FileNotFoundError(
            f"Forecast input artifact not found: "
            f"{forecast_input_path}"
        )

    frame = pd.read_parquet(
        forecast_input_path
    )

    required = [
        "grid_id",
        "forecast_hour",
        "cluster_number",
        "support_sensor_id",
        "support_sensor_model",
        "support_sensor_f1",
        "support_sensor_precision",
        "support_sensor_recall",
        "forecast_mode",
        "target_date_nyc",
        *FEATURES,
    ]

    require_columns(
        frame,
        required,
        frame_name="grid forecast inputs",
    )

    frame = frame.copy()

    frame[
        "forecast_hour"
    ] = pd.to_datetime(
        frame[
            "forecast_hour"
        ],
        utc=True,
    )

    for feature in FEATURES:

        frame[
            feature
        ] = pd.to_numeric(
            frame[
                feature
            ],
            errors="coerce",
        )

    missing_features = (
        frame[
            FEATURES
        ]
        .isna()
        .any(
            axis=1
        )
    )

    if missing_features.any():

        raise RuntimeError(
            "Forecast inputs contain "
            f"{int(missing_features.sum()):,} rows "
            "with missing precipitation predictors."
        )

    duplicate_keys = (
        frame.duplicated(
            [
                "grid_id",
                "forecast_hour",
            ]
        )
    )

    if duplicate_keys.any():

        raise RuntimeError(
            "Forecast inputs contain duplicate "
            "(grid_id, forecast_hour) rows."
        )

    model_names = set(
        frame[
            "support_sensor_model"
        ]
        .astype(str)
        .str.lower()
        .unique()
        .tolist()
    )

    expected_models = {
        "gcn",
        "logistic",
    }

    unknown_models = (
        model_names
        -
        expected_models
    )

    if unknown_models:

        raise RuntimeError(
            "Forecast inputs contain unsupported "
            "selected models: "
            + ", ".join(
                sorted(
                    unknown_models
                )
            )
        )

    frame[
        "support_sensor_model"
    ] = (
        frame[
            "support_sensor_model"
        ]
        .astype(str)
        .str.lower()
    )

    forecast_modes = (
        frame[
            "forecast_mode"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .str.lower()
        .unique()
        .tolist()
    )

    if len(
        forecast_modes
    ) != 1:

        raise RuntimeError(
            "Forecast inputs must contain exactly one "
            f"forecast_mode. Found: {forecast_modes}"
        )

    if forecast_modes[
        0
    ] not in SUPPORTED_FORECAST_MODES:

        raise RuntimeError(
            "Forecast input artifact contains unsupported "
            f"forecast_mode: {forecast_modes[0]}"
        )

    frame[
        "forecast_mode"
    ] = forecast_modes[
        0
    ]

    target_dates = (
        frame[
            "target_date_nyc"
        ]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if len(
        target_dates
    ) != 1:

        raise RuntimeError(
            "Forecast inputs must contain exactly one "
            f"target_date_nyc. Found: {target_dates}"
        )

    frame[
        "target_date_nyc"
    ] = target_dates[
        0
    ]

    return frame


# ============================================================
# LOGISTIC MODEL
# ============================================================

def load_logistic_artifacts():

    for path in [
        LOGISTIC_MODEL_PATH,
        LOGISTIC_SCALER_PATH,
        LOGISTIC_METADATA_PATH,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                f"Required logistic artifact not found: "
                f"{path}"
            )

    model = load_pickle(
        LOGISTIC_MODEL_PATH
    )

    scaler_artifact = load_pickle(
        LOGISTIC_SCALER_PATH
    )

    with LOGISTIC_METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        metadata = json.load(
            file
        )

    if not isinstance(
        scaler_artifact,
        dict,
    ):

        raise RuntimeError(
            "Logistic scaler artifact must be a dictionary."
        )

    if "feature_scaler" not in scaler_artifact:

        raise RuntimeError(
            "Logistic scaler artifact does not contain "
            "'feature_scaler'."
        )

    scaler = scaler_artifact[
        "feature_scaler"
    ]

    artifact_features = scaler_artifact.get(
        "features"
    )

    if artifact_features is None:

        raise RuntimeError(
            "Logistic scaler artifact does not contain "
            "its feature contract."
        )

    if list(
        artifact_features
    ) != FEATURES:

        raise RuntimeError(
            "Logistic scaler feature order does not match "
            "the operational forecast feature order.\n"
            f"Saved: {artifact_features}\n"
            f"Expected: {FEATURES}"
        )

    metadata_features = metadata.get(
        "features"
    )

    if list(
        metadata_features or []
    ) != FEATURES:

        raise RuntimeError(
            "Logistic metadata feature contract does not "
            "match the operational forecast feature order."
        )

    if not hasattr(
        model,
        "predict_proba",
    ):

        raise RuntimeError(
            "Saved logistic model does not expose predict_proba()."
        )

    if not hasattr(
        scaler,
        "transform",
    ):

        raise RuntimeError(
            "Saved logistic scaler does not expose transform()."
        )

    classes = list(
        model.classes_
    )

    if 1 not in classes:

        raise RuntimeError(
            "Logistic model does not contain positive class 1."
        )

    return (
        model,
        scaler,
        metadata,
    )


def score_logistic_rows(
    rows: pd.DataFrame,
    *,
    model,
    scaler,
) -> pd.DataFrame:

    result = rows.copy()

    if result.empty:

        result[
            "logistic_event_probability"
        ] = pd.Series(
            dtype=float
        )

        result[
            "predicted_flood_event"
        ] = pd.Series(
            dtype=bool
        )

        return result

    features = (
        result[
            FEATURES
        ]
        .astype(float)
    )

    scaled = scaler.transform(
        features
    )

    probabilities = model.predict_proba(
        scaled
    )

    positive_class_index = (
        list(
            model.classes_
        )
        .index(
            1
        )
    )

    event_probability = probabilities[
        :,
        positive_class_index
    ]

    result[
        "logistic_event_probability"
    ] = event_probability

    result[
        "gcn_predicted_minutes_above_1inch"
    ] = np.nan

    result[
        "predicted_flood_event"
    ] = (
        result[
            "logistic_event_probability"
        ]
        >=
        LOGISTIC_THRESHOLD
    )

    result[
        "native_prediction_type"
    ] = (
        "event_probability"
    )

    result[
        "native_prediction_value"
    ] = (
        result[
            "logistic_event_probability"
        ]
    )

    result[
        "classification_threshold"
    ] = (
        LOGISTIC_THRESHOLD
    )

    return result


# ============================================================
# GCN MODEL
# ============================================================

def load_gcn_model():

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    return mlflow.pyfunc.load_model(
        GCN_MODEL_URI
    )


def normalize_gcn_prediction(
    prediction,
    *,
    expected_rows: int,
) -> np.ndarray:
    """
    Normalize the registered pyfunc model output to one
    predicted-duration value per input row.
    """

    if isinstance(
        prediction,
        pd.DataFrame,
    ):

        preferred_columns = [
            "predicted_minutes_above_1inch",
            "prediction",
            "predicted_minutes",
        ]

        prediction_column = None

        for column in preferred_columns:

            if column in prediction.columns:

                prediction_column = column
                break

        if prediction_column is None:

            numeric_columns = (
                prediction
                .select_dtypes(
                    include=[
                        np.number
                    ]
                )
                .columns
                .tolist()
            )

            if len(
                numeric_columns
            ) != 1:

                raise RuntimeError(
                    "Could not determine the GCN prediction "
                    "column from registered model output. "
                    f"Columns: {prediction.columns.tolist()}"
                )

            prediction_column = numeric_columns[
                0
            ]

        values = (
            prediction[
                prediction_column
            ]
            .to_numpy(
                dtype=float
            )
        )

    elif isinstance(
        prediction,
        pd.Series,
    ):

        values = prediction.to_numpy(
            dtype=float
        )

    else:

        values = np.asarray(
            prediction,
            dtype=float,
        )

        if values.ndim == 2:

            if values.shape[
                1
            ] != 1:

                raise RuntimeError(
                    "Registered GCN returned a multi-column "
                    "array that cannot be interpreted as "
                    "predicted duration."
                )

            values = values[
                :,
                0
            ]

        elif values.ndim != 1:

            raise RuntimeError(
                "Registered GCN returned an unsupported "
                f"prediction shape: {values.shape}"
            )

    if len(
        values
    ) != expected_rows:

        raise RuntimeError(
            "Registered GCN prediction row count does not "
            "match inference input row count. "
            f"Expected {expected_rows:,}; "
            f"received {len(values):,}."
        )

    return values


def score_gcn_rows(
    rows: pd.DataFrame,
    *,
    model,
) -> pd.DataFrame:
    """
    Score all target-grid rows routed to the GCN.

    The registered GCN permits at most one row per deployment_id in a
    single model.predict() call. Many 1-km target grids can share the same
    support sensor while having different HRRR precipitation scenarios.

    This function therefore:

        1. identifies unique support-sensor + rainfall scenarios;
        2. assigns a scenario rank within each support sensor;
        3. batches scenarios so each deployment_id appears at most once
           per model.predict() call;
        4. scores each valid batch;
        5. maps each scenario prediction back to all original grid/hour rows.

    Exact duplicate sensor/weather scenarios are scored only once.
    """

    result = rows.copy()

    if result.empty:
        result["gcn_predicted_minutes_above_1inch"] = pd.Series(dtype=float)
        result["logistic_event_probability"] = pd.Series(dtype=float)
        result["predicted_flood_event"] = pd.Series(dtype=bool)
        result["native_prediction_type"] = pd.Series(dtype=str)
        result["native_prediction_value"] = pd.Series(dtype=float)
        result["classification_threshold"] = pd.Series(dtype=float)
        result["gcn_scenario_id"] = pd.Series(dtype="int64")
        return result

    required_columns = [
        "support_sensor_id",
        "precip_current_hour_mm",
        "precip_previous_6h_mm",
        "daily_total_precip_mm",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise RuntimeError(
            "GCN forecast rows are missing required columns: "
            + ", ".join(missing_columns)
        )

    result["support_sensor_id"] = (
        result["support_sensor_id"]
        .astype(str)
        .str.strip()
    )

    for column in [
        "precip_current_hour_mm",
        "precip_previous_6h_mm",
        "daily_total_precip_mm",
    ]:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    invalid_feature_mask = (
        result[[
            "precip_current_hour_mm",
            "precip_previous_6h_mm",
            "daily_total_precip_mm",
        ]]
        .isna()
        .any(axis=1)
    )

    if invalid_feature_mask.any():
        raise RuntimeError(
            "GCN inference contains "
            f"{int(invalid_feature_mask.sum()):,} rows with "
            "missing or non-numeric precipitation predictors."
        )

    scenario_columns = [
        "support_sensor_id",
        "precip_current_hour_mm",
        "precip_previous_6h_mm",
        "daily_total_precip_mm",
    ]

    scenarios = (
        result[scenario_columns]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    scenarios["_gcn_scenario_id"] = np.arange(
        len(scenarios),
        dtype=np.int64,
    )

    scenarios = (
        scenarios
        .sort_values(
            [
                "support_sensor_id",
                "precip_current_hour_mm",
                "precip_previous_6h_mm",
                "daily_total_precip_mm",
            ],
            ascending=[True, True, True, True],
        )
        .reset_index(drop=True)
    )

    scenarios["_gcn_batch_number"] = (
        scenarios
        .groupby(
            "support_sensor_id",
            sort=False,
        )
        .cumcount()
    )

    batch_count = int(
        scenarios["_gcn_batch_number"].max() + 1
    )

    print()
    print("GCN SCENARIO BATCHING")
    print("---------------------")
    print(
        "Original GCN grid/hour rows:",
        f"{len(result):,}",
    )
    print(
        "Unique sensor/weather scenarios:",
        f"{len(scenarios):,}",
    )
    print(
        "Unique GCN support sensors:",
        f"{scenarios['support_sensor_id'].nunique():,}",
    )
    print(
        "GCN inference batches:",
        f"{batch_count:,}",
    )

    prediction_frames: list[pd.DataFrame] = []

    grouped_batches = scenarios.groupby(
        "_gcn_batch_number",
        sort=True,
    )

    for batch_number, batch in grouped_batches:
        batch = batch.copy()

        duplicate_sensor_mask = batch[
            "support_sensor_id"
        ].duplicated()

        if duplicate_sensor_mask.any():
            duplicate_ids = (
                batch.loc[
                    duplicate_sensor_mask,
                    "support_sensor_id",
                ]
                .drop_duplicates()
                .head(20)
                .tolist()
            )
            raise RuntimeError(
                "Internal GCN batching error: duplicate support sensors "
                "were placed in the same model.predict() call. "
                "Duplicate IDs: "
                + ", ".join(duplicate_ids)
            )

        if (
            batch_number == 0
            or (batch_number + 1) % 25 == 0
            or (batch_number + 1) == batch_count
        ):
            print(
                "Scoring GCN batch "
                f"{batch_number + 1:,}/{batch_count:,} "
                f"({len(batch):,} sensors)"
            )

        inference = pd.DataFrame(
            {
                "deployment_id":
                    batch["support_sensor_id"]
                    .astype(str)
                    .to_numpy(),
                "precip_current_hour_mm":
                    batch["precip_current_hour_mm"]
                    .astype(float)
                    .to_numpy(),
                "precip_previous_6h_mm":
                    batch["precip_previous_6h_mm"]
                    .astype(float)
                    .to_numpy(),
                "daily_total_precip_mm":
                    batch["daily_total_precip_mm"]
                    .astype(float)
                    .to_numpy(),
            }
        )

        prediction = model.predict(inference)

        if isinstance(prediction, pd.DataFrame):
            if "predicted_minutes_above_1inch" not in prediction.columns:
                raise RuntimeError(
                    "Registered GCN output is missing "
                    "'predicted_minutes_above_1inch'. "
                    f"Columns: {prediction.columns.tolist()}"
                )

            if len(prediction) != len(batch):
                raise RuntimeError(
                    "Registered GCN output row count does not match "
                    "the inference batch. "
                    f"Input rows={len(batch):,}; "
                    f"output rows={len(prediction):,}."
                )

            if "deployment_id" in prediction.columns:
                input_ids = (
                    inference["deployment_id"]
                    .astype(str)
                    .tolist()
                )
                output_ids = (
                    prediction["deployment_id"]
                    .astype(str)
                    .tolist()
                )
                if input_ids != output_ids:
                    raise RuntimeError(
                        "Registered GCN changed deployment_id ordering "
                        "during inference."
                    )

            predicted_minutes = (
                prediction["predicted_minutes_above_1inch"]
                .to_numpy(dtype=float)
            )

        else:
            predicted_minutes = normalize_gcn_prediction(
                prediction,
                expected_rows=len(batch),
            )

        if len(predicted_minutes) != len(batch):
            raise RuntimeError(
                "GCN prediction count does not match batch size."
            )

        if not np.isfinite(predicted_minutes).all():
            raise RuntimeError(
                "Registered GCN returned non-finite predicted-minute "
                "values."
            )

        batch_predictions = pd.DataFrame(
            {
                "_gcn_scenario_id":
                    batch["_gcn_scenario_id"]
                    .to_numpy(dtype=np.int64),
                "gcn_predicted_minutes_above_1inch":
                    predicted_minutes,
            }
        )

        prediction_frames.append(batch_predictions)

    if not prediction_frames:
        raise RuntimeError(
            "No GCN prediction batches were produced."
        )

    scenario_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    if len(scenario_predictions) != len(scenarios):
        raise RuntimeError(
            "GCN scenario prediction count does not match "
            "the number of unique GCN scenarios. "
            f"Scenarios={len(scenarios):,}; "
            f"predictions={len(scenario_predictions):,}."
        )

    if scenario_predictions["_gcn_scenario_id"].duplicated().any():
        raise RuntimeError(
            "GCN scenario prediction table contains duplicate "
            "scenario IDs."
        )

    result = result.merge(
        scenarios[
            [
                *scenario_columns,
                "_gcn_scenario_id",
            ]
        ],
        on=scenario_columns,
        how="left",
        validate="many_to_one",
    )

    if result["_gcn_scenario_id"].isna().any():
        raise RuntimeError(
            "Some original GCN grid/hour rows could not be assigned "
            "to a unique GCN sensor/weather scenario."
        )

    result = result.merge(
        scenario_predictions,
        on="_gcn_scenario_id",
        how="left",
        validate="many_to_one",
    )

    if result["gcn_predicted_minutes_above_1inch"].isna().any():
        raise RuntimeError(
            "Some GCN grid/hour rows did not receive a model prediction."
        )

    result["logistic_event_probability"] = np.nan

    result["predicted_flood_event"] = (
        result["gcn_predicted_minutes_above_1inch"]
        > GCN_DURATION_THRESHOLD_MINUTES
    )

    result["native_prediction_type"] = (
        "predicted_minutes_above_1inch"
    )

    result["native_prediction_value"] = (
        result["gcn_predicted_minutes_above_1inch"]
    )

    result["classification_threshold"] = (
        GCN_DURATION_THRESHOLD_MINUTES
    )

    result["gcn_scenario_id"] = (
        result["_gcn_scenario_id"]
        .astype(np.int64)
    )

    result = result.drop(
        columns=["_gcn_scenario_id"]
    )

    if len(result) != len(rows):
        raise RuntimeError(
            "GCN scenario expansion changed the original grid/hour "
            "row count. "
            f"Before={len(rows):,}; "
            f"after={len(result):,}."
        )

    return result


# ============================================================
# COMBINED FORECAST
# ============================================================

def run_selected_model_inference(
    inputs: pd.DataFrame,
) -> pd.DataFrame:

    logistic_mask = (
        inputs[
            "support_sensor_model"
        ]
        ==
        "logistic"
    )

    gcn_mask = (
        inputs[
            "support_sensor_model"
        ]
        ==
        "gcn"
    )

    logistic_rows = (
        inputs.loc[
            logistic_mask
        ]
        .copy()
    )

    gcn_rows = (
        inputs.loc[
            gcn_mask
        ]
        .copy()
    )

    print()
    print(
        "SELECTED-MODEL ROUTING"
    )
    print(
        "----------------------"
    )
    print(
        "Logistic rows:",
        f"{len(logistic_rows):,}",
    )
    print(
        "GCN rows:     ",
        f"{len(gcn_rows):,}",
    )

    print()
    print(
        "Loading logistic model + scaler..."
    )

    (
        logistic_model,
        logistic_scaler,
        logistic_metadata,
    ) = load_logistic_artifacts()

    print(
        "Logistic artifacts loaded."
    )

    print()
    print(
        "LOGISTIC INFERENCE"
    )
    print(
        "------------------"
    )

    logistic_scored = (
        score_logistic_rows(
            logistic_rows,
            model=logistic_model,
            scaler=logistic_scaler,
        )
    )

    print(
        "Rows scored:",
        f"{len(logistic_scored):,}",
    )

    if len(
        logistic_scored
    ):

        print(
            "Predicted flood events:",
            f"{int(logistic_scored['predicted_flood_event'].sum()):,}",
        )

        print(
            "Probability range:",
            f"{logistic_scored['logistic_event_probability'].min():.6f}",
            "->",
            f"{logistic_scored['logistic_event_probability'].max():.6f}",
        )

    print()
    print(
        "Loading registered GCN champion..."
    )

    gcn_model = load_gcn_model()

    print(
        "Registered GCN loaded."
    )

    print()
    print(
        "GCN INFERENCE"
    )
    print(
        "-------------"
    )

    gcn_scored = score_gcn_rows(
        gcn_rows,
        model=gcn_model,
    )

    print(
        "Rows scored:",
        f"{len(gcn_scored):,}",
    )

    if len(
        gcn_scored
    ):

        print(
            "Predicted flood events:",
            f"{int(gcn_scored['predicted_flood_event'].sum()):,}",
        )

        print(
            "Predicted-minute range:",
            f"{gcn_scored['gcn_predicted_minutes_above_1inch'].min():.6f}",
            "->",
            f"{gcn_scored['gcn_predicted_minutes_above_1inch'].max():.6f}",
        )

    result = pd.concat(
        [
            logistic_scored,
            gcn_scored,
        ],
        ignore_index=True,
    )

    if len(
        result
    ) != len(
        inputs
    ):

        raise RuntimeError(
            "Combined model inference changed row count."
        )

    result[
        "forecast_generated_at"
    ] = pd.Timestamp.now(
        tz="UTC"
    )

    result[
        "gcn_model_uri"
    ] = np.where(
        result[
            "support_sensor_model"
        ]
        ==
        "gcn",
        GCN_MODEL_URI,
        None,
    )

    result[
        "logistic_model_artifact"
    ] = np.where(
        result[
            "support_sensor_model"
        ]
        ==
        "logistic",
        str(
            LOGISTIC_MODEL_PATH
        ),
        None,
    )

    result[
        "operational_prediction_source"
    ] = (
        result[
            "support_sensor_model"
        ]
    )

    result = (
        result
        .sort_values(
            [
                "forecast_hour",
                "grid_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    return (
        result,
        logistic_metadata,
    )


# ============================================================
# VALIDATION
# ============================================================

def validate_final_forecast(
    result: pd.DataFrame,
    inputs: pd.DataFrame,
) -> None:

    if len(
        result
    ) != len(
        inputs
    ):

        raise RuntimeError(
            "Final forecast row count does not match "
            "forecast input row count."
        )

    expected_keys = set(
        zip(
            inputs[
                "grid_id"
            ].astype(str),
            inputs[
                "forecast_hour"
            ],
        )
    )

    actual_keys = set(
        zip(
            result[
                "grid_id"
            ].astype(str),
            result[
                "forecast_hour"
            ],
        )
    )

    if expected_keys != actual_keys:

        raise RuntimeError(
            "Final forecast does not preserve the complete "
            "grid/hour population."
        )

    logistic_rows = (
        result[
            "support_sensor_model"
        ]
        ==
        "logistic"
    )

    gcn_rows = (
        result[
            "support_sensor_model"
        ]
        ==
        "gcn"
    )

    if (
        result.loc[
            logistic_rows,
            "logistic_event_probability",
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Some logistic rows are missing logistic probability."
        )

    if (
        result.loc[
            gcn_rows,
            "gcn_predicted_minutes_above_1inch",
        ]
        .isna()
        .any()
    ):

        raise RuntimeError(
            "Some GCN rows are missing predicted duration."
        )

    if (
        result.loc[
            logistic_rows,
            "gcn_predicted_minutes_above_1inch",
        ]
        .notna()
        .any()
    ):

        raise RuntimeError(
            "Logistic rows unexpectedly contain GCN predictions."
        )

    if (
        result.loc[
            gcn_rows,
            "logistic_event_probability",
        ]
        .notna()
        .any()
    ):

        raise RuntimeError(
            "GCN rows unexpectedly contain logistic probabilities."
        )

    if result[
        "predicted_flood_event"
    ].isna().any():

        raise RuntimeError(
            "Some forecast rows are missing the operational "
            "flood classification."
        )


# ============================================================
# SAVE
# ============================================================

def save_forecast(
    result: pd.DataFrame,
    *,
    logistic_metadata: dict,
    output_dir: Path,
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_parquet_path = (
        output_dir
        / "grid_flood_forecast.parquet"
    )

    output_csv_path = (
        output_dir
        / "grid_flood_forecast.csv"
    )

    output_summary_path = (
        output_dir
        / "grid_flood_forecast_summary.json"
    )

    result.to_parquet(
        output_parquet_path,
        index=False,
    )

    result.to_csv(
        output_csv_path,
        index=False,
    )

    logistic_rows = (
        result[
            "support_sensor_model"
        ]
        ==
        "logistic"
    )

    gcn_rows = (
        result[
            "support_sensor_model"
        ]
        ==
        "gcn"
    )

    summary = {
        "generated_at":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "forecast_mode":
            str(
                result[
                    "forecast_mode"
                ].iloc[0]
            ),

        "target_date_nyc":
            str(
                result[
                    "target_date_nyc"
                ].iloc[0]
            ),

        "rows":
            len(
                result
            ),

        "grid_cells":
            int(
                result[
                    "grid_id"
                ].nunique()
            ),

        "forecast_hours":
            int(
                result[
                    "forecast_hour"
                ].nunique()
            ),

        "support_sensors":
            int(
                result[
                    "support_sensor_id"
                ].nunique()
            ),

        "logistic_rows":
            int(
                logistic_rows.sum()
            ),

        "gcn_rows":
            int(
                gcn_rows.sum()
            ),

        "predicted_flood_rows":
            int(
                result[
                    "predicted_flood_event"
                ].sum()
            ),

        "logistic_predicted_flood_rows":
            int(
                result.loc[
                    logistic_rows,
                    "predicted_flood_event",
                ].sum()
            ),

        "gcn_predicted_flood_rows":
            int(
                result.loc[
                    gcn_rows,
                    "predicted_flood_event",
                ].sum()
            ),

        "logistic_threshold":
            LOGISTIC_THRESHOLD,

        "gcn_duration_threshold_minutes":
            GCN_DURATION_THRESHOLD_MINUTES,

        "gcn_model_uri":
            GCN_MODEL_URI,

        "mlflow_tracking_uri":
            MLFLOW_TRACKING_URI,

        "logistic_model_path":
            str(
                LOGISTIC_MODEL_PATH
            ),

        "logistic_features":
            logistic_metadata.get(
                "features",
                FEATURES,
            ),

        "forecast_start":
            result[
                "forecast_hour"
            ].min(),

        "forecast_end":
            result[
                "forecast_hour"
            ].max(),
    }

    summary = {
        key:
            safe_json_value(
                value
            )
        for key, value
        in summary.items()
    }

    with output_summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
            default=str,
        )

    return output_parquet_path, output_csv_path, output_summary_path


# ============================================================
# REPORT
# ============================================================

def print_final_report(
    result: pd.DataFrame,
    *,
    output_dir: Path,
) -> None:

    output_parquet_path = (
        output_dir
        / "grid_flood_forecast.parquet"
    )

    output_csv_path = (
        output_dir
        / "grid_flood_forecast.csv"
    )

    output_summary_path = (
        output_dir
        / "grid_flood_forecast_summary.json"
    )

    logistic_rows = (
        result[
            "support_sensor_model"
        ]
        ==
        "logistic"
    )

    gcn_rows = (
        result[
            "support_sensor_model"
        ]
        ==
        "gcn"
    )

    print()
    print(
        "FINAL NYC GRID FLOOD FORECAST"
    )
    print(
        "============================="
    )

    print(
        "Rows:                   ",
        f"{len(result):,}",
    )

    print(
        "Grid cells:             ",
        f"{result['grid_id'].nunique():,}",
    )

    print(
        "Forecast hours:         ",
        f"{result['forecast_hour'].nunique():,}",
    )

    print(
        "Unique support sensors: ",
        f"{result['support_sensor_id'].nunique():,}",
    )

    print()
    print(
        "MODEL ROUTING"
    )
    print(
        "-------------"
    )

    print(
        "Logistic rows:",
        f"{int(logistic_rows.sum()):,}",
    )

    print(
        "GCN rows:     ",
        f"{int(gcn_rows.sum()):,}",
    )

    print()
    print(
        "PREDICTED FLOOD EVENTS"
    )
    print(
        "----------------------"
    )

    print(
        "All rows:",
        f"{int(result['predicted_flood_event'].sum()):,}",
    )

    print(
        "Logistic:",
        f"{int(result.loc[logistic_rows, 'predicted_flood_event'].sum()):,}",
    )

    print(
        "GCN:",
        f"{int(result.loc[gcn_rows, 'predicted_flood_event'].sum()):,}",
    )

    if logistic_rows.any():

        print()
        print(
            "LOGISTIC EVENT PROBABILITY"
        )
        print(
            "--------------------------"
        )

        print(
            result.loc[
                logistic_rows,
                "logistic_event_probability",
            ]
            .describe()
            .to_string()
        )

    if gcn_rows.any():

        print()
        print(
            "GCN PREDICTED MINUTES ABOVE 1-INCH"
        )
        print(
            "----------------------------------"
        )

        print(
            result.loc[
                gcn_rows,
                "gcn_predicted_minutes_above_1inch",
            ]
            .describe()
            .to_string()
        )

    print()
    print(
        "FORECASTS BY HOUR"
    )
    print(
        "-----------------"
    )

    hourly = (
        result
        .groupby(
            "forecast_hour"
        )
        .agg(
            grid_cells=(
                "grid_id",
                "nunique",
            ),
            predicted_flood_grids=(
                "predicted_flood_event",
                "sum",
            ),
        )
    )

    print(
        hourly.to_string()
    )

    print()
    print(
        "SAVED OPERATIONAL FORECAST"
    )
    print(
        "--------------------------"
    )

    print(
        output_parquet_path
    )

    print(
        output_csv_path
    )

    print(
        output_summary_path
    )

    print()
    print(
        "NYC OPERATIONAL FLOOD FORECAST COMPLETE"
    )


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NYC 1-km operational flood-model inference."
    )
    parser.add_argument(
        "--mode",
        choices=SUPPORTED_FORECAST_MODES,
        default="tomorrow",
        help="Operational forecast product to score.",
    )
    parser.add_argument(
        "--inputs",
        default=None,
        help="Optional grid_forecast_inputs.parquet override.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    forecast_input_path, output_dir = resolve_operational_paths(
        mode=args.mode,
        inputs=args.inputs,
        output_dir=args.output_dir,
    )

    print("RUN NYC 1-KM OPERATIONAL FLOOD FORECAST")
    print("=======================================")
    print()
    print("OPERATIONAL CONTRACT")
    print("--------------------")
    print(f"Forecast mode:          {args.mode}")
    print("Training rainfall:      MRMS")
    print("Operational rainfall:   HRRR")
    print("Forecast geography:     NYC 1-km grids")
    print("Model routing:          selected support sensor")
    print(f"Logistic event rule:    probability >= {LOGISTIC_THRESHOLD}")
    print(
        "GCN event rule:         predicted minutes > "
        f"{GCN_DURATION_THRESHOLD_MINUTES}"
    )
    print(f"Input artifact:         {forecast_input_path}")
    print(f"Output directory:       {output_dir}")

    inputs = load_forecast_inputs(forecast_input_path)

    if "forecast_mode" in inputs.columns:
        input_modes = set(inputs["forecast_mode"].dropna().astype(str).str.lower())
        if input_modes and input_modes != {args.mode}:
            raise RuntimeError(
                f"Input forecast_mode {sorted(input_modes)} does not match "
                f"requested mode {args.mode!r}."
            )

    print()
    print("INPUT AUDIT")
    print("-----------")
    print("Rows:           ", f"{len(inputs):,}")
    print("Grid cells:     ", f"{inputs['grid_id'].nunique():,}")
    print("Forecast hours: ", f"{inputs['forecast_hour'].nunique():,}")
    print("Support sensors:", f"{inputs['support_sensor_id'].nunique():,}")

    result, logistic_metadata = run_selected_model_inference(inputs)
    validate_final_forecast(result, inputs)

    save_forecast(
        result,
        logistic_metadata=logistic_metadata,
        output_dir=output_dir,
    )

    print_final_report(
        result,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()