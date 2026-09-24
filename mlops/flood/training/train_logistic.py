"""
Precipitation-only logistic-regression baseline for NYC flood prediction.

This pipeline intentionally mirrors the data contract used by train_gcn.py:

    predictors:
        precip_current_hour_mm
        precip_previous_6h_mm
        daily_total_precip_mm

    source response:
        minutes_above_1p5_inch
        (legacy canonical column name)

    active semantic response:
        minutes_above_1inch

Because sklearn LogisticRegression is a binary classifier, the continuous
GCN duration response is converted to:

    actual_event = minutes_above_1inch > 0

The logistic model is deliberately simple and preserves the configuration
used by the existing project baseline:

    LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )

The model is trained globally across eligible sensor-hour observations.
Evaluation is reported both overall and per sensor so it can later be
compared with GCN event performance sensor by sensor.

No sensor/model selection is performed in this script.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

from mlops.flood.training import train_gcn as gcn


# ============================================================
# MODEL CONTRACT
# ============================================================

FEATURES = list(
    gcn.FEATURES
)

TARGET_COLUMN = (
    gcn.TARGET_COLUMN
)

SENSOR_ID = (
    gcn.SENSOR_ID
)

TIME_COLUMN = (
    gcn.TIME_COLUMN
)

TRAIN_FRACTION = (
    gcn.TRAIN_FRACTION
)

VALIDATION_FRACTION = (
    gcn.VALIDATION_FRACTION
)

# The observed target is a duration in minutes.
# Any scientifically positive observed duration represents an event.
#
# Use the GCN's tiny zero tolerance so numerical noise around zero is
# not interpreted as an observed event.
EVENT_ZERO_TOLERANCE_MINUTES = float(
    getattr(
        gcn,
        "RAIN_ZERO_TOLERANCE_MM",
        1e-6,
    )
)

# Standard sklearn classification threshold.
CLASSIFICATION_THRESHOLD = 0.50


# ============================================================
# OUTPUT PATHS
# ============================================================

LOGISTIC_OUTPUT_DIR = Path(
    "artifacts"
) / "flood" / "logistic"

MODEL_PATH = (
    LOGISTIC_OUTPUT_DIR
    / "precipitation_only_logistic.pkl"
)

SCALER_PATH = (
    LOGISTIC_OUTPUT_DIR
    / "feature_scaler.pkl"
)

TEST_PREDICTIONS_PATH = (
    LOGISTIC_OUTPUT_DIR
    / "test_predictions.parquet"
)

VALIDATION_PREDICTIONS_PATH = (
    LOGISTIC_OUTPUT_DIR
    / "validation_predictions.parquet"
)

OVERALL_METRICS_PATH = (
    LOGISTIC_OUTPUT_DIR
    / "metrics.csv"
)

SENSOR_METRICS_PATH = (
    LOGISTIC_OUTPUT_DIR
    / "sensor_metrics.csv"
)

METADATA_PATH = (
    LOGISTIC_OUTPUT_DIR
    / "metadata.json"
)


# ============================================================
# TARGET CONSTRUCTION
# ============================================================


def build_event_target(
    target: pd.Series,
) -> pd.Series:
    """
    Convert the GCN continuous duration response to a binary event target.

    Positive class:
        minutes_above_1inch > zero tolerance

    Missing continuous responses remain missing so rows without an observed
    response are never silently converted to negative examples.
    """

    numeric = pd.to_numeric(
        target,
        errors="coerce",
    )

    result = pd.Series(
        pd.NA,
        index=numeric.index,
        dtype="Int64",
    )

    observed = numeric.notna()

    result.loc[
        observed
    ] = (
        numeric.loc[
            observed
        ]
        > EVENT_ZERO_TOLERANCE_MINUTES
    ).astype(int)

    return result


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================


def chronological_split(
    df: pd.DataFrame,
    *,
    split_reference: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Timestamp,
    pd.Timestamp,
]:
    """
    Apply the exact GCN chronological split boundaries to logistic rows.

    Split boundaries are calculated from the same dirty-filtered GCN
    training view BEFORE logistic-specific removal of rows with missing
    targets or predictors.

    This keeps the GCN and logistic test periods directly comparable.
    """

    if TIME_COLUMN not in split_reference.columns:
        raise ValueError(
            f"Split reference is missing {TIME_COLUMN!r}."
        )

    hours = (
        split_reference[TIME_COLUMN]
        .dropna()
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )

    n_hours = len(hours)

    if n_hours < 3:
        raise RuntimeError(
            "Not enough unique hours for "
            "train/validation/test split."
        )

    train_end_pos = int(
        n_hours * TRAIN_FRACTION
    )

    validation_end_pos = int(
        n_hours
        * (
            TRAIN_FRACTION
            + VALIDATION_FRACTION
        )
    )

    if (
        train_end_pos <= 0
        or validation_end_pos <= train_end_pos
        or validation_end_pos >= n_hours
    ):
        raise RuntimeError(
            "Chronological split positions are invalid."
        )

    train_end_hour = hours.iloc[
        train_end_pos - 1
    ]

    validation_end_hour = hours.iloc[
        validation_end_pos - 1
    ]

    train = df.loc[
        df[TIME_COLUMN] <= train_end_hour
    ].copy()

    validation = df.loc[
        (
            df[TIME_COLUMN] > train_end_hour
        )
        & (
            df[TIME_COLUMN]
            <= validation_end_hour
        )
    ].copy()

    test = df.loc[
        df[TIME_COLUMN] > validation_end_hour
    ].copy()

    if train.empty:
        raise RuntimeError(
            "Logistic training split is empty."
        )

    if validation.empty:
        raise RuntimeError(
            "Logistic validation split is empty."
        )

    if test.empty:
        raise RuntimeError(
            "Logistic test split is empty."
        )

    print(
        "\nCHRONOLOGICAL SPLIT"
    )
    print(
        "-------------------"
    )
    print(
        f"Reference unique hours: {n_hours:,}"
    )
    print(
        f"Train through:          {train_end_hour}"
    )
    print(
        f"Validation through:     {validation_end_hour}"
    )
    print(
        f"Test begins after:      {validation_end_hour}"
    )
    print(
        f"Train logistic rows:    {len(train):,}"
    )
    print(
        f"Validation rows:        {len(validation):,}"
    )
    print(
        f"Test logistic rows:     {len(test):,}"
    )

    return (
        train,
        validation,
        test,
        train_end_hour,
        validation_end_hour,
    )


# ============================================================
# MODELING FRAME
# ============================================================


def prepare_modeling_rows(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the logistic-regression modeling frame.

    Rows must have:
        - deployment_id
        - hour
        - complete precipitation predictors
        - an observed duration response

    The same canonical duration target used by the GCN is retained for
    later sensor-level comparison.
    """

    required = {
        SENSOR_ID,
        TIME_COLUMN,
        TARGET_COLUMN,
        *FEATURES,
    }

    missing = (
        required
        - set(
            df.columns
        )
    )

    if missing:
        raise ValueError(
            "Logistic training data is missing "
            "required columns: "
            + ", ".join(
                sorted(
                    missing
                )
            )
        )

    result = df[
        [
            TIME_COLUMN,
            SENSOR_ID,
            TARGET_COLUMN,
            *FEATURES,
        ]
    ].copy()

    result[
        SENSOR_ID
    ] = (
        result[
            SENSOR_ID
        ]
        .astype(str)
    )

    result[
        TIME_COLUMN
    ] = pd.to_datetime(
        result[
            TIME_COLUMN
        ],
        utc=True,
        errors="coerce",
    )

    result[
        TARGET_COLUMN
    ] = pd.to_numeric(
        result[
            TARGET_COLUMN
        ],
        errors="coerce",
    )

    for column in FEATURES:
        result[
            column
        ] = pd.to_numeric(
            result[
                column
            ],
            errors="coerce",
        )

    result[
        "actual_event"
    ] = build_event_target(
        result[
            TARGET_COLUMN
        ]
    )

    result = result.replace(
        [
            np.inf,
            -np.inf,
        ],
        np.nan,
    )

    valid_mask = (
        result[
            TIME_COLUMN
        ].notna()
        & result[
            SENSOR_ID
        ].notna()
        & result[
            TARGET_COLUMN
        ].notna()
        & result[
            "actual_event"
        ].notna()
        & result[
            FEATURES
        ].notna().all(
            axis=1
        )
    )

    result = (
        result.loc[
            valid_mask
        ]
        .copy()
        .sort_values(
            [
                TIME_COLUMN,
                SENSOR_ID,
            ]
        )
        .reset_index(
            drop=True
        )
    )

    result[
        "actual_event"
    ] = (
        result[
            "actual_event"
        ]
        .astype(int)
    )

    if result.empty:
        raise RuntimeError(
            "No complete observed rows are "
            "available for logistic training."
        )

    return result


# ============================================================
# TRAIN
# ============================================================


def fit_model(
    train: pd.DataFrame,
) -> tuple[
    StandardScaler,
    LogisticRegression,
]:
    """
    Fit StandardScaler on training predictors only, followed by the project
    logistic-regression baseline.
    """

    if train.empty:
        raise RuntimeError(
            "Logistic training split is empty."
        )

    class_counts = (
        train[
            "actual_event"
        ]
        .value_counts()
        .sort_index()
    )

    if train[
        "actual_event"
    ].nunique() < 2:
        raise RuntimeError(
            "Logistic training split does not "
            "contain both event classes."
        )

    print(
        "\nTRAINING CLASS COUNTS"
    )

    print(
        "---------------------"
    )

    for value, count in (
        class_counts.items()
    ):
        print(
            f"class {value}: "
            f"{count:,}"
        )

    scaler = StandardScaler()

    scaler.fit(
        train[
            FEATURES
        ]
    )

    x_train = scaler.transform(
        train[
            FEATURES
        ]
    )

    y_train = (
        train[
            "actual_event"
        ]
        .astype(int)
        .to_numpy()
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
    )

    model.fit(
        x_train,
        y_train,
    )

    print(
        "\nLOGISTIC MODEL"
    )

    print(
        "--------------"
    )

    print(
        "max_iter:     2000"
    )

    print(
        "class_weight: balanced"
    )

    print(
        f"iterations:   "
        f"{int(model.n_iter_[0])}"
    )

    return (
        scaler,
        model,
    )


# ============================================================
# PREDICTION
# ============================================================


def predict_split(
    df: pd.DataFrame,
    *,
    scaler: StandardScaler,
    model: LogisticRegression,
    split_name: str,
) -> pd.DataFrame:
    """
    Score one chronological split.
    """

    if df.empty:
        raise RuntimeError(
            f"{split_name} split is empty."
        )

    x = scaler.transform(
        df[
            FEATURES
        ]
    )

    probability = (
        model.predict_proba(
            x
        )[
            :,
            1,
        ]
    )

    predicted_event = (
        probability
        >= CLASSIFICATION_THRESHOLD
    )

    result = df[
        [
            TIME_COLUMN,
            SENSOR_ID,
            TARGET_COLUMN,
            "actual_event",
        ]
    ].copy()

    result = result.rename(
        columns={
            TARGET_COLUMN:
                "actual_minutes_above_1inch",
        }
    )

    result[
        "logistic_event_probability"
    ] = probability.astype(
        float
    )

    result[
        "logistic_predicted_event"
    ] = predicted_event.astype(
        bool
    )

    result[
        "split"
    ] = split_name

    return result


# ============================================================
# METRICS
# ============================================================


def safe_roc_auc(
    y_true: pd.Series,
    probability: pd.Series,
) -> float:
    if y_true.nunique() < 2:
        return float(
            "nan"
        )

    return float(
        roc_auc_score(
            y_true,
            probability,
        )
    )


def safe_average_precision(
    y_true: pd.Series,
    probability: pd.Series,
) -> float:
    if y_true.nunique() < 2:
        return float(
            "nan"
        )

    return float(
        average_precision_score(
            y_true,
            probability,
        )
    )


def calculate_binary_metrics(
    predictions: pd.DataFrame,
) -> dict[str, float | int]:
    """
    Calculate binary event metrics for one prediction frame.
    """

    y_true = (
        predictions[
            "actual_event"
        ]
        .astype(int)
    )

    y_pred = (
        predictions[
            "logistic_predicted_event"
        ]
        .astype(int)
    )

    probability = (
        predictions[
            "logistic_event_probability"
        ]
        .astype(float)
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = (
        matrix.ravel()
    )

    return {
        "n_rows": int(
            len(
                predictions
            )
        ),
        "actual_events": int(
            y_true.sum()
        ),
        "predicted_events": int(
            y_pred.sum()
        ),
        "true_positive": int(
            tp
        ),
        "false_positive": int(
            fp
        ),
        "true_negative": int(
            tn
        ),
        "false_negative": int(
            fn
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "roc_auc": safe_roc_auc(
            y_true,
            probability,
        ),
        "average_precision": (
            safe_average_precision(
                y_true,
                probability,
            )
        ),
    }


def calculate_sensor_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate held-out event metrics independently for each sensor.

    These are the metrics that will later be joined to the GCN sensor-level
    results for model selection.
    """

    rows: list[
        dict
    ] = []

    for sensor_id, group in (
        predictions.groupby(
            SENSOR_ID,
            sort=True,
        )
    ):

        metrics = (
            calculate_binary_metrics(
                group
            )
        )

        rows.append(
            {
                SENSOR_ID:
                    sensor_id,
                **metrics,
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        raise RuntimeError(
            "No per-sensor logistic metrics "
            "were produced."
        )

    return (
        result
        .sort_values(
            SENSOR_ID
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# SAVE
# ============================================================


def save_outputs(
    *,
    scaler: StandardScaler,
    model: LogisticRegression,
    validation_predictions: pd.DataFrame,
    test_predictions: pd.DataFrame,
    overall_metrics: dict,
    sensor_metrics: pd.DataFrame,
    train_end_hour: pd.Timestamp,
    validation_end_hour: pd.Timestamp,
) -> None:

    LOGISTIC_OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with MODEL_PATH.open(
        "wb"
    ) as handle:
        pickle.dump(
            model,
            handle,
        )

    with SCALER_PATH.open(
        "wb"
    ) as handle:
        pickle.dump(
            {
                "feature_scaler":
                    scaler,
                "features":
                    FEATURES,
            },
            handle,
        )

    validation_predictions.to_parquet(
        VALIDATION_PREDICTIONS_PATH,
        index=False,
    )

    test_predictions.to_parquet(
        TEST_PREDICTIONS_PATH,
        index=False,
    )

    pd.DataFrame(
        [
            {
                "metric":
                    key,
                "value":
                    value,
            }
            for key, value
            in overall_metrics.items()
        ]
    ).to_csv(
        OVERALL_METRICS_PATH,
        index=False,
    )

    sensor_metrics.to_csv(
        SENSOR_METRICS_PATH,
        index=False,
    )

    metadata = {
        "model_family":
            "logistic-regression",
        "predictor_family":
            "precipitation-only",
        "features":
            FEATURES,
        "source_target":
            gcn.SOURCE_TARGET_COLUMN,
        "continuous_target":
            TARGET_COLUMN,
        "binary_target":
            "actual_event",
        "binary_target_rule":
            (
                "minutes_above_1inch "
                f"> {EVENT_ZERO_TOLERANCE_MINUTES}"
            ),
        "classification_threshold":
            CLASSIFICATION_THRESHOLD,
        "max_iter":
            2000,
        "class_weight":
            "balanced",
        "train_fraction":
            TRAIN_FRACTION,
        "validation_fraction":
            VALIDATION_FRACTION,
        "test_fraction":
            (
                1.0
                - TRAIN_FRACTION
                - VALIDATION_FRACTION
            ),
        "train_end_hour":
            str(
                train_end_hour
            ),
        "validation_end_hour":
            str(
                validation_end_hour
            ),
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )

    print(
        "\nSAVED OUTPUTS"
    )

    print(
        "-------------"
    )

    print(
        LOGISTIC_OUTPUT_DIR
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "PRECIPITATION-ONLY LOGISTIC BASELINE"
    )

    print(
        "===================================="
    )

    print(
        f"Canonical source target: "
        f"{gcn.SOURCE_TARGET_COLUMN}"
    )

    print(
        f"Continuous target: "
        f"{TARGET_COLUMN}"
    )

    print(
        "Binary response: "
        "minutes_above_1inch > 0"
    )

    print(
        f"Features: "
        f"{FEATURES}"
    )

    # --------------------------------------------------------
    # Same canonical data as GCN
    # --------------------------------------------------------

    df = (
        gcn.load_training_features()
    )

    # --------------------------------------------------------
    # Same dirty sensor-day exclusion as GCN
    # --------------------------------------------------------

    df, _ = (
        gcn.remove_dirty_sensor_days(
            df
        )
    )

    modeling = (
        prepare_modeling_rows(
            df
        )
    )

    print(
        "\nLOGISTIC MODELING VIEW"
    )

    print(
        "----------------------"
    )

    print(
        f"Rows:       "
        f"{len(modeling):,}"
    )

    print(
        f"Sensors:    "
        f"{modeling[SENSOR_ID].nunique():,}"
    )

    print(
        f"Time range: "
        f"{modeling[TIME_COLUMN].min()} "
        f"-> "
        f"{modeling[TIME_COLUMN].max()}"
    )

    print(
        f"Events:     "
        f"{int(modeling['actual_event'].sum()):,}"
    )

    # --------------------------------------------------------
    # Same chronological hour boundaries as GCN
    # --------------------------------------------------------

    (
        train,
        validation,
        test,
        train_end_hour,
        validation_end_hour,
    ) = chronological_split(
        modeling,
        split_reference=df,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    scaler, model = (
        fit_model(
            train
        )
    )

    # --------------------------------------------------------
    # Validation predictions retained for diagnostics only.
    #
    # We do NOT tune the classification threshold here.
    # Keeping 0.50 avoids introducing a new hyperparameter
    # after observing validation performance.
    # --------------------------------------------------------

    validation_predictions = (
        predict_split(
            validation,
            scaler=scaler,
            model=model,
            split_name="validation",
        )
    )

    # --------------------------------------------------------
    # Final held-out test predictions
    # --------------------------------------------------------

    test_predictions = (
        predict_split(
            test,
            scaler=scaler,
            model=model,
            split_name="test",
        )
    )

    overall_metrics = (
        calculate_binary_metrics(
            test_predictions
        )
    )

    sensor_metrics = (
        calculate_sensor_metrics(
            test_predictions
        )
    )

    print(
        "\nTEST EVENT METRICS"
    )

    print(
        "------------------"
    )

    print(
        f"Rows:              "
        f"{overall_metrics['n_rows']:,}"
    )

    print(
        f"Actual events:     "
        f"{overall_metrics['actual_events']:,}"
    )

    print(
        f"Predicted events:  "
        f"{overall_metrics['predicted_events']:,}"
    )

    print(
        f"Precision:         "
        f"{overall_metrics['precision']:.6f}"
    )

    print(
        f"Recall:            "
        f"{overall_metrics['recall']:.6f}"
    )

    print(
        f"F1:                "
        f"{overall_metrics['f1']:.6f}"
    )

    print(
        f"ROC-AUC:           "
        f"{overall_metrics['roc_auc']:.6f}"
    )

    print(
        f"Average precision: "
        f"{overall_metrics['average_precision']:.6f}"
    )

    save_outputs(
        scaler=scaler,
        model=model,
        validation_predictions=(
            validation_predictions
        ),
        test_predictions=(
            test_predictions
        ),
        overall_metrics=(
            overall_metrics
        ),
        sensor_metrics=(
            sensor_metrics
        ),
        train_end_hour=(
            train_end_hour
        ),
        validation_end_hour=(
            validation_end_hour
        ),
    )

    print(
        "\nLOGISTIC TRAINING COMPLETE"
    )


if __name__ == "__main__":
    main()