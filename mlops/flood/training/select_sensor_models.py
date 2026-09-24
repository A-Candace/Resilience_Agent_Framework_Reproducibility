"""
Per-sensor model selection and eligibility for NYC flood prediction.

This script compares held-out test predictions from:

    1. precipitation-only GCN
    2. precipitation-only logistic regression

The purpose is NOT to identify one globally superior model.

Instead, for each FloodNet sensor:

    1. evaluate GCN event performance;
    2. evaluate logistic event performance;
    3. determine whether each model independently meets the minimum
       precision and recall requirements;
    4. choose the highest-F1 model ONLY from the models that satisfy
       those minimum requirements;
    5. mark the sensor ineligible if neither model qualifies.

Quality gates
-------------

A model is eligible for a sensor only when:

    precision >= MIN_PRECISION
    recall    >= MIN_RECALL

Current project requirements:

    MIN_PRECISION = 0.25
    MIN_RECALL    = 0.30

Model selection
---------------

If both models qualify:

    higher F1 wins.

Tie-breakers:

    1. higher recall
    2. higher precision
    3. fewer false-alarm event hours
    4. GCN as final deterministic tie-break

If only one model qualifies:

    select that model.

If neither qualifies:

    selected_model = None
    sensor_eligible = False

Event definitions
-----------------

Observed event:

    actual_minutes_above_1inch > EVENT_ZERO_TOLERANCE_MINUTES

GCN predicted event:

    predicted_minutes_above_1inch
        > GCN_PREDICTED_EVENT_THRESHOLD_MINUTES

Logistic predicted event:

    logistic_event_probability
        >= LOGISTIC_CLASSIFICATION_THRESHOLD

Event matching
--------------

To remain aligned with the existing GCN event-evaluation methodology,
an observed event hour is considered captured when at least one predicted
event occurs within +/- MATCH_WINDOW_HOURS.

Likewise, a predicted event is considered matched when at least one actual
event occurs within +/- MATCH_WINDOW_HOURS.

Outputs
-------

artifacts/flood/model_selection/

    sensor_model_selection.csv
    sensor_model_selection.parquet
    eligible_sensor_model_registry.csv
    eligible_sensor_model_registry.parquet
    model_selection_summary.csv
    matched_test_predictions.parquet
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# INPUT PATHS
# ============================================================

GCN_TEST_PREDICTIONS_PATH = (
    Path("artifacts")
    / "flood"
    / "gcn"
    / "test_predictions.parquet"
)

LOGISTIC_TEST_PREDICTIONS_PATH = (
    Path("artifacts")
    / "flood"
    / "logistic"
    / "test_predictions.parquet"
)


# ============================================================
# OUTPUT PATHS
# ============================================================

OUTPUT_DIR = (
    Path("artifacts")
    / "flood"
    / "model_selection"
)

SENSOR_SELECTION_PATH = (
    OUTPUT_DIR
    / "sensor_model_selection.csv"
)

SENSOR_SELECTION_PARQUET_PATH = (
    OUTPUT_DIR
    / "sensor_model_selection.parquet"
)

ELIGIBLE_SENSOR_REGISTRY_PATH = (
    OUTPUT_DIR
    / "eligible_sensor_model_registry.csv"
)

ELIGIBLE_SENSOR_REGISTRY_PARQUET_PATH = (
    OUTPUT_DIR
    / "eligible_sensor_model_registry.parquet"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "model_selection_summary.csv"
)

MATCHED_TEST_PATH = (
    OUTPUT_DIR
    / "matched_test_predictions.parquet"
)


# ============================================================
# COLUMN CONTRACT
# ============================================================

TIME_COLUMN = "hour"

SENSOR_ID = "deployment_id"

ACTUAL_MINUTES_COLUMN = (
    "actual_minutes_above_1inch"
)

GCN_PREDICTION_COLUMN = (
    "predicted_minutes_above_1inch"
)

LOGISTIC_PROBABILITY_COLUMN = (
    "logistic_event_probability"
)

LOGISTIC_PREDICTED_EVENT_COLUMN = (
    "logistic_predicted_event"
)


# ============================================================
# EVENT DEFINITIONS
# ============================================================

EVENT_ZERO_TOLERANCE_MINUTES = 1e-6

GCN_PREDICTED_EVENT_THRESHOLD_MINUTES = 1.0

LOGISTIC_CLASSIFICATION_THRESHOLD = 0.50

MATCH_WINDOW_HOURS = 24


# ============================================================
# MODEL / SENSOR ELIGIBILITY REQUIREMENTS
# ============================================================

MIN_PRECISION = 0.25

MIN_RECALL = 0.30


# ============================================================
# INPUT VALIDATION
# ============================================================


def require_file(
    path: Path,
    *,
    description: str,
) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found: {path}"
        )


def validate_gcn_predictions(
    df: pd.DataFrame,
) -> None:
    required = {
        TIME_COLUMN,
        SENSOR_ID,
        ACTUAL_MINUTES_COLUMN,
        GCN_PREDICTION_COLUMN,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "GCN test predictions are missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )


def validate_logistic_predictions(
    df: pd.DataFrame,
) -> None:
    required = {
        TIME_COLUMN,
        SENSOR_ID,
        ACTUAL_MINUTES_COLUMN,
        LOGISTIC_PROBABILITY_COLUMN,
        LOGISTIC_PREDICTED_EVENT_COLUMN,
    }

    missing = (
        required
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Logistic test predictions are missing required columns: "
            + ", ".join(
                sorted(missing)
            )
        )


# ============================================================
# LOAD TEST PREDICTIONS
# ============================================================


def load_test_predictions() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    require_file(
        GCN_TEST_PREDICTIONS_PATH,
        description="GCN test predictions",
    )

    require_file(
        LOGISTIC_TEST_PREDICTIONS_PATH,
        description="Logistic test predictions",
    )

    gcn = pd.read_parquet(
        GCN_TEST_PREDICTIONS_PATH
    )

    logistic = pd.read_parquet(
        LOGISTIC_TEST_PREDICTIONS_PATH
    )

    validate_gcn_predictions(
        gcn
    )

    validate_logistic_predictions(
        logistic
    )

    for df in (
        gcn,
        logistic,
    ):
        df[TIME_COLUMN] = pd.to_datetime(
            df[TIME_COLUMN],
            utc=True,
            errors="coerce",
        )

        df[SENSOR_ID] = (
            df[SENSOR_ID]
            .astype(str)
            .str.strip()
        )

    return (
        gcn,
        logistic,
    )


# ============================================================
# MATCH TEST POPULATIONS
# ============================================================


def build_matched_test_frame(
    gcn: pd.DataFrame,
    logistic: pd.DataFrame,
) -> pd.DataFrame:
    keys = [
        TIME_COLUMN,
        SENSOR_ID,
    ]

    gcn_duplicate_count = int(
        gcn.duplicated(
            keys
        ).sum()
    )

    logistic_duplicate_count = int(
        logistic.duplicated(
            keys
        ).sum()
    )

    if gcn_duplicate_count:
        raise ValueError(
            "GCN test predictions contain duplicate "
            f"sensor-hour keys: {gcn_duplicate_count:,}"
        )

    if logistic_duplicate_count:
        raise ValueError(
            "Logistic test predictions contain duplicate "
            f"sensor-hour keys: {logistic_duplicate_count:,}"
        )

    joined = gcn.merge(
        logistic,
        on=keys,
        how="outer",
        indicator=True,
        suffixes=(
            "_gcn",
            "_logistic",
        ),
    )

    join_counts = (
        joined["_merge"]
        .value_counts()
    )

    left_only = int(
        join_counts.get(
            "left_only",
            0,
        )
    )

    right_only = int(
        join_counts.get(
            "right_only",
            0,
        )
    )

    if (
        left_only > 0
        or right_only > 0
    ):
        raise ValueError(
            "GCN and logistic test populations do not match exactly. "
            f"GCN-only rows: {left_only:,}; "
            f"logistic-only rows: {right_only:,}."
        )

    joined = (
        joined.loc[
            joined["_merge"]
            == "both"
        ]
        .drop(
            columns=[
                "_merge"
            ]
        )
        .copy()
    )

    gcn_actual = pd.to_numeric(
        joined[
            f"{ACTUAL_MINUTES_COLUMN}_gcn"
        ],
        errors="coerce",
    )

    logistic_actual = pd.to_numeric(
        joined[
            f"{ACTUAL_MINUTES_COLUMN}_logistic"
        ],
        errors="coerce",
    )

    # The logistic artifact retains the original unscaled response and is
    # therefore used as the common observed value for event comparison.
    #
    # The saved GCN target differs only by tiny inverse-scaling floating-point
    # residues in a very small fraction of rows.
    joined[
        ACTUAL_MINUTES_COLUMN
    ] = logistic_actual

    joined[
        "actual_target_absolute_difference"
    ] = (
        gcn_actual
        - logistic_actual
    ).abs()

    joined[
        "actual_event"
    ] = (
        joined[
            ACTUAL_MINUTES_COLUMN
        ]
        > EVENT_ZERO_TOLERANCE_MINUTES
    )

    joined[
        "gcn_predicted_event"
    ] = (
        pd.to_numeric(
            joined[
                GCN_PREDICTION_COLUMN
            ],
            errors="coerce",
        )
        > GCN_PREDICTED_EVENT_THRESHOLD_MINUTES
    )

    joined[
        "logistic_predicted_event_comparison"
    ] = (
        pd.to_numeric(
            joined[
                LOGISTIC_PROBABILITY_COLUMN
            ],
            errors="coerce",
        )
        >= LOGISTIC_CLASSIFICATION_THRESHOLD
    )

    required_non_null = [
        ACTUAL_MINUTES_COLUMN,
        GCN_PREDICTION_COLUMN,
        LOGISTIC_PROBABILITY_COLUMN,
    ]

    invalid_mask = (
        joined[
            required_non_null
        ]
        .isna()
        .any(
            axis=1
        )
    )

    if invalid_mask.any():
        raise ValueError(
            "Matched test frame contains "
            f"{int(invalid_mask.sum()):,} rows with missing "
            "actual or predicted values."
        )

    return (
        joined
        .sort_values(
            [
                SENSOR_ID,
                TIME_COLUMN,
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# EVENT MATCHING
# ============================================================


def evaluate_event_times(
    actual_times: list[pd.Timestamp],
    predicted_times: list[pd.Timestamp],
) -> dict[str, float | int]:
    """
    Evaluate one model for one sensor using the existing +/- 24-hour
    event-matching concept.
    """

    window = pd.Timedelta(
        hours=MATCH_WINDOW_HOURS
    )

    actual_total = len(
        actual_times
    )

    predicted_total = len(
        predicted_times
    )

    if actual_total:
        captured_actual = sum(
            int(
                any(
                    abs(
                        predicted_time
                        - actual_time
                    )
                    <= window
                    for predicted_time
                    in predicted_times
                )
            )
            for actual_time
            in actual_times
        )
    else:
        captured_actual = 0

    if predicted_total:
        matched_predictions = sum(
            int(
                any(
                    abs(
                        actual_time
                        - predicted_time
                    )
                    <= window
                    for actual_time
                    in actual_times
                )
            )
            for predicted_time
            in predicted_times
        )
    else:
        matched_predictions = 0

    false_alarms = (
        predicted_total
        - matched_predictions
    )

    missed_events = (
        actual_total
        - captured_actual
    )

    precision = (
        matched_predictions
        / predicted_total
        if predicted_total
        else 0.0
    )

    recall = (
        captured_actual
        / actual_total
        if actual_total
        else 0.0
    )

    if (
        precision
        + recall
    ) > 0:
        f1 = (
            2.0
            * precision
            * recall
            / (
                precision
                + recall
            )
        )
    else:
        f1 = 0.0

    return {
        "actual_events":
            int(
                actual_total
            ),
        "predicted_events":
            int(
                predicted_total
            ),
        "captured_actual_events":
            int(
                captured_actual
            ),
        "matched_predicted_events":
            int(
                matched_predictions
            ),
        "false_alarm_events":
            int(
                false_alarms
            ),
        "missed_events":
            int(
                missed_events
            ),
        "precision":
            float(
                precision
            ),
        "recall":
            float(
                recall
            ),
        "f1":
            float(
                f1
            ),
    }


# ============================================================
# SENSOR-LEVEL EVALUATION
# ============================================================


def evaluate_sensor(
    group: pd.DataFrame,
) -> dict:
    sensor_id = str(
        group[
            SENSOR_ID
        ].iloc[
            0
        ]
    )

    group = (
        group
        .sort_values(
            TIME_COLUMN
        )
        .copy()
    )

    actual_times = (
        group.loc[
            group[
                "actual_event"
            ],
            TIME_COLUMN,
        ]
        .tolist()
    )

    gcn_times = (
        group.loc[
            group[
                "gcn_predicted_event"
            ],
            TIME_COLUMN,
        ]
        .tolist()
    )

    logistic_times = (
        group.loc[
            group[
                "logistic_predicted_event_comparison"
            ],
            TIME_COLUMN,
        ]
        .tolist()
    )

    gcn_metrics = (
        evaluate_event_times(
            actual_times,
            gcn_times,
        )
    )

    logistic_metrics = (
        evaluate_event_times(
            actual_times,
            logistic_times,
        )
    )

    result = {
        SENSOR_ID:
            sensor_id,
        "test_rows":
            int(
                len(
                    group
                )
            ),
        "test_start":
            group[
                TIME_COLUMN
            ].min(),
        "test_end":
            group[
                TIME_COLUMN
            ].max(),
        "actual_events":
            int(
                len(
                    actual_times
                )
            ),
    }

    for key, value in (
        gcn_metrics.items()
    ):
        if key == "actual_events":
            continue

        result[
            f"gcn_{key}"
        ] = value

    for key, value in (
        logistic_metrics.items()
    ):
        if key == "actual_events":
            continue

        result[
            f"logistic_{key}"
        ] = value

    return result


def calculate_sensor_comparison(
    matched: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict
    ] = []

    grouped = matched.groupby(
        SENSOR_ID,
        sort=True,
    )

    total_sensors = matched[
        SENSOR_ID
    ].nunique()

    print(
        "\nPER-SENSOR EVENT EVALUATION"
    )

    print(
        "---------------------------"
    )

    print(
        f"Sensors: {total_sensors:,}"
    )

    for index, (
        _,
        group,
    ) in enumerate(
        grouped,
        start=1,
    ):

        if (
            index == 1
            or index % 50 == 0
            or index == total_sensors
        ):
            print(
                f"Evaluating sensor "
                f"{index:,}/{total_sensors:,}"
            )

        rows.append(
            evaluate_sensor(
                group
            )
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        raise RuntimeError(
            "No sensor-level model comparison "
            "rows were produced."
        )

    return result


# ============================================================
# MODEL ELIGIBILITY
# ============================================================


def add_model_eligibility(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply minimum precision and recall requirements separately to GCN and
    logistic regression before model selection.
    """

    result = comparison.copy()

    result[
        "gcn_meets_min_precision"
    ] = (
        result[
            "gcn_precision"
        ]
        >= MIN_PRECISION
    )

    result[
        "gcn_meets_min_recall"
    ] = (
        result[
            "gcn_recall"
        ]
        >= MIN_RECALL
    )

    result[
        "gcn_eligible"
    ] = (
        result[
            "gcn_meets_min_precision"
        ]
        & result[
            "gcn_meets_min_recall"
        ]
    )

    result[
        "logistic_meets_min_precision"
    ] = (
        result[
            "logistic_precision"
        ]
        >= MIN_PRECISION
    )

    result[
        "logistic_meets_min_recall"
    ] = (
        result[
            "logistic_recall"
        ]
        >= MIN_RECALL
    )

    result[
        "logistic_eligible"
    ] = (
        result[
            "logistic_meets_min_precision"
        ]
        & result[
            "logistic_meets_min_recall"
        ]
    )

    return result


# ============================================================
# ELIGIBLE-MODEL SELECTION
# ============================================================


def choose_between_eligible_models(
    row: pd.Series,
) -> tuple[
    str,
    str,
]:
    """
    Choose the best model only from models that already satisfy the minimum
    precision and recall requirements.
    """

    gcn_eligible = bool(
        row[
            "gcn_eligible"
        ]
    )

    logistic_eligible = bool(
        row[
            "logistic_eligible"
        ]
    )

    # --------------------------------------------------------
    # Neither qualifies
    # --------------------------------------------------------

    if (
        not gcn_eligible
        and not logistic_eligible
    ):
        return (
            "none",
            "no_model_meets_minimums",
        )

    # --------------------------------------------------------
    # GCN only
    # --------------------------------------------------------

    if (
        gcn_eligible
        and not logistic_eligible
    ):
        return (
            "gcn",
            "gcn_only_eligible",
        )

    # --------------------------------------------------------
    # Logistic only
    # --------------------------------------------------------

    if (
        logistic_eligible
        and not gcn_eligible
    ):
        return (
            "logistic",
            "logistic_only_eligible",
        )

    # --------------------------------------------------------
    # Both qualify: F1 is primary selector
    # --------------------------------------------------------

    gcn_f1 = float(
        row[
            "gcn_f1"
        ]
    )

    logistic_f1 = float(
        row[
            "logistic_f1"
        ]
    )

    if gcn_f1 > logistic_f1:
        return (
            "gcn",
            "both_eligible_gcn_higher_f1",
        )

    if logistic_f1 > gcn_f1:
        return (
            "logistic",
            "both_eligible_logistic_higher_f1",
        )

    # --------------------------------------------------------
    # Tie-break 1: recall
    # --------------------------------------------------------

    gcn_recall = float(
        row[
            "gcn_recall"
        ]
    )

    logistic_recall = float(
        row[
            "logistic_recall"
        ]
    )

    if gcn_recall > logistic_recall:
        return (
            "gcn",
            "both_eligible_f1_tie_gcn_higher_recall",
        )

    if logistic_recall > gcn_recall:
        return (
            "logistic",
            "both_eligible_f1_tie_logistic_higher_recall",
        )

    # --------------------------------------------------------
    # Tie-break 2: precision
    # --------------------------------------------------------

    gcn_precision = float(
        row[
            "gcn_precision"
        ]
    )

    logistic_precision = float(
        row[
            "logistic_precision"
        ]
    )

    if gcn_precision > logistic_precision:
        return (
            "gcn",
            "both_eligible_f1_recall_tie_gcn_higher_precision",
        )

    if logistic_precision > gcn_precision:
        return (
            "logistic",
            "both_eligible_f1_recall_tie_logistic_higher_precision",
        )

    # --------------------------------------------------------
    # Tie-break 3: fewer false alarms
    # --------------------------------------------------------

    gcn_false_alarms = int(
        row[
            "gcn_false_alarm_events"
        ]
    )

    logistic_false_alarms = int(
        row[
            "logistic_false_alarm_events"
        ]
    )

    if gcn_false_alarms < logistic_false_alarms:
        return (
            "gcn",
            "both_eligible_metric_tie_gcn_fewer_false_alarms",
        )

    if logistic_false_alarms < gcn_false_alarms:
        return (
            "logistic",
            "both_eligible_metric_tie_logistic_fewer_false_alarms",
        )

    # --------------------------------------------------------
    # Final deterministic tie-break
    # --------------------------------------------------------

    return (
        "gcn",
        "both_eligible_complete_tie_gcn_default",
    )


def add_model_selection(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply eligibility first, then choose the strongest qualifying model.
    """

    result = add_model_eligibility(
        comparison
    )

    choices = result.apply(
        choose_between_eligible_models,
        axis=1,
    )

    result[
        "selected_model"
    ] = [
        choice[
            0
        ]
        for choice
        in choices
    ]

    result[
        "selection_reason"
    ] = [
        choice[
            1
        ]
        for choice
        in choices
    ]

    result[
        "sensor_eligible"
    ] = (
        result[
            "selected_model"
        ]
        != "none"
    )

    # --------------------------------------------------------
    # Selected metrics
    #
    # Ineligible sensors receive NaN because no model passed.
    # --------------------------------------------------------

    result[
        "selected_precision"
    ] = np.where(
        result[
            "selected_model"
        ]
        == "gcn",
        result[
            "gcn_precision"
        ],
        np.where(
            result[
                "selected_model"
            ]
            == "logistic",
            result[
                "logistic_precision"
            ],
            np.nan,
        ),
    )

    result[
        "selected_recall"
    ] = np.where(
        result[
            "selected_model"
        ]
        == "gcn",
        result[
            "gcn_recall"
        ],
        np.where(
            result[
                "selected_model"
            ]
            == "logistic",
            result[
                "logistic_recall"
            ],
            np.nan,
        ),
    )

    result[
        "selected_f1"
    ] = np.where(
        result[
            "selected_model"
        ]
        == "gcn",
        result[
            "gcn_f1"
        ],
        np.where(
            result[
                "selected_model"
            ]
            == "logistic",
            result[
                "logistic_f1"
            ],
            np.nan,
        ),
    )

    result[
        "selected_predicted_events"
    ] = np.where(
        result[
            "selected_model"
        ]
        == "gcn",
        result[
            "gcn_predicted_events"
        ],
        np.where(
            result[
                "selected_model"
            ]
            == "logistic",
            result[
                "logistic_predicted_events"
            ],
            np.nan,
        ),
    )

    result[
        "selected_false_alarm_events"
    ] = np.where(
        result[
            "selected_model"
        ]
        == "gcn",
        result[
            "gcn_false_alarm_events"
        ],
        np.where(
            result[
                "selected_model"
            ]
            == "logistic",
            result[
                "logistic_false_alarm_events"
            ],
            np.nan,
        ),
    )

    result[
        "selected_missed_events"
    ] = np.where(
        result[
            "selected_model"
        ]
        == "gcn",
        result[
            "gcn_missed_events"
        ],
        np.where(
            result[
                "selected_model"
            ]
            == "logistic",
            result[
                "logistic_missed_events"
            ],
            np.nan,
        ),
    )

    result[
        "selection_status"
    ] = np.where(
        result[
            "sensor_eligible"
        ],
        "eligible_model_selected",
        "sensor_excluded",
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
# ELIGIBLE SENSOR REGISTRY
# ============================================================


def build_eligible_sensor_registry(
    selection: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce the authoritative set of sensors allowed to move forward to the
    spatial stage.
    """

    eligible = (
        selection.loc[
            selection[
                "sensor_eligible"
            ]
        ]
        .copy()
    )

    columns = [
        SENSOR_ID,
        "selected_model",
        "selection_reason",
        "actual_events",
        "selected_precision",
        "selected_recall",
        "selected_f1",
        "selected_predicted_events",
        "selected_false_alarm_events",
        "selected_missed_events",
        "gcn_eligible",
        "gcn_precision",
        "gcn_recall",
        "gcn_f1",
        "logistic_eligible",
        "logistic_precision",
        "logistic_recall",
        "logistic_f1",
        "test_rows",
        "test_start",
        "test_end",
    ]

    return (
        eligible[
            columns
        ]
        .sort_values(
            SENSOR_ID
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# SUMMARY
# ============================================================


def build_summary(
    selection: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[
        dict
    ] = []

    total = len(
        selection
    )

    eligible = selection.loc[
        selection[
            "sensor_eligible"
        ]
    ]

    ineligible = selection.loc[
        ~selection[
            "sensor_eligible"
        ]
    ]

    rows.extend(
        [
            {
                "metric":
                    "minimum_precision",
                "value":
                    MIN_PRECISION,
            },
            {
                "metric":
                    "minimum_recall",
                "value":
                    MIN_RECALL,
            },
            {
                "metric":
                    "sensors_total",
                "value":
                    total,
            },
            {
                "metric":
                    "sensors_eligible",
                "value":
                    len(
                        eligible
                    ),
            },
            {
                "metric":
                    "sensors_excluded",
                "value":
                    len(
                        ineligible
                    ),
            },
            {
                "metric":
                    "eligible_fraction",
                "value":
                    (
                        len(
                            eligible
                        )
                        / total
                        if total
                        else np.nan
                    ),
            },
            {
                "metric":
                    "gcn_eligible_sensor_model_pairs",
                "value":
                    int(
                        selection[
                            "gcn_eligible"
                        ].sum()
                    ),
            },
            {
                "metric":
                    "logistic_eligible_sensor_model_pairs",
                "value":
                    int(
                        selection[
                            "logistic_eligible"
                        ].sum()
                    ),
            },
        ]
    )

    for model_name in [
        "gcn",
        "logistic",
    ]:
        count = int(
            (
                selection[
                    "selected_model"
                ]
                == model_name
            ).sum()
        )

        rows.append(
            {
                "metric":
                    f"sensors_selected_{model_name}",
                "value":
                    count,
            }
        )

    both_eligible = int(
        (
            selection[
                "gcn_eligible"
            ]
            & selection[
                "logistic_eligible"
            ]
        ).sum()
    )

    gcn_only = int(
        (
            selection[
                "gcn_eligible"
            ]
            & ~selection[
                "logistic_eligible"
            ]
        ).sum()
    )

    logistic_only = int(
        (
            ~selection[
                "gcn_eligible"
            ]
            & selection[
                "logistic_eligible"
            ]
        ).sum()
    )

    neither = int(
        (
            ~selection[
                "gcn_eligible"
            ]
            & ~selection[
                "logistic_eligible"
            ]
        ).sum()
    )

    rows.extend(
        [
            {
                "metric":
                    "sensors_both_models_eligible",
                "value":
                    both_eligible,
            },
            {
                "metric":
                    "sensors_gcn_only_eligible",
                "value":
                    gcn_only,
            },
            {
                "metric":
                    "sensors_logistic_only_eligible",
                "value":
                    logistic_only,
            },
            {
                "metric":
                    "sensors_no_model_eligible",
                "value":
                    neither,
            },
            {
                "metric":
                    "sensors_with_actual_events",
                "value":
                    int(
                        (
                            selection[
                                "actual_events"
                            ]
                            > 0
                        ).sum()
                    ),
            },
            {
                "metric":
                    "sensors_without_actual_events",
                "value":
                    int(
                        (
                            selection[
                                "actual_events"
                            ]
                            == 0
                        ).sum()
                    ),
            },
        ]
    )

    if not eligible.empty:
        rows.extend(
            [
                {
                    "metric":
                        "selected_f1_mean",
                    "value":
                        float(
                            eligible[
                                "selected_f1"
                            ].mean()
                        ),
                },
                {
                    "metric":
                        "selected_f1_median",
                    "value":
                        float(
                            eligible[
                                "selected_f1"
                            ].median()
                        ),
                },
                {
                    "metric":
                        "selected_precision_mean",
                    "value":
                        float(
                            eligible[
                                "selected_precision"
                            ].mean()
                        ),
                },
                {
                    "metric":
                        "selected_recall_mean",
                    "value":
                        float(
                            eligible[
                                "selected_recall"
                            ].mean()
                        ),
                },
            ]
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# SAVE OUTPUTS
# ============================================================


def save_outputs(
    matched: pd.DataFrame,
    selection: pd.DataFrame,
    eligible_registry: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    matched.to_parquet(
        MATCHED_TEST_PATH,
        index=False,
    )

    selection.to_csv(
        SENSOR_SELECTION_PATH,
        index=False,
    )

    selection.to_parquet(
        SENSOR_SELECTION_PARQUET_PATH,
        index=False,
    )

    eligible_registry.to_csv(
        ELIGIBLE_SENSOR_REGISTRY_PATH,
        index=False,
    )

    eligible_registry.to_parquet(
        ELIGIBLE_SENSOR_REGISTRY_PARQUET_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print(
        "\nSAVED OUTPUTS"
    )

    print(
        "-------------"
    )

    print(
        SENSOR_SELECTION_PATH
    )

    print(
        SENSOR_SELECTION_PARQUET_PATH
    )

    print(
        ELIGIBLE_SENSOR_REGISTRY_PATH
    )

    print(
        ELIGIBLE_SENSOR_REGISTRY_PARQUET_PATH
    )

    print(
        SUMMARY_PATH
    )

    print(
        MATCHED_TEST_PATH
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:
    print(
        "PER-SENSOR FLOOD MODEL SELECTION"
    )

    print(
        "================================"
    )

    print()

    print(
        "MODEL QUALITY REQUIREMENTS"
    )

    print(
        "--------------------------"
    )

    print(
        f"Minimum precision: "
        f"{MIN_PRECISION:.2f}"
    )

    print(
        f"Minimum recall:    "
        f"{MIN_RECALL:.2f}"
    )

    print()

    print(
        "Selection order:"
    )

    print(
        "    1. apply minimum precision/recall to each model"
    )

    print(
        "    2. discard models that fail"
    )

    print(
        "    3. choose highest F1 among qualifying models"
    )

    print(
        "    4. exclude sensor if neither model qualifies"
    )

    print()

    print(
        "Event matching:"
    )

    print(
        f"    +/- {MATCH_WINDOW_HOURS} hours"
    )

    print()

    print(
        "GCN predicted-event threshold:"
    )

    print(
        "    predicted_minutes_above_1inch "
        f"> {GCN_PREDICTED_EVENT_THRESHOLD_MINUTES}"
    )

    print()

    print(
        "Logistic predicted-event threshold:"
    )

    print(
        "    probability "
        f">= {LOGISTIC_CLASSIFICATION_THRESHOLD}"
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    gcn, logistic = (
        load_test_predictions()
    )

    print(
        "\nTEST POPULATIONS"
    )

    print(
        "----------------"
    )

    print(
        f"GCN rows:        "
        f"{len(gcn):,}"
    )

    print(
        f"Logistic rows:   "
        f"{len(logistic):,}"
    )

    print(
        f"GCN sensors:     "
        f"{gcn[SENSOR_ID].nunique():,}"
    )

    print(
        f"Logistic sensors:"
        f" {logistic[SENSOR_ID].nunique():,}"
    )

    # --------------------------------------------------------
    # Match
    # --------------------------------------------------------

    matched = (
        build_matched_test_frame(
            gcn,
            logistic,
        )
    )

    print(
        "\nMATCHED TEST POPULATION"
    )

    print(
        "-----------------------"
    )

    print(
        f"Rows:       "
        f"{len(matched):,}"
    )

    print(
        f"Sensors:    "
        f"{matched[SENSOR_ID].nunique():,}"
    )

    print(
        f"Hours:      "
        f"{matched[TIME_COLUMN].nunique():,}"
    )

    print(
        f"First hour: "
        f"{matched[TIME_COLUMN].min()}"
    )

    print(
        f"Last hour:  "
        f"{matched[TIME_COLUMN].max()}"
    )

    print(
        "Target-value differences:"
    )

    print(
        f"    nonzero: "
        f"{int((matched['actual_target_absolute_difference'] > 0).sum()):,}"
    )

    print(
        f"    max abs: "
        f"{matched['actual_target_absolute_difference'].max()}"
    )

    # --------------------------------------------------------
    # Sensor-level metrics
    # --------------------------------------------------------

    comparison = (
        calculate_sensor_comparison(
            matched
        )
    )

    # --------------------------------------------------------
    # Eligibility first, selection second
    # --------------------------------------------------------

    selection = (
        add_model_selection(
            comparison
        )
    )

    eligible_registry = (
        build_eligible_sensor_registry(
            selection
        )
    )

    summary = (
        build_summary(
            selection
        )
    )

    # --------------------------------------------------------
    # Console results
    # --------------------------------------------------------

    print(
        "\nMODEL ELIGIBILITY RESULTS"
    )

    print(
        "-------------------------"
    )

    both_eligible = int(
        (
            selection[
                "gcn_eligible"
            ]
            & selection[
                "logistic_eligible"
            ]
        ).sum()
    )

    gcn_only = int(
        (
            selection[
                "gcn_eligible"
            ]
            & ~selection[
                "logistic_eligible"
            ]
        ).sum()
    )

    logistic_only = int(
        (
            ~selection[
                "gcn_eligible"
            ]
            & selection[
                "logistic_eligible"
            ]
        ).sum()
    )

    neither = int(
        (
            ~selection[
                "gcn_eligible"
            ]
            & ~selection[
                "logistic_eligible"
            ]
        ).sum()
    )

    print(
        f"Both models eligible:     "
        f"{both_eligible:,}"
    )

    print(
        f"GCN only eligible:        "
        f"{gcn_only:,}"
    )

    print(
        f"Logistic only eligible:   "
        f"{logistic_only:,}"
    )

    print(
        f"No model eligible:        "
        f"{neither:,}"
    )

    print(
        "\nFINAL SENSOR SELECTION"
    )

    print(
        "----------------------"
    )

    counts = (
        selection[
            "selected_model"
        ]
        .value_counts()
    )

    print(
        counts.to_string()
    )

    eligible_count = int(
        selection[
            "sensor_eligible"
        ].sum()
    )

    excluded_count = int(
        (
            ~selection[
                "sensor_eligible"
            ]
        ).sum()
    )

    print()

    print(
        f"Eligible sensors: "
        f"{eligible_count:,}"
    )

    print(
        f"Excluded sensors: "
        f"{excluded_count:,}"
    )

    print()

    print(
        "SELECTION REASONS"
    )

    print(
        "-----------------"
    )

    print(
        selection[
            "selection_reason"
        ]
        .value_counts()
        .to_string()
    )

    if not eligible_registry.empty:
        print()

        print(
            "ELIGIBLE SENSOR F1 DISTRIBUTION"
        )

        print(
            "-------------------------------"
        )

        print(
            eligible_registry[
                "selected_f1"
            ]
            .describe()
            .to_string()
        )

        print()

        print(
            "ELIGIBLE SENSOR PRECISION DISTRIBUTION"
        )

        print(
            "--------------------------------------"
        )

        print(
            eligible_registry[
                "selected_precision"
            ]
            .describe()
            .to_string()
        )

        print()

        print(
            "ELIGIBLE SENSOR RECALL DISTRIBUTION"
        )

        print(
            "-----------------------------------"
        )

        print(
            eligible_registry[
                "selected_recall"
            ]
            .describe()
            .to_string()
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_outputs(
        matched,
        selection,
        eligible_registry,
        summary,
    )

    print(
        "\nMODEL / SENSOR SELECTION COMPLETE"
    )

    print(
        "Only eligible sensors should move forward "
        "to the spatial 1 km x 1 km grid stage."
    )


if __name__ == "__main__":
    main()