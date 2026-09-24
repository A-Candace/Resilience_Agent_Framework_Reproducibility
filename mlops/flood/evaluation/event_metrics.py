"""
Shared flood-event evaluation utilities.

This module preserves the event-level evaluation methodology
currently used by the GCN research notebook.

The same evaluation functions will be used for every candidate
model (GCN, Logistic Regression, and future models) so that
per-sensor model selection is based on directly comparable
precision, recall, and F1 scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from mlops.flood.shared.config import EVENT_MATCH_WINDOW_HOURS


# ============================================================
# DEFAULT EVENT-DETECTION SETTINGS
# ============================================================

# Existing GCN notebook methodology:
#
# Actual flood event:
#     actual predicted-duration target > 0 minutes
#
# Predicted flood event:
#     model prediction > 1 minute
#
# Matching:
#     prediction occurs within +/- 24 hours
#     of an actual event for the same sensor.
#
PREDICTED_DURATION_THRESHOLD_MINUTES = 1.0


# ============================================================
# RESULT CONTAINER
# ============================================================

@dataclass
class EventEvaluationResult:
    """
    Container for per-sensor and aggregate flood-event metrics.
    """

    sensor_metrics: pd.DataFrame

    overall_precision: float
    overall_recall: float
    overall_f1: float

    total_actual_events: int
    total_predicted_events: int
    total_captured_events: int
    total_missed_events: int
    total_matched_predictions: int
    total_false_alarms: int


# ============================================================
# PER-SENSOR EVENT EVALUATION
# ============================================================

def evaluate_events_by_sensor(
    predictions: pd.DataFrame,
    *,
    sensor_col: str,
    time_col: str,
    actual_minutes_col: str,
    predicted_minutes_col: str,
    model_name: str,
    predicted_duration_threshold_minutes: float = (
        PREDICTED_DURATION_THRESHOLD_MINUTES
    ),
    match_window_hours: int = EVENT_MATCH_WINDOW_HOURS,
) -> EventEvaluationResult:
    """
    Evaluate flood-event predictions separately for each sensor.

    Parameters
    ----------
    predictions:
        Hourly prediction dataframe.

    sensor_col:
        Column containing the sensor identifier.

    time_col:
        Column containing prediction timestamps.

    actual_minutes_col:
        Column containing the observed flood-duration target.

    predicted_minutes_col:
        Column containing model-predicted flood duration.

    model_name:
        Name of the candidate model being evaluated.
        Examples: "gcn", "logistic".

    predicted_duration_threshold_minutes:
        Minimum predicted duration required to classify an
        hourly prediction as a flood event.

    match_window_hours:
        Maximum time difference allowed between an actual
        event and predicted event.

    Returns
    -------
    EventEvaluationResult
        Per-sensor precision, recall, F1, event counts,
        and overall aggregate metrics.

    Notes
    -----
    This intentionally preserves the existing GCN notebook
    evaluation methodology.

    An actual event is counted as captured when at least one
    predicted event occurs within +/- match_window_hours.

    A predicted event is counted as matched when at least one
    actual event occurs within +/- match_window_hours.
    """

    required_columns = {
        sensor_col,
        time_col,
        actual_minutes_col,
        predicted_minutes_col,
    }

    missing_columns = (
        required_columns
        - set(predictions.columns)
    )

    if missing_columns:
        raise ValueError(
            "Prediction dataframe is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    event_source = (
        predictions[
            [
                time_col,
                sensor_col,
                actual_minutes_col,
                predicted_minutes_col,
            ]
        ]
        .rename(
            columns={
                time_col: "event_time",
                actual_minutes_col: "actual_minutes",
                predicted_minutes_col: "predicted_minutes",
            }
        )
        .copy()
    )

    event_source["event_time"] = pd.to_datetime(
        event_source["event_time"],
        utc=True,
    )

    event_source["actual_event"] = (
        event_source["actual_minutes"] > 0
    ).astype(int)

    event_source["predicted_event"] = (
        event_source["predicted_minutes"]
        > predicted_duration_threshold_minutes
    ).astype(int)

    match_window = pd.Timedelta(
        hours=match_window_hours
    )

    evaluation_rows: list[dict] = []

    # ========================================================
    # Evaluate each sensor independently
    # ========================================================

    for sensor_id, group in event_source.groupby(
        sensor_col
    ):
        group = (
            group
            .sort_values("event_time")
            .copy()
        )

        actual_times = (
            group.loc[
                group["actual_event"] == 1,
                "event_time",
            ]
            .sort_values()
            .tolist()
        )

        predicted_times = (
            group.loc[
                group["predicted_event"] == 1,
                "event_time",
            ]
            .sort_values()
            .tolist()
        )

        # ----------------------------------------------------
        # Recall:
        #
        # For every actual flood event, determine whether
        # at least one prediction occurred within +/- window.
        # ----------------------------------------------------

        actual_matches: list[int] = []

        for actual_time in actual_times:
            matched = any(
                abs(
                    predicted_time
                    - actual_time
                )
                <= match_window

                for predicted_time
                in predicted_times
            )

            actual_matches.append(
                int(matched)
            )

        # ----------------------------------------------------
        # Precision:
        #
        # For every predicted flood event, determine whether
        # at least one actual flood occurred within +/- window.
        # ----------------------------------------------------

        prediction_matches: list[int] = []

        for predicted_time in predicted_times:
            matched = any(
                abs(
                    actual_time
                    - predicted_time
                )
                <= match_window

                for actual_time
                in actual_times
            )

            prediction_matches.append(
                int(matched)
            )

        matched_predictions = int(
            sum(prediction_matches)
        )

        false_alarm_predictions = int(
            len(prediction_matches)
            - matched_predictions
        )

        captured_actual_events = int(
            sum(actual_matches)
        )

        missed_actual_events = int(
            len(actual_matches)
            - captured_actual_events
        )

        sensor_precision = (
            matched_predictions
            / max(
                len(predicted_times),
                1,
            )
        )

        sensor_recall = (
            captured_actual_events
            / max(
                len(actual_times),
                1,
            )
        )

        if (
            sensor_precision
            + sensor_recall
            > 0
        ):
            sensor_f1 = (
                2
                * sensor_precision
                * sensor_recall
                / (
                    sensor_precision
                    + sensor_recall
                )
            )

        else:
            sensor_f1 = 0.0

        evaluation_rows.append(
            {
                sensor_col:
                    sensor_id,

                "model":
                    model_name,

                "actual_flood_events":
                    len(actual_times),

                "predicted_flood_events":
                    len(predicted_times),

                "captured_actual_events":
                    captured_actual_events,

                "missed_actual_events":
                    missed_actual_events,

                "matched_predictions":
                    matched_predictions,

                "false_alarm_predictions":
                    false_alarm_predictions,

                "precision":
                    float(sensor_precision),

                "recall":
                    float(sensor_recall),

                "f1":
                    float(sensor_f1),
            }
        )

    sensor_metrics = pd.DataFrame(
        evaluation_rows
    )

    # ========================================================
    # Aggregate event metrics
    # ========================================================

    if sensor_metrics.empty:
        return EventEvaluationResult(
            sensor_metrics=sensor_metrics,
            overall_precision=0.0,
            overall_recall=0.0,
            overall_f1=0.0,
            total_actual_events=0,
            total_predicted_events=0,
            total_captured_events=0,
            total_missed_events=0,
            total_matched_predictions=0,
            total_false_alarms=0,
        )

    total_actual_events = int(
        sensor_metrics[
            "actual_flood_events"
        ].sum()
    )

    total_predicted_events = int(
        sensor_metrics[
            "predicted_flood_events"
        ].sum()
    )

    total_captured_events = int(
        sensor_metrics[
            "captured_actual_events"
        ].sum()
    )

    total_missed_events = int(
        sensor_metrics[
            "missed_actual_events"
        ].sum()
    )

    total_matched_predictions = int(
        sensor_metrics[
            "matched_predictions"
        ].sum()
    )

    total_false_alarms = int(
        sensor_metrics[
            "false_alarm_predictions"
        ].sum()
    )

    overall_precision = (
        total_matched_predictions
        / max(
            total_predicted_events,
            1,
        )
    )

    overall_recall = (
        total_captured_events
        / max(
            total_actual_events,
            1,
        )
    )

    if (
        overall_precision
        + overall_recall
        > 0
    ):
        overall_f1 = (
            2
            * overall_precision
            * overall_recall
            / (
                overall_precision
                + overall_recall
            )
        )

    else:
        overall_f1 = 0.0

    return EventEvaluationResult(
        sensor_metrics=sensor_metrics,

        overall_precision=float(
            overall_precision
        ),

        overall_recall=float(
            overall_recall
        ),

        overall_f1=float(
            overall_f1
        ),

        total_actual_events=
            total_actual_events,

        total_predicted_events=
            total_predicted_events,

        total_captured_events=
            total_captured_events,

        total_missed_events=
            total_missed_events,

        total_matched_predictions=
            total_matched_predictions,

        total_false_alarms=
            total_false_alarms,
    )