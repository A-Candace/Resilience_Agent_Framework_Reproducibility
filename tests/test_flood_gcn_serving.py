from __future__ import annotations

import pandas as pd
import pytest

from mlops.flood.shared import model_interface


# ============================================================
# TEST FIXTURES
# ============================================================


@pytest.fixture
def valid_snapshot() -> pd.DataFrame:
    """
    Canonical valid hourly snapshot used by the serving tests.
    """

    return pd.DataFrame(
        {
            "deployment_id": [
                "Abruptly-delayed-doormouse",
                "Sadly-blue-beetle",
                "Wonderfully-adept-chihuahua",
            ],
            "precip_current_hour_mm": [
                0.0,
                1.0,
                5.0,
            ],
            "precip_previous_6h_mm": [
                0.0,
                2.0,
                10.0,
            ],
            "daily_total_precip_mm": [
                0.0,
                3.0,
                15.0,
            ],
        }
    )


class FakeChampionModel:
    """
    Minimal MLflow-PyFunc-like model used to isolate serving-interface tests
    from the external MLflow server.

    The real champion model is validated separately in integration testing.
    """

    def predict(
        self,
        snapshot: pd.DataFrame,
    ) -> pd.DataFrame:

        prediction_lookup = {
            "Abruptly-delayed-doormouse": (
                0.07455337792634964
            ),
            "Sadly-blue-beetle": (
                0.4277450442314148
            ),
            "Wonderfully-adept-chihuahua": (
                2.1637558937072754
            ),
        }

        output = snapshot[
            ["deployment_id"]
        ].copy()

        output[
            "predicted_minutes_above_1inch"
        ] = output[
            "deployment_id"
        ].map(
            prediction_lookup
        ).fillna(
            0.0
        )

        output[
            "predicted_event"
        ] = (
            output[
                "predicted_minutes_above_1inch"
            ]
            > 0.5
        )

        return output


# ============================================================
# MODEL AVAILABILITY
# ============================================================


def test_gcn_availability_uses_mlflow_model_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    GCN availability should describe the promoted MLflow-backed model rather
    than depend on the legacy local .pt model artifact.
    """

    monkeypatch.setattr(
        model_interface,
        "_gcn_registry_available",
        lambda: (
            True,
            "GCN available from MLflow test registry.",
        ),
    )

    availability = (
        model_interface.get_model_availability()
    )

    assert "gcn" in availability

    gcn = availability["gcn"]

    assert gcn.name == "gcn"
    assert gcn.available is True
    assert gcn.model_uri == (
        model_interface.GCN_MODEL_URI
    )

    assert (
        model_interface.GCN_MODEL_URI
        in {
            gcn.model_uri,
        }
    )


# ============================================================
# VALID SNAPSHOT / OUTPUT CONTRACT
# ============================================================


def test_predict_gcn_snapshot_returns_expected_schema(
    monkeypatch: pytest.MonkeyPatch,
    valid_snapshot: pd.DataFrame,
) -> None:
    """
    A valid hourly snapshot should produce the public serving schema.
    """

    monkeypatch.setattr(
        model_interface,
        "load_gcn_champion",
        lambda: FakeChampionModel(),
    )

    result = (
        model_interface.predict_gcn_snapshot(
            valid_snapshot
        )
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert list(
        result.columns
    ) == [
        "deployment_id",
        "predicted_minutes_above_1inch",
        "predicted_event",
    ]

    assert len(result) == 3


def test_predict_gcn_snapshot_preserves_sensor_ids(
    monkeypatch: pytest.MonkeyPatch,
    valid_snapshot: pd.DataFrame,
) -> None:
    """
    Output rows must remain associated with the input deployment IDs.
    """

    monkeypatch.setattr(
        model_interface,
        "load_gcn_champion",
        lambda: FakeChampionModel(),
    )

    result = (
        model_interface.predict_gcn_snapshot(
            valid_snapshot
        )
    )

    assert (
        result["deployment_id"].tolist()
        == valid_snapshot[
            "deployment_id"
        ].tolist()
    )


def test_predict_gcn_snapshot_returns_numeric_predictions(
    monkeypatch: pytest.MonkeyPatch,
    valid_snapshot: pd.DataFrame,
) -> None:
    """
    The serving layer must return finite numeric duration predictions and
    boolean event indicators.
    """

    monkeypatch.setattr(
        model_interface,
        "load_gcn_champion",
        lambda: FakeChampionModel(),
    )

    result = (
        model_interface.predict_gcn_snapshot(
            valid_snapshot
        )
    )

    assert pd.api.types.is_numeric_dtype(
        result[
            "predicted_minutes_above_1inch"
        ]
    )

    assert result[
        "predicted_minutes_above_1inch"
    ].notna().all()

    assert (
        result[
            "predicted_minutes_above_1inch"
        ]
        >= 0.0
    ).all()

    assert pd.api.types.is_bool_dtype(
        result[
            "predicted_event"
        ]
    )


# ============================================================
# INPUT VALIDATION
# ============================================================


@pytest.mark.parametrize(
    "missing_column",
    [
        "deployment_id",
        "precip_current_hour_mm",
        "precip_previous_6h_mm",
        "daily_total_precip_mm",
    ],
)
def test_predict_gcn_snapshot_rejects_missing_columns(
    monkeypatch: pytest.MonkeyPatch,
    valid_snapshot: pd.DataFrame,
    missing_column: str,
) -> None:
    """
    The serving contract requires all model predictors plus deployment_id.
    """

    monkeypatch.setattr(
        model_interface,
        "load_gcn_champion",
        lambda: FakeChampionModel(),
    )

    invalid = valid_snapshot.drop(
        columns=[
            missing_column
        ]
    )

    with pytest.raises(
        Exception
    ):
        model_interface.predict_gcn_snapshot(
            invalid
        )


def test_predict_gcn_snapshot_rejects_duplicate_sensor_rows(
    monkeypatch: pytest.MonkeyPatch,
    valid_snapshot: pd.DataFrame,
) -> None:
    """
    One hourly snapshot must contain at most one row per deployment_id.
    """

    monkeypatch.setattr(
        model_interface,
        "load_gcn_champion",
        lambda: FakeChampionModel(),
    )

    duplicate = pd.concat(
        [
            valid_snapshot,
            valid_snapshot.iloc[
                [0]
            ],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        Exception
    ):
        model_interface.predict_gcn_snapshot(
            duplicate
        )


def test_predict_gcn_snapshot_rejects_null_sensor_id(
    monkeypatch: pytest.MonkeyPatch,
    valid_snapshot: pd.DataFrame,
) -> None:
    """
    Sensor identity is mandatory for graph-node lookup.
    """

    monkeypatch.setattr(
        model_interface,
        "load_gcn_champion",
        lambda: FakeChampionModel(),
    )

    invalid = valid_snapshot.copy()

    invalid.loc[
        0,
        "deployment_id",
    ] = None

    with pytest.raises(
        Exception
    ):
        model_interface.predict_gcn_snapshot(
            invalid
        )


def test_predict_gcn_snapshot_rejects_null_predictor(
    monkeypatch: pytest.MonkeyPatch,
    valid_snapshot: pd.DataFrame,
) -> None:
    """
    Serving should not silently score incomplete precipitation features.
    """

    monkeypatch.setattr(
        model_interface,
        "load_gcn_champion",
        lambda: FakeChampionModel(),
    )

    invalid = valid_snapshot.copy()

    invalid.loc[
        0,
        "precip_current_hour_mm",
    ] = None

    with pytest.raises(
        Exception
    ):
        model_interface.predict_gcn_snapshot(
            invalid
        )


# ============================================================
# MODEL OUTPUT VALIDATION
# ============================================================


def test_predict_gcn_snapshot_rejects_non_dataframe_model_output(
    monkeypatch: pytest.MonkeyPatch,
    valid_snapshot: pd.DataFrame,
) -> None:
    """
    Guard against a corrupted or incompatible MLflow model returning an
    unexpected result type.
    """

    class InvalidModel:
        def predict(
            self,
            snapshot: pd.DataFrame,
        ):
            return [
                1.0,
                2.0,
                3.0,
            ]

    monkeypatch.setattr(
        model_interface,
        "load_gcn_champion",
        lambda: InvalidModel(),
    )

    with pytest.raises(
        RuntimeError
    ):
        model_interface.predict_gcn_snapshot(
            valid_snapshot
        )


def test_predict_gcn_snapshot_rejects_missing_output_columns(
    monkeypatch: pytest.MonkeyPatch,
    valid_snapshot: pd.DataFrame,
) -> None:
    """
    MLflow output must honor the public flood prediction contract.
    """

    class InvalidModel:
        def predict(
            self,
            snapshot: pd.DataFrame,
        ) -> pd.DataFrame:

            return pd.DataFrame(
                {
                    "deployment_id": (
                        snapshot[
                            "deployment_id"
                        ]
                    ),
                    "wrong_prediction_column": [
                        0.0
                    ]
                    * len(
                        snapshot
                    ),
                }
            )

    monkeypatch.setattr(
        model_interface,
        "load_gcn_champion",
        lambda: InvalidModel(),
    )

    with pytest.raises(
        RuntimeError
    ):
        model_interface.predict_gcn_snapshot(
            valid_snapshot
        )