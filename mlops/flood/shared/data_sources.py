"""
Shared data-source interfaces for the flood MLOps pipeline.

Production training and daily inference should depend on these
standardized functions rather than directly depending on local files,
notebook paths, or specific APIs.

The goal is to make the downstream model code independent of whether
data came from:
- a live API,
- an S3 snapshot,
- a cached response,
- or a local test fixture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


# ============================================================
# STANDARDIZED DATA CONTAINERS
# ============================================================

@dataclass
class HistoricalFloodData:
    """
    Historical data used for retraining and evaluation.

    sensor_observations:
        Sensor-level flood/depth observations.

    precipitation:
        Historical precipitation or weather features.

    metadata:
        Optional source metadata, timestamps, or version identifiers.
    """

    sensor_observations: pd.DataFrame
    precipitation: pd.DataFrame
    metadata: dict


@dataclass
class ForecastFloodData:
    """
    Forecast data used for daily inference.

    forecast:
        Future precipitation/weather forecast features.

    metadata:
        Optional source metadata, timestamps, or version identifiers.
    """

    forecast: pd.DataFrame
    metadata: dict


# ============================================================
# DATA SOURCE INTERFACE
# ============================================================

class FloodDataSource(Protocol):
    """
    Interface that every production flood data source should support.

    The concrete implementation may call one or more APIs,
    load an S3 snapshot, or use another source.
    """

    def load_historical_data(
        self,
    ) -> HistoricalFloodData:
        """
        Retrieve historical sensor and precipitation data
        required for model retraining.
        """
        ...

    def load_forecast_data(
        self,
    ) -> ForecastFloodData:
        """
        Retrieve weather forecast data required for
        day-ahead flood inference.
        """
        ...


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_sensor_observations(
    df: pd.DataFrame,
) -> None:
    """
    Validate the minimum schema required from the
    historical sensor observation source.
    """

    required_columns = {
        "deployment_id",
        "time",
        "depth_proc_mm",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Sensor observation data is missing required columns: "
            + ", ".join(sorted(missing))
        )


def validate_precipitation_data(
    df: pd.DataFrame,
) -> None:
    """
    Validate the minimum schema required from the
    historical precipitation source.
    """

    required_columns = {
        "deployment_id",
        "hour",
        "precip_current_hour_mm",
        "precip_previous_6h_mm",
        "daily_total_precip_mm",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Precipitation data is missing required columns: "
            + ", ".join(sorted(missing))
        )


def validate_forecast_data(
    df: pd.DataFrame,
) -> None:
    """
    Validate the minimum schema required for daily
    day-ahead inference.
    """

    required_columns = {
        "hour",
        "precip_current_hour_mm",
        "precip_previous_6h_mm",
        "daily_total_precip_mm",
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:
        raise ValueError(
            "Forecast data is missing required columns: "
            + ", ".join(sorted(missing))
        )