"""Typed request/response schemas for the forecasting capability."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class AgentResult:
    data: Any
    source: str = "application"

@dataclass(frozen=True)
class SensorForecastInput:
    deployment_id: str
    precip_current_hour_mm: float
    precip_previous_6h_mm: float
    daily_total_precip_mm: float

@dataclass(frozen=True)
class SensorForecastOutput:
    deployment_id: str
    predicted_minutes_above_1inch: float
    predicted_event: bool