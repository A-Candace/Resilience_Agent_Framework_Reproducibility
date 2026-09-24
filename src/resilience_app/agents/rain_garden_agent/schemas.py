"""Typed request/response schemas for this capability."""
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class AgentResult:
    data: Any
    source: str = "application"
