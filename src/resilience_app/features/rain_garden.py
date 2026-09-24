"""Backward-compatible imports for the relocated rain_garden_agent capability.

New code should import from ``resilience_app.agents.rain_garden_agent``.
"""
from resilience_app.agents.rain_garden_agent.page import page_rain_garden

__all__ = ['page_rain_garden']
