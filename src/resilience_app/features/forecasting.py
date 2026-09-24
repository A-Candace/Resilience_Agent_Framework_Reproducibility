"""Backward-compatible imports for the relocated forecasting_agent capability.

New code should import from ``resilience_app.agents.forecasting_agent``.
"""
from resilience_app.agents.forecasting_agent.page import forecasting_page

__all__ = ['forecasting_page']
