"""Backward-compatible imports for the relocated green_roof_agent capability.

New code should import from ``resilience_app.agents.green_roof_agent``.
"""
from resilience_app.agents.green_roof_agent.page import page_green_roof

__all__ = ['page_green_roof']
