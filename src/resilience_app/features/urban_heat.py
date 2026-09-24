"""Backward-compatible imports for the relocated heat_agent capability.

New code should import from ``resilience_app.agents.heat_agent``.
"""
from resilience_app.agents.heat_agent.page import page_uhi

__all__ = ['page_uhi']
