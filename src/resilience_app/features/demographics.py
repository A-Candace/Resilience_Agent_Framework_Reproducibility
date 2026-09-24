"""Backward-compatible imports for the relocated demographics_agent capability.

New code should import from ``resilience_app.agents.demographics_agent``.
"""
from resilience_app.agents.demographics_agent.page import page_demographics

__all__ = ['page_demographics']
