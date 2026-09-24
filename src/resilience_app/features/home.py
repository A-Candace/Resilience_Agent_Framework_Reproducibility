"""Backward-compatible imports for the relocated home_agent capability.

New code should import from ``resilience_app.agents.home_agent``.
"""
from resilience_app.agents.home_agent.page import home_page, page_landing

__all__ = ['home_page', 'page_landing']
