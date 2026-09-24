"""Backward-compatible imports for the relocated ai_query_agent capability.

New code should import from ``resilience_app.agents.ai_query_agent``.
"""
from resilience_app.agents.ai_query_agent.page import page_ai_query

__all__ = ['page_ai_query']
