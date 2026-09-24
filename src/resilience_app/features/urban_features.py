"""Backward-compatible imports for the relocated urban_features_agent capability.

New code should import from ``resilience_app.agents.urban_features_agent``.
"""
from resilience_app.agents.urban_features_agent.page import urban_features_page

__all__ = ['urban_features_page']
