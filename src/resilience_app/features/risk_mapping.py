"""Backward-compatible imports for the relocated flood_risk_agent capability.

New code should import from ``resilience_app.agents.flood_risk_agent``.
"""
from resilience_app.agents.flood_risk_agent.page import list_numeric_candidates, pick_field_with_fallback, rescale_score_component, page_risk_mapping

__all__ = ['list_numeric_candidates', 'pick_field_with_fallback', 'rescale_score_component', 'page_risk_mapping']
