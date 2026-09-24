"""Single source of truth for UI routes and agent ownership."""
from dataclasses import dataclass
from typing import Callable

from resilience_app.agents.home_agent import page_landing
from resilience_app.agents.urban_features_agent import urban_features_page
from resilience_app.agents.flood_risk_agent import page_risk_mapping
from resilience_app.agents.demographics_agent import page_demographics
from resilience_app.agents.forecasting_agent import forecasting_page
from resilience_app.agents.heat_agent import page_uhi
from resilience_app.agents.ai_query_agent import page_ai_query
from resilience_app.agents.green_roof_agent import page_green_roof
from resilience_app.agents.rain_garden_agent import page_rain_garden
from resilience_app.agents.chat_agent import page_chat

@dataclass(frozen=True)
class AgentRoute:
    key: str
    label: str
    owner: str
    page: Callable[[], None]

AGENT_ROUTES = (
    AgentRoute("landing", "Home", "home_agent", page_landing),
    AgentRoute("urban", "Urban Features", "urban_features_agent", urban_features_page),
    AgentRoute("risk", "Risk Mapping", "flood_risk_agent", page_risk_mapping),
    AgentRoute("demographics", "Socio-Demographics", "demographics_agent", page_demographics),
    AgentRoute("forecast", "Forecasting", "forecasting_agent", forecasting_page),
    AgentRoute("uhi", "Urban Heat Island", "heat_agent", page_uhi),
    AgentRoute("query", "Multi-risk Identification Tool", "ai_query_agent", page_ai_query),
    AgentRoute("green_roof", "Green Roof Calculator", "green_roof_agent", page_green_roof),
    AgentRoute("rain_garden", "Rain Garden Estimator", "rain_garden_agent", page_rain_garden),
    AgentRoute("chat", "Chat", "chat_agent", page_chat),
)

ROUTES = {route.key: route.page for route in AGENT_ROUTES}
ROUTE_OWNERS = {route.key: route.owner for route in AGENT_ROUTES}
