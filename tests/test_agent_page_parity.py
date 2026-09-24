from pathlib import Path
MAP={"home":"home_agent","urban_features":"urban_features_agent","risk_mapping":"flood_risk_agent","demographics":"demographics_agent","forecasting":"forecasting_agent","urban_heat":"heat_agent","ai_query":"ai_query_agent","green_roof":"green_roof_agent","rain_garden":"rain_garden_agent","chat":"chat_agent"}
def test_compatibility_wrappers_target_all_agents():
    root=Path("src/resilience_app")
    for old,agent in MAP.items():
        text=(root/"features"/f"{old}.py").read_text()
        assert f"resilience_app.agents.{agent}.page" in text
