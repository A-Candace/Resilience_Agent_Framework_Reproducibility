from pathlib import Path
import ast
EXPECTED = {
 "home_agent":"page_landing", "urban_features_agent":"urban_features_page",
 "flood_risk_agent":"page_risk_mapping", "demographics_agent":"page_demographics",
 "forecasting_agent":"forecasting_page", "heat_agent":"page_uhi",
 "ai_query_agent":"page_ai_query", "green_roof_agent":"page_green_roof",
 "rain_garden_agent":"page_rain_garden", "chat_agent":"page_chat",
}
def test_all_agent_entrypoints_are_present():
    root=Path("src/resilience_app/agents")
    for agent, fn in EXPECTED.items():
        tree=ast.parse((root/agent/"page.py").read_text(encoding="utf-8"))
        actual={n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
        assert fn in actual, f"Missing {fn} from {agent}"
def test_compatibility_modules_exist():
    for name in ["home","urban_features","risk_mapping","demographics","forecasting","urban_heat","ai_query","green_roof","rain_garden","chat"]:
        assert Path(f"src/resilience_app/features/{name}.py").exists()
def test_legacy_monolith_retained():
    assert Path("legacy/app_monolith_original.py").exists()
