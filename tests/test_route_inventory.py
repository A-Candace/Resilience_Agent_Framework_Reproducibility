from resilience_app.agents.registry import ROUTES, ROUTE_OWNERS
EXPECTED={"landing","urban","risk","demographics","forecast","uhi","query","green_roof","rain_garden","chat"}
def test_route_keys_preserved(): assert set(ROUTES)==EXPECTED
def test_every_route_has_agent_owner(): assert set(ROUTE_OWNERS)==EXPECTED
