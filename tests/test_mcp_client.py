"""mcp_client.py against a mocked fastmcp Client — no real Docker network calls."""
import asyncio

import pytest

from agentic.orchestrator import mcp_client


class _FakeTool:
    def __init__(self, name, description, input_schema):
        self.name = name
        self.description = description
        self.inputSchema = input_schema


class _FakeCallResult:
    def __init__(self, data=None, structured_content=None, content=None):
        self.data = data
        self.structured_content = structured_content
        self.content = content or []


class _FakeTextBlock:
    def __init__(self, text):
        self.text = text


def _make_fake_client_class(tools_by_url, results_by_name=None, raise_for_names=None):
    """Build a fastmcp.Client stand-in keyed by server URL and tool name."""
    results_by_name = results_by_name or {}
    raise_for_names = raise_for_names or set()

    class _FakeClient:
        def __init__(self, url, *args, **kwargs):
            self.url = url

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def list_tools(self):
            return tools_by_url.get(self.url, [])

        async def call_tool(self, name, arguments):
            if name in raise_for_names:
                raise RuntimeError(f"simulated failure calling {name}")
            return results_by_name[name]

    return _FakeClient


@pytest.fixture(autouse=True)
def _reset_mcp_cache():
    mcp_client.reset_cache()
    yield
    mcp_client.reset_cache()


def test_get_tool_catalog_discovers_expected_shape(monkeypatch):
    settings = mcp_client.get_settings()
    tools_by_url = {
        settings.mcp_geospatial_url: [
            _FakeTool("flood_grid_summary", "Summarize flood grid.", {"type": "object", "properties": {}}),
        ],
        settings.mcp_calculators_url: [
            _FakeTool(
                "green_roof_estimate",
                "Estimate green roof capture.",
                {"type": "object", "properties": {"roof_area_sqft": {"type": "number"}}},
            ),
        ],
        settings.mcp_data_url: [],
        settings.mcp_model_url: [],
    }
    monkeypatch.setattr(mcp_client, "Client", _make_fake_client_class(tools_by_url))

    catalog = asyncio.run(mcp_client.get_tool_catalog())

    assert catalog == [
        {
            "name": "flood_grid_summary",
            "description": "Summarize flood grid.",
            "input_schema": {"type": "object", "properties": {}},
        },
        {
            "name": "green_roof_estimate",
            "description": "Estimate green roof capture.",
            "input_schema": {"type": "object", "properties": {"roof_area_sqft": {"type": "number"}}},
        },
    ]


def test_get_tool_catalog_falls_back_for_missing_description(monkeypatch, caplog):
    import logging

    settings = mcp_client.get_settings()
    tools_by_url = {
        settings.mcp_geospatial_url: [_FakeTool("flood_grid_summary", "", {})],
        settings.mcp_calculators_url: [_FakeTool("green_roof_estimate", None, {})],
    }
    monkeypatch.setattr(mcp_client, "Client", _make_fake_client_class(tools_by_url))

    with caplog.at_level(logging.WARNING, logger="agentic.orchestrator.mcp_client"):
        catalog = asyncio.run(mcp_client.get_tool_catalog())

    descriptions = {tool["name"]: tool["description"] for tool in catalog}
    assert descriptions["flood_grid_summary"] == "Tool: flood_grid_summary"
    assert descriptions["green_roof_estimate"] == "Tool: green_roof_estimate"
    assert all(tool["description"] for tool in catalog)  # Bedrock rejects empty descriptions

    warned_tools = {
        record.args[0] if record.args else ""
        for record in caplog.records
        if "no description" in record.message.lower() or "no description" in record.msg.lower()
    }
    assert "flood_grid_summary" in warned_tools
    assert "green_roof_estimate" in warned_tools


def test_get_tool_catalog_is_cached(monkeypatch):
    settings = mcp_client.get_settings()
    call_count = {"n": 0}
    tools_by_url = {settings.mcp_geospatial_url: [_FakeTool("t1", "d", {})]}
    fake_client_cls = _make_fake_client_class(tools_by_url)

    class _CountingFakeClient(fake_client_cls):
        async def list_tools(self):
            call_count["n"] += 1
            return await super().list_tools()

    monkeypatch.setattr(mcp_client, "Client", _CountingFakeClient)

    asyncio.run(mcp_client.get_tool_catalog())
    asyncio.run(mcp_client.get_tool_catalog())

    # 4 servers queried once on first call; second call must not re-query any of them.
    assert call_count["n"] == 4


def test_call_tool_dispatches_to_owning_server(monkeypatch):
    settings = mcp_client.get_settings()
    tools_by_url = {
        settings.mcp_calculators_url: [_FakeTool("green_roof_estimate", "d", {})],
    }
    results_by_name = {
        "green_roof_estimate": _FakeCallResult(data={"captured_gallons": 123.4}),
    }
    monkeypatch.setattr(
        mcp_client, "Client", _make_fake_client_class(tools_by_url, results_by_name)
    )

    asyncio.run(mcp_client.get_tool_catalog())
    result = asyncio.run(
        mcp_client.call_tool("green_roof_estimate", {"roof_area_sqft": 1000})
    )

    assert result == {"captured_gallons": 123.4}


def test_call_tool_falls_back_to_text_content_when_no_data(monkeypatch):
    settings = mcp_client.get_settings()
    tools_by_url = {settings.mcp_data_url: [_FakeTool("list_runtime_assets", "d", {})]}
    results_by_name = {
        "list_runtime_assets": _FakeCallResult(content=[_FakeTextBlock("asset1.geojson")]),
    }
    monkeypatch.setattr(
        mcp_client, "Client", _make_fake_client_class(tools_by_url, results_by_name)
    )

    asyncio.run(mcp_client.get_tool_catalog())
    result = asyncio.run(mcp_client.call_tool("list_runtime_assets", {}))

    assert result == "asset1.geojson"


def test_call_tool_dispatches_get_methodology_to_data_server(monkeypatch):
    settings = mcp_client.get_settings()
    tools_by_url = {
        settings.mcp_data_url: [_FakeTool("get_methodology", "d", {})],
    }
    results_by_name = {
        "get_methodology": _FakeCallResult(
            data={"topic": "heat", "methodology": "**Urban Heat Island / Heat-Wave Risk (methodology)**"}
        ),
    }
    monkeypatch.setattr(
        mcp_client, "Client", _make_fake_client_class(tools_by_url, results_by_name)
    )

    asyncio.run(mcp_client.get_tool_catalog())
    result = asyncio.run(mcp_client.call_tool("get_methodology", {"topic": "heat"}))

    assert result == {
        "topic": "heat",
        "methodology": "**Urban Heat Island / Heat-Wave Risk (methodology)**",
    }


def test_call_tool_unknown_name_returns_error_without_raising(monkeypatch):
    monkeypatch.setattr(mcp_client, "Client", _make_fake_client_class({}))

    result = asyncio.run(mcp_client.call_tool("does_not_exist", {}))

    assert result == {"error": "Unknown tool: 'does_not_exist'"}


def test_call_tool_dispatch_failure_returns_error_without_raising(monkeypatch):
    settings = mcp_client.get_settings()
    tools_by_url = {settings.mcp_model_url: [_FakeTool("reload_production_model", "d", {})]}
    monkeypatch.setattr(
        mcp_client,
        "Client",
        _make_fake_client_class(
            tools_by_url, raise_for_names={"reload_production_model"}
        ),
    )

    asyncio.run(mcp_client.get_tool_catalog())
    result = asyncio.run(mcp_client.call_tool("reload_production_model", {}))

    assert result == {"error": "simulated failure calling reload_production_model"}
