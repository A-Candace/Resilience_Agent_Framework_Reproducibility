"""invoke_claude() against a mocked bedrock-runtime client — no real AWS calls."""
from agentic.orchestrator import bedrock as bedrock_module


class _FakeBedrockClient:
    """Stands in for boto3's bedrock-runtime client; records the last converse() call."""

    def __init__(self, response):
        self._response = response
        self.last_request = None

    def converse(self, **kwargs):
        self.last_request = kwargs
        return self._response


def _end_turn_response(text: str) -> dict:
    return {
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
        "stopReason": "end_turn",
    }


def _tool_use_response(tool_use_id: str, name: str, tool_input: dict, text: str | None = None) -> dict:
    content = []
    if text:
        content.append({"text": text})
    content.append(
        {"toolUse": {"toolUseId": tool_use_id, "name": name, "input": tool_input}}
    )
    return {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": "tool_use",
    }


def test_text_only_response_with_end_turn(monkeypatch):
    fake_client = _FakeBedrockClient(_end_turn_response("The flood risk here is moderate."))
    monkeypatch.setattr(bedrock_module, "bedrock_client", lambda: fake_client)

    messages = [bedrock_module.user_message("What's the flood risk near the Rockaways?")]
    result = bedrock_module.invoke_claude(messages)

    assert result == {
        "stop_reason": "end_turn",
        "text": "The flood risk here is moderate.",
        "tool_calls": [],
    }


def test_tool_use_response_preserves_tool_call(monkeypatch):
    fake_client = _FakeBedrockClient(
        _tool_use_response(
            tool_use_id="tooluse_abc123",
            name="flood_grid_summary",
            tool_input={"prefix": "nyc-resilience/data/current/"},
            text="Let me check the flood grid first.",
        )
    )
    monkeypatch.setattr(bedrock_module, "bedrock_client", lambda: fake_client)

    tools = [
        {
            "name": "flood_grid_summary",
            "description": "Summarize the current flood risk grid.",
            "input_schema": {"type": "object", "properties": {}},
        }
    ]

    messages = [bedrock_module.user_message("Is my block at flood risk?")]
    result = bedrock_module.invoke_claude(messages, tools=tools)

    assert result["stop_reason"] == "tool_use"
    assert result["text"] == "Let me check the flood grid first."
    assert result["tool_calls"] == [
        {
            "tool_use_id": "tooluse_abc123",
            "name": "flood_grid_summary",
            "input": {"prefix": "nyc-resilience/data/current/"},
        }
    ]


def test_tools_are_translated_into_converse_tool_config(monkeypatch):
    fake_client = _FakeBedrockClient(_end_turn_response("ok"))
    monkeypatch.setattr(bedrock_module, "bedrock_client", lambda: fake_client)

    tools = [
        {
            "name": "green_roof_estimate",
            "description": "Estimate green roof stormwater capture.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "roof_area_sqft": {"type": "number"},
                    "coverage_percent": {"type": "number"},
                },
                "required": ["roof_area_sqft", "coverage_percent"],
            },
        }
    ]

    messages = [bedrock_module.user_message("Estimate a green roof")]
    bedrock_module.invoke_claude(messages, tools=tools)

    assert fake_client.last_request["toolConfig"] == {
        "tools": [
            {
                "toolSpec": {
                    "name": "green_roof_estimate",
                    "description": "Estimate green roof stormwater capture.",
                    "inputSchema": {"json": tools[0]["input_schema"]},
                }
            }
        ]
    }


def test_no_tools_omits_tool_config_from_request(monkeypatch):
    fake_client = _FakeBedrockClient(_end_turn_response("ok"))
    monkeypatch.setattr(bedrock_module, "bedrock_client", lambda: fake_client)

    messages = [bedrock_module.user_message("Just answer normally")]
    bedrock_module.invoke_claude(messages)

    assert "toolConfig" not in fake_client.last_request
    assert fake_client.last_request["messages"] is messages
    assert fake_client.last_request["messages"] == [
        {"role": "user", "content": [{"text": "Just answer normally"}]}
    ]
