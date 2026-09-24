"""run_agent() must return the Claude response even if persisting the AgentRun fails."""
import asyncio
import logging

from agentic.orchestrator import service as orchestrator_service


class _BrokenSession:
    """Stands in for a session whose connection is broken/closed mid-request."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def add(self, obj):
        pass

    async def commit(self):
        raise ConnectionError("simulated broken database connection")


async def _async_return(value):
    return value


def test_run_agent_survives_db_write_failure(monkeypatch, caplog):
    monkeypatch.setattr(
        orchestrator_service,
        "invoke_claude",
        lambda messages, tools=None: {
            "stop_reason": "end_turn",
            "text": f"echo: {messages[-1]['content'][0]['text']}",
            "tool_calls": [],
        },
    )
    monkeypatch.setattr(
        orchestrator_service.mcp_client, "get_tool_catalog", lambda: _async_return([])
    )
    monkeypatch.setattr(orchestrator_service, "SessionLocal", lambda: _BrokenSession())

    with caplog.at_level(logging.ERROR, logger="agentic.orchestrator.service"):
        result = asyncio.run(orchestrator_service.run_agent("hello", "session-broken"))

    assert result == {
        "session_id": "session-broken",
        "response": "echo: hello",
        "tool_calls": [],
    }
    assert any(
        "Failed to persist AgentRun" in record.message for record in caplog.records
    )
