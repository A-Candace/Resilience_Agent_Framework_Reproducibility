"""Integration test: run_agent() against a real Postgres (DATABASE_URL). No DB mocking."""
import asyncio
import uuid

import pytest
from sqlalchemy import select

from agentic.db.models import AgentRun, Base
from agentic.db.session import SessionLocal, _engine
from agentic.orchestrator import service as orchestrator_service


async def _async_return(value):
    return value


async def _fetch_agent_run(session_id: str) -> AgentRun:
    async with SessionLocal() as session:
        result = await session.execute(
            select(AgentRun).where(AgentRun.session_id == session_id)
        )
        return result.scalar_one()


async def _delete_agent_run(session_id: str) -> None:
    async with SessionLocal() as session:
        result = await session.execute(
            select(AgentRun).where(AgentRun.session_id == session_id)
        )
        row = result.scalar_one_or_none()
        if row is not None:
            await session.delete(row)
            await session.commit()


async def _run_test(session_id: str, message: str) -> tuple[dict, AgentRun]:
    # A single event loop for the whole test: agentic/db/session.py builds its
    # async engine once at import time, so asyncpg connections in that pool
    # are bound to whichever event loop first used them. Spanning multiple
    # asyncio.run() calls (each its own loop) breaks that pool.
    try:
        async with _engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
    except OSError as exc:
        pytest.skip(f"Postgres not reachable via DATABASE_URL: {exc}")

    result = await orchestrator_service.run_agent(message, session_id)
    row = await _fetch_agent_run(session_id)
    await _delete_agent_run(session_id)
    return result, row


def test_run_agent_persists_agent_run(monkeypatch):
    fake_tool_calls = [{"tool_use_id": "tooluse_1", "name": "flood_grid_summary", "input": {}}]
    # Two scripted rounds: tool_use (dispatches once) then end_turn (stops the
    # loop). A mock that returns tool_use forever would run all 10 rounds and
    # re-append the same tool call each time, which isn't what this test —
    # persistence of whatever invoke_claude/dispatch produce — is checking.
    responses = [
        {"stop_reason": "tool_use", "text": "", "tool_calls": fake_tool_calls},
        {
            "stop_reason": "end_turn",
            "text": "final answer using the flood grid",
            "tool_calls": [],
        },
    ]
    calls = {"n": 0}

    def fake_invoke_claude(message, tools=None):
        index = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return responses[index]

    monkeypatch.setattr(orchestrator_service, "invoke_claude", fake_invoke_claude)
    monkeypatch.setattr(
        orchestrator_service.mcp_client, "get_tool_catalog", lambda: _async_return([])
    )

    async def fake_call_tool(name, arguments):
        return {"available": True}

    monkeypatch.setattr(orchestrator_service.mcp_client, "call_tool", fake_call_tool)

    session_id = f"test-{uuid.uuid4()}"
    message = "what is the flood risk near the Rockaways?"

    result, row = asyncio.run(_run_test(session_id, message))

    assert result["session_id"] == session_id
    assert result["response"] == "final answer using the flood grid"
    assert result["tool_calls"] == fake_tool_calls
    assert row.session_id == session_id
    assert row.request_text == message
    assert row.response_text == result["response"]
    assert row.tool_calls == fake_tool_calls
