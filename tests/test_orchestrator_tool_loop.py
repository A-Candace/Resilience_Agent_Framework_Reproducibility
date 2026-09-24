"""run_agent()'s tool-use loop and cross-request history memory, mocking
invoke_claude, the MCP client, and SessionLocal — no real Bedrock calls, no
real Postgres, no real Docker network calls."""
import asyncio

from agentic.db.models import AgentRun
from agentic.orchestrator import service as orchestrator_service


class _FakeResult:
    """Stands in for the SQLAlchemy Result returned by session.execute()."""

    def __init__(self, rows):
        self._rows = rows

    def scalars(self):
        return self

    def all(self):
        return self._rows


class _FakeSession:
    """Records what would have been persisted and answers history queries with
    canned rows, without needing real Postgres."""

    def __init__(self, store, history_rows=None, history_error=None):
        self._store = store
        self._history_rows = history_rows or []
        self._history_error = history_error

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def add(self, obj):
        self._store.append(obj)

    async def commit(self):
        pass

    async def execute(self, stmt):
        if self._history_error is not None:
            raise self._history_error
        return _FakeResult(self._history_rows)


class _ScriptedInvokeClaude:
    """Returns each response in order (repeating the last once exhausted); records every call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages, tools=None):
        # service.py mutates the same `messages` list in place across rounds —
        # snapshot it now so later appends don't retroactively change what
        # earlier calls appear to have received.
        self.calls.append({"messages": list(messages), "tools": tools})
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]

    @property
    def call_count(self):
        return len(self.calls)


def _patch_common(
    monkeypatch, invoke_claude_stub, call_tool_stub, catalog=None, history_rows=None, history_error=None
):
    monkeypatch.setattr(orchestrator_service, "invoke_claude", invoke_claude_stub)
    monkeypatch.setattr(
        orchestrator_service.mcp_client,
        "get_tool_catalog",
        lambda: _async_return(catalog or []),
    )
    monkeypatch.setattr(orchestrator_service.mcp_client, "call_tool", call_tool_stub)
    store: list = []
    monkeypatch.setattr(
        orchestrator_service,
        "SessionLocal",
        lambda: _FakeSession(store, history_rows=history_rows, history_error=history_error),
    )
    return store


async def _async_return(value):
    return value


def _agent_run(session_id, request_text, response_text, tool_calls=None):
    return AgentRun(
        session_id=session_id,
        request_text=request_text,
        response_text=response_text,
        tool_calls=tool_calls or [],
    )


def test_single_round_no_tool_use_works_as_before(monkeypatch):
    invoke = _ScriptedInvokeClaude(
        [{"stop_reason": "end_turn", "text": "The flood risk is moderate.", "tool_calls": []}]
    )

    async def call_tool_stub(name, arguments):
        raise AssertionError("no tool should be dispatched in a no-tool-use conversation")

    store = _patch_common(monkeypatch, invoke, call_tool_stub)

    result = asyncio.run(orchestrator_service.run_agent("What's the flood risk?", "s1"))

    assert invoke.call_count == 1
    assert invoke.calls[0]["messages"] == [
        {"role": "user", "content": [{"text": "What's the flood risk?"}]}
    ]
    assert result == {
        "session_id": "s1",
        "response": "The flood risk is moderate.",
        "tool_calls": [],
    }
    assert store[0].response_text == "The flood risk is moderate."
    assert store[0].tool_calls == []


def test_single_tool_call_round_trip(monkeypatch):
    tool_call = {"tool_use_id": "t1", "name": "flood_grid_summary", "input": {"prefix": "x"}}
    invoke = _ScriptedInvokeClaude(
        [
            {"stop_reason": "tool_use", "text": "", "tool_calls": [tool_call]},
            {
                "stop_reason": "end_turn",
                "text": "Based on the flood grid, risk is low.",
                "tool_calls": [],
            },
        ]
    )
    catalog = [{"name": "flood_grid_summary", "description": "d", "input_schema": {}}]

    async def call_tool_stub(name, arguments):
        assert name == "flood_grid_summary"
        assert arguments == {"prefix": "x"}
        return {"available": True, "rows": 42}

    store = _patch_common(monkeypatch, invoke, call_tool_stub, catalog=catalog)

    result = asyncio.run(orchestrator_service.run_agent("Is my block at risk?", "s2"))

    assert invoke.call_count == 2
    assert invoke.calls[0]["tools"] == catalog
    assert result["response"] == "Based on the flood grid, risk is low."
    assert result["tool_calls"] == [tool_call]
    assert store[0].tool_calls == [tool_call]

    # the second round's messages must carry the first round's tool call and
    # result forward as proper structured assistant/user turns
    assert invoke.calls[1]["messages"] == [
        {"role": "user", "content": [{"text": "Is my block at risk?"}]},
        {
            "role": "assistant",
            "content": [
                {
                    "toolUse": {
                        "toolUseId": "t1",
                        "name": "flood_grid_summary",
                        "input": {"prefix": "x"},
                    }
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "toolResult": {
                        "toolUseId": "t1",
                        "content": [{"json": {"available": True, "rows": 42}}],
                    }
                }
            ],
        },
    ]


def test_two_sequential_tool_calls(monkeypatch):
    call_a = {"tool_use_id": "t1", "name": "flood_grid_summary", "input": {}}
    call_b = {"tool_use_id": "t2", "name": "forecast_heatmap", "input": {"forecast_inches": 1.5}}
    invoke = _ScriptedInvokeClaude(
        [
            {"stop_reason": "tool_use", "text": "", "tool_calls": [call_a]},
            {"stop_reason": "tool_use", "text": "", "tool_calls": [call_b]},
            {"stop_reason": "end_turn", "text": "Combining both tools: moderate risk.", "tool_calls": []},
        ]
    )
    catalog = [
        {"name": "flood_grid_summary", "description": "d", "input_schema": {}},
        {"name": "forecast_heatmap", "description": "d", "input_schema": {}},
    ]

    async def call_tool_stub(name, arguments):
        return {"flood_grid_summary": {"rows": 10}, "forecast_heatmap": [{"grid_id": "1"}]}[name]

    store = _patch_common(monkeypatch, invoke, call_tool_stub, catalog=catalog)

    result = asyncio.run(orchestrator_service.run_agent("Assess risk with two tools", "s3"))

    assert invoke.call_count == 3
    assert result["response"] == "Combining both tools: moderate risk."
    assert result["tool_calls"] == [call_a, call_b]
    assert store[0].tool_calls == [call_a, call_b]

    # third round's messages: original turn + both rounds' assistant/user pairs, in order
    third_round_messages = invoke.calls[2]["messages"]
    assert third_round_messages[0] == {
        "role": "user",
        "content": [{"text": "Assess risk with two tools"}],
    }
    assert third_round_messages[1] == {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": "t1", "name": "flood_grid_summary", "input": {}}}],
    }
    assert third_round_messages[2] == {
        "role": "user",
        "content": [{"toolResult": {"toolUseId": "t1", "content": [{"json": {"rows": 10}}]}}],
    }
    assert third_round_messages[3] == {
        "role": "assistant",
        "content": [
            {
                "toolUse": {
                    "toolUseId": "t2",
                    "name": "forecast_heatmap",
                    "input": {"forecast_inches": 1.5},
                }
            }
        ],
    }
    assert third_round_messages[4] == {
        "role": "user",
        "content": [
            {"toolResult": {"toolUseId": "t2", "content": [{"json": [{"grid_id": "1"}]}]}}
        ],
    }
    assert len(third_round_messages) == 5


def test_ten_round_cap_stops_the_loop(monkeypatch):
    always_tool_use = {
        "stop_reason": "tool_use",
        "text": "still working",
        "tool_calls": [{"tool_use_id": "loop", "name": "loop_tool", "input": {}}],
    }
    invoke = _ScriptedInvokeClaude([always_tool_use])
    catalog = [{"name": "loop_tool", "description": "d", "input_schema": {}}]

    async def call_tool_stub(name, arguments):
        return {"ok": True}

    store = _patch_common(monkeypatch, invoke, call_tool_stub, catalog=catalog)

    result = asyncio.run(orchestrator_service.run_agent("loop forever please", "s4"))

    assert invoke.call_count == orchestrator_service.MAX_TOOL_ROUNDS == 10
    assert len(result["tool_calls"]) == 10
    assert "still working" in result["response"]
    assert "10 rounds" in result["response"]
    assert store[0].tool_calls == result["tool_calls"]

    # messages grow by one assistant+user pair per round: 1 initial + 2*9 appended
    # ahead of the 10th call
    assert len(invoke.calls[9]["messages"]) == 1 + 2 * 9
    assert len(invoke.calls[0]["messages"]) == 1


def test_tool_dispatch_exception_does_not_crash_run_agent(monkeypatch):
    tool_call = {"tool_use_id": "t1", "name": "flaky_tool", "input": {}}
    invoke = _ScriptedInvokeClaude(
        [
            {"stop_reason": "tool_use", "text": "", "tool_calls": [tool_call]},
            {
                "stop_reason": "end_turn",
                "text": "I couldn't complete that lookup, but here's what I know.",
                "tool_calls": [],
            },
        ]
    )
    catalog = [{"name": "flaky_tool", "description": "d", "input_schema": {}}]

    async def call_tool_stub(name, arguments):
        raise RuntimeError("mcp server unreachable")

    store = _patch_common(monkeypatch, invoke, call_tool_stub, catalog=catalog)

    result = asyncio.run(orchestrator_service.run_agent("Try the flaky tool", "s5"))

    assert invoke.call_count == 2
    assert result["response"] == "I couldn't complete that lookup, but here's what I know."
    assert result["tool_calls"] == [tool_call]
    assert store[0].tool_calls == [tool_call]

    # the failure must have been fed back to the model as a structured tool
    # result, not swallowed silently
    tool_result_message = invoke.calls[1]["messages"][2]
    assert tool_result_message["role"] == "user"
    error_content = tool_result_message["content"][0]["toolResult"]["content"][0]["json"]
    assert error_content == {"error": "mcp server unreachable"}


def test_history_from_two_prior_turns_is_prepended_in_order(monkeypatch):
    invoke = _ScriptedInvokeClaude(
        [{"stop_reason": "end_turn", "text": "Third answer.", "tool_calls": []}]
    )

    async def call_tool_stub(name, arguments):
        raise AssertionError("no tool should be dispatched")

    history_rows = [
        _agent_run("s6", "First question?", "First answer."),
        _agent_run("s6", "Second question?", "Second answer."),
    ]
    _patch_common(monkeypatch, invoke, call_tool_stub, history_rows=history_rows)

    asyncio.run(orchestrator_service.run_agent("Third question?", "s6"))

    assert invoke.calls[0]["messages"] == [
        {"role": "user", "content": [{"text": "First question?"}]},
        {"role": "assistant", "content": [{"text": "First answer."}]},
        {"role": "user", "content": [{"text": "Second question?"}]},
        {"role": "assistant", "content": [{"text": "Second answer."}]},
        {"role": "user", "content": [{"text": "Third question?"}]},
    ]


def test_history_is_capped_to_most_recent_five_of_six_rows(monkeypatch):
    invoke = _ScriptedInvokeClaude(
        [{"stop_reason": "end_turn", "text": "Seventh answer.", "tool_calls": []}]
    )

    async def call_tool_stub(name, arguments):
        raise AssertionError("no tool should be dispatched")

    history_rows = [
        _agent_run("s7", f"Question {i}?", f"Answer {i}.") for i in range(1, 7)
    ]
    _patch_common(monkeypatch, invoke, call_tool_stub, history_rows=history_rows)

    asyncio.run(orchestrator_service.run_agent("Question 7?", "s7"))

    sent_messages = invoke.calls[0]["messages"]
    # 5 prior turns (2 messages each) + the current turn's user message
    assert len(sent_messages) == 5 * 2 + 1
    assert sent_messages[0] == {"role": "user", "content": [{"text": "Question 2?"}]}
    assert sent_messages[-1] == {"role": "user", "content": [{"text": "Question 7?"}]}
    # the oldest row (Question 1) must have been dropped
    assert all("Question 1?" not in str(m) for m in sent_messages)


def test_brand_new_session_with_no_prior_rows_proceeds_with_current_message_only(monkeypatch):
    invoke = _ScriptedInvokeClaude(
        [{"stop_reason": "end_turn", "text": "First answer ever.", "tool_calls": []}]
    )

    async def call_tool_stub(name, arguments):
        raise AssertionError("no tool should be dispatched")

    _patch_common(monkeypatch, invoke, call_tool_stub, history_rows=[])

    result = asyncio.run(orchestrator_service.run_agent("Brand new question?", "s8-new"))

    assert invoke.calls[0]["messages"] == [
        {"role": "user", "content": [{"text": "Brand new question?"}]}
    ]
    assert result["response"] == "First answer ever."


def test_history_fetch_exception_falls_back_to_current_message_only(monkeypatch, caplog):
    import logging

    invoke = _ScriptedInvokeClaude(
        [{"stop_reason": "end_turn", "text": "Answer despite broken history.", "tool_calls": []}]
    )

    async def call_tool_stub(name, arguments):
        raise AssertionError("no tool should be dispatched")

    _patch_common(
        monkeypatch,
        invoke,
        call_tool_stub,
        history_error=ConnectionError("simulated broken database connection"),
    )

    with caplog.at_level(logging.ERROR, logger="agentic.orchestrator.service"):
        result = asyncio.run(orchestrator_service.run_agent("Question despite outage?", "s9"))

    assert result["response"] == "Answer despite broken history."
    assert invoke.calls[0]["messages"] == [
        {"role": "user", "content": [{"text": "Question despite outage?"}]}
    ]
    assert any(
        "Failed to fetch conversation history" in record.message for record in caplog.records
    )
