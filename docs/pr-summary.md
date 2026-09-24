# PR Summary: Wiring the Agent Orchestrator Up For Real

Branch: `feature/agent-run-persistence` → `main`
Base compared against: `main` @ `a154727` (`git diff main`)

A note on how this doc was written: every claim below is checked against the
actual diff, actual test runs, or an actual live request I made against a
running docker-compose stack in this session — not against docstrings, commit
messages, or what the code is "supposed to" do. Where I only ran something
with mocks, it says so. Where I ran something for real against live Bedrock,
live MCP servers, and live Postgres, it says that too, with the evidence.

---

## What this PR does, in one paragraph

Before this branch, the repo had two completely separate things that looked
related but weren't wired together: the Streamlit app (which talks to Claude
directly, one question in / one answer out, no memory, no tools) and an
"agent orchestrator" subsystem (`agentic/`) that existed in the codebase but
wasn't called by anything, wasn't part of the deployed app, and — even if you
ran it — didn't actually pass any tools to Claude or save anything anywhere.
This PR connects the wiring: Claude, through the orchestrator, can now
actually call real tools (look up methodology text, check the flood grid,
run calculators) instead of just answering from what's in the prompt, and
every exchange gets saved to Postgres. The Chat page now talks to this
orchestrator, with a safety net that falls back to the old behavior if the
orchestrator isn't reachable. One unrelated, purely cosmetic change (a page
rename) is also included in this branch's history.

**Important scope note up front:** none of this is live in production. This
PR doesn't touch the Dockerfile, the Elastic Beanstalk deployment config, or
the GitHub Actions workflow. Everything above only runs when the full
`docker compose` stack is running locally (or wherever that compose file
gets deployed in the future). See "Known limitations" below.

---

## Before vs after

| | Before this PR | After this PR (when running the full `docker compose` stack) |
|---|---|---|
| **User asks a question in Chat** | Streamlit calls Bedrock directly (`call_claude()`), single request, single response. | Streamlit sends the question to the orchestrator (`POST /v1/agent`). |
| **Can Claude use tools?** | No — the Bedrock call never included a `tools` parameter, and the orchestrator's version of this call didn't either. | Yes — Claude sees a catalog of real tools (flood grid summary, forecast heuristic, green roof / rain garden calculators, S3 asset listing, and the new methodology lookup) and can call up to 10 rounds of them before giving a final answer. |
| **Does it remember anything?** | No persistence anywhere — the `agent_runs` Postgres table existed in the schema but nothing ever wrote to it. | Every exchange (question, which tools were called, final answer) is saved to Postgres, tagged with a `session_id`. |
| **What if the orchestrator is down?** | N/A (it was never called). | The chat page catches the failure, shows a warning, and falls back to the exact same direct-Bedrock call as before — so the page still works, just without tools/memory. |
| **Methodology context** | The full methodology text for every topic (urban features, coastal flood, riverine flood, custom risk, heat) was stuffed into every single chat prompt as JSON, whether relevant or not. | Claude fetches only the methodology text relevant to the question, on demand, via a real tool call. |
| **"AI Query" page** | Labeled "AI Query" everywhere in the UI. | Labeled "Multi-risk Identification Tool" everywhere in the UI. Same page, same code, same route key — label only. |

---

## What changed, file by file

### 1. Tool-use plumbing — `agentic/orchestrator/bedrock.py`

- **Before:** `invoke_claude(message)` called Bedrock's raw `invoke_model` API with a hand-built JSON payload. No `tools` parameter existed anywhere in the request. Returned a plain string.
- **After:** `invoke_claude(message, tools=None)` uses boto3's `converse()` API instead, translates a simple tool-dict list into Converse's `toolConfig` format, and returns a dict: `{"stop_reason", "text", "tool_calls"}` instead of a bare string.
- **Verified with mocks:** `tests/test_bedrock_converse.py` (4 tests) — a fake object stands in for the boto3 bedrock-runtime client's `.converse()` method. No real AWS calls in these tests. I ran them this session: all pass.
- **Verified live:** see the orchestrator round-trip below — a real `converse()` call against real Bedrock actually happened as part of that test.

### 2. Dispatch loop & persistence — `agentic/orchestrator/service.py` + new `agentic/orchestrator/mcp_client.py`

- **New file, `mcp_client.py`:** discovers tools from all four MCP servers (geospatial, calculators, data, model) using the `fastmcp` client library, caches the combined catalog for the process's lifetime, and routes a named tool call to whichever server owns it. An unknown tool name or a failed dispatch (server down, tool-side error, timeout) both come back as `{"error": "..."}` instead of raising — one bad tool call can't take down the whole request.
- **`service.py`'s `run_agent()` rewritten:** it now loops up to 10 rounds — ask Claude with the tool catalog attached, and if it asks for a tool, run it and feed the result back in, repeat — until Claude gives a final (non-tool-use) answer, or the 10-round cap is hit (in which case a note is appended saying so). At the end, the whole exchange is written to Postgres as an `AgentRun` row (`session_id`, request text, response text, the list of tool calls made).
- **Caveat on how "conversation" is carried between rounds:** the message sent back to Claude on each round is a growing plain-text string — the original question, plus each tool call and its result, appended as text — not Converse's structured multi-turn `messages` array. This works fine for the in-request tool loop it's used for, but it's a workaround, not real multi-turn history. See Known Limitations.
- **Verified with mocks:** `tests/test_orchestrator_tool_loop.py` (5 tests — no-tool passthrough, one tool call, two sequential tool calls, the 10-round cap, and a tool-dispatch-exception case) and `tests/test_mcp_client.py` (8 tests, including one for the new `get_methodology` tool) both fake the pieces they touch (`invoke_claude`, the MCP client, `fastmcp.Client`) — no real Bedrock or Docker network calls. `tests/test_orchestrator_resilience.py` (1 test) confirms a broken DB write doesn't crash the request, using a fake session object. All ran this session: pass.
- **`tests/test_orchestrator_persistence.py` is different — it's a real integration test, no DB mocking.** It needs an actual reachable Postgres via `DATABASE_URL`. Run without one, it skips itself (I saw this happen: `pytest.skip(...)`, reason `nodename nor servname provided` — the default `DATABASE_URL` points at the docker-compose hostname `postgres`, which doesn't resolve outside that network). **I also ran it against a real Postgres this session (see "How to verify" for the exact steps) and it passed for real** — a row genuinely landed in the `agent_runs` table.
- **Why `docker-compose.yml` changed too:** this PR publishes ports 8001–8004 (the four MCP servers) and 5432 (Postgres) to the host, which is what makes it possible to run the persistence test, or hit an MCP server directly, from outside the Docker network at all. Before this PR those ports weren't published.

### 3. New `get_methodology` MCP tool — `agentic/mcp_servers/data_access.py`

- Added to the existing data-access MCP server (already used for `list_runtime_assets`).
- Takes a `topic` argument — one of `urban_features`, `coastal_flood`, `riverine_flood`, `custom_risk`, `heat` — and returns the corresponding methodology reference text. The text itself isn't new or rewritten; it's the same `METH_URBAN` / `METH_RISK[...]` / `METH_UHI` constants already defined in `core/shared.py`, just made reachable as a tool instead of always being pasted into every chat prompt.
- Invalid topic → raises `ValueError` listing the five valid names (fastmcp turns that into a normal MCP tool-error response; it doesn't crash the server).
- **Verified with mocks:** one dispatch test in `test_mcp_client.py`, faking `fastmcp.Client` the same way as the other MCP client tests.
- **Verified directly (no mocks):** in this session I imported the function and called it in-process — `get_methodology("heat")` returned the real `METH_UHI` text; an invalid topic raised `ValueError` listing all five valid names.
- **Verified live, end to end:** see below — in a real conversation against real Bedrock, Claude chose to call this exact tool and used its result correctly.

### 4. Chat page wiring — `src/resilience_app/agents/chat_agent/page.py`

- **Before:** always called `call_claude()` in-process (direct Bedrock, as described above), with a large JSON blob of context (stats + all five methodology texts) baked into every prompt.
- **After:** sends just the user's raw question, plus a `session_id` (created once per browser tab via `uuid4()`, reused for that tab's lifetime), as a POST to `{orchestrator_url}/v1/agent`. On success, shows the response, and — if any tools were used — a small "🔧 Used N tool call(s)" expander naming them. On *any* failure (timeout, connection error, bad response), it catches the exception, shows a warning, and falls back to the original direct `call_claude()` call, unchanged.
- New setting: `orchestrator_url` in `agentic/common/settings.py`, default `http://orchestrator:8090` (the docker-compose service name), overridable via `ORCHESTRATOR_URL`.
- **Verified:** the JSON shape the page sends/expects (`{message, session_id}` in, `{session_id, response, tool_calls}` out) matches exactly what I verified live against the orchestrator's `/v1/agent` endpoint (see below). I also confirmed, from inside the running `app` container, that `http://orchestrator:8090/health` resolves and responds — i.e., the network path this code depends on is actually up.
- **Not verified this session:** I did not click through the actual Streamlit page in a browser to visually confirm the chat bubble and tool-expander render correctly. That's a real gap — worth a quick manual check (or a Playwright script) before merging, and is the one item in "How to verify" below marked as a manual step.

**Live, end-to-end verification (this session), no mocks anywhere in this path:**

A docker-compose stack (postgres, all 4 MCP servers, orchestrator, phoenix, app) was already running with real AWS credentials configured. I sent a real request straight to the orchestrator:

```
curl -X POST http://localhost:8090/v1/agent -H "Content-Type: application/json" \
  -d '{"message": "What methodology do you use to calculate coastal flood risk? Please look it up rather than guessing.", "session_id": "pr-summary-verify-1"}'
```

Result (8.5 seconds, real Bedrock round trip):

```json
{
  "session_id": "pr-summary-verify-1",
  "response": "... NRI Coastal Flooding Risk Methodology ... Source: FEMA National Risk Index (NRI) ... Metric used: CFLD_RISKS ...",
  "tool_calls": [
    {"tool_use_id": "tooluse_AhYZzr8AGnch13fjZIdUzi", "name": "get_methodology", "input": {"topic": "coastal_flood"}}
  ]
}
```

Claude genuinely chose to call `get_methodology` with `topic: "coastal_flood"`, got back the real text, and its final answer matches the real `METH_RISK["NRI Coastal"]` content. I then queried Postgres directly:

```sql
select session_id, tool_calls from agent_runs where session_id='pr-summary-verify-1';
```

which returned exactly that row and tool-call list — confirming the persistence path works end to end too, not just in the mocked resilience test.

### 5. "AI Query" → "Multi-risk Identification Tool" rename

- Pure display-string change across 6 files: sidebar button, page title, home-page landing button, the route label in `registry.py`, `README.md`, and two docs files.
- Route key (`"query"`), module path (`ai_query_agent`), and function name (`page_ai_query`) are all untouched — nothing that routes on those breaks.
- **Verified:** full test suite passes unchanged after this commit; no test asserts on the old label text.
- Unrelated to the orchestrator work above — it's in this branch's history, but it's a separate, independent change.

---

## Known limitations / explicitly out of scope

- **Flat-string conversation history, not real multi-turn `messages`.** Within one request, the tool-use loop resends a growing text blob (question + each tool call + each tool result), not Converse's structured `messages` array. And while each turn is now persisted to Postgres with a `session_id`, nothing currently reads a prior turn back out of the database and replays it to Claude on the next request — so multi-turn *memory* isn't actually implemented yet, only multi-turn *storage*. From Claude's point of view, each new chat message is still a fresh conversation (tool-use round-trips within that one message aside).
- **Production is unaffected by this PR.** No changes to `Dockerfile`, `deployment/`, or `.github/workflows/`. Elastic Beanstalk still deploys the single-container image (Streamlit + model-reload API + nginx only, per `deployment/supervisor/supervisord.conf`) — the orchestrator, MCP servers, and Postgres exist only in `docker-compose.yml` (local/dev). In the deployed app today, the chat page's request to the orchestrator will fail to even resolve the hostname, and it'll silently take the fallback path every time — in production specifically, behavior is unchanged from before this PR.
- **The flood-risk grid file isn't in S3 yet.** `flood_grid_summary` and `forecast_heatmap` depend on a grid file that hasn't been uploaded to the runtime S3 location — those two tools will raise `FileNotFoundError` until that data exists, independent of anything in this PR.
- **No heat-wave forecasting exists anywhere in the codebase.** Only a flood-rainfall forecast heuristic (`forecast_heatmap`) exists. The existing "Urban Heat Island" page is a static FEMA NRI risk map, not a forecast — there's nothing to wire up for heat forecasting because it doesn't exist yet.
- **`forecast_heatmap` is an explicit placeholder**, by its own (newly added, for accuracy) docstring: a simple historical-rainfall-threshold heuristic, standing in ahead of the project's planned GraphSAGE-based model. Not a validated prediction. This PR documented that fact; it didn't change the underlying logic.
- **Observability (Phoenix/OTel) is untouched.** `trace_span()` still wraps `run_agent()` and still silently swallows any error connecting to Phoenix. This PR doesn't add, fix, or verify tracing.
- **Page layout / default landing page** was deliberately left exactly as-is, pending a separate decision from the team.

---

## How to verify this locally

**1. Environment**

```bash
cp .env.example .env
# fill in AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY / AWS_REGION for a Bedrock-enabled account
```

**2. Bring up the full stack**

```bash
docker compose build
docker compose up -d
```

This starts `app` (Streamlit, port 8080), `orchestrator` (port 8090), all four MCP servers (ports 8001–8004, now published to the host by this PR), `postgres` (port 5432, also now published), and `phoenix`.

**3. Run the test suite**

From the host, against the Postgres port this PR now exposes:

```bash
PYTHONPATH=$(pwd):$(pwd)/src \
DATABASE_URL="postgresql+asyncpg://resilience:resilience@localhost:5432/resilience" \
python -m pytest tests/ -v
```

Expect **26 passed**. If you omit `DATABASE_URL` (or Postgres isn't reachable), you'll instead see **25 passed, 1 skipped** — the skipped one is `test_orchestrator_persistence.py`, the only test that needs a real database.

**4. Prove a tool actually gets called (no browser needed)**

```bash
curl -X POST http://localhost:8090/v1/agent \
  -H "Content-Type: application/json" \
  -d '{"message": "What methodology do you use for coastal flood risk?", "session_id": "manual-test-1"}'
```

Look for a non-empty `tool_calls` array in the response — Claude *may or may not* choose to call a tool depending on the exact wording of your question (that's normal model behavior, not something to force), but for a methodology question it reliably called `get_methodology` when I tried it.

Then confirm it was persisted:

```bash
docker compose exec postgres psql -U resilience -d resilience \
  -c "select session_id, request_text, tool_calls from agent_runs order by id desc limit 1;"
```

**5. Manual UI check (not automated in this PR — do this yourself before approving)**

Open `http://localhost:8080`, go to Chat, ask a question. Look for the "🔧 Used N tool call(s)" expander under the answer — if it's there, you're going through the orchestrator/tool path. If you only see a `st.warning` about the orchestrator being unavailable followed by an immediate plain answer, the fallback path fired instead (check `docker compose logs orchestrator` for why).
