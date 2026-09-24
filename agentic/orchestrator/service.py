from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import select

from agentic.common.observability import trace_span
from agentic.db.models import AgentRun
from agentic.db.session import SessionLocal
from agentic.orchestrator import mcp_client
from agentic.orchestrator.bedrock import invoke_claude, user_message


logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 10

# Only the final Q&A text of each prior turn is replayed as history,
# not the tool calls that produced it. Tool results are a snapshot
# of "now" at the time they were fetched; replaying them into a new
# request would present stale data to the model as if it were current.
MAX_HISTORY_TURNS = 5

NYC_TIMEZONE = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------

async def _fetch_history_messages(
    session_id: str,
) -> list[dict]:
    """Load recent persisted conversation turns."""

    async with SessionLocal() as session:
        result = await session.execute(
            select(AgentRun)
            .where(AgentRun.session_id == session_id)
            .order_by(AgentRun.id.asc())
        )

        rows = result.scalars().all()

    recent_rows = rows[-MAX_HISTORY_TURNS:]

    messages: list[dict] = []

    for row in recent_rows:
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "text": row.request_text,
                    }
                ],
            }
        )

        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "text": row.response_text,
                    }
                ],
            }
        )

    return messages


# ---------------------------------------------------------------------
# MCP dispatch
# ---------------------------------------------------------------------

async def _dispatch_tool_call(
    call: dict,
) -> object:
    """Dispatch one Bedrock-requested tool call to MCP."""

    try:
        return await mcp_client.call_tool(
            call["name"],
            call["input"],
        )

    except Exception as exc:
        logger.exception(
            "Unhandled error dispatching tool call name=%s",
            call["name"],
        )

        return {
            "error": str(exc),
        }


def _bedrock_json_object(
    value: Any,
) -> dict:
    """Normalize an MCP result for Bedrock Converse toolResult JSON.

    Bedrock's toolResult.content[].json field must contain a JSON
    object. MCP tools, however, may legitimately return dictionaries,
    lists, strings, numbers, booleans, or None.

    Dictionaries are preserved exactly. Other JSON-compatible values
    are wrapped under the ``result`` key.
    """

    if isinstance(value, dict):
        return value

    if isinstance(value, list):
        return {
            "result": value,
        }

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return {
            "result": value,
        }

    if value is None:
        return {
            "result": None,
        }

    return {
        "result": str(value),
    }


# ---------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------

async def _persist_agent_run(
    *,
    session_id: str,
    request_text: str,
    response_text: str,
    tool_calls: list[dict],
) -> None:
    """Persist an agent interaction without breaking the live response."""

    try:
        async with SessionLocal() as session:
            session.add(
                AgentRun(
                    session_id=session_id,
                    request_text=request_text,
                    response_text=response_text,
                    tool_calls=tool_calls,
                )
            )

            await session.commit()

    except Exception:
        logger.exception(
            "Failed to persist AgentRun "
            "for session_id=%s",
            session_id,
        )


# ---------------------------------------------------------------------
# Deterministic operational forecast routing
# ---------------------------------------------------------------------

def _normalize_message(
    message: str,
) -> str:
    """Normalize a user message for lightweight intent routing."""

    return re.sub(
        r"\s+",
        " ",
        message.strip().lower(),
    )


def _looks_like_forecast_request(
    message: str,
) -> bool:
    """Return True for clear flood/weather forecasting requests.

    This deliberately stays conservative. Ambiguous questions continue
    through the normal Bedrock orchestration path.
    """

    text = _normalize_message(message)

    forecast_terms = (
        "forecast",
        "flooding expected",
        "flood expected",
        "flood risk",
        "flooding tomorrow",
        "flood tomorrow",
        "flooding today",
        "flood today",
        "may experience flooding",
        "may flood",
        "flood susceptibility",
        "flood susceptible",
        "flood-susceptible",
        "flood vulnerability",
        "flood vulnerable",
        "flood-vulnerable",
    )

    return any(
        term in text
        for term in forecast_terms
    )


def _resolve_forecast_mode(
    message: str,
) -> str | None:
    """Resolve the supported operational forecast product."""

    text = _normalize_message(message)

    if "tomorrow" in text:
        return "tomorrow"

    if "today" in text:
        return "today"

    today_phrases = (
        "this morning",
        "this afternoon",
        "this evening",
        "tonight",
        "this night",
    )

    if any(
        phrase in text
        for phrase in today_phrases
    ):
        return "today"

    return None


def _resolve_daypart(
    message: str,
) -> str | None:
    """Resolve supported NYC-local daypart language."""

    text = _normalize_message(message)

    # More-specific phrases should be checked before broader ones.
    mappings = (
        (
            "overnight",
            (
                "overnight",
                "late night",
                "late tonight",
            ),
        ),
        (
            "morning",
            (
                "morning",
                "this morning",
            ),
        ),
        (
            "afternoon",
            (
                "afternoon",
                "this afternoon",
            ),
        ),
        (
            "evening",
            (
                "evening",
                "this evening",
            ),
        ),
        (
            "night",
            (
                "tonight",
                "nighttime",
                "at night",
            ),
        ),
    )

    for daypart, phrases in mappings:
        if any(
            phrase in text
            for phrase in phrases
        ):
            return daypart

    return None


def _format_nyc_datetime(
    value: Any,
) -> str | None:
    """Format an ISO datetime as a readable NYC-local timestamp."""

    if value is None:
        return None

    try:
        timestamp = datetime.fromisoformat(
            str(value).replace(
                "Z",
                "+00:00",
            )
        )

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(
                tzinfo=NYC_TIMEZONE
            )
        else:
            timestamp = timestamp.astimezone(
                NYC_TIMEZONE
            )

        return timestamp.strftime(
            "%A, %B %d at %-I:%M %p %Z"
        )

    except Exception:
        return str(value)


def _safe_number(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def _format_forecast_summary_response(
    data: dict,
    *,
    mode: str,
) -> str:
    """Turn the compact operational summary into user-facing text."""

    if not data.get(
        "forecast_available",
        False,
    ):
        return (
            f"The operational flood forecast for {mode} "
            "is not currently available."
        )

    forecast_hours = _safe_int(
        data.get("forecast_hours")
    )

    requested_hours = _safe_int(
        data.get(
            "requested_forecast_hours",
            forecast_hours,
        )
    )

    flood_grids = _safe_int(
        data.get("predicted_flood_grids")
    )

    flood_rows = _safe_int(
        data.get("predicted_flood_rows")
    )

    available_through = (
        _format_nyc_datetime(
            data.get(
                "forecast_available_through_nyc"
            )
            or data.get(
                "available_through_nyc"
            )
        )
    )

    label = (
        "today"
        if mode == "today"
        else "tomorrow"
    )

    parts = [
        (
            f"The authoritative NYC operational flood forecast "
            f"for {label} contains {forecast_hours} published "
            f"hour{'s' if forecast_hours != 1 else ''} across "
            f"837 1-km forecast grids."
        )
    ]

    if flood_grids > 0:
        parts.append(
            f"{flood_grids} unique 1-km grid "
            f"{'areas are' if flood_grids != 1 else 'area is'} "
            f"forecast as flood-susceptible, representing "
            f"{flood_rows} forecast-susceptible grid-hours."
        )
    else:
        parts.append(
            "No forecast grid currently crosses the operational "
            "flood-event threshold."
        )

    if available_through:
        parts.append(
            f"The forecast is available through "
            f"{available_through}."
        )

    if data.get(
        "forecast_horizon_truncated",
        False,
    ):
        parts.append(
            f"This is an adaptive forecast horizon: "
            f"{forecast_hours} of the requested "
            f"{requested_hours} hours are currently available."
        )

    parts.append(
        "The operational forecast uses HRRR precipitation on the "
        "NYC 1-km prediction grid and routes each grid through its "
        "selected support-sensor model."
    )

    return " ".join(parts)


def _extract_daypart_hours(
    data: dict,
) -> list[dict]:
    """Find an hourly summary list despite minor service schema changes."""

    candidate_keys = (
        "hourly_summary",
        "hours",
        "hourly_forecast",
        "hourly_counts",
        "available_hours",
    )

    for key in candidate_keys:
        value = data.get(key)

        if isinstance(value, list):
            return [
                row
                for row in value
                if isinstance(row, dict)
            ]

    return []


def _extract_top_risk_rows(
    data: dict,
) -> list[dict]:
    """Find compact top-risk rows returned by the daypart service."""

    candidate_keys = (
        "top_risk_grids",
        "top_risk",
        "highest_risk_grids",
        "risk_sample",
        "top_grids",
    )

    for key in candidate_keys:
        value = data.get(key)

        if isinstance(value, list):
            return [
                row
                for row in value
                if isinstance(row, dict)
            ]

    return []


def _format_daypart_response(
    data: dict,
    *,
    mode: str,
    daypart: str,
) -> str:
    """Create a compact deterministic answer from a daypart product."""

    if not data.get(
        "forecast_available",
        True,
    ):
        return (
            f"The operational forecast for {mode} "
            f"{daypart} is not currently available."
        )

    requested_hours = _safe_int(
        data.get(
            "requested_hour_count",
            data.get(
                "requested_hours",
                0,
            ),
        )
    )

    available_hours = _safe_int(
        data.get(
            "available_hour_count",
            data.get(
                "available_hours_count",
                data.get(
                    "published_hour_count",
                    0,
                ),
            ),
        )
    )

    hourly_rows = _extract_daypart_hours(
        data
    )

    if available_hours == 0 and hourly_rows:
        available_hours = len(hourly_rows)

    unique_flood_grids = _safe_int(
        data.get(
            "predicted_flood_grids",
            data.get(
                "unique_predicted_flood_grids",
                data.get(
                    "unique_flood_grids",
                    0,
                ),
            ),
        )
    )

    flood_grid_hours = _safe_int(
        data.get(
            "predicted_flood_rows",
            data.get(
                "predicted_flood_grid_hours",
                data.get(
                    "flood_grid_hours",
                    0,
                ),
            ),
        )
    )

    available_through = (
        _format_nyc_datetime(
            data.get(
                "forecast_available_through_nyc"
            )
            or data.get(
                "available_through_nyc"
            )
        )
    )

    coverage_complete = data.get(
        "coverage_complete"
    )

    if coverage_complete is None:
        coverage_complete = data.get(
            "daypart_fully_available"
        )

    label = f"{mode} {daypart}"

    # No hours at all for the requested daypart.
    if available_hours == 0 and not hourly_rows:
        if available_through:
            return (
                f"The operational forecast does not currently "
                f"extend into {label}. The forecast is available "
                f"through {available_through}. I won't infer flood "
                f"conditions beyond the published forecast horizon."
            )

        return (
            f"No published operational forecast hours are currently "
            f"available for {label}."
        )

    parts: list[str] = []

    if unique_flood_grids > 0:
        parts.append(
            f"For {label}, "
            f"{unique_flood_grids} unique 1-km grid "
            f"{'areas are' if unique_flood_grids != 1 else 'area is'} "
            f"forecast as flood-susceptible."
        )

        if flood_grid_hours > 0:
            parts.append(
                f"Those forecasts account for "
                f"{flood_grid_hours} forecast-susceptible grid-hours "
                f"within the available portion of the daypart."
            )
    else:
        parts.append(
            f"For the available forecast hours in {label}, "
            "no 1-km grid currently crosses the operational "
            "flood-event threshold."
        )

    if (
        coverage_complete is False
        or (
            requested_hours > 0
            and available_hours > 0
            and available_hours < requested_hours
        )
    ):
        parts.append(
            "Only part of this daypart is currently covered by "
            "the published HRRR-supported forecast horizon."
        )

    if available_through:
        parts.append(
            f"The forecast is available through "
            f"{available_through}."
        )

    # Give the user useful temporal information without dumping
    # hundreds or thousands of grid records.
    if hourly_rows:
        peak_row = max(
            hourly_rows,
            key=lambda row: _safe_number(
                row.get(
                    "predicted_flood_grids",
                    row.get(
                        "flood_grids",
                        row.get(
                            "predicted_grids",
                            0,
                        ),
                    ),
                )
            ),
        )

        peak_count = _safe_int(
            peak_row.get(
                "predicted_flood_grids",
                peak_row.get(
                    "flood_grids",
                    peak_row.get(
                        "predicted_grids",
                        0,
                    ),
                ),
            )
        )

        peak_time = (
            _format_nyc_datetime(
                peak_row.get(
                    "forecast_hour_nyc"
                )
                or peak_row.get(
                    "forecast_hour"
                )
                or peak_row.get(
                    "forecast_hour_utc"
                )
            )
        )

        if (
            peak_count > 0
            and peak_time
        ):
            parts.append(
                f"The largest hourly footprint in this period is "
                f"{peak_count} forecast flood-susceptible grids around "
                f"{peak_time}."
            )

    top_rows = _extract_top_risk_rows(
        data
    )

    if top_rows:
        parts.append(
            "The highest-risk locations are available to the "
            "forecast map as grid geometries; the user interface "
            "should present these spatially rather than exposing "
            "raw grid identifiers as the primary location."
        )

    return " ".join(parts)


async def _try_deterministic_forecast_route(
    message: str,
) -> dict | None:
    """Handle common operational forecast questions without Bedrock.

    Returning None means the request should continue through the normal
    LLM orchestration path.

    This route intentionally uses compact MCP products. It must never
    retrieve the full 837-grid x multi-hour forecast merely to answer a
    broad natural-language question.
    """

    if not _looks_like_forecast_request(
        message
    ):
        return None

    mode = _resolve_forecast_mode(
        message
    )

    # Keep ambiguous date references in the general agent path.
    if mode is None:
        return None

    daypart = _resolve_daypart(
        message
    )

    if daypart is not None:
        tool_name = (
            "get_daypart_forecast_summary"
        )

        tool_input = {
            "mode": mode,
            "daypart": daypart,
            "top_n": 20,
        }

        logger.info(
            "Deterministic forecast route: "
            "tool=%s input=%s",
            tool_name,
            tool_input,
        )

        output = await mcp_client.call_tool(
            tool_name,
            tool_input,
        )

        if not isinstance(output, dict):
            logger.warning(
                "Unexpected daypart forecast result "
                "type=%s; falling back to Bedrock",
                type(output).__name__,
            )
            return None

        response_text = (
            _format_daypart_response(
                output,
                mode=mode,
                daypart=daypart,
            )
        )

        return {
            "response": response_text,
            "tool_calls": [
                {
                    "name": tool_name,
                    "input": tool_input,
                    "routing": "deterministic",
                }
            ],
        }

    # General today/tomorrow summary.
    tool_name = "get_forecast_summary"

    tool_input = {
        "mode": mode,
    }

    logger.info(
        "Deterministic forecast route: "
        "tool=%s input=%s",
        tool_name,
        tool_input,
    )

    output = await mcp_client.call_tool(
        tool_name,
        tool_input,
    )

    if not isinstance(output, dict):
        logger.warning(
            "Unexpected forecast summary result "
            "type=%s; falling back to Bedrock",
            type(output).__name__,
        )
        return None

    response_text = (
        _format_forecast_summary_response(
            output,
            mode=mode,
        )
    )

    return {
        "response": response_text,
        "tool_calls": [
            {
                "name": tool_name,
                "input": tool_input,
                "routing": "deterministic",
            }
        ],
        "visualization": {
            "type": "forecast_susceptibility_map",
            "mode": mode,
            "daypart": daypart,
        },
    }


# ---------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------

def _visualization_from_tool_calls(
    tool_calls: list[dict],
    *,
    message: str,
) -> dict | None:
    """Build UI visualization metadata from actual forecasting tool use.

    The visualization is intentionally additive. Claude still receives
    the forecasting MCP outputs and produces the detailed narrative,
    tables, trends, peak-hour discussion, and grid-level explanation.

    This helper only tells Streamlit which authoritative operational
    forecast slice should be drawn underneath that response.
    """

    forecasting_tool_names = {
        "get_daypart_forecast_summary",
        "get_forecast_summary",
        "get_available_forecast_hours",
        "get_hourly_flood_forecast",
        "get_predicted_flood_events",
        "get_highest_risk_grids",
    }

    saw_forecasting_tool = False
    resolved_mode: str | None = None
    resolved_daypart: str | None = None
    resolved_forecast_hour: str | None = None

    for call in reversed(tool_calls):
        name = str(
            call.get("name", "")
        ).strip()

        if name not in forecasting_tool_names:
            continue

        saw_forecasting_tool = True

        raw_input = call.get("input")
        tool_input = (
            raw_input
            if isinstance(raw_input, dict)
            else {}
        )

        mode = str(
            tool_input.get("mode", "")
        ).strip().lower()

        if mode in {
            "today",
            "tomorrow",
        }:
            resolved_mode = mode

        # A specific-hour forecast is the most precise map request.
        if name in {
            "get_hourly_flood_forecast",
            "get_predicted_flood_events",
            "get_highest_risk_grids",
        }:
            forecast_hour = tool_input.get(
                "forecast_hour"
            )

            if forecast_hour:
                resolved_forecast_hour = str(
                    forecast_hour
                )
                break

        if name == "get_daypart_forecast_summary":
            daypart = str(
                tool_input.get("daypart", "")
            ).strip().lower()

            if daypart in {
                "overnight",
                "morning",
                "afternoon",
                "evening",
                "night",
            }:
                resolved_daypart = daypart

            if (
                resolved_mode is not None
                and resolved_daypart is not None
            ):
                break

    if not saw_forecasting_tool:
        return None

    if resolved_mode is None:
        resolved_mode = _resolve_forecast_mode(
            message
        )

    if resolved_daypart is None:
        resolved_daypart = _resolve_daypart(
            message
        )

    if resolved_mode not in {
        "today",
        "tomorrow",
    }:
        return None

    visualization = {
        "type": "forecast_susceptibility_map",
        "mode": resolved_mode,
    }

    if resolved_forecast_hour is not None:
        visualization["forecast_hour"] = (
            resolved_forecast_hour
        )
        return visualization

    if resolved_daypart is not None:
        visualization["daypart"] = (
            resolved_daypart
        )

    return visualization

def _compact_tool_output_for_bedrock(
    tool_name: str,
    normalized_output: dict,
) -> dict:
    """
    Keep Bedrock tool context bounded without changing the
    authoritative MCP result or visualization data.

    Large grid-level forecasting tools can legitimately return
    hundreds or thousands of rows. Claude only needs a compact
    analytical representation for synthesis; Streamlit reads the
    authoritative forecast artifacts separately for visualization.
    """

    large_forecast_tools = {
        "get_hourly_flood_forecast",
        "get_predicted_flood_events",
        "get_highest_risk_grids",
    }

    if tool_name not in large_forecast_tools:
        return normalized_output

    rows = normalized_output.get(
        "result"
    )

    if not isinstance(
        rows,
        list,
    ):
        return normalized_output

    if len(rows) <= 100:
        return normalized_output

    unique_grids = set()
    hourly_counts: dict[str, int] = {}
    predicted_event_count = 0

    for row in rows:
        if not isinstance(
            row,
            dict,
        ):
            continue

        grid_id = row.get(
            "grid_id"
        )

        if grid_id is not None:
            unique_grids.add(
                str(grid_id)
            )

        forecast_hour = row.get(
            "forecast_hour"
        )

        if forecast_hour is not None:
            hour_key = str(
                forecast_hour
            )

            hourly_counts[
                hour_key
            ] = (
                hourly_counts.get(
                    hour_key,
                    0,
                )
                + 1
            )

        if bool(
            row.get(
                "predicted_flood_event",
                False,
            )
        ):
            predicted_event_count += 1

    # Prefer rows that actually cross the event threshold so
    # Claude can still describe representative susceptible grids.
    event_rows = [
        row
        for row in rows
        if isinstance(
            row,
            dict,
        )
        and bool(
            row.get(
                "predicted_flood_event",
                False,
            )
        )
    ]

    sample_source = (
        event_rows
        if event_rows
        else rows
    )

    sample_rows = (
        sample_source[:75]
    )

    return {
        "result_type":
            "compact_forecast_tool_result",
        "tool_name":
            tool_name,
        "total_rows":
            len(rows),
        "unique_grid_count":
            len(unique_grids),
        "predicted_event_rows":
            predicted_event_count,
        "hourly_row_counts":
            hourly_counts,
        "sample_row_count":
            len(sample_rows),
        "sample_rows":
            sample_rows,
        "truncated_for_llm_context":
            True,
        "note": (
            "The complete authoritative forecast "
            "result remains available to the "
            "application. This tool result was "
            "compacted only for LLM synthesis "
            "to keep Bedrock context bounded."
        ),
    }

async def run_agent(
    message: str,
    session_id: str,
) -> dict:
    """Run one user request through the NYC Resilience orchestrator.

    Forecast questions use the normal Bedrock + MCP orchestration path
    so Claude can synthesize the full forecasting tool output. Any map
    visualization metadata is attached after tool use and does not
    replace or shorten Claude's response.
    """

    with trace_span(
        "agent.run",
        session_id=session_id,
    ):
        # -------------------------------------------------------------
        # GENERAL AGENT PATH:
        # Preserve Claude's full Bedrock/MCP synthesis for forecasting
        # and non-forecasting questions alike.
        # -------------------------------------------------------------

        catalog = (
            await mcp_client.get_tool_catalog()
        )

        try:
            messages = (
                await _fetch_history_messages(
                    session_id
                )
            )

        except Exception:
            logger.exception(
                "Failed to fetch conversation history "
                "for session_id=%s",
                session_id,
            )

            messages = []

        messages.append(
            user_message(message)
        )

        all_tool_calls: list[dict] = []

        result = {
            "stop_reason": "tool_use",
            "text": "",
            "tool_calls": [],
        }

        truncated = True

        for round_number in range(
            1,
            MAX_TOOL_ROUNDS + 1,
        ):
            result = invoke_claude(
                messages,
                tools=catalog,
            )

            if (
                result["stop_reason"]
                != "tool_use"
            ):
                truncated = False
                break

            tool_use_blocks: list[dict] = []
            tool_result_blocks: list[dict] = []

            for call in result["tool_calls"]:
                all_tool_calls.append(
                    call
                )

                tool_use_blocks.append(
                    {
                        "toolUse": {
                            "toolUseId": call[
                                "tool_use_id"
                            ],
                            "name": call["name"],
                            "input": call["input"],
                        }
                    }
                )

                output = (
                    await _dispatch_tool_call(
                        call
                    )
                )

                normalized_output = (
                    _bedrock_json_object(
                        output
                    )
                )

                bedrock_output = (
                    _compact_tool_output_for_bedrock(
                        call["name"],
                        normalized_output,
                    )
                )

                logger.debug(
                    "Tool result normalized for Bedrock: "
                    "round=%s tool=%s "
                    "raw_type=%s normalized_keys=%s",
                    round_number,
                    call["name"],
                    type(output).__name__,
                    list(
                        normalized_output.keys()
                    ),
                )

                tool_result_blocks.append(
                    {
                        "toolResult": {
                            "toolUseId": call[
                                "tool_use_id"
                            ],
                            "content": [
                                {
                                    "json": (
                                        bedrock_output
                                    ),
                                }
                            ],
                        }
                    }
                )

            messages.append(
                {
                    "role": "assistant",
                    "content": tool_use_blocks,
                }
            )

            messages.append(
                {
                    "role": "user",
                    "content": tool_result_blocks,
                }
            )

        response_text = result["text"]

        if truncated:
            note = (
                "[Tool-use loop stopped after "
                f"{MAX_TOOL_ROUNDS} rounds without "
                "a final answer.]"
            )

            if response_text:
                response_text = (
                    f"{response_text}\n\n{note}"
                )
            else:
                response_text = note

    visualization = _visualization_from_tool_calls(
        all_tool_calls,
        message=message,
    )

    await _persist_agent_run(
        session_id=session_id,
        request_text=message,
        response_text=response_text,
        tool_calls=all_tool_calls,
    )

    return {
        "session_id": session_id,
        "response": response_text,
        "tool_calls": all_tool_calls,
        "visualization": visualization,
    }