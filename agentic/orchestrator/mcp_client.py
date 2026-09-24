"""Client for the four MCP tool servers.

Discovers each server's tools once (cached for the process lifetime) and
dispatches a named tool call to whichever server actually owns it. Uses the
fastmcp Client against each server's streamable-HTTP endpoint — no
hand-rolled HTTP/JSON-RPC.
"""
import asyncio
import logging
from typing import Any

from fastmcp import Client

from agentic.common.settings import get_settings

logger = logging.getLogger(__name__)

_catalog: list[dict] | None = None
_tool_servers: dict[str, str] | None = None
_lock = asyncio.Lock()


def _server_urls() -> dict[str, str]:
    cfg = get_settings()
    return {
        "geospatial": cfg.mcp_geospatial_url,
        "calculators": cfg.mcp_calculators_url,
        "data": cfg.mcp_data_url,
        "model": cfg.mcp_model_url,
        "forecasting": cfg.mcp_forecasting_url,
    }


async def get_tool_catalog() -> list[dict]:
    """Discover tools from all MCP servers, caching the result for the process lifetime.

    Returns tool definitions in the {"name", "description", "input_schema"}
    shape that invoke_claude()'s `tools` parameter expects. A server that
    can't be reached is skipped (logged), not fatal to the others.
    """
    global _catalog, _tool_servers
    if _catalog is not None:
        return _catalog

    async with _lock:
        if _catalog is not None:
            return _catalog

        catalog: list[dict] = []
        tool_servers: dict[str, str] = {}

        for server_name, url in _server_urls().items():
            try:
                async with Client(url) as client:
                    tools = await client.list_tools()
            except Exception:
                logger.exception(
                    "Failed to discover tools from MCP server %s (%s)", server_name, url
                )
                continue

            for tool in tools:
                description = tool.description or ""
                if not description:
                    description = f"Tool: {tool.name}"
                    logger.warning(
                        "MCP tool %s (%s) has no description (missing docstring?) — "
                        "using placeholder %r. Bedrock rejects empty descriptions.",
                        tool.name,
                        url,
                        description,
                    )
                catalog.append(
                    {
                        "name": tool.name,
                        "description": description,
                        "input_schema": tool.inputSchema,
                    }
                )
                tool_servers[tool.name] = url

        _catalog = catalog
        _tool_servers = tool_servers

    return _catalog


async def call_tool(name: str, arguments: dict[str, Any]) -> Any:
    """Dispatch a tool call to whichever MCP server owns `name`.

    Never raises: an unknown tool name and a failed dispatch (unreachable
    server, tool-side error, timeout, ...) both come back as
    {"error": "..."} instead of propagating, so one bad tool call can't take
    down the whole request.
    """
    if _tool_servers is None:
        await get_tool_catalog()

    url = (_tool_servers or {}).get(name)
    if url is None:
        return {"error": f"Unknown tool: {name!r}"}

    try:
        async with Client(url) as client:
            result = await client.call_tool(name, arguments)
    except Exception as exc:
        logger.exception("Tool call failed: name=%s arguments=%s", name, arguments)
        return {"error": str(exc)}

    return _extract_result(result)


def _extract_result(result: Any) -> Any:
    if getattr(result, "data", None) is not None:
        return result.data
    if getattr(result, "structured_content", None) is not None:
        return result.structured_content
    texts = [block.text for block in getattr(result, "content", []) if hasattr(block, "text")]
    return "\n".join(texts) if texts else None


def reset_cache() -> None:
    """Test hook: clear the cached tool catalog so the next call re-discovers.

    Also rebuilds `_lock`: asyncio.Lock binds to whichever event loop first
    acquires it, so a lock left over from a previous asyncio.run() (a closed
    loop) would raise "bound to a different event loop" the next time it's
    acquired under a new one.
    """
    global _catalog, _tool_servers, _lock
    _catalog = None
    _tool_servers = None
    _lock = asyncio.Lock()
