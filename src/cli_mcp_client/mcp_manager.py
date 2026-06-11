"""Manage stdio and SSE MCP servers and expose their tools to the LLM.

The MCP Python SDK is async and uses anyio cancel scopes that are task-bound, so
we run a single dedicated asyncio event loop on a background thread. Each server
connection lives inside one long-running task (so its context managers are entered
and exited on the same task); tool calls are dispatched onto the same loop.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

NAMESPACE_SEP = "__"


@dataclass
class _Server:
    session: ClientSession
    tools: list[Any]
    stop: asyncio.Event


@dataclass
class MCPManager:
    _loop: asyncio.AbstractEventLoop = field(init=False)
    _thread: threading.Thread = field(init=False)
    _servers: dict[str, _Server] = field(init=False, default_factory=dict)
    _tasks: dict[str, asyncio.Task] = field(init=False, default_factory=dict)

    def __init__(self) -> None:
        self._servers = {}
        self._tasks = {}
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _call(self, coro, timeout: float = 60.0):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    # ---- connection lifecycle ------------------------------------------
    async def _run_server(self, name, transport, config, ready: asyncio.Future):
        try:
            async with AsyncExitStack() as stack:
                if transport == "stdio":
                    params = StdioServerParameters(
                        command=config["command"],
                        args=config.get("args", []),
                        env=config.get("env") or None,
                    )
                    read, write = await stack.enter_async_context(stdio_client(params))
                elif transport == "sse":
                    read, write = await stack.enter_async_context(
                        sse_client(config["url"], headers=config.get("headers"))
                    )
                else:
                    raise ValueError(f"unknown transport: {transport}")

                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                tools = (await session.list_tools()).tools
                stop = asyncio.Event()
                self._servers[name] = _Server(session=session, tools=tools, stop=stop)
                if not ready.done():
                    ready.set_result(tools)
                await stop.wait()  # keep contexts alive until disconnect
        except Exception as e:  # noqa: BLE001
            if not ready.done():
                ready.set_exception(e)
        finally:
            self._servers.pop(name, None)

    async def _connect(self, name, transport, config):
        await self._disconnect(name)
        ready: asyncio.Future = self._loop.create_future()
        task = self._loop.create_task(self._run_server(name, transport, config, ready))
        self._tasks[name] = task
        return await ready

    async def _disconnect(self, name):
        srv = self._servers.get(name)
        if srv:
            srv.stop.set()
        task = self._tasks.pop(name, None)
        if task:
            try:
                await asyncio.wait_for(task, timeout=10)
            except Exception:  # noqa: BLE001
                task.cancel()

    def connect(self, name: str, transport: str, config: dict[str, Any]) -> list[Any]:
        """Connect a server; returns its tool list. Raises on failure."""
        return self._call(self._connect(name, transport, config), timeout=60)

    def disconnect(self, name: str) -> None:
        self._call(self._disconnect(name), timeout=20)

    def connected(self) -> list[str]:
        return list(self._servers.keys())

    # ---- tool exposure --------------------------------------------------
    def openai_tools(self) -> list[dict[str, Any]]:
        """All connected tools in OpenAI function-tool format, namespaced by server."""
        out: list[dict[str, Any]] = []
        for sname, srv in self._servers.items():
            for tool in srv.tools:
                out.append(
                    {
                        "type": "function",
                        "function": {
                            "name": f"{sname}{NAMESPACE_SEP}{tool.name}",
                            "description": tool.description or "",
                            "parameters": tool.inputSchema or {"type": "object", "properties": {}},
                        },
                    }
                )
        return out

    def list_tools(self) -> dict[str, list[tuple[str, str]]]:
        """{server: [(tool_name, description), ...]} for display."""
        return {
            s: [(t.name, t.description or "") for t in srv.tools]
            for s, srv in self._servers.items()
        }

    def call_tool(self, namespaced: str, arguments: dict[str, Any]) -> str:
        """Invoke a namespaced tool ('server__tool') and return text output."""
        if NAMESPACE_SEP not in namespaced:
            raise ValueError(f"bad tool name: {namespaced}")
        sname, tool = namespaced.split(NAMESPACE_SEP, 1)
        srv = self._servers.get(sname)
        if not srv:
            raise RuntimeError(f"server not connected: {sname}")
        result = self._call(srv.session.call_tool(tool, arguments), timeout=120)
        return self._render_result(result)

    @staticmethod
    def _render_result(result: Any) -> str:
        parts: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text is not None:
                parts.append(text)
            else:
                parts.append(str(block))
        if getattr(result, "isError", False):
            return "ERROR: " + ("\n".join(parts) or "tool call failed")
        return "\n".join(parts) if parts else "(no output)"

    def shutdown(self) -> None:
        for name in list(self._servers.keys()):
            try:
                self.disconnect(name)
            except Exception:  # noqa: BLE001
                pass
        self._loop.call_soon_threadsafe(self._loop.stop)
