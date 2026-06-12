"""Metasploit-styled REPL tying together storage, the LLM, and MCP servers."""

from __future__ import annotations

import json
import shlex
from typing import Any, Optional

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.table import Table

from . import ui
from .agent import ABORT, ALWAYS, MEMORY_TOOL, ONCE, SKIP, AgentRunner
from .config import Config
from .llm import LLMClient, trim_history
from .mcp_manager import MCPManager
from .storage import Storage

BANNER = r"""
       =[ cli-mcp-client v0.1.0                         ]
+ -- --=[ remote LLM + MCP servers, REPL edition        ]
+ -- --=[ type 'help' for commands                      ]
"""

PROMPT_STYLE = Style.from_dict({"prompt": "ansired bold"})


class Repl:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.console = Console()
        self.storage = Storage(cfg.db_path)
        self.llm = LLMClient(
            cfg.base_url,
            cfg.model,
            cfg.auth_token,
            cfg.temperature,
            parse_text_tool_calls=cfg.parse_text_tool_calls,
        )
        self.mcp = MCPManager()
        self.session_id: Optional[int] = None
        self.psession: PromptSession = PromptSession(
            history=FileHistory(str(cfg.db_path.parent / "history"))
        )

    # ---- prompt ---------------------------------------------------------
    def _prompt_text(self) -> str:
        ctx = f"session-{self.session_id}" if self.session_id else "no-session"
        connected = self.mcp.connected()
        mcp_part = f" mcp({len(connected)})" if connected else ""
        return f"mcpc{mcp_part} {ctx} > "

    def run(self) -> None:
        self.console.print(BANNER, style="red")
        missing = self.cfg.missing()
        if missing:
            self.console.print(
                f"[yellow]warning:[/] missing config: {', '.join(missing)} "
                f"(set them in .env; chat will fail until then)"
            )
        self._autoconnect_servers()
        self._ensure_session()

        while True:
            try:
                line = self.psession.prompt(
                    [("class:prompt", self._prompt_text())], style=PROMPT_STYLE
                ).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
            if not self._dispatch(line):
                break
        self._shutdown()

    # ---- command dispatch ----------------------------------------------
    def _dispatch(self, line: str) -> bool:
        parts = shlex.split(line)
        cmd, args = parts[0], parts[1:]
        handlers = {
            "help": self.cmd_help,
            "?": self.cmd_help,
            "chat": lambda a: self.cmd_chat(" ".join(a)),
            "sessions": self.cmd_sessions,
            "session": self.cmd_session,
            "memory": self.cmd_memory,
            "agent": self.cmd_agent,
            "mcp": self.cmd_mcp,
            "set": self.cmd_set,
            "clear": lambda a: self.console.clear(),
            "exit": lambda a: False,
            "quit": lambda a: False,
        }
        handler = handlers.get(cmd)
        if handler is None:
            # Bare text → treat as a chat message.
            self.cmd_chat(line)
            return True
        result = handler(args)
        return result is not False

    # ---- commands -------------------------------------------------------
    def cmd_help(self, args) -> None:
        t = Table(title="commands", show_header=True, header_style="bold red")
        t.add_column("command")
        t.add_column("description")
        rows = [
            ("chat <msg>  (or just type)", "send a message to the LLM"),
            ("sessions", "list saved sessions"),
            ("session new [name]", "start a new session"),
            ("session resume <id>", "load a past session"),
            ("session rename <name>", "rename current session"),
            ("session delete <id>", "delete a session"),
            ("memory list", "show saved long-term memories"),
            ("memory add <fact>", "manually save a memory"),
            ("memory del <id>", "delete a memory"),
            ("agent run <objective>", "run an autonomous ReAct agent toward a goal"),
            ("agent runs", "list past agent runs"),
            ("agent show <id>", "show the step-by-step trace of a run"),
            ("mcp list", "list configured/connected MCP servers"),
            ("mcp add stdio <name> <cmd> [args...]", "add a stdio server"),
            ("mcp add sse <name> <url>", "add an SSE server"),
            ("mcp connect <name>", "connect a configured server"),
            ("mcp disconnect <name>", "disconnect a server"),
            ("mcp tools", "list tools from connected servers"),
            ("mcp del <name>", "remove a server config"),
            ("set", "show current LLM config"),
            ("clear", "clear the screen"),
            ("exit / quit", "leave the client"),
        ]
        for c, d in rows:
            t.add_row(c, d)
        self.console.print(t)

    # ---- sessions -------------------------------------------------------
    def _ensure_session(self) -> None:
        if self.session_id is None:
            self.session_id = self.storage.create_session()
            self.console.print(f"[green]*[/] started session {self.session_id}")

    def cmd_sessions(self, args) -> None:
        rows = self.storage.list_sessions()
        if not rows:
            self.console.print("[dim]no sessions yet[/]")
            return
        t = Table(show_header=True, header_style="bold red")
        for col in ("id", "name", "messages", "updated"):
            t.add_column(col)
        import time as _t
        for r in rows:
            marker = " *" if r["id"] == self.session_id else ""
            t.add_row(
                str(r["id"]) + marker,
                r["name"],
                str(r["msg_count"]),
                _t.strftime("%Y-%m-%d %H:%M", _t.localtime(r["updated_at"])),
            )
        self.console.print(t)

    def cmd_session(self, args) -> None:
        if not args:
            self.console.print("usage: session new|resume|rename|delete ...")
            return
        sub = args[0]
        if sub == "new":
            name = " ".join(args[1:]) or None
            self.session_id = self.storage.create_session(name)
            self.console.print(f"[green]*[/] started session {self.session_id}")
        elif sub == "resume":
            if len(args) < 2 or not args[1].isdigit():
                self.console.print("usage: session resume <id>")
                return
            sid = int(args[1])
            if not self.storage.get_session(sid):
                self.console.print(f"[red]no such session: {sid}[/]")
                return
            self.session_id = sid
            n = len(self.storage.get_messages(sid))
            self.console.print(f"[green]*[/] resumed session {sid} ({n} messages)")
        elif sub == "rename":
            name = " ".join(args[1:])
            if not name:
                self.console.print("usage: session rename <name>")
                return
            self.storage.rename_session(self.session_id, name)
            self.console.print("[green]*[/] renamed")
        elif sub == "delete":
            if len(args) < 2 or not args[1].isdigit():
                self.console.print("usage: session delete <id>")
                return
            sid = int(args[1])
            self.storage.delete_session(sid)
            if sid == self.session_id:
                self.session_id = None
                self._ensure_session()
            self.console.print(f"[green]*[/] deleted session {sid}")
        else:
            self.console.print(f"unknown session subcommand: {sub}")

    # ---- memory ---------------------------------------------------------
    def cmd_memory(self, args) -> None:
        sub = args[0] if args else "list"
        if sub == "list":
            mems = self.storage.list_memories()
            if not mems:
                self.console.print("[dim]no memories saved[/]")
                return
            t = Table(show_header=True, header_style="bold red")
            t.add_column("id")
            t.add_column("fact")
            for m in mems:
                t.add_row(str(m["id"]), m["content"])
            self.console.print(t)
        elif sub == "add":
            fact = " ".join(args[1:])
            if not fact:
                self.console.print("usage: memory add <fact>")
                return
            mid = self.storage.add_memory(fact)
            self.console.print(f"[green]*[/] saved memory {mid}")
        elif sub in ("del", "delete"):
            if len(args) < 2 or not args[1].isdigit():
                self.console.print("usage: memory del <id>")
                return
            self.storage.delete_memory(int(args[1]))
            self.console.print("[green]*[/] deleted")
        else:
            self.console.print(f"unknown memory subcommand: {sub}")

    # ---- agent ----------------------------------------------------------
    def cmd_agent(self, args) -> None:
        if not args:
            self.console.print("usage: agent run <objective> | agent runs | agent show <id>")
            return
        sub = args[0]
        if sub == "run":
            objective = " ".join(args[1:]).strip()
            if not objective:
                self.console.print("usage: agent run <objective>")
                return
            self._agent_run(objective)
        elif sub == "runs":
            self._agent_runs()
        elif sub == "show":
            if len(args) < 2 or not args[1].isdigit():
                self.console.print("usage: agent show <id>")
                return
            self._agent_show(int(args[1]))
        else:
            self.console.print(f"unknown agent subcommand: {sub}")

    def _agent_run(self, objective: str) -> None:
        if self.cfg.missing():
            self.console.print("[red]LLM not configured.[/] Set values in .env, then restart.")
            return
        self._ensure_session()
        mode = "autonomous" if self.cfg.agent_auto_approve_all else "hybrid (will confirm writes)"
        self.console.print(f"[dim]approval mode: {mode}; max steps: {self.cfg.agent_max_steps}[/]")
        runner = AgentRunner(
            self.cfg, self.console, self.llm, self.mcp, self.storage, self._confirm_tool
        )
        try:
            runner.run(objective, self.session_id)
        except Exception as e:  # noqa: BLE001
            self.console.print(f"[red]agent error:[/] {e}")

    def _confirm_tool(self, name: str, args: dict[str, Any], reason: str) -> str:
        """Interactive approval prompt for a state-changing tool call (hybrid mode)."""
        self.console.print(
            f"[yellow]⚠ approval needed[/] [bold]{name}[/] [dim]({reason})[/]\n"
            f"  args: {json.dumps(args)[:300]}"
        )
        while True:
            try:
                ans = self.psession.prompt(
                    [("class:prompt", "  approve? [y]es / [a]lways / [s]kip / a[b]ort > ")],
                    style=PROMPT_STYLE,
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return ABORT
            if ans in ("y", "yes"):
                return ONCE
            if ans in ("a", "always"):
                return ALWAYS
            if ans in ("s", "skip", "n", "no"):
                return SKIP
            if ans in ("b", "abort", "q"):
                return ABORT
            self.console.print("  [dim]please answer y / a / s / b[/]")

    def _agent_runs(self) -> None:
        runs = self.storage.list_runs()
        if not runs:
            self.console.print("[dim]no agent runs yet[/]")
            return
        import time as _t
        t = Table(show_header=True, header_style="bold red")
        for col in ("id", "status", "steps", "objective", "updated"):
            t.add_column(col)
        for r in runs:
            obj = r["objective"]
            t.add_row(
                str(r["id"]),
                r["status"],
                str(r["steps"]),
                (obj[:50] + "…") if len(obj) > 50 else obj,
                _t.strftime("%Y-%m-%d %H:%M", _t.localtime(r["updated_at"])),
            )
        self.console.print(t)

    def _agent_show(self, run_id: int) -> None:
        run = self.storage.get_run(run_id)
        if not run:
            self.console.print(f"[red]no such run: {run_id}[/]")
            return
        self.console.print(
            f"[bold]run {run['id']}[/] — [italic]{run['objective']}[/]\n"
            f"status: {run['status']}  steps: {run['steps']}"
        )
        if run["summary"]:
            self.console.print(f"summary: {run['summary']}")
        steps = self.storage.get_run_steps(run_id)
        glyph = {"thought": "[blue]…[/]", "tool": "[cyan]→[/]", "observation": "[dim]←[/]", "final": "[green]✔[/]"}
        for s in steps:
            mark = glyph.get(s["kind"], " ")
            label = f"[s{s['step']}] {mark} {s['kind']}"
            if s["tool_name"]:
                label += f" {s['tool_name']}"
            if s["approved"] is not None:
                label += " [green](approved)[/]" if s["approved"] else " [red](skipped)[/]"
            detail = (s["detail"] or "").replace("\n", " ")
            self.console.print(f"{label}: [dim]{detail[:200]}{'…' if len(detail) > 200 else ''}[/]")

    # ---- mcp ------------------------------------------------------------
    def _autoconnect_servers(self) -> None:
        for srv in self.storage.list_mcp_servers():
            try:
                self.mcp.connect(srv["name"], srv["transport"], srv["config"])
                self.console.print(f"[green]*[/] connected MCP server '{srv['name']}'")
            except Exception as e:  # noqa: BLE001
                self.console.print(f"[yellow]![/] could not connect '{srv['name']}': {e}")

    def cmd_mcp(self, args) -> None:
        if not args:
            self.console.print("usage: mcp list|add|connect|disconnect|tools|del ...")
            return
        sub = args[0]
        if sub == "list":
            self._mcp_list()
        elif sub == "add":
            self._mcp_add(args[1:])
        elif sub == "connect":
            self._mcp_connect(args[1:])
        elif sub == "disconnect":
            if len(args) < 2:
                self.console.print("usage: mcp disconnect <name>")
                return
            self.mcp.disconnect(args[1])
            self.console.print(f"[green]*[/] disconnected '{args[1]}'")
        elif sub == "tools":
            self._mcp_tools()
        elif sub in ("del", "delete"):
            if len(args) < 2:
                self.console.print("usage: mcp del <name>")
                return
            self.storage.delete_mcp_server(args[1])
            try:
                self.mcp.disconnect(args[1])
            except Exception:  # noqa: BLE001
                pass
            self.console.print(f"[green]*[/] removed '{args[1]}'")
        else:
            self.console.print(f"unknown mcp subcommand: {sub}")

    def _mcp_list(self) -> None:
        servers = self.storage.list_mcp_servers()
        if not servers:
            self.console.print("[dim]no MCP servers configured[/]")
            return
        connected = set(self.mcp.connected())
        t = Table(show_header=True, header_style="bold red")
        for col in ("name", "transport", "status", "config"):
            t.add_column(col)
        for s in servers:
            status = "[green]connected[/]" if s["name"] in connected else "[dim]offline[/]"
            cfg = s["config"]
            summary = cfg.get("url") or " ".join([cfg.get("command", "")] + cfg.get("args", []))
            t.add_row(s["name"], s["transport"], status, summary)
        self.console.print(t)

    def _mcp_add(self, args) -> None:
        if len(args) < 3:
            self.console.print(
                "usage: mcp add stdio <name> <cmd> [args...]  |  mcp add sse <name> <url>"
            )
            return
        transport, name = args[0], args[1]
        if transport == "stdio":
            config = {"command": args[2], "args": list(args[3:])}
        elif transport == "sse":
            config = {"url": args[2]}
        else:
            self.console.print("transport must be 'stdio' or 'sse'")
            return
        self.storage.save_mcp_server(name, transport, config)
        self.console.print(f"[green]*[/] added {transport} server '{name}' (run: mcp connect {name})")

    def _mcp_connect(self, args) -> None:
        if not args:
            self.console.print("usage: mcp connect <name>")
            return
        name = args[0]
        match = [s for s in self.storage.list_mcp_servers() if s["name"] == name]
        if not match:
            self.console.print(f"[red]no such server: {name}[/]")
            return
        srv = match[0]
        try:
            tools = self.mcp.connect(name, srv["transport"], srv["config"])
            self.console.print(f"[green]*[/] connected '{name}' ({len(tools)} tools)")
        except Exception as e:  # noqa: BLE001
            self.console.print(f"[red]connect failed:[/] {e}")

    def _mcp_tools(self) -> None:
        tools = self.mcp.list_tools()
        if not tools:
            self.console.print("[dim]no connected servers[/]")
            return
        for server, tlist in tools.items():
            t = Table(title=server, show_header=True, header_style="bold red")
            t.add_column("tool")
            t.add_column("description")
            for name, desc in tlist:
                t.add_row(name, desc)
            self.console.print(t)

    # ---- config ---------------------------------------------------------
    def cmd_set(self, args) -> None:
        t = Table(show_header=False)
        t.add_column("k", style="bold")
        t.add_column("v")
        t.add_row("base_url", self.cfg.base_url or "[red]unset[/]")
        t.add_row("model", self.cfg.model or "[red]unset[/]")
        t.add_row("auth_token", "set" if self.cfg.auth_token else "[red]unset[/]")
        t.add_row("max_context_tokens", str(self.cfg.max_context_tokens))
        t.add_row("max_tool_output_tokens", str(self.cfg.max_tool_output_tokens))
        t.add_row("parse_text_tool_calls", str(self.cfg.parse_text_tool_calls))
        t.add_row("temperature", str(self.cfg.temperature))
        t.add_row("db_path", str(self.cfg.db_path))
        t.add_row("agent_max_steps", str(self.cfg.agent_max_steps))
        t.add_row("agent_auto_approve_all", str(self.cfg.agent_auto_approve_all))
        t.add_row("agent_allow", ", ".join(self.cfg.agent_allow) or "[dim]none[/]")
        t.add_row("agent_deny", ", ".join(self.cfg.agent_deny) or "[dim]none[/]")
        self.console.print(t)

    # ---- chat -----------------------------------------------------------
    def _system_prompt(self) -> dict[str, Any]:
        mems = self.storage.list_memories()
        base = (
            "You are a helpful assistant running inside a CLI client. "
            "You may call MCP tools that are available, and use save_memory to "
            "remember durable facts about the user.\n\n"
            "When you decide to use a tool, actually invoke it — do not just describe "
            "the command. If for any reason you cannot emit a native tool call, output "
            "exactly one tool call as a single line of the form "
            '<tool_call>{"name": "<tool_name>", "arguments": {<json args>}}</tool_call> '
            "and nothing else, so the client can execute it."
        )
        if mems:
            facts = "\n".join(f"- {m['content']}" for m in mems)
            base += f"\n\nWhat you know about the user:\n{facts}"
        return {"role": "system", "content": base}

    def cmd_chat(self, text: str) -> None:
        if not text:
            return
        if self.cfg.missing():
            self.console.print("[red]LLM not configured.[/] Set values in .env, then restart.")
            return
        self._ensure_session()
        self.storage.add_message(self.session_id, "user", text)

        tools = self.mcp.openai_tools() + [MEMORY_TOOL]

        # Tool-call loop: keep going until the model returns a plain answer.
        for _ in range(10):
            history = [self._system_prompt()] + self.storage.get_messages(self.session_id)
            history = trim_history(
                history,
                self.cfg.max_context_tokens,
                tools=tools,
                max_tool_output_tokens=self.cfg.max_tool_output_tokens,
            )

            try:
                assistant_msg = ui.collect_stream(self.console, self.llm, history, tools)
            except Exception as e:  # noqa: BLE001
                self.console.print(f"[red]LLM error:[/] {e}")
                return

            extra = {k: v for k, v in assistant_msg.items() if k not in ("role", "content")}
            self.storage.add_message(
                self.session_id, "assistant", assistant_msg.get("content"), extra or None
            )

            tool_calls = assistant_msg.get("tool_calls")
            if not tool_calls:
                # Plain answer: this is the only model text the user sees.
                ui.answer(self.console, assistant_msg.get("content"))
                return

            # Intermediate narration is suppressed; only tool activity is shown.
            for tc in tool_calls:
                self._handle_tool_call(tc)

    def _handle_tool_call(self, tc: dict[str, Any]) -> None:
        fn = tc.get("function", {})
        name = fn.get("name", "")
        try:
            arguments = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            arguments = {}

        if name == "save_memory":
            self.storage.add_memory(arguments.get("fact", ""))
            output, state, meta = f"saved: {arguments.get('fact', '')}", "ok", "saved"
        else:
            with ui.running(self.console, name):
                try:
                    output = self.mcp.call_tool(name, arguments)
                except Exception as e:  # noqa: BLE001
                    output = f"ERROR: {e}"
            state, meta = ui.output_meta(output)

        ui.tool_line(self.console, name, state, meta)
        self.storage.add_message(
            self.session_id,
            "tool",
            output,
            {"tool_call_id": tc.get("id", ""), "name": name},
        )

    # ---- shutdown -------------------------------------------------------
    def _shutdown(self) -> None:
        self.console.print("\n[dim]bye.[/]")
        try:
            self.mcp.shutdown()
        except Exception:  # noqa: BLE001
            pass
        self.llm.close()
        self.storage.close()
