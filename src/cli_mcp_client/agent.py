"""Single-loop ReAct agent runner with hybrid (human-in-the-loop) approval.

The chat path (`Repl.cmd_chat`) already loops over tool calls, but it terminates
implicitly (when the model stops calling tools) and auto-executes everything. An
*agent* run differs in three ways:

  * it is driven by an explicit **objective** and ends via a deliberate
    ``task_complete`` tool call (clean termination), with a configurable hard cap
    on steps as a safety net;
  * every tool call passes through an **approval policy** — read-only tools run
    automatically, state-changing ones are confirmed by the user (hybrid mode);
  * each step (thought / tool / observation / final) is **persisted** to the
    ``runs`` + ``run_steps`` tables so a run can be inspected after the fact.

`llm.py` and `mcp_manager.py` are reused unchanged.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from rich.console import Console

from .config import Config
from .llm import LLMClient, trim_history
from .mcp_manager import MCPManager
from .storage import Storage

# Built-in tool: persist a durable fact across sessions (shared with chat mode).
MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "save_memory",
        "description": (
            "Save a durable fact about the user or their preferences so it can be "
            "recalled in future sessions. Use for stable info (name, role, "
            "preferences, ongoing projects), not transient conversation details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {"type": "string", "description": "The fact to remember, one sentence."}
            },
            "required": ["fact"],
        },
    },
}

# Built-in tool: the agent calls this to end the run with a final summary.
TASK_COMPLETE_TOOL = {
    "type": "function",
    "function": {
        "name": "task_complete",
        "description": (
            "Call this when the objective is fully accomplished (or cannot be "
            "accomplished) to end the run. Provide a concise summary of what was "
            "done and the outcome. Do not call any other tool in the same step."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {"type": "string", "description": "What was accomplished and the result."},
                "success": {"type": "boolean", "description": "Whether the objective was met."},
            },
            "required": ["summary"],
        },
    },
}

# Approval decisions returned by the confirm callback.
ONCE, ALWAYS, SKIP, ABORT = "once", "always", "skip", "abort"

# How many identical (name+args) tool calls before we intervene / abort.
_STALL_WARN = 3
_STALL_ABORT = 5


class AgentAbort(Exception):
    """Raised internally to unwind the run loop on user abort."""


def _agent_system_prompt(memories: list[str], objective: str) -> dict[str, Any]:
    base = (
        "You are an autonomous agent operating inside a CLI client. You pursue a "
        "single OBJECTIVE by reasoning and acting in a loop:\n"
        "  1. Think briefly about the next concrete step.\n"
        "  2. Call exactly the tool(s) needed for that step.\n"
        "  3. Read the tool results (observations) and decide the next step.\n"
        "Repeat until the objective is met, then call `task_complete` with a "
        "summary. Always finish by calling `task_complete` — do not just describe "
        "that you are done.\n\n"
        "Guidelines: prefer one tool call per step so you can react to each result. "
        "Do not invent tool outputs — only use what tools actually return. If you "
        "are blocked or the objective is impossible, call `task_complete` with "
        "success=false and explain why. Be efficient; avoid repeating identical "
        "calls.\n\n"
        "If you cannot emit a native tool call, output exactly one tool call as a "
        'single line: <tool_call>{"name": "<tool>", "arguments": {<json>}}</tool_call>\n\n'
        f"OBJECTIVE: {objective}"
    )
    if memories:
        base += "\n\nWhat you know about the user:\n" + "\n".join(f"- {m}" for m in memories)
    return {"role": "system", "content": base}


class AgentRunner:
    """Drives one ReAct run end-to-end."""

    def __init__(
        self,
        cfg: Config,
        console: Console,
        llm: LLMClient,
        mcp: MCPManager,
        storage: Storage,
        confirm: Callable[[str, dict[str, Any], str], str],
    ):
        self.cfg = cfg
        self.console = console
        self.llm = llm
        self.mcp = mcp
        self.storage = storage
        # confirm(name, args, reason) -> ONCE | ALWAYS | SKIP | ABORT
        self.confirm = confirm
        # Tools auto-approved for the duration of this run (user chose "always").
        self._session_allow: set[str] = set()

    # ---- approval policy ------------------------------------------------
    def _classify(self, name: str) -> tuple[str, str]:
        """Return (decision, reason): 'auto' | 'prompt' | 'deny'."""
        bare = name.split("__", 1)[-1] if "__" in name else name
        if name in self.cfg.agent_deny:
            return "deny", "on deny list"
        if name in ("task_complete", "save_memory"):
            return "auto", "built-in safe tool"
        if self.cfg.agent_auto_approve_all:
            return "auto", "auto-approve-all enabled"
        if name in self._session_allow or name in self.cfg.agent_allow:
            return "auto", "allow-listed"
        low = bare.lower()
        if any(low.startswith(p) for p in self.cfg.agent_readonly_prefixes):
            return "auto", "read-only tool"
        return "prompt", "state-changing tool"

    def _approve(self, name: str, args: dict[str, Any]) -> tuple[bool, str]:
        """Resolve whether a tool call may run. Returns (allowed, note)."""
        decision, reason = self._classify(name)
        if decision == "deny":
            return False, f"denied ({reason})"
        if decision == "auto":
            return True, f"auto-approved ({reason})"
        # decision == "prompt": ask the user.
        choice = self.confirm(name, args, reason)
        if choice == ALWAYS:
            self._session_allow.add(name)
            return True, "approved (always, this run)"
        if choice == ONCE:
            return True, "approved"
        if choice == ABORT:
            raise AgentAbort()
        return False, "skipped by user"

    # ---- main loop ------------------------------------------------------
    def run(self, objective: str, session_id: int) -> int:
        """Execute the run; returns the run id. Persists everything to storage."""
        run_id = self.storage.create_run(objective, session_id)
        # Record the objective in the conversation so history/trimming includes it.
        self.storage.add_message(session_id, "user", f"[AGENT OBJECTIVE] {objective}")

        tools = self.mcp.openai_tools() + [MEMORY_TOOL, TASK_COMPLETE_TOOL]
        memories = [m["content"] for m in self.storage.list_memories()]

        sig_counts: dict[str, int] = {}
        status, summary, step = "max_steps", None, 0

        self.console.print(
            f"[bold red]▶ agent run {run_id}[/] — objective: [italic]{objective}[/]"
        )

        try:
            for step in range(1, self.cfg.agent_max_steps + 1):
                self.console.rule(f"[dim]step {step}/{self.cfg.agent_max_steps}[/]", style="red")
                assistant_msg = self._llm_step(session_id, memories, objective, tools)

                content = assistant_msg.get("content")
                if content:
                    self.storage.add_run_step(run_id, step, "thought", detail=content)

                tool_calls = assistant_msg.get("tool_calls")
                if not tool_calls:
                    # Plain answer with no tool call — treat as implicit completion.
                    status, summary = "done", content or "(no summary)"
                    self.storage.add_run_step(run_id, step, "final", detail=summary)
                    break

                done = False
                for tc in tool_calls:
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = {}

                    if name == "task_complete":
                        status = "done" if args.get("success", True) else "failed"
                        summary = args.get("summary", "(no summary)")
                        self.storage.add_run_step(run_id, step, "final", detail=summary)
                        self.console.print(f"[bold green]✔ task_complete:[/] {summary}")
                        done = True
                        break

                    # Stall guard: identical name+args repeated too often.
                    sig = name + "|" + json.dumps(args, sort_keys=True)
                    sig_counts[sig] = sig_counts.get(sig, 0) + 1
                    if sig_counts[sig] >= _STALL_ABORT:
                        status, summary = "aborted", "aborted: repeated the same call too many times"
                        self.storage.add_run_step(run_id, step, "observation", detail=summary, tool_name=name)
                        self.console.print(f"[red]✖ {summary}[/]")
                        done = True
                        break

                    self._run_tool(run_id, step, session_id, tc, name, args)
                    if sig_counts[sig] == _STALL_WARN:
                        nudge = (
                            "You have repeated this exact call several times. Try a "
                            "different approach or call task_complete."
                        )
                        self.storage.add_message(session_id, "user", f"[SYSTEM] {nudge}")

                if done:
                    break
            else:
                # Loop fell through without break → hit the step cap.
                summary = f"stopped after reaching the step cap ({self.cfg.agent_max_steps})"
                self.console.print(f"[yellow]⚠ {summary}[/]")
        except AgentAbort:
            status, summary, = "aborted", "aborted by user"
            self.console.print("[yellow]⚠ run aborted by user[/]")
        except KeyboardInterrupt:
            status, summary = "aborted", "interrupted (Ctrl-C)"
            self.console.print("\n[yellow]⚠ interrupted[/]")

        self.storage.update_run(run_id, status=status, summary=summary, steps=step)
        self.console.print(f"[bold]run {run_id} finished:[/] {status} — {summary}")
        return run_id

    def _llm_step(self, session_id, memories, objective, tools) -> dict[str, Any]:
        history = [_agent_system_prompt(memories, objective)] + self.storage.get_messages(session_id)
        history = trim_history(
            history,
            self.cfg.max_context_tokens,
            tools=tools,
            max_tool_output_tokens=self.cfg.max_tool_output_tokens,
        )
        assistant_msg: dict[str, Any] = {}
        self.console.print("[bold green]agent[/] ", end="")
        for event in self.llm.stream_chat(history, tools=tools):
            if event["type"] == "text":
                self.console.print(event["data"], end="")
            elif event["type"] == "recovered_tool_calls":
                self.console.print(f"\n[yellow]·[/] recovered {event['count']} tool call(s) from text")
            elif event["type"] == "done":
                assistant_msg = event["message"]
        self.console.print()
        extra = {k: v for k, v in assistant_msg.items() if k not in ("role", "content")}
        self.storage.add_message(session_id, "assistant", assistant_msg.get("content"), extra or None)
        return assistant_msg

    def _run_tool(self, run_id, step, session_id, tc, name, args) -> None:
        allowed, note = self._approve(name, args)
        self.storage.add_run_step(
            run_id, step, "tool",
            detail=json.dumps(args), tool_name=name,
            approved=allowed,
        )
        self.console.print(f"[cyan]→ tool[/] {name} {args} [dim]({note})[/]")

        if not allowed:
            output = f"SKIPPED: tool '{name}' was not approved and did not run."
        elif name == "save_memory":
            self.storage.add_memory(args.get("fact", ""))
            output = f"saved: {args.get('fact', '')}"
        else:
            try:
                output = self.mcp.call_tool(name, args)
            except Exception as e:  # noqa: BLE001
                output = f"ERROR: {e}"

        self.console.print(f"[dim]{output[:500]}{'…' if len(output) > 500 else ''}[/]")
        self.storage.add_run_step(run_id, step, "observation", detail=output, tool_name=name)
        self.storage.add_message(
            session_id, "tool", output, {"tool_call_id": tc.get("id", ""), "name": name}
        )
