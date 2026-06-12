"""Clean, Claude-app-style rendering for chat and agent output.

Only two kinds of thing reach the screen:

  * the model's **final answer** (prose), and
  * a compact **one-line status per tool call** (e.g. ``✓ files__read_file  1.2 KB``).

The model's intermediate narration and the raw tool output are deliberately *not*
printed — they are still stored in SQLite and still sent back to the model, but
showing them turns the REPL into a debug dump. Keeping them off-screen is what
makes this feel like a chat client rather than a log tail.
"""

from __future__ import annotations

import time
from typing import Any, Optional

from rich.console import Console

# state -> (glyph, short fallback label)
_GLYPH = {
    "ok": "[green]✓[/]",
    "error": "[red]✗[/]",
    "skipped": "[yellow]⊘[/]",
    "denied": "[red]⊘[/]",
}


def thinking(console: Console, label: str = "thinking"):
    """Transient spinner shown while the model is generating (no text streamed)."""
    return console.status(f"[dim]{label}…[/]", spinner="dots")


def running(console: Console, name: str):
    """Transient spinner shown while a tool call is executing."""
    return console.status(f"[cyan]{name}[/][dim] …[/]", spinner="dots")


def tool_line(console: Console, name: str, state: str, meta: Optional[str] = None) -> None:
    """Print one compact status line for a finished tool call."""
    glyph = _GLYPH.get(state, "[dim]•[/]")
    extra = f"  [dim]{meta}[/]" if meta else ""
    console.print(f"  {glyph} [cyan]{name}[/]{extra}")


def answer(console: Console, text: Optional[str]) -> None:
    """Print the model's final answer, if any."""
    if text and text.strip():
        console.print()
        console.print(text.strip())


def output_meta(output: str) -> tuple[str, str]:
    """Summarize a tool's output into (state, short_meta) WITHOUT showing it."""
    if output.startswith("ERROR:"):
        return "error", output.split("\n", 1)[0][7:].strip()[:60] or "error"
    if not output or output == "(no output)":
        return "ok", "no output"
    n = len(output)
    size = f"{n} B" if n < 1024 else f"{n / 1024:.1f} KB"
    return "ok", size


def collect_stream(
    console: Console,
    llm: Any,
    messages: list[dict[str, Any]],
    tools: Optional[list[dict[str, Any]]],
    label: str = "thinking",
) -> dict[str, Any]:
    """Run one streamed completion under a live spinner, buffering (not printing) text.

    The spinner label is updated as deltas arrive — showing elapsed time and how
    much has been generated — so the user can see the model is actively producing
    tokens (not hung or timing out), even though the text itself stays off-screen.
    Returns the assembled assistant message; the caller decides what to surface.
    """
    assistant_msg: dict[str, Any] = {"role": "assistant", "content": None}
    start = time.monotonic()
    chars = 0
    with thinking(console, label) as status:
        for event in llm.stream_chat(messages, tools=tools):
            etype = event["type"]
            if etype == "text":
                chars += len(event.get("data") or "")
                elapsed = time.monotonic() - start
                status.update(f"[dim]{label}… {elapsed:.0f}s · {chars} chars[/]")
            elif etype == "done":
                assistant_msg = event["message"]
    return assistant_msg
