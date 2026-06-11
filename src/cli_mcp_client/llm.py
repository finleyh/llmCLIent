"""OpenAI-compatible LLM client with streaming + tool-call support.

Short-term memory note
----------------------
The model itself is stateless; "memory" within a conversation is just the full
message list we re-send on every call. ``trim_history`` keeps that list under the
configured context budget by dropping the oldest non-system messages first.
"""

from __future__ import annotations

import json
from typing import Any, Iterator, Optional

import httpx

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - tiktoken optional at runtime
    _ENC = None


def count_tokens(text: str) -> int:
    if _ENC is not None:
        return len(_ENC.encode(text))
    # Rough fallback: ~4 chars per token.
    return max(1, len(text) // 4)


def _message_tokens(msg: dict[str, Any]) -> int:
    total = 4  # per-message overhead
    for v in msg.values():
        if isinstance(v, str):
            total += count_tokens(v)
        else:
            total += count_tokens(json.dumps(v))
    return total


def count_tools_tokens(tools: Optional[list[dict[str, Any]]]) -> int:
    """Approximate tokens consumed by the `tools` array in a request.

    Tool schemas are a separate payload from `messages` but share the model's
    context window, so they must be subtracted from the history budget. With a
    server like HexStrike (150+ tools) this can be tens of thousands of tokens.
    """
    if not tools:
        return 0
    return count_tokens(json.dumps(tools)) + 8 * len(tools)


def _truncate_tool_output(content: Optional[str], max_tokens: int) -> Optional[str]:
    if content is None or max_tokens <= 0:
        return content
    if count_tokens(content) <= max_tokens:
        return content
    # cl100k averages ~4 chars/token; trim with margin and mark it.
    keep = max_tokens * 4
    dropped = len(content) - keep
    return content[:keep] + f"\n…[truncated {dropped} chars of tool output]"


def _build_blocks(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group messages into atomic blocks so tool-call pairs are never split.

    An assistant message carrying `tool_calls` is bundled with the `tool`
    messages that respond to its call ids; everything else is its own block.
    """
    blocks: list[list[dict[str, Any]]] = []
    i, n = 0, len(messages)
    while i < n:
        m = messages[i]
        if m.get("role") == "assistant" and m.get("tool_calls"):
            ids = {tc.get("id") for tc in m["tool_calls"]}
            block = [m]
            j = i + 1
            while (
                j < n
                and messages[j].get("role") == "tool"
                and messages[j].get("tool_call_id") in ids
            ):
                block.append(messages[j])
                j += 1
            blocks.append(block)
            i = j
        else:
            blocks.append([m])
            i += 1
    return blocks


def trim_history(
    messages: list[dict[str, Any]],
    max_tokens: int,
    tools: Optional[list[dict[str, Any]]] = None,
    reserve: int = 1500,
    max_tool_output_tokens: int = 0,
) -> list[dict[str, Any]]:
    """Fit history into the context window without breaking tool-call pairing.

    - Subtracts the `tools` schema size and a completion `reserve` from the budget.
    - Optionally truncates oversized `tool` outputs (0 = disabled).
    - Trims oldest-first by *block*, so an assistant tool_calls message and its
      tool results are always kept or dropped together. System messages are always
      retained.
    """
    budget = max_tokens - reserve - count_tools_tokens(tools)

    system = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]

    # Work on copies so we never mutate caller/stored dicts when truncating.
    if max_tool_output_tokens > 0:
        trimmed_rest = []
        for m in rest:
            if m.get("role") == "tool":
                m = dict(m)
                m["content"] = _truncate_tool_output(m.get("content"), max_tool_output_tokens)
            trimmed_rest.append(m)
        rest = trimmed_rest

    blocks = _build_blocks(rest)

    used = sum(_message_tokens(m) for m in system)
    kept_rev: list[list[dict[str, Any]]] = []
    for block in reversed(blocks):
        bt = sum(_message_tokens(m) for m in block)
        if used + bt > budget and kept_rev:
            break
        used += bt
        kept_rev.append(block)

    kept = [m for block in reversed(kept_rev) for m in block]
    # Safety net: never lead with an orphan tool result whose call was dropped.
    while kept and kept[0].get("role") == "tool":
        kept.pop(0)
    return system + kept


class LLMClient:
    def __init__(self, base_url: str, model: str, auth_token: str, temperature: float = 0.7):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {auth_token}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(120.0, connect=10.0),
        )

    def close(self) -> None:
        self._client.close()

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        tools: Optional[list[dict[str, Any]]] = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield events while streaming a completion.

        Events:
            {"type": "text",  "data": <delta str>}
            {"type": "done",  "message": <assembled assistant message dict>}
        The assembled message includes ``tool_calls`` when the model requests tools.
        """
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        content_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}

        with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            if resp.status_code >= 400:
                body = resp.read().decode("utf-8", "replace")
                raise RuntimeError(f"LLM API error {resp.status_code}: {body}")
            for line in resp.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})

                if delta.get("content"):
                    content_parts.append(delta["content"])
                    yield {"type": "text", "data": delta["content"]}

                for tc in delta.get("tool_calls", []) or []:
                    idx = tc.get("index", 0)
                    slot = tool_calls.setdefault(
                        idx,
                        {"id": "", "type": "function",
                         "function": {"name": "", "arguments": ""}},
                    )
                    if tc.get("id"):
                        slot["id"] = tc["id"]
                    fn = tc.get("function", {})
                    if fn.get("name"):
                        slot["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        slot["function"]["arguments"] += fn["arguments"]

        message: dict[str, Any] = {"role": "assistant"}
        message["content"] = "".join(content_parts) or None
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        yield {"type": "done", "message": message}
