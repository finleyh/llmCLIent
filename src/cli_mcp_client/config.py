"""Configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()


def _csv(value: str) -> list[str]:
    """Parse a comma-separated env value into a clean list."""
    return [item.strip() for item in value.split(",") if item.strip()]


# Verb prefixes that mark a tool as read-only, so it can be auto-approved in
# hybrid mode. Matched against the *bare* tool name (after the 'server__' prefix
# is stripped). Conservative on purpose — anything not matching needs confirmation.
DEFAULT_READONLY_PREFIXES = [
    "read", "get", "list", "search", "find", "fetch", "view", "show",
    "describe", "query", "lookup", "inspect", "scan", "check", "count",
]


@dataclass
class Config:
    base_url: str
    model: str
    auth_token: str
    max_context_tokens: int
    max_tool_output_tokens: int
    temperature: float
    parse_text_tool_calls: bool
    db_path: Path
    # ---- agent settings ----
    agent_max_steps: int
    agent_auto_approve_all: bool
    agent_readonly_prefixes: list[str]
    agent_allow: list[str]
    agent_deny: list[str]

    @classmethod
    def load(cls) -> "Config":
        base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
        model = os.getenv("LLM_MODEL", "")
        auth_token = os.getenv("LLM_AUTH_TOKEN", "")
        max_ctx = int(os.getenv("LLM_MAX_CONTEXT_TOKENS", "128000"))
        max_tool_out = int(os.getenv("LLM_MAX_TOOL_OUTPUT_TOKENS", "4000"))
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        parse_text_tc = os.getenv("LLM_PARSE_TEXT_TOOLCALLS", "1").lower() not in (
            "0", "false", "no", "off",
        )
        db_path = _expand(os.getenv("MCPC_DB_PATH", "~/.mcpc/mcpc.db"))
        db_path.parent.mkdir(parents=True, exist_ok=True)

        agent_max_steps = int(os.getenv("AGENT_MAX_STEPS", "25"))
        agent_auto_approve_all = os.getenv("AGENT_AUTO_APPROVE_ALL", "0").lower() in (
            "1", "true", "yes", "on",
        )
        readonly = os.getenv("AGENT_READONLY_PREFIXES", "")
        agent_readonly_prefixes = _csv(readonly) if readonly else list(DEFAULT_READONLY_PREFIXES)
        agent_allow = _csv(os.getenv("AGENT_ALLOW", ""))
        agent_deny = _csv(os.getenv("AGENT_DENY", ""))

        return cls(
            base_url=base_url,
            model=model,
            auth_token=auth_token,
            max_context_tokens=max_ctx,
            max_tool_output_tokens=max_tool_out,
            temperature=temperature,
            parse_text_tool_calls=parse_text_tc,
            db_path=db_path,
            agent_max_steps=agent_max_steps,
            agent_auto_approve_all=agent_auto_approve_all,
            agent_readonly_prefixes=agent_readonly_prefixes,
            agent_allow=agent_allow,
            agent_deny=agent_deny,
        )

    def missing(self) -> list[str]:
        """Return names of required settings that are not set."""
        out = []
        if not self.base_url:
            out.append("LLM_BASE_URL")
        if not self.model:
            out.append("LLM_MODEL")
        if not self.auth_token:
            out.append("LLM_AUTH_TOKEN")
        return out
