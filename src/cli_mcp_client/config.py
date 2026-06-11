"""Configuration loaded from environment / .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p)).resolve()


@dataclass
class Config:
    base_url: str
    model: str
    auth_token: str
    max_context_tokens: int
    max_tool_output_tokens: int
    temperature: float
    db_path: Path

    @classmethod
    def load(cls) -> "Config":
        base_url = os.getenv("LLM_BASE_URL", "").rstrip("/")
        model = os.getenv("LLM_MODEL", "")
        auth_token = os.getenv("LLM_AUTH_TOKEN", "")
        max_ctx = int(os.getenv("LLM_MAX_CONTEXT_TOKENS", "128000"))
        max_tool_out = int(os.getenv("LLM_MAX_TOOL_OUTPUT_TOKENS", "4000"))
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
        db_path = _expand(os.getenv("MCPC_DB_PATH", "~/.mcpc/mcpc.db"))
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return cls(
            base_url=base_url,
            model=model,
            auth_token=auth_token,
            max_context_tokens=max_ctx,
            max_tool_output_tokens=max_tool_out,
            temperature=temperature,
            db_path=db_path,
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
