"""SQLite persistence: sessions, messages, long-term memory, and MCP server configs."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Optional

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL,          -- system | user | assistant | tool
    content     TEXT,                   -- text content (may be NULL for tool-call-only)
    extra       TEXT,                   -- JSON blob: tool_calls, tool_call_id, name, etc.
    ts          REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS memories (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    content     TEXT NOT NULL,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS mcp_servers (
    name        TEXT PRIMARY KEY,
    transport   TEXT NOT NULL,          -- stdio | sse
    config      TEXT NOT NULL,          -- JSON: command/args/env  OR  url/headers
    created_at  REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
"""


class Storage:
    def __init__(self, db_path: Path):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---- sessions -------------------------------------------------------
    def create_session(self, name: Optional[str] = None) -> int:
        now = time.time()
        name = name or time.strftime("session-%Y%m%d-%H%M%S", time.localtime(now))
        cur = self.conn.execute(
            "INSERT INTO sessions(name, created_at, updated_at) VALUES (?,?,?)",
            (name, now, now),
        )
        self.conn.commit()
        return cur.lastrowid

    def list_sessions(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            """SELECT s.*, COUNT(m.id) AS msg_count
               FROM sessions s LEFT JOIN messages m ON m.session_id = s.id
               GROUP BY s.id ORDER BY s.updated_at DESC"""
        ).fetchall()

    def get_session(self, session_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()

    def rename_session(self, session_id: int, name: str) -> None:
        self.conn.execute(
            "UPDATE sessions SET name = ?, updated_at = ? WHERE id = ?",
            (name, time.time(), session_id),
        )
        self.conn.commit()

    def delete_session(self, session_id: int) -> None:
        self.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self.conn.commit()

    # ---- messages -------------------------------------------------------
    def add_message(
        self,
        session_id: int,
        role: str,
        content: Optional[str],
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO messages(session_id, role, content, extra, ts) VALUES (?,?,?,?,?)",
            (session_id, role, content, json.dumps(extra) if extra else None, time.time()),
        )
        self.conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (time.time(), session_id),
        )
        self.conn.commit()

    def get_messages(self, session_id: int) -> list[dict[str, Any]]:
        """Return messages as OpenAI-style dicts ready to send to the API."""
        rows = self.conn.execute(
            "SELECT role, content, extra FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            msg: dict[str, Any] = {"role": r["role"]}
            if r["content"] is not None:
                msg["content"] = r["content"]
            if r["extra"]:
                msg.update(json.loads(r["extra"]))
            out.append(msg)
        return out

    # ---- memories -------------------------------------------------------
    def add_memory(self, content: str) -> int:
        cur = self.conn.execute(
            "INSERT INTO memories(content, created_at) VALUES (?,?)",
            (content, time.time()),
        )
        self.conn.commit()
        return cur.lastrowid

    def list_memories(self) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM memories ORDER BY id"
        ).fetchall()

    def delete_memory(self, memory_id: int) -> None:
        self.conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
        self.conn.commit()

    # ---- mcp server configs --------------------------------------------
    def save_mcp_server(self, name: str, transport: str, config: dict[str, Any]) -> None:
        self.conn.execute(
            """INSERT INTO mcp_servers(name, transport, config, created_at)
               VALUES (?,?,?,?)
               ON CONFLICT(name) DO UPDATE SET transport=excluded.transport,
                                               config=excluded.config""",
            (name, transport, json.dumps(config), time.time()),
        )
        self.conn.commit()

    def list_mcp_servers(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT name, transport, config FROM mcp_servers ORDER BY name"
        ).fetchall()
        return [
            {"name": r["name"], "transport": r["transport"], "config": json.loads(r["config"])}
            for r in rows
        ]

    def delete_mcp_server(self, name: str) -> None:
        self.conn.execute("DELETE FROM mcp_servers WHERE name = ?", (name,))
        self.conn.commit()
