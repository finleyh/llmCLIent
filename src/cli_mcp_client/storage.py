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

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER REFERENCES sessions(id) ON DELETE SET NULL,
    objective   TEXT NOT NULL,
    status      TEXT NOT NULL,          -- running | done | failed | aborted | max_steps
    summary     TEXT,                   -- final summary from task_complete (if any)
    steps       INTEGER NOT NULL DEFAULT 0,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS run_steps (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    step        INTEGER NOT NULL,       -- 1-based iteration number
    kind        TEXT NOT NULL,          -- thought | tool | observation | final
    tool_name   TEXT,                   -- set for kind=tool/observation
    detail      TEXT,                   -- thought text / args JSON / observation text
    approved    INTEGER,                -- 1/0/NULL: was the tool call approved?
    ts          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id);
CREATE INDEX IF NOT EXISTS idx_runsteps_run ON run_steps(run_id, id);
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

    # ---- agent runs -----------------------------------------------------
    def create_run(self, objective: str, session_id: Optional[int] = None) -> int:
        now = time.time()
        cur = self.conn.execute(
            """INSERT INTO runs(session_id, objective, status, steps, created_at, updated_at)
               VALUES (?,?,?,?,?,?)""",
            (session_id, objective, "running", 0, now, now),
        )
        self.conn.commit()
        return cur.lastrowid

    def update_run(
        self,
        run_id: int,
        status: Optional[str] = None,
        summary: Optional[str] = None,
        steps: Optional[int] = None,
    ) -> None:
        sets, vals = ["updated_at = ?"], [time.time()]
        if status is not None:
            sets.append("status = ?"); vals.append(status)
        if summary is not None:
            sets.append("summary = ?"); vals.append(summary)
        if steps is not None:
            sets.append("steps = ?"); vals.append(steps)
        vals.append(run_id)
        self.conn.execute(f"UPDATE runs SET {', '.join(sets)} WHERE id = ?", vals)
        self.conn.commit()

    def add_run_step(
        self,
        run_id: int,
        step: int,
        kind: str,
        detail: Optional[str] = None,
        tool_name: Optional[str] = None,
        approved: Optional[bool] = None,
    ) -> None:
        self.conn.execute(
            """INSERT INTO run_steps(run_id, step, kind, tool_name, detail, approved, ts)
               VALUES (?,?,?,?,?,?,?)""",
            (
                run_id, step, kind, tool_name, detail,
                None if approved is None else int(approved), time.time(),
            ),
        )
        self.conn.commit()

    def list_runs(self, limit: int = 50) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()

    def get_run(self, run_id: int) -> Optional[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()

    def get_run_steps(self, run_id: int) -> list[sqlite3.Row]:
        return self.conn.execute(
            "SELECT * FROM run_steps WHERE run_id = ? ORDER BY id", (run_id,)
        ).fetchall()

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
