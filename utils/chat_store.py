"""SQLite chat history (local file, optional). Failures never break the app."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "data" / "chat_history.db"


def db_path() -> Path:
    """Resolved on every call so tests can point FINWISE_DB at a temp file."""
    return Path(os.getenv("FINWISE_DB", str(DEFAULT_DB)))


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS messages ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL, role TEXT NOT NULL, "
        "text TEXT NOT NULL, source TEXT, citations TEXT)"
    )
    return conn


def add_message(role: str, text: str, source: str | None = None, citations: list | None = None) -> None:
    try:
        with closing(_connect()) as conn, conn:
            conn.execute(
                "INSERT INTO messages (ts, role, text, source, citations) VALUES (?, ?, ?, ?, ?)",
                (datetime.now().isoformat(timespec="seconds"), role, text, source,
                 json.dumps(citations) if citations else None),
            )
    except (sqlite3.Error, OSError):
        pass


def load_messages(limit: int = 100) -> list[dict]:
    try:
        with closing(_connect()) as conn:
            rows = conn.execute(
                "SELECT role, text, source, citations FROM messages ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
    except (sqlite3.Error, OSError):
        return []
    out = []
    for role, text, source, citations in reversed(rows):
        msg = {"role": role, "text": text}
        if source:
            msg["source"] = source
        if citations:
            try:
                msg["citations"] = json.loads(citations)
            except ValueError:
                pass
        out.append(msg)
    return out


def clear() -> None:
    try:
        with closing(_connect()) as conn, conn:
            conn.execute("DELETE FROM messages")
    except (sqlite3.Error, OSError):
        pass
