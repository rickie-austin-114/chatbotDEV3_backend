"""
Persistent chat history using SQLite (via aiosqlite).

Schema
------
sessions  (chat_id TEXT PK, created_at TEXT)
messages  (id INTEGER PK, chat_id TEXT FK, role TEXT, content TEXT, created_at TEXT)

Usage
-----
  await init_db()                              # call once at startup
  history = await load_history(chat_id)        # returns list[{role, content}]
  await append_messages(chat_id, role, content) # call once per role per turn
"""

from __future__ import annotations

import os
from pathlib import Path

import aiosqlite
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

DB_PATH = Path(os.getenv("DB_PATH", "./data/chat_history.db"))

# Keep only the most recent N messages per session when loading history
# (avoids unbounded token growth for very long conversations)
HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "20"))


async def init_db() -> None:
    """Create tables if they don't exist. Called once at app startup."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                chat_id    TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id    TEXT NOT NULL REFERENCES sessions(chat_id),
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)
        await db.execute("CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id)")
        await db.commit()


async def load_history(chat_id: str) -> list[dict]:
    """
    Return the last HISTORY_LIMIT messages for the given chat_id,
    ordered oldest-first so they can be passed directly to the LLM.
    Returns an empty list if the chat_id does not exist.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            """
            SELECT role, content FROM (
                SELECT role, content, id
                FROM messages
                WHERE chat_id = ?
                ORDER BY id DESC
                LIMIT ?
            ) ORDER BY id ASC
            """,
            (chat_id, HISTORY_LIMIT),
        ) as cursor:
            rows = await cursor.fetchall()
    return [{"role": row["role"], "content": row["content"]} for row in rows]


async def save_turn(chat_id: str, user_content: str, assistant_content: str) -> None:
    """
    Persist a single conversation turn (user + assistant messages).
    Creates the session row on first use.
    """
    async with aiosqlite.connect(DB_PATH) as db:
        # Upsert session
        await db.execute(
            "INSERT OR IGNORE INTO sessions (chat_id) VALUES (?)",
            (chat_id,),
        )
        # Insert both messages in one transaction
        await db.executemany(
            "INSERT INTO messages (chat_id, role, content) VALUES (?, ?, ?)",
            [
                (chat_id, "user", user_content),
                (chat_id, "assistant", assistant_content),
            ],
        )
        await db.commit()
