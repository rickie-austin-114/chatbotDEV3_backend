"""
Per-request file logger.

Creates one .log file per user message under LOG_DIR:
  {LOG_DIR}/{YYYY-MM-DD_HH-MM-SS}-{sanitised_query}.log

Each file contains the full turn details: chat_id, detected language,
query, expanded query (if different), answer, and top sources.
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))


def _sanitise(query: str, max_len: int = 50) -> str:
    """Strip filesystem-unsafe characters and truncate for use in a filename."""
    sanitised = re.sub(r'[\\/*?:"<>|\n\r\t]', "_", query)
    sanitised = sanitised.strip().strip(".")
    return sanitised[:max_len] if sanitised else "query"


def write_log(
    chat_id: str,
    user_query: str,
    expanded_query: str,
    language: str,
    answer: str,
    sources: list[dict],
) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}-{_sanitise(user_query)}.log"
    log_path = LOG_DIR / filename

    lines = [
        f"Timestamp : {datetime.now().isoformat()}",
        f"Chat ID   : {chat_id}",
        f"Language  : {language}",
        f"Query     : {user_query}",
    ]

    if expanded_query != user_query:
        lines.append(f"Expanded  : {expanded_query}")

    lines += [
        "",
        "=== Answer ===",
        answer,
        "",
        "=== Sources ===",
    ]

    for i, src in enumerate(sources, 1):
        lines.append(f"[{i}] {src.get('source', '')}  Q: {src.get('q', '')}")

    log_path.write_text("\n".join(lines), encoding="utf-8")
