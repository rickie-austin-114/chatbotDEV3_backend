"""
Lexicon loader — reads domain terminology from a JSON file and provides:
  1. expand_query(query, lang) — Option 3: appends term definitions to the
     query before embedding so retrieval finds the right documents even when
     the user uses abbreviations or colloquial terms.
  2. glossary_text(lang)      — Option 2: returns a formatted glossary block
     to inject into the LLM system prompt so the model understands the terms.

Expected JSON format:
  [
    {"term": "...", "zh_def": "...", "en_def": "..."},
    ...
  ]
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def _resolve_lexicon_json_path() -> Path:
    raw = os.getenv("LEXICON_JSON_PATH", "../FAQ_combined/lexicon/custom_lexicon.json")
    p = Path(raw)
    if not p.is_absolute():
        p = (Path(__file__).parent / p).resolve()
    return p


class Lexicon:
    def __init__(self):
        self._terms: list[dict] = []  # [{term, zh_def, en_def}]
        self._load()

    def _load(self) -> None:
        path = _resolve_lexicon_json_path()
        if not path.exists():
            print(f"[Lexicon] WARNING: JSON not found at {path}. Lexicon disabled.")
            return

        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
        for entry in entries:
            self._terms.append({
                "term":   str(entry.get("term", "")).strip(),
                "zh_def": str(entry.get("zh_def", "")).strip(),
                "en_def": str(entry.get("en_def", "")).strip(),
            })
        print(f"[Lexicon] Loaded {len(self._terms)} terms from {path.name}.")

    # ------------------------------------------------------------------
    # Option 3 — Query expansion
    # ------------------------------------------------------------------

    def expand_query(self, query: str, lang: str) -> str:
        """
        Scan the query for known lexicon terms and append their definitions.
        This enriches the embedding query so FAISS retrieves better candidates
        even when the user uses abbreviations or colloquial expressions.

        Example (zh_hant):
          "公屋輪候冊幾耐？"
          → "公屋輪候冊幾耐？ [公屋: 公共租住房屋，政府資助的廉租屋]
                              [輪候冊: 公屋申請名單]"
        """
        if not self._terms:
            return query

        expansions: list[str] = []
        for entry in self._terms:
            term = entry["term"]
            if term in query:
                definition = entry["zh_def"] if lang == "zh_hant" else entry["en_def"]
                if definition:
                    expansions.append(f"[{term}: {definition}]")

        if not expansions:
            return query

        return query + " " + " ".join(expansions)

    # ------------------------------------------------------------------
    # Option 2 — System prompt glossary
    # ------------------------------------------------------------------

    def glossary_text(self, lang: str) -> str:
        """
        Return a formatted glossary of all terms for injection into the
        LLM system prompt so the model understands domain terminology.
        """
        if not self._terms:
            return ""

        if lang == "zh_hant":
            lines = [f"- {e['term']}：{e['zh_def']}" for e in self._terms if e["zh_def"]]
            header = "【術語表】以下為本領域常用術語，請參考作答：\n"
        else:
            lines = [f"- {e['term']}: {e['en_def']}" for e in self._terms if e["en_def"]]
            header = "Terminology glossary (use these definitions when interpreting the user's query):\n"

        return header + "\n".join(lines)
