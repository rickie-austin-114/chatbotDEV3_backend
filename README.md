# ChatbotDEV3 RAG Backend

FastAPI backend for `chatbotDemoDEV3.html` using BAAI/bge-m3 embeddings and bge-reranker-v2-m3.

## Setup

```bash
cd chatbotDEV3_backend
pip install -r requirements.txt
```

## Knowledge Base

Set `KB_PATH` in `.env` to point at your knowledge base. Accepts:
- **A directory** containing one or more `.xlsx` files
- **A single `.xlsx` file** (path with or without the `.xlsx` extension)

Default: `../FAQ_combined` (i.e., a `FAQ_combined/` folder next to this backend directory)

Each Excel worksheet must have this column layout (row 1 = header):

| Col | Field       |
|-----|-------------|
| 0   | No.         |
| 1   | Source      |
| 2   | lang        | ← `"en"` or `"zh_hant"`
| 3   | definition  |
| 4   | question    |
| 5   | answer      |
| 6   | category    |
| 7   | active      | ← `True`/`False`; `False` rows are skipped
| 8   | status      |

## Key .env Variables

| Variable      | Default | Description |
|---------------|---------|-------------|
| `KB_PATH`     | `../FAQ_combined` | Path to knowledge base directory or file |
| `EMBED_TOP_K` | `15`    | Candidates retrieved by bge-m3 (FAISS) before reranking |
| `RERANK_TOP_K`| `5`     | Top docs kept after bge-reranker-v2-m3; passed as LLM context |

## Language Detection

Language is detected from the **message charset**, not the UI `lang` field:
- CJK character ratio > 10 % → `zh_hant` → searches Chinese knowledge base
- Otherwise → `en` → searches English knowledge base

## Running

```bash
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000
```

The first run encodes all documents and saves FAISS indexes to `cache/`.
Subsequent runs load from cache instantly.

## Cache Invalidation

Delete the `cache/` directory to force a rebuild after updating the knowledge base:

```bash
rm -rf cache/
```

## API

**POST /chat**

Request:
```json
{
  "type": "cms",
  "message": "What is public housing?",
  "lang": "en",
  "env": "test",
  "channels": {},
  "chatId": null
}
```

Response:
```json
{
  "chatId": "uuid-...",
  "responseMsg": "Public housing in Hong Kong...",
  "language": "en",
  "sources": [...]
}
```
