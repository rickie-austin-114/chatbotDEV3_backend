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

Default (Docker): `/FAQ_combined` (mounted from `../FAQ_combined` on the host)
Default (local): `../FAQ_combined` (relative to the backend directory)

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

### Option 1 — Docker (recommended)

**Prerequisites:** Docker and Docker Compose installed.

1. Place your `.xlsx` knowledge base files in a `FAQ_combined/` folder next to this backend directory:

```
Documents/claude/
├── FAQ_combined/          ← put .xlsx files here
│   └── FAQ_Combined.xlsx
└── chatbotDEV3_backend/
    └── ...
```

2. Build and start the container:

```bash
cd chatbotDEV3_backend
docker compose up --build
```

The first start will download the bge-m3 and bge-reranker-v2-m3 models and encode all documents — this may take several minutes. Subsequent starts load from the `cache/` volume instantly.

To run in the background:

```bash
docker compose up --build -d
```

To stop:

```bash
docker compose down
```

### Option 2 — Local Python

```bash
cd chatbotDEV3_backend
pip install -r requirements.txt
python main.py
```

The server starts on **http://localhost:8014**.

## Cache Invalidation

Delete the `cache/` directory to force a rebuild after updating the knowledge base:

```bash
rm -rf cache/
```

When running via Docker the cache is mounted as a volume, so it persists across container restarts. To force a rebuild with Docker:

```bash
docker compose down
rm -rf cache/
docker compose up --build
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
