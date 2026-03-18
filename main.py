"""
FastAPI backend for chatbotDemoDEV3.html

POST /chat
  Request  { type, message, lang, env, channels, chatId? }
  Response { chatId, responseMsg }

The backend auto-detects the actual query language from the message charset
(Chinese vs English) regardless of the UI lang preference sent in the request.

Conversation history is persisted in SQLite. If no chatId is provided the
request is treated as a new conversation and a fresh UUID is issued.
"""

from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncAzureOpenAI
from pydantic import BaseModel

load_dotenv(Path(__file__).parent / ".env")

from chat_logger import write_log                       # noqa: E402
from database import init_db, load_history, save_turn  # noqa: E402
from knowledge_base import KnowledgeBase               # noqa: E402
from rag_engine import RAGEngine                       # noqa: E402

# ---------------------------------------------------------------------------
# Azure OpenAI client
# ---------------------------------------------------------------------------
azure_client = AsyncAzureOpenAI(
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_key=os.environ["AZURE_API_KEY"],
    api_version=os.environ.get("AZURE_API_VERSION", "2024-12-01-preview"),
)

# ---------------------------------------------------------------------------
# App-level singletons
# ---------------------------------------------------------------------------
kb: KnowledgeBase | None = None
rag: RAGEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kb, rag
    print("=== Initialising RAG chatbot (chatbotDEV3) ===")
    await init_db()
    kb = KnowledgeBase()
    rag = RAGEngine(kb, azure_client)
    print("=== Ready ===")
    yield
    print("Shutting down.")


app = FastAPI(title="ChatbotDEV3 RAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    type: str = "cms"
    message: str
    lang: str = "en"           # UI language preference (en / zh_hant)
    env: str = "test"
    channels: dict = {}
    chatId: str | None = None  # absent → new conversation


class SourceItem(BaseModel):
    q: str
    answer: str
    source: str = ""


class ChatResponse(BaseModel):
    chatId: str
    responseMsg: str
    language: str              # detected language used for retrieval
    sources: list[SourceItem] = []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/chat/chatbot/HousingAuthority/b_0ETZJljBxa/config")
async def chatbot_config():
    return {
        "success": True,
        "config": {
            "org": "HousingAuthority",
            "name": "b_0ETZJljBxa",
            "streamMode": False,
            "detectLang": True,
            "historyLimit": 5,
            "settings": {
                "welcomeMsg": {},
                "allowedDirectChat": True,
                "allowedGroupChat": True,
                "allowedLangs": ["en", "zh_hant"],
            },
        },
    }


@app.get("/health")
async def health():
    return {"status": "ok", "ready": rag is not None}


@app.post("/chat/chatbot/HousingAuthority/b_0ETZJljBxa/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if rag is None:
        raise HTTPException(status_code=503, detail="Service not ready yet.")

    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # No chatId → new conversation; existing chatId → load its history
    if request.chatId:
        chat_id = request.chatId
        history = await load_history(chat_id)
    else:
        chat_id = str(uuid.uuid4())
        history = []

    # RAG query with conversation history for multi-turn context
    result = await rag.query(user_message, history=history)

    # Persist this turn to the database
    await save_turn(chat_id, user_message, result["answer"])

    # Write per-request log file
    write_log(
        chat_id=chat_id,
        user_query=user_message,
        expanded_query=result.get("expanded_query", user_message),
        language=result["language"],
        answer=result["answer"],
        sources=result["sources"],
    )

    return ChatResponse(
        chatId=chat_id,
        responseMsg=result["answer"],
        language=result["language"],
        sources=[SourceItem(**s) for s in result["sources"]],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8014")),
        reload=False,
    )
