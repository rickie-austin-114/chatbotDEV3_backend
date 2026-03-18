"""
FastAPI backend for chatbotDemoDEV3.html

POST /chat
  Request  { type, message, lang, env, channels, chatId? }
  Response { chatId, responseMsg }

The backend auto-detects the actual query language from the message charset
(Chinese vs English) regardless of the UI lang preference sent in the request.
Conversation history is kept in memory per chatId for multi-turn context.
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

from knowledge_base import KnowledgeBase  # noqa: E402  (after load_dotenv)
from rag_engine import RAGEngine  # noqa: E402

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

# In-memory conversation store:  chatId -> list of {role, content} messages
sessions: dict[str, list[dict]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kb, rag
    print("=== Initialising RAG chatbot (chatbotDEV3) ===")
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
    lang: str = "en"          # UI language preference (en / zh_hant)
    env: str = "test"
    channels: dict = {}
    chatId: str | None = None  # absent on the first turn


class SourceItem(BaseModel):
    q: str
    answer: str
    source: str = ""


class ChatResponse(BaseModel):
    chatId: str
    responseMsg: str
    language: str            # detected language used for retrieval
    sources: list[SourceItem] = []


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    return {"status": "ok", "ready": rag is not None}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if rag is None:
        raise HTTPException(status_code=503, detail="Service not ready yet.")

    user_message = request.message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Resolve or create session
    chat_id = request.chatId or str(uuid.uuid4())
    history = sessions.get(chat_id, [])

    # RAG query
    result = await rag.query(user_message, history=history)

    # Append this turn to history (store only user + assistant content, not context)
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": result["answer"]})

    # Trim history to last 10 turns (20 messages) to avoid token bloat
    if len(history) > 20:
        history = history[-20:]
    sessions[chat_id] = history

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
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )
