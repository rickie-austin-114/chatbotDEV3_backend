"""
RAG Engine: charset-based language detection → dense retrieval → cross-encoder
reranking → GPT answer generation.

Models:
  Embedding : BAAI/bge-m3                (handled by KnowledgeBase)
  Reranker  : BAAI/bge-reranker-v2-m3   (multilingual cross-encoder)
  LLM       : Azure OpenAI (configurable via .env)
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from sentence_transformers import CrossEncoder

from knowledge_base import KnowledgeBase

load_dotenv(Path(__file__).parent / ".env")

RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

EN_SYSTEM_PROMPT = (
    "You are a helpful Housing Authority knowledge base assistant. "
    "Answer the user's question accurately and concisely based solely on the provided context.\n"
    "- If the context contains the answer, provide it clearly.\n"
    "- Always include any contact information that appears in the context.\n"
    "- If the context is insufficient, say so honestly.\n"
    "- Do not fabricate information not present in the context."
)

ZH_SYSTEM_PROMPT = (
    "你是房屋局知識庫助手。請根據所提供的內容，準確簡潔地回答用戶問題。\n"
    "- 若內容包含答案，請清楚說明。\n"
    "- 如有聯絡資訊，請一併提供。\n"
    "- 若內容不足以回答問題，請如實說明。\n"
    "- 切勿捏造內容以外的資訊。"
)


class RAGEngine:
    def __init__(self, knowledge_base: KnowledgeBase, azure_client: AsyncAzureOpenAI):
        self.kb = knowledge_base
        self.client = azure_client
        self.azure_model = os.getenv("AZURE_MODEL", "gpt-4o-mini")

        # Read retrieval limits from .env (with safe integer parsing)
        self.embed_top_k = int(os.getenv("EMBED_TOP_K", "15"))
        self.rerank_top_k = int(os.getenv("RERANK_TOP_K", "5"))

        print(f"Loading reranker model: {RERANK_MODEL_NAME}")
        self.reranker = CrossEncoder(RERANK_MODEL_NAME)
        print(
            f"RAG engine ready  "
            f"[embed_top_k={self.embed_top_k}, rerank_top_k={self.rerank_top_k}]"
        )

    # ------------------------------------------------------------------
    # Language detection — charset-based (CJK ratio threshold)
    # ------------------------------------------------------------------

    @staticmethod
    def detect_language(text: str) -> str:
        """
        Return 'zh_hant' if more than 10 % of the characters fall in the
        CJK Unified Ideographs block (U+4E00–U+9FFF), otherwise 'en'.

        Assumes the user inputs only Chinese or English.
        """
        if not text:
            return "en"
        cjk_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        ratio = cjk_count / len(text)
        return "zh_hant" if ratio > 0.1 else "en"

    # ------------------------------------------------------------------
    # RAG pipeline
    # ------------------------------------------------------------------

    async def query(
        self,
        user_input: str,
        history: list[dict] | None = None,
    ) -> dict:
        """
        Full RAG pipeline:
          1. Detect language from input charset.
          2. Dense retrieval with bge-m3 (EMBED_TOP_K candidates).
          3. Cross-encoder reranking with bge-reranker-v2-m3.
          4. Keep RERANK_TOP_K docs as grounded context.
          5. Call Azure OpenAI with conversation history preserved.

        Returns dict with keys: answer, sources, language.
        """
        lang = self.detect_language(user_input)

        # 1. Dense retrieval
        candidates = self.kb.search(user_input, lang, top_k=self.embed_top_k)
        if not candidates:
            fallback = (
                "I'm sorry, I couldn't find relevant information for your query. "
                "Please try rephrasing or contact our service centre."
                if lang == "en"
                else "抱歉，找不到相關資訊。請換個方式提問，或聯絡我們的服務中心。"
            )
            return {"answer": fallback, "sources": [], "language": lang}

        # 2. Cross-encoder reranking
        pairs = [
            [user_input, f"{doc['q']}\n{doc['answer']}"]
            for doc in candidates
        ]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: float(x[0]), reverse=True)
        top_docs = [doc for _, doc in ranked[: self.rerank_top_k]]

        # 3. Build grounded context string
        context_parts: list[str] = []
        for i, doc in enumerate(top_docs, 1):
            part = f"[Source {i}]\nQ: {doc['q']}\nA: {doc['answer']}"
            context_parts.append(part)
        context = "\n\n".join(context_parts)

        # 4. Build messages list (system + optional history + current turn)
        system_prompt = EN_SYSTEM_PROMPT if lang == "en" else ZH_SYSTEM_PROMPT
        messages: list[dict] = [{"role": "system", "content": system_prompt}]

        if history:
            messages.extend(history)

        messages.append({
            "role": "user",
            "content": f"Context:\n{context}\n\nUser Question: {user_input}",
        })

        # 5. LLM generation
        response = await self.client.chat.completions.create(
            model=self.azure_model,
            messages=messages,
            temperature=0.2,
            max_tokens=1200,
        )
        answer = response.choices[0].message.content

        sources = [
            {
                "q": doc["q"],
                "answer": doc["answer"],
                "source": doc.get("source", ""),
            }
            for doc in top_docs[:3]
        ]

        return {"answer": answer, "sources": sources, "language": lang}
