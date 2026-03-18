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

import torch
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from sentence_transformers import CrossEncoder

from knowledge_base import EMBED_DEVICE, KnowledgeBase

RERANK_DEVICE = (
    "cuda:1" if torch.cuda.device_count() > 1
    else EMBED_DEVICE  # fall back to same GPU (or CPU) if only one is available
)
from lexicon import Lexicon

load_dotenv(Path(__file__).parent / ".env")

RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

_EN_SYSTEM_PROMPT_BASE = """You are a helpful Housing Authority knowledge base assistant. \
Answer the user's question accurately and concisely based solely on the provided context.

**Response Rules:**
- If the context contains the answer, provide it clearly and completely.
- Always include any contact information (phone numbers, emails, addresses) that appears in the context.
- If the context is insufficient to answer the question, say so honestly and suggest the user contact the Housing Authority directly.
- Do not fabricate or infer information not explicitly present in the context.
- Do not reference or cite sources in your response (e.g. never write "Source 1", "Question 2", "[Source 3]", or any similar citation). Present the answer as direct, natural prose.

**URL Formatting:**
- If the context contains raw URLs (e.g. https://www.housingauthority.gov.hk/...), always convert them into clickable Markdown links using the format [descriptive link text](URL).
- Choose meaningful link text based on the surrounding context (e.g. [Housing Authority Official Website](https://...) rather than repeating the raw URL).
- Never output a bare URL — every URL in your response must be wrapped in a Markdown link.

**Image Rendering:**
- If the context contains image tags in any markdown or HTML format (e.g. ![alt](url) or <img src="...">), pass them through to your response exactly as they appear in the source — do not alter, remove, or describe them as text.
- Preserve the original image syntax so the frontend can render the image correctly."""

_ZH_SYSTEM_PROMPT_BASE = """你是房屋局知識庫助手。請根據所提供的內容，準確簡潔地回答用戶問題。

**回答規則：**
- 若內容包含答案，請清楚完整地說明。
- 如有聯絡資訊（電話、電郵、地址），請一併提供。
- 若內容不足以回答問題，請如實說明，並建議用戶直接聯絡房屋局。
- 切勿捏造或推斷內容以外的資訊。
- 回覆中不得引用或標注來源（例如不得寫「來源1」、「問題2」、「[Source 3]」等）。請以自然流暢的語句直接作答。

**URL 格式化：**
- 若內容包含原始網址（如 https://www.housingauthority.gov.hk/...），請一律將其轉換為可點擊的 Markdown 連結，格式為 [描述性連結文字](URL)。
- 根據上下文選擇合適的連結文字（如 [房屋局官方網站](https://...)，而非直接重複網址）。
- 回覆中不得出現裸露網址，所有網址必須包裝為 Markdown 連結。

**圖片渲染：**
- 若內容包含任何 Markdown 或 HTML 格式的圖片標記（如 ![alt](url) 或 <img src="...">），請原樣保留並輸出至回覆中，不得修改、刪除或以文字描述代替。
- 保留原始圖片語法，以確保前端能正確渲染圖片。"""


class RAGEngine:
    def __init__(self, knowledge_base: KnowledgeBase, azure_client: AsyncAzureOpenAI):
        self.kb = knowledge_base
        self.client = azure_client
        self.azure_model = os.getenv("AZURE_MODEL", "gpt-4o-mini")

        # Read retrieval limits from .env (with safe integer parsing)
        self.embed_top_k = int(os.getenv("EMBED_TOP_K", "15"))
        self.rerank_top_k = int(os.getenv("RERANK_TOP_K", "5"))

        print(f"Loading reranker model: {RERANK_MODEL_NAME}  [device={RERANK_DEVICE}]")
        self.reranker = CrossEncoder(RERANK_MODEL_NAME, device=RERANK_DEVICE)

        self.lexicon = Lexicon()

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

        # Option 3 — expand query with lexicon definitions before retrieval
        expanded_query = self.lexicon.expand_query(user_input, lang)

        # 1. Dense retrieval (use expanded query for better embedding match)
        candidates = self.kb.search(expanded_query, lang, top_k=self.embed_top_k)
        if not candidates:
            fallback = (
                "I'm sorry, I couldn't find relevant information for your query. "
                "Please try rephrasing or contact our service centre."
                if lang == "en"
                else "抱歉，找不到相關資訊。請換個方式提問，或聯絡我們的服務中心。"
            )
            return {"answer": fallback, "sources": [], "language": lang, "expanded_query": expanded_query}

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
        # Option 2 — append glossary to system prompt so LLM understands domain terms
        base_prompt = _EN_SYSTEM_PROMPT_BASE if lang == "en" else _ZH_SYSTEM_PROMPT_BASE
        glossary = self.lexicon.glossary_text(lang)
        system_prompt = f"{base_prompt}\n\n{glossary}" if glossary else base_prompt
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

        return {"answer": answer, "sources": sources, "language": lang, "expanded_query": expanded_query}
