"""
Knowledge Base loader — reads from a directory of .xlsx files OR a single .xlsx file,
splits documents by language (en / zh_hant), and builds per-language FAISS indexes.

Column layout expected in each worksheet (row 1 = header):
  [0] No.        [1] Source     [2] lang       [3] definition
  [4] question   [5] answer     [6] category   [7] active   [8] status
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import faiss
import numpy as np
import openpyxl
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv(Path(__file__).parent / ".env")

EMBED_MODEL_NAME = "BAAI/bge-m3"
CACHE_DIR = Path(__file__).parent / "cache"

# Use CUDA if available, fall back to CPU gracefully
EMBED_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def _resolve_kb_path() -> Path:
    """Resolve KB_PATH from .env relative to this file's directory."""
    raw = os.getenv("KB_PATH", "../FAQ_combined")
    p = Path(raw)
    if not p.is_absolute():
        p = (Path(__file__).parent / p).resolve()
    return p


def _load_xlsx(file_path: Path) -> tuple[list[dict], list[dict]]:
    """Parse a single xlsx file and return (en_docs, zh_docs)."""
    en_docs: list[dict] = []
    zh_docs: list[dict] = []

    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    for ws in wb.worksheets:
        for row in ws.iter_rows(min_row=2, values_only=True):
            # Guard against completely empty rows
            if not any(row):
                continue

            lang = row[2] if len(row) > 2 else None
            question = row[4] if len(row) > 4 else None
            answer = row[5] if len(row) > 5 else None
            active = row[7] if len(row) > 7 else True  # default active if missing

            # Skip inactive or empty entries
            if active is False:
                continue
            if not question or not answer:
                continue

            doc = {
                "q": str(question).strip(),
                "answer": str(answer).strip(),
                "source": str(row[1]).strip() if row[1] else "",
                "definition": str(row[3]).strip() if row[3] else "",
                "lang": lang,
            }

            if lang == "en":
                en_docs.append(doc)
            elif lang == "zh_hant":
                zh_docs.append(doc)

    wb.close()
    return en_docs, zh_docs


def load_all_docs() -> tuple[list[dict], list[dict]]:
    """
    Load documents from KB_PATH.

    - If KB_PATH is a directory:  read every *.xlsx inside it.
    - If KB_PATH looks like a file (or ends with .xlsx):  read that file only.
    - Fallback: try appending .xlsx to the path.
    """
    kb_path = _resolve_kb_path()
    xlsx_files: list[Path] = []

    if kb_path.is_dir():
        xlsx_files = sorted(kb_path.glob("*.xlsx"))
        if not xlsx_files:
            raise FileNotFoundError(f"No .xlsx files found in directory: {kb_path}")
    elif kb_path.is_file():
        xlsx_files = [kb_path]
    else:
        # Try adding .xlsx suffix
        candidate = kb_path.with_suffix(".xlsx")
        if candidate.is_file():
            xlsx_files = [candidate]
        else:
            raise FileNotFoundError(
                f"Knowledge base not found at '{kb_path}'. "
                "Set KB_PATH in .env to a directory of .xlsx files or a single .xlsx file."
            )

    all_en: list[dict] = []
    all_zh: list[dict] = []
    for f in xlsx_files:
        print(f"  Loading {f.name}…")
        en, zh = _load_xlsx(f)
        all_en.extend(en)
        all_zh.extend(zh)

    print(f"Loaded {len(all_en)} EN docs and {len(all_zh)} ZH docs from {len(xlsx_files)} file(s).")
    return all_en, all_zh


class KnowledgeBase:
    def __init__(self):
        print(f"Loading embedding model: {EMBED_MODEL_NAME}  [device={EMBED_DEVICE}]")
        self.embed_model = SentenceTransformer(EMBED_MODEL_NAME, device=EMBED_DEVICE)

        self.en_docs: list[dict] = []
        self.zh_docs: list[dict] = []
        self.en_index: faiss.Index | None = None
        self.zh_index: faiss.Index | None = None

        self._load_or_build()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gpu(index: faiss.Index) -> faiss.Index:
        """Move a FAISS index to GPU 0 if CUDA is available."""
        if EMBED_DEVICE.startswith("cuda"):
            res = faiss.StandardGpuResources()
            return faiss.index_cpu_to_gpu(res, 0, index)
        return index

    def _build_index(self, docs: list[dict]) -> faiss.Index:
        # Concatenate question + answer for richer embeddings
        texts = [f"{doc['q']} {doc['answer']}" for doc in docs]
        print(f"    Encoding {len(texts)} documents…")
        embeddings = self.embed_model.encode(
            texts,
            normalize_embeddings=True,  # cosine similarity via inner product
            batch_size=32,
            show_progress_bar=True,
        )
        embeddings = np.array(embeddings, dtype=np.float32)
        cpu_index = faiss.IndexFlatIP(embeddings.shape[1])
        cpu_index.add(embeddings)
        return self._to_gpu(cpu_index)

    def _load_or_build(self) -> None:
        CACHE_DIR.mkdir(exist_ok=True)
        en_idx = CACHE_DIR / "en_index.faiss"
        zh_idx = CACHE_DIR / "zh_index.faiss"
        en_pkl = CACHE_DIR / "en_docs.pkl"
        zh_pkl = CACHE_DIR / "zh_docs.pkl"

        if all(p.exists() for p in [en_idx, zh_idx, en_pkl, zh_pkl]):
            print(f"Loading cached FAISS indexes…  [device={DEVICE}]")
            self.en_index = self._to_gpu(faiss.read_index(str(en_idx)))
            self.zh_index = self._to_gpu(faiss.read_index(str(zh_idx)))
            with open(en_pkl, "rb") as f:
                self.en_docs = pickle.load(f)
            with open(zh_pkl, "rb") as f:
                self.zh_docs = pickle.load(f)
            print(f"  EN: {len(self.en_docs)} docs | ZH: {len(self.zh_docs)} docs")
        else:
            print("Building knowledge base from source files…")
            self.en_docs, self.zh_docs = load_all_docs()

            print("  Building EN index…")
            self.en_index = self._build_index(self.en_docs)
            print("  Building ZH index…")
            self.zh_index = self._build_index(self.zh_docs)

            # Persist — GPU indexes must be moved back to CPU before writing
            faiss.write_index(faiss.index_gpu_to_cpu(self.en_index) if EMBED_DEVICE.startswith("cuda") else self.en_index, str(en_idx))
            faiss.write_index(faiss.index_gpu_to_cpu(self.zh_index) if EMBED_DEVICE.startswith("cuda") else self.zh_index, str(zh_idx))
            with open(en_pkl, "wb") as f:
                pickle.dump(self.en_docs, f)
            with open(zh_pkl, "wb") as f:
                pickle.dump(self.zh_docs, f)
            print("FAISS indexes saved to cache.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, lang: str, top_k: int) -> list[dict]:
        """Dense retrieval — returns up to top_k docs for the given language."""
        if lang == "zh_hant":
            index, docs = self.zh_index, self.zh_docs
        else:
            index, docs = self.en_index, self.en_docs

        if index is None or not docs:
            return []

        top_k = min(top_k, len(docs))
        query_vec = self.embed_model.encode(
            [query], normalize_embeddings=True
        )
        query_vec = np.array(query_vec, dtype=np.float32)
        _, indices = index.search(query_vec, top_k)
        return [docs[i] for i in indices[0] if i < len(docs)]
