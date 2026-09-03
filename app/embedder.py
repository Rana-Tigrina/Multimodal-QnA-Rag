"""
Universal AI Knowledge Assistant — Local Embedder Module
=========================================================
100% Local ONNX embeddings using all-MiniLM-L6-v2 (384 dimensions).
Completely offline, fast, zero API quotas or network dependencies.
"""

import sys
import logging
from typing import List
import chromadb.utils.embedding_functions as ef

from config import EMBEDDING_MODEL, EMBEDDING_DIM, LOG_LEVEL, LOG_FORMAT

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("embedder")

_local_ef = None


def get_local_embedding_function():
    """Get or initialize singleton local ONNX all-MiniLM-L6-v2 embedding function."""
    global _local_ef
    if _local_ef is None:
        _local_ef = ef.DefaultEmbeddingFunction()
    return _local_ef


def build_embed_text(chunk: dict) -> str:
    """Combine document text and synthetic HyDE questions for optimal retrieval matching."""
    parts = []
    embed_text = chunk.get("embed_text", "").strip() or chunk.get("content", "").strip()
    if embed_text:
        parts.append(embed_text)

    hyde_questions = chunk.get("hyde_questions", [])
    if hyde_questions and isinstance(hyde_questions, list):
        parts.append(" ".join(str(q) for q in hyde_questions))

    return " ".join(parts)


def embed_texts(texts: List[str], is_query: bool = False) -> List[List[float]]:
    """Generate 384d vector embeddings for a list of strings using local ONNX model."""
    if not texts:
        return []
    local_ef = get_local_embedding_function()
    raw_embs = local_ef(texts)
    return [list(map(float, e)) for e in raw_embs]


def embed_query(query: str) -> List[float]:
    """Generate 384d vector embedding for a single user search query."""
    res = embed_texts([query], is_query=True)
    return res[0] if res else []


def embed_chunk_list(chunks: List[dict]) -> List[dict]:
    """
    Generate and attach embeddings to an in-memory list of chunks.
    Used for live document and URL ingestion.
    """
    needs = [c for c in chunks if "embedding" not in c]
    if not needs:
        return chunks

    texts = [build_embed_text(c) for c in needs]
    embeddings = embed_texts(texts=texts, is_query=False)

    for i, c in enumerate(needs):
        c["embedding"] = embeddings[i]

    return chunks


if __name__ == "__main__":
    v = embed_query("What is the refund policy?")
    print(f"Generated embedding vector: dimension={len(v)}, sample={v[:3]}")