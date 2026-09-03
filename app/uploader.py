"""
Universal AI Knowledge Assistant — ChromaDB Vector Store Manager
================================================================
Handles vector storage, batch upserts, metadata normalization,
and collection queries with local ChromaDB.
"""

import json
import logging
from typing import Optional
import chromadb

from config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("uploader")

BATCH_SIZE = 100
_chroma_client: Optional[chromadb.PersistentClient] = None


def get_chroma_client() -> chromadb.PersistentClient:
    """Get or initialize singleton local ChromaDB PersistentClient."""
    global _chroma_client
    if _chroma_client is None:
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    return _chroma_client


def get_chroma_collection(client: Optional[chromadb.PersistentClient] = None):
    """Get or create the ChromaDB collection configured with cosine distance."""
    if client is None:
        client = get_chroma_client()
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"}
    )


def build_metadata(chunk: dict) -> dict:
    """Flatten chunk attributes into primitive types for ChromaDB storage."""
    chunk_type = chunk.get("chunk_type", "text")

    refs = chunk.get("references", [])
    refs_str = json.dumps(refs) if isinstance(refs, (list, dict)) else str(refs or "[]")

    hyde = chunk.get("hyde_questions", [])
    hyde_str = json.dumps(hyde) if isinstance(hyde, (list, dict)) else str(hyde or "[]")

    metadata = {
        "chunk_id": str(chunk.get("chunk_id", "") or ""),
        "chunk_type": str(chunk_type or "text"),
        "heading": str(chunk.get("heading", "") or ""),
        "heading_level": int(chunk.get("heading_level", 0) or 0),
        "doc_title": str(chunk.get("doc_title", "") or ""),
        "section": str(chunk.get("section", "") or ""),
        "breadcrumb": str(chunk.get("breadcrumb", "") or ""),
        "source_url": str(chunk.get("source_url", "") or ""),
        "parent_doc": str(chunk.get("parent_doc", "") or ""),
        "references": refs_str,
        "hyde_questions": hyde_str,
        "created_at": str(chunk.get("created_at", "") or ""),
        "version": str(chunk.get("version", "1") or "1"),
        "token_count": int(chunk.get("token_count", 0) or 0),
    }

    if chunk_type == "image":
        metadata["image_file"] = str(chunk.get("image_file", "") or "")
        metadata["image_content"] = str(chunk.get("image_content", "") or "")
        metadata["scan_method"] = str(chunk.get("scan_method", "") or "")

    return metadata


def upload_chunks_batch(chunks: list[dict]) -> int:
    """
    Safely upsert a batch of vectorized chunks into local ChromaDB.
    Preserves data integrity and prevents dimension mismatch errors.
    """
    if not chunks:
        return 0

    client = get_chroma_client()
    collection = get_chroma_collection(client)

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for chunk in chunks:
        embedding = chunk.get("embedding", [])
        chunk_id = chunk.get("chunk_id", "")
        if not embedding or not chunk_id:
            continue

        doc_text = chunk.get("content", "") or chunk.get("embed_text", "")
        metadata = build_metadata(chunk)

        ids.append(str(chunk_id))
        embeddings.append(embedding)
        documents.append(doc_text)
        metadatas.append(metadata)

    if not ids:
        return 0

    for i in range(0, len(ids), BATCH_SIZE):
        batch_ids = ids[i:i + BATCH_SIZE]
        batch_embs = embeddings[i:i + BATCH_SIZE]
        batch_docs = documents[i:i + BATCH_SIZE]
        batch_meta = metadatas[i:i + BATCH_SIZE]
        collection.upsert(
            ids=batch_ids,
            embeddings=batch_embs,
            documents=batch_docs,
            metadatas=batch_meta,
        )

    return len(ids)


def delete_document_chunks(doc_title: str) -> bool:
    """Delete all chunks belonging to a document by title."""
    try:
        collection = get_chroma_collection()
        collection.delete(where={"doc_title": doc_title})
        logger.info(f"Deleted all indexed chunks for document '{doc_title}'")
        return True
    except Exception as e:
        logger.error(f"Failed to delete chunks for '{doc_title}': {e}")
        return False


def list_indexed_documents() -> list[dict]:
    """Return distinct indexed documents with chunk count and source metadata."""
    docs_map = {}
    try:
        collection = get_chroma_collection()
        results = collection.get(include=["metadatas"])
        metadatas = results.get("metadatas", []) or []
        for metadata in metadatas:
            if not metadata:
                continue
            doc_title = metadata.get("doc_title", "Untitled Document")
            source_url = metadata.get("source_url", "")
            created_at = metadata.get("created_at", "")
            if doc_title not in docs_map:
                docs_map[doc_title] = {
                    "doc_title": doc_title,
                    "source_url": source_url,
                    "chunk_count": 0,
                    "created_at": created_at,
                }
            docs_map[doc_title]["chunk_count"] += 1
    except Exception as e:
        logger.warning(f"Error listing documents from ChromaDB: {e}")
        return []
    return list(docs_map.values())