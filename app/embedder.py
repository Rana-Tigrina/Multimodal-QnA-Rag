"""
IITM BS RAG Pipeline — Stage 5: Embedder
==========================================
INPUT:  output/chunks/all_chunks.json
OUTPUT: output/chunks/all_chunks_embedded.json

Provider: Google Gemini text-embedding-004 (free, no card needed)
  - 768 dim embeddings
  - 100 RPM free tier
  - No rate limit issues
"""

import json
import logging
import time
import google.generativeai as genai
from pathlib import Path
from tqdm import tqdm

from config import (
    ALL_CHUNKS_FILE,
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIM,
    EMBEDDING_BATCH,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("embedder")

EMBEDDED_FILE  = ALL_CHUNKS_FILE.parent / "all_chunks_embedded.json"
PROGRESS_FILE  = ALL_CHUNKS_FILE.parent / "embedding_progress.json"
BATCH_SIZE     = 1


def build_embed_text(chunk: dict) -> str:
    parts = []
    embed_text = chunk.get("embed_text", "").strip()
    if embed_text:
        parts.append(embed_text)
    hyde_questions = chunk.get("hyde_questions", [])
    if hyde_questions:
        parts.append(" ".join(hyde_questions))
    return " ".join(parts)


def embed_chunk_list(chunks: list[dict]) -> list[dict]:
    """
    Generate embeddings for a list of chunk dicts in memory using Gemini.
    Used for live document ingestion.
    """
    genai.configure(api_key=GEMINI_API_KEY)
    needs = [c for c in chunks if "embedding" not in c]
    if not needs:
        return chunks

    texts = [build_embed_text(c) for c in needs]
    embeddings = embed_batch_with_retry(texts=texts, model=EMBEDDING_MODEL, input_type="document")
    for i, c in enumerate(needs):
        c["embedding"] = embeddings[i]

    return chunks



def load_progress() -> set:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
        logger.info(f"Resuming — {len(data['embedded'])} chunks already done")
        return set(data["embedded"])
    return set()


def save_progress(embedded_ids: set):
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"embedded": list(embedded_ids)}, f)


def embed_batch_with_retry(
    texts: list[str],
    model: str,
    input_type: str = "document",
    max_retries: int = 5,
) -> list[list[float]]:
    task_type = "retrieval_document" if input_type == "document" else "retrieval_query"
    embeddings = []
    for text in texts:
        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=model,
                    content=text,
                    task_type=task_type,
                )
                embeddings.append(result["embedding"])
                break
            except Exception as e:
                wait = 10
                logger.warning(f"Error — waiting {wait}s before retry {attempt + 1}/{max_retries}: {e}")
                time.sleep(wait)
        else:
            raise Exception(f"Embedding failed after {max_retries} retries")
    return embeddings


def run():
    print("\n" + "═" * 65)
    print("  IITM BS RAG Pipeline — Stage 5: Embedder")
    print("  Provider: Google Gemini (text-embedding-004)")
    print("═" * 65)

    if not ALL_CHUNKS_FILE.exists():
        print(f"\n  ❌ {ALL_CHUNKS_FILE} not found")
        print(f"     Run python hyde_generator.py first")
        return

    with open(ALL_CHUNKS_FILE) as f:
        chunks = json.load(f)
    print(f"\n  Loaded {len(chunks)} chunks")

    existing_embedded = {}
    if EMBEDDED_FILE.exists():
        print(f"  Found existing embedded file — loading for resume...")
        with open(EMBEDDED_FILE) as f:
            existing = json.load(f)
        existing_embedded = {c["chunk_id"]: c["embedding"] for c in existing if "embedding" in c}
        print(f"  Already embedded: {len(existing_embedded)} chunks")

    print(f"\n  Initialising Gemini client...")
    genai.configure(api_key=GEMINI_API_KEY)
    print(f"  ✅ Gemini ready — model: {EMBEDDING_MODEL} (dim: {EMBEDDING_DIM})")

    needs_embedding = []
    already_done    = 0

    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        if chunk_id in existing_embedded:
            chunk["embedding"] = existing_embedded[chunk_id]
            already_done += 1
        else:
            needs_embedding.append(chunk)

    print(f"  Already embedded: {already_done}")
    print(f"  To embed:         {len(needs_embedding)}")

    if not needs_embedding:
        print(f"\n  ✅ All chunks already embedded")
        _save_and_report(chunks)
        return

    texts = [build_embed_text(c) for c in needs_embedding]

    with_hyde    = sum(1 for c in needs_embedding if c.get("hyde_questions"))
    without_hyde = sum(1 for c in needs_embedding if not c.get("hyde_questions"))
    print(f"\n  HyDE coverage:")
    print(f"    With questions:    {with_hyde}")
    print(f"    Without questions: {without_hyde}")

    print(f"\n  Embedding {len(texts)} chunks via Gemini")
    print(f"  Batch size: {BATCH_SIZE} | task_type: retrieval_document\n")

    embedded_ids = set(existing_embedded.keys())
    batch_count  = 0

    with tqdm(total=len(needs_embedding), desc="  Embedding") as pbar:
        for i in range(0, len(needs_embedding), BATCH_SIZE):
            batch_chunks = needs_embedding[i:i + BATCH_SIZE]
            batch_texts  = texts[i:i + BATCH_SIZE]

            embeddings = embed_batch_with_retry(
                texts=batch_texts,
                model=EMBEDDING_MODEL,
                input_type="document",
            )

            for j, chunk in enumerate(batch_chunks):
                chunk["embedding"] = embeddings[j]
                embedded_ids.add(chunk["chunk_id"])

            batch_count += 1
            pbar.update(len(batch_chunks))

            if batch_count % 5 == 0:
                save_progress(embedded_ids)
                logger.info(f"Progress saved — {len(embedded_ids)} chunks embedded")

            time.sleep(0.5)  # gentle rate limiting, 100 RPM is generous

    save_progress(embedded_ids)
    _save_and_report(chunks)

    if PROGRESS_FILE.exists():
        PROGRESS_FILE.unlink()
        logger.info("Progress file cleaned up")


def _save_and_report(chunks: list[dict]):
    missing = [c["chunk_id"] for c in chunks if "embedding" not in c]
    if missing:
        logger.warning(f"{len(missing)} chunks missing embeddings!")

    sample = next((c for c in chunks if "embedding" in c), None)
    if sample:
        actual_dim = len(sample["embedding"])
        if actual_dim != EMBEDDING_DIM:
            logger.warning(
                f"Embedding dim mismatch: got {actual_dim}, config says {EMBEDDING_DIM}. "
                f"Update EMBEDDING_DIM in config.py to {actual_dim}"
            )

    with open(EMBEDDED_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)

    size_mb = EMBEDDED_FILE.stat().st_size / (1024 * 1024)

    print(f"\n  {'═' * 40}")
    print(f"  Total chunks:      {len(chunks)}")
    print(f"  With embeddings:   {sum(1 for c in chunks if 'embedding' in c)}")
    print(f"  Missing:           {len(missing)}")
    print(f"  File size:         {size_mb:.1f} MB")

    if sample:
        print(f"\n  Sample verification:")
        print(f"    chunk_id:    {sample['chunk_id']}")
        print(f"    chunk_type:  {sample['chunk_type']}")
        print(f"    embed_dim:   {len(sample['embedding'])}")
        print(f"    hyde_q:      {len(sample.get('hyde_questions', []))} questions")
        print(f"    embed_text:  {sample.get('embed_text','')[:60]}...")
        print(f"    provider:    Gemini ({EMBEDDING_MODEL})")

    print(f"\n  Saved to: {EMBEDDED_FILE}")
    print(f"\n  Next step: python uploader.py")
    print("═" * 65)


if __name__ == "__main__":
    run()