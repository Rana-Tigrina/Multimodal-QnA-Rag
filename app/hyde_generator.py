"""
IITM BS RAG Pipeline — Stage 4: HyDE Generator
=================================================
INPUT:  output/chunks/all_chunks.json   (from chunker.py)
OUTPUT: output/chunks/all_chunks.json   (UPDATED — hyde_questions added)

What it does:
  - Reads every chunk from all_chunks.json
  - For each chunk (except skipped types), calls LLM to generate
    3 questions a student would ask whose answer is in that chunk
  - Writes hyde_questions: [...] back into each chunk
  - Saves progress after every chunk — crash-safe
  - Skips already-done chunks (resume on crash)
  - Skips section_index and restricted_doc (see config.HYDE_SKIP_TYPES)

Why HyDE helps:
  At retrieval time, student question is matched against similar
  questions (hyde_questions) not raw text — much more accurate
  because questions match questions, not questions matching paragraphs.

Run:
  python hyde_generator.py
"""
import sys

import json
import time
import logging
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

from config import (
    ALL_CHUNKS_FILE,
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_PROVIDER,
    LLM_FALLBACK_MODELS,
    HYDE_QUESTIONS_PER_CHUNK,
    HYDE_SKIP_TYPES,
    HYDE_PROMPT,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("hyde_generator")

# ══════════════════════════════════════════════════════════════════
# LLM CLIENT
# ══════════════════════════════════════════════════════════════════

client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
)

# ══════════════════════════════════════════════════════════════════
# LLM CALLER WITH FALLBACK
# ══════════════════════════════════════════════════════════════════

def call_llm(prompt: str, max_tokens: int = 200) -> str | None:
    """
    INPUT:  prompt string
    OUTPUT: LLM response string or None on failure

    Tries primary model first.
    On rate limit (Groq), cycles through fallback models.
    On Ollama, no fallback needed — just retries once.
    """
    models_to_try = [LLM_MODEL] + LLM_FALLBACK_MODELS

    for model in models_to_try:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()

        except Exception as e:
            err = str(e).lower()
            if "rate_limit" in err or "429" in err:
                logger.warning(f"Rate limit on {model}, trying next...")
                time.sleep(2)
                continue
            if "decommissioned" in err or "model_not_found" in err:
                logger.warning(f"Model {model} not available, trying next...")
                continue
            # Unexpected error — log and skip this chunk
            logger.error(f"LLM error on {model}: {e}")
            return None

    logger.error("All models exhausted")
    return None

# ══════════════════════════════════════════════════════════════════
# QUESTION GENERATOR
# ══════════════════════════════════════════════════════════════════

def generate_questions(chunk: dict) -> list[str]:
    """
    INPUT:  one chunk dict
    OUTPUT: list of N questions (N = HYDE_QUESTIONS_PER_CHUNK)

    Returns [] if LLM fails or chunk is too short.
    """
    content   = chunk.get("content", "").strip()
    doc_title = chunk.get("doc_title", "")
    section   = chunk.get("section", "")

    # Skip very short chunks — not enough content for meaningful questions
    if len(content) < 50:
        return []

    prompt = HYDE_PROMPT.format(
        doc_title = doc_title,
        section   = section,
        content   = content[:1500],   # cap at 1500 chars to stay within token limits
        n         = HYDE_QUESTIONS_PER_CHUNK,
    )

    response = call_llm(prompt, max_tokens=200)
    if not response:
        return []

    # Parse — one question per line, strip numbering if present
    lines = []
    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove common numbering patterns: "1.", "1)", "-", "*"
        line = line.lstrip("0123456789.-)*• ").strip()
        if len(line) > 10 and "?" in line:
            lines.append(line)

    # Return exactly N questions (trim if LLM returned more)
    return lines[:HYDE_QUESTIONS_PER_CHUNK]


def generate_hyde_for_chunks(chunks: list[dict]) -> list[dict]:
    """
    Generate HyDE synthetic questions for a given list of chunks in memory.
    Used for live document ingestion.
    """
    count = 0
    for chunk in chunks:
        chunk_type = chunk.get("chunk_type", "text")
        if chunk_type in HYDE_SKIP_TYPES:
            chunk["hyde_questions"] = []
            continue

        content = chunk.get("content", "").strip()
        if len(content) < 80:
            chunk["hyde_questions"] = []
            continue

        if not chunk.get("hyde_questions"):
            questions = generate_questions(chunk)
            chunk["hyde_questions"] = questions or []
            count += 1
            if LLM_PROVIDER == "groq" and count > 3:
                time.sleep(1.2)

    return chunks



# ══════════════════════════════════════════════════════════════════
# PROGRESS SAVER
# Saves after every chunk — crash safe
# ══════════════════════════════════════════════════════════════════


def save_progress(chunks: list[dict]):
    """Save chunks back to all_chunks.json."""
    with open(ALL_CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run():
    print("\n" + "═" * 65)
    force = "--force" in sys.argv
    print("  IITM BS RAG Pipeline — Stage 4: HyDE Generator")
    print("═" * 65)

    if not ALL_CHUNKS_FILE.exists():
        print(f"\n  ❌ {ALL_CHUNKS_FILE} not found")
        print(f"     Run python chunker.py first")
        return

    with open(ALL_CHUNKS_FILE) as f:
        chunks = json.load(f)

    total = len(chunks)
    print(f"\n  Total chunks:    {total}")
    print(f"  LLM provider:    {LLM_PROVIDER} → {LLM_MODEL}")
    print(f"  Questions/chunk: {HYDE_QUESTIONS_PER_CHUNK}")
    print(f"  Skip types:      {HYDE_SKIP_TYPES}")

    # Count what needs processing
    needs_processing = []
    already_done     = 0
    skipped_type     = 0

    for chunk in chunks:
        chunk_type = chunk.get("chunk_type", "text")

        # Skip configured types
        if chunk_type in HYDE_SKIP_TYPES:
            skipped_type += 1
            continue

        # Skip already done (resume on crash)
        if chunk.get("hyde_questions") and not force:
            already_done += 1
            continue

        needs_processing.append(chunk)

    print(f"\n  Already done:    {already_done}")
    print(f"  Skipped (type):  {skipped_type}")
    print(f"  To process:      {len(needs_processing)}")

    if not needs_processing:
        print(f"\n  ✅ All chunks already have HyDE questions")
        _print_summary(chunks)
        return

    # Estimate time
    est_seconds = len(needs_processing) * 1.5  # ~1.5s per chunk on Groq
    est_minutes = est_seconds / 60
    print(f"\n  Estimated time:  ~{est_minutes:.0f} minutes")
    print(f"\n  Starting generation...\n")

    # Stats
    stats = {
        "generated":  0,
        "skipped":    0,
        "failed":     0,
        "total_questions": 0,
    }

    with tqdm(total=len(needs_processing), desc="  Generating") as pbar:
        for chunk in needs_processing:
            chunk_id   = chunk.get("chunk_id", "?")
            chunk_type = chunk.get("chunk_type", "text")
            section    = chunk.get("section", "")[:50]

            questions = generate_questions(chunk)

            if questions:
                chunk["hyde_questions"] = questions
                stats["generated"]      += 1
                stats["total_questions"] += len(questions)
            else:
                chunk["hyde_questions"] = []
                stats["failed"]         += 1

            # Save after every chunk — crash safe
            save_progress(chunks)

            pbar.set_postfix({
                "done": stats["generated"],
                "fail": stats["failed"],
            })
            pbar.update(1)

            # Small delay to avoid rate limits on Groq
            if LLM_PROVIDER == "groq":
                time.sleep(1.5)

    # Final summary
    _print_summary(chunks, stats)


def _print_summary(chunks: list[dict], stats: dict = None):
    """Print final summary of HyDE generation."""

    with_hyde    = sum(1 for c in chunks if c.get("hyde_questions"))
    without_hyde = sum(1 for c in chunks if not c.get("hyde_questions"))
    total_q      = sum(len(c.get("hyde_questions", [])) for c in chunks)

    # Sample
    sample = next((c for c in chunks if c.get("hyde_questions")), None)

    print(f"\n  {'═' * 40}")
    print(f"  Total chunks:         {len(chunks)}")
    print(f"  With HyDE questions:  {with_hyde}")
    print(f"  Without (skipped):    {without_hyde}")
    print(f"  Total questions gen:  {total_q}")

    if stats:
        print(f"\n  This run:")
        print(f"    Generated:  {stats['generated']}")
        print(f"    Failed:     {stats['failed']}")

    if sample:
        print(f"\n  Sample HyDE questions from: {sample.get('section','')[:50]}")
        for q in sample["hyde_questions"]:
            print(f"    → {q}")

    print(f"\n  Saved to: {ALL_CHUNKS_FILE}")
    print(f"\n  Next step: python embedder.py")
    print("═" * 65)


if __name__ == "__main__":
    run()


    