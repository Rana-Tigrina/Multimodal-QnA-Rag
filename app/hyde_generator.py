"""
Universal AI Knowledge Assistant — HyDE Generator Module
=========================================================
Generates hypothetical questions per chunk during document ingestion to boost
retrieval accuracy (matching question-to-question rather than question-to-paragraph).
"""

import time
import logging
from config import HYDE_QUESTIONS_PER_CHUNK, HYDE_PROMPT, LLM_PROVIDER, LOG_LEVEL, LOG_FORMAT
from model_service import generate_chat_text

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("hyde_generator")


def generate_questions_for_chunk(chunk: dict) -> list[str]:
    """Generate 2-3 hypothetical user questions whose answer is found in the chunk."""
    content = chunk.get("content", "").strip()
    if len(content) < 80:
        return []

    doc_title = chunk.get("doc_title", "Document")
    section = chunk.get("section", doc_title)

    prompt = HYDE_PROMPT.format(
        doc_title=doc_title,
        section=section,
        content=content[:1500],
        n=HYDE_QUESTIONS_PER_CHUNK,
    )

    try:
        response = generate_chat_text(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=200,
        )
        if not response:
            return []

        questions = []
        for line in response.splitlines():
            line = line.strip().lstrip("0123456789.-)*• ").strip()
            if len(line) > 10 and "?" in line:
                questions.append(line)

        return questions[:HYDE_QUESTIONS_PER_CHUNK]
    except Exception as e:
        logger.debug(f"HyDE generation skipped for chunk: {e}")
        return []


def generate_hyde_for_chunks(chunks: list[dict]) -> list[dict]:
    """
    Enrich an in-memory list of document chunks with HyDE questions during ingestion.
    Gracefully skips on rate limits or API errors without failing ingestion.
    """
    count = 0
    for chunk in chunks:
        if chunk.get("chunk_type") in ("image", "section_index"):
            chunk["hyde_questions"] = []
            continue

        if not chunk.get("hyde_questions"):
            q_list = generate_questions_for_chunk(chunk)
            chunk["hyde_questions"] = q_list
            count += 1
            # Micro-throttle if using Groq API
            if LLM_PROVIDER == "groq" and count > 2:
                time.sleep(0.5)

    return chunks