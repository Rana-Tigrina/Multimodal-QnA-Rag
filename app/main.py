"""
IITM BS RAG Pipeline — Stage 11: FastAPI Backend
=================================================
Provider: Google Gemini (text-embedding-004, free, no card needed)

Endpoints:
  POST /ask        → streaming SSE answer
  GET  /health     → check all services are up

Run:
  uvicorn main:app --reload --port 8000
"""

import os
import json
import hashlib
import asyncio
import logging
import time
from pathlib import Path
import google.generativeai as genai

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import ScoredPoint
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config import (
    LLM_BASE_URL,
    LLM_API_KEY,
    LLM_MODEL,
    LLM_FALLBACK_MODELS,
    LLM_PROVIDER,
    GEMINI_API_KEY,
    EMBEDDING_MODEL,
    RERANKER_MODEL,
    RERANKER_TOP_K,
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    RETRIEVAL_TOP_K,
    RERANK_TOP_K,
    VECTOR_WEIGHT,
    BM25_WEIGHT,
    RAG_SYSTEM_PROMPT,
    STREAM_RESPONSE,
    LOG_LEVEL,
    LOG_FORMAT,
    UPLOAD_DIR,
    ALLOWED_EXTENSIONS,
)

from ingestion_service import ingest_file_stream, ingest_url_stream
from uploader import list_indexed_documents, delete_document_chunks, get_qdrant_client


logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("main")

MAX_HISTORY = 6
RETRIEVAL_TOP_K_EXPANDED = 20
CONFIDENCE_THRESHOLD = 0.3


# ══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are the AI Document Knowledge Assistant — an expert system for answering questions based on uploaded documents and web pages.

## YOUR KNOWLEDGE SOURCE
Answer ONLY from the context chunks provided below.
Never guess or make up policy details, numbers, dates, or specifications.

## HANDLING DIFFERENT QUESTION TYPES

OFF-TOPIC:
→ Politely redirect:
  "I am your Document AI Assistant. I can help answer questions based on your uploaded documents and web pages. What would you like to know?"

VAGUE QUESTION (context has partial answer):
→ Give what you found, then ask ONE specific follow-up.

SPECIFIC QUESTION:
→ Answer directly and completely from context.

FOLLOW-UP QUESTION:
→ Use conversation history to understand what "it", "that", "this" refers to.
→ Combine history + current question for a complete answer.

## FORMAT RULES
- Simple fact → 1-3 sentences, then detail if needed
- Criteria/Requirements → bullet list
- Process/Steps → numbered list
- Comparison → markdown table
- Use **bold** for key dates, metrics, numbers, and important terms
- Use `code` for codes, parameters, or technical identifiers
- Never pad with filler ("Great question!", "Certainly!", etc.)

## DIRECT ANSWER RULE
- Questions starting with "Can I", "Is there", "Does", "Will", "Am I" MUST start with YES or NO
- Never hedge with "it seems", "it appears", "however", "based on the context"
- If you know the answer from context → state it directly and confidently
- If context only partially answers → give what you know, then state exactly what is missing

## CITATIONS — MANDATORY SOURCE REQUIRED
End EVERY answer with source citation in this exact format:

📎 **Source:** [section name] — [document title] 🔗 [URL or filename]

RULES:
- Citation is MANDATORY — never omit it
- If context has a source_url → use it exactly
- If multiple sources → list each on its own line with its own source URL/file
- NEVER include internal labels like [SOURCE:...] or [CONTEXT N] in citations

## LINKS & IMAGES
If context contains reference links or image descriptions — include them inline where helpful.

## CANNOT ANSWER
If answer truly not in context:
"I could not find this information in the current indexed documents."
"""

# ══════════════════════════════════════════════════════════════════
# INIT
# ══════════════════════════════════════════════════════════════════

print("Initialising Gemini client...")
genai.configure(api_key=GEMINI_API_KEY)
print(f"✅ Gemini ready: {EMBEDDING_MODEL}")

print(f"Connecting to Qdrant...")
qdrant = get_qdrant_client()
print("✅ Qdrant database ready")


print(f"Initialising LLM client ({LLM_PROVIDER})...")
llm_client = OpenAI(
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    timeout=20,
    max_retries=0,
)
print(f"✅ LLM client ready: {LLM_MODEL}")
print("✅ Backend ready\n")

# ══════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════

app = FastAPI(title="Universal AI Knowledge Assistant API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════
# SCHEMAS
# ══════════════════════════════════════════════════════════════════

class ChatMessage(BaseModel):
    role: str
    content: str

class AskRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []

class UrlIngestRequest(BaseModel):
    url: str



# ══════════════════════════════════════════════════════════════════
# LLM CALLER WITH FALLBACK + BACKOFF
# ══════════════════════════════════════════════════════════════════

def llm_call(messages: list, max_tokens: int = 200, stream: bool = False):
    models_to_try = [LLM_MODEL] + LLM_FALLBACK_MODELS
    last_error    = None

    for i, model in enumerate(models_to_try):
        try:
            if i > 0:
                wait = min(2 ** i, 8)
                logger.info(f"Waiting {wait}s before trying {model}...")
                time.sleep(wait)

            resp = llm_client.chat.completions.create(
                model       = model,
                messages    = messages,
                max_tokens  = max_tokens,
                temperature = 0.1,
                stream      = stream,
            )
            if model != LLM_MODEL:
                logger.info(f"Using fallback model: {model}")
            return resp

        except Exception as e:
            err = str(e).lower()
            if any(x in err for x in [
                "rate_limit", "429", "decommissioned",
                "model_not_found", "timed out", "timeout",
                "connection", "read timed out"
            ]):
                logger.warning(f"Model {model} failed ({type(e).__name__}), trying next...")
                last_error = e
                continue
            raise e

    raise Exception(f"All LLM models exhausted. Last error: {last_error}")


# ══════════════════════════════════════════════════════════════════
# QUERY CLASSIFIER & REWRITER
# ══════════════════════════════════════════════════════════════════

GREETINGS = ["hello", "hi", "hey", "good morning", "good evening", "thanks", "thank you", "bye"]

def is_greeting(question: str) -> bool:
    q = question.lower().strip()
    return q in GREETINGS or any(q == g or q.startswith(g + " ") for g in GREETINGS)


def rewrite_query(question: str, history: list[ChatMessage]) -> str:
    """
    Rewrite user query for better retrieval using conversation context.
    Includes timeout protection to prevent hanging.
    """
    question = question.strip() if question else ""
    if not question:
        return question

    # Skip LLM rewrite for simple questions to save time and reduce failures
    if len(question.split()) < 4 and not history:
        logger.info(f"Using original query (short): '{question}'")
        return question

    history_text = ""
    if history:
        history_text = "\n".join(
            f"{m.role}: {m.content}" for m in history[-2:]
        )

    try:
        prompt = (
            f"Rewrite this user question to be explicit, specific, and highly searchable "
            f"in a document knowledge vector index.\n\n"
            f"Rules:\n"
            f"- Expand abbreviations, pronouns, or vague references using conversation context\n"
            f"- Make implicit topics explicit\n"
            f"- Return ONLY the rewritten question string, nothing else\n\n"
        )
        if history_text:
            prompt += f"Conversation so far:\n{history_text}\n\n"
        prompt += f"Question to rewrite: {question}"

        resp = llm_call(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
        )
        rewritten = resp.choices[0].message.content.strip() if resp and resp.choices else ""
        final_query = rewritten if (rewritten and len(rewritten) > 3) else question
        logger.info(f"Rewritten: '{question}' → '{final_query}'")
        return final_query

    except Exception as e:
        logger.warning(f"Rewrite failed: {e}, using original query")
        return question


# ══════════════════════════════════════════════════════════════════
# SPARSE VECTOR BUILDER (BM25 approximation)
# ══════════════════════════════════════════════════════════════════

def build_sparse_vector(text: str) -> tuple[list[int], list[float]]:
    import re
    from collections import Counter

    words = re.findall(r'\b[a-z]{2,}\b', text.lower())
    if not words:
        return [0], [0.0]

    tf    = Counter(words)
    total = len(words)

    index_map = {}
    for word, count in tf.items():
        word_idx = int(hashlib.md5(word.encode()).hexdigest()[:6], 16) % 100000
        tf_score = count / total
        if word_idx in index_map:
            index_map[word_idx] += tf_score
        else:
            index_map[word_idx] = tf_score

    return list(index_map.keys()), [float(v) for v in index_map.values()]


# ══════════════════════════════════════════════════════════════════
# HYBRID RETRIEVAL
# ══════════════════════════════════════════════════════════════════

def retrieve_chunks(question: str, top_k: int = RETRIEVAL_TOP_K_EXPANDED) -> tuple[list[dict], float]:
    """
    Retrieve relevant chunks from Qdrant using hybrid search (dense + sparse).
    Includes error handling and fallbacks.
    """
    question = question.strip() if question else ""
    if not question:
        return [], 0.0

    # ── Step 1: Embed query with Gemini ─────────────────────────
    try:
        embed_result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=question,
            task_type="retrieval_query",
        )
        query_vec = embed_result["embedding"]
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return [], 0.0

    # ── Step 2: Build sparse vector for BM25 ────────────────────
    sparse_indices, sparse_values = build_sparse_vector(question)

    # ── Step 3: Dense search ────────────────────────────────────
    dense_results = []
    try:
        dense_results = qdrant.query_points(
            collection_name = QDRANT_COLLECTION,
            query           = query_vec,
            using           = "dense",
            limit           = top_k,
            with_payload    = True,
        ).points
    except Exception as e:
        logger.warning(f"Dense search failed: {e}, trying without 'using' parameter")
        try:
            dense_results = qdrant.query_points(
                collection_name = QDRANT_COLLECTION,
                query           = query_vec,
                limit           = top_k,
                with_payload    = True,
            ).points
        except Exception as e2:
            logger.error(f"Dense search completely failed: {e2}")
            dense_results = []

    # ── Step 4: Sparse search (BM25) ────────────────────────────
    sparse_results = []
    try:
        from qdrant_client.models import SparseVector
        sparse_results = qdrant.query_points(
            collection_name = QDRANT_COLLECTION,
            query           = SparseVector(indices=sparse_indices, values=sparse_values),
            using           = "sparse",
            limit           = top_k,
            with_payload    = True,
        ).points
    except Exception as e:
        logger.warning(f"Sparse search failed: {e}")
        sparse_results = []

    # ── Step 5: Merge dense + sparse scores ─────────────────────
    seen_ids = {}
    for point in dense_results:
        seen_ids[point.id] = {"point": point, "score": point.score * VECTOR_WEIGHT}
    for point in sparse_results:
        if point.id in seen_ids:
            seen_ids[point.id]["score"] += point.score * BM25_WEIGHT
        else:
            seen_ids[point.id] = {"point": point, "score": point.score * BM25_WEIGHT}

    merged     = sorted(seen_ids.values(), key=lambda x: x["score"], reverse=True)
    top_points = [m["point"] for m in merged[:top_k]]

    if not top_points:
        return [], 0.0

    # ── Step 6: Cross-reference expansion ───────────────────────
    all_points   = list(top_points)
    ref_sections = set()
    for point in top_points:
        refs = point.payload.get("references", [])
        for ref in refs[:3]:
            ref_sections.add(ref)

    if ref_sections:
        try:
            from qdrant_client.models import Filter, FieldCondition, MatchAny
            ref_results = qdrant.query_points(
                collection_name = QDRANT_COLLECTION,
                query           = query_vec,
                using           = "dense",
                query_filter    = Filter(
                    must=[FieldCondition(
                        key   = "section",
                        match = MatchAny(any=list(ref_sections))
                    )]
                ),
                limit        = 5,
                with_payload = True,
            ).points
            existing_ids = {p.id for p in all_points}
            for p in ref_results:
                if p.id not in existing_ids:
                    all_points.append(p)
                    existing_ids.add(p.id)
        except Exception as e:
            logger.warning(f"Cross-ref fetch failed: {e}")

    # ── Step 7: Return top chunks by hybrid score (no reranker) ─
    top_score = merged[0]["score"] if merged else 0.0

    chunks = []
    for point in top_points[:RERANK_TOP_K]:
        payload = dict(point.payload)
        payload["rerank_score"] = float(point.score)
        chunks.append(payload)

    return chunks, top_score


# ══════════════════════════════════════════════════════════════════
# CONTEXT BUILDER
# ══════════════════════════════════════════════════════════════════

def build_context(chunks: list[dict]) -> tuple[str, list[dict], list[dict]]:
    context_parts = []
    sources       = []
    images        = []
    seen_sources  = set()

    for i, chunk in enumerate(chunks):
        chunk_type = chunk.get("chunk_type", "text")
        content    = chunk.get("content", "")
        doc_title  = chunk.get("doc_title", "")
        section    = chunk.get("section", "")
        source_url = chunk.get("source_url", "")

        noise_sections = [
            "this will be in effect",
            "important advisory",
            "section index",
        ]
        is_noise = any(n in section.lower() for n in noise_sections)

        if chunk_type in ("text", "table", "section_index"):
            context_parts.append(
                f"[SOURCE: {doc_title} — {section} | URL: {source_url}]\n{content}"
            )
            source_key   = f"{doc_title}|{section}"
            rerank_score = chunk.get("rerank_score", 0)
            if source_key not in seen_sources and rerank_score > 0.4 and not is_noise:
                seen_sources.add(source_key)
                sources.append({
                    "doc":     doc_title,
                    "section": section,
                    "url":     source_url,
                    "type":    "document",
                })

        elif chunk_type == "image":
            image_content = chunk.get("image_content", "")
            context_parts.append(
                f"[IMAGE: {section} | URL: {source_url}]\n"
                f"Image file: {chunk.get('image_file','')}\n"
                f"Content: {image_content or content}"
            )
            if chunk.get("image_file"):
                images.append({
                    "file":    chunk["image_file"],
                    "section": section,
                    "url":     source_url,
                })
            source_key = f"{doc_title}|{section}"
            if source_key not in seen_sources:
                seen_sources.add(source_key)
                sources.append({
                    "doc":     doc_title,
                    "section": section,
                    "url":     source_url,
                    "type":    "image",
                })

        elif chunk_type == "reference_link":
            when_to_refer    = chunk.get("when_to_refer", "")
            what_it_contains = chunk.get("what_it_contains", "")
            link_url         = chunk.get("link_url", source_url)
            context_parts.append(
                f"[REFERENCE LINK: {section} | URL: {link_url}]\n"
                f"Title: {chunk.get('link_text','')}\n"
                f"URL: {link_url}\n"
                f"Contains: {what_it_contains}\n"
                f"Use when: {when_to_refer}"
            )
            sources.append({
                "doc":              doc_title,
                "section":          section,
                "url":              link_url,
                "text":             chunk.get("link_text", ""),
                "what_it_contains": what_it_contains,
                "type":             "link",
                "is_link":          True,
            })

        elif chunk_type == "restricted_doc":
            link_url = chunk.get("link_url", source_url)
            context_parts.append(
                f"[RESTRICTED DOCUMENT: {section} | URL: {link_url}]\n"
                f"URL: {link_url}\n"
                f"Note: {chunk.get('note', 'Requires login to access.')}\n"
                f"Context: {content}"
            )
            sources.append({
                "doc":     doc_title,
                "section": section,
                "url":     link_url,
                "type":    "restricted",
                "note":    chunk.get("note", ""),
            })


    return "\n\n".join(context_parts), sources, images


# ══════════════════════════════════════════════════════════════════
# SSE HELPER
# ══════════════════════════════════════════════════════════════════

def sse(event_type: str, data: dict) -> str:
    return f"data: {json.dumps({'type': event_type, **data})}\n\n"


# ══════════════════════════════════════════════════════════════════
# INGESTION ENDPOINTS
# ══════════════════════════════════════════════════════════════════

@app.post("/ingest/file")
async def ingest_file(file: UploadFile = File(...)):
    import tempfile
    import shutil
    
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Create temp file with proper cleanup
    temp_path = None
    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix=ext, prefix="upload_")
        os.close(temp_fd)
        
        content = await file.read()
        with open(temp_path, "wb") as buffer:
            buffer.write(content)

        return StreamingResponse(
            ingest_file_stream(Path(temp_path), file.filename),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            }
        )
    except Exception as e:
        logger.error(f"Upload error: {e}")
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/ingest/url")
async def ingest_url(req: UrlIngestRequest):
    url = req.url.strip()
    if not url.startswith("http://") and not url.startswith("https://"):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")

    return StreamingResponse(
        ingest_url_stream(url),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
    )


@app.get("/documents")
def get_documents():
    return {"documents": list_indexed_documents()}


@app.delete("/documents/{doc_title}")
def delete_document(doc_title: str):
    success = delete_document_chunks(doc_title)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to delete document '{doc_title}'")
    return {"status": "ok", "deleted": doc_title}


# ══════════════════════════════════════════════════════════════════
# ASK ENDPOINT
# ══════════════════════════════════════════════════════════════════

@app.post("/ask")
async def ask(req: AskRequest):

    async def stream():
        try:
            question = req.question.strip()

            if is_greeting(question):
                yield sse("token", {"text": (
                    "Hello! 👋 I am your AI Document Knowledge Assistant.\n\n"
                    "You can upload files (PDF, DOCX, TXT, MD, Images) or paste web URLs into the Knowledge Base, "
                    "and I will answer your questions directly from your documents!"
                )})
                yield sse("done", {})
                return

            yield sse("status", {"text": "Understanding your question..."})
            rewritten = rewrite_query(question, req.history)

            yield sse("status", {"text": "Searching document database..."})
            chunks, top_score = retrieve_chunks(rewritten)

            if top_score < CONFIDENCE_THRESHOLD and chunks:
                logger.info(f"Low confidence ({top_score:.2f}), broadening search...")
                yield sse("status", {"text": "Searching deeper..."})
                chunks2, score2 = retrieve_chunks(question, top_k=30)
                if score2 > top_score:
                    chunks    = chunks2
                    top_score = score2

            if not chunks:
                yield sse("token", {"text": (
                    "I could not find relevant information in the currently indexed documents.\n"
                    "Please upload a document or paste a webpage URL using the Knowledge Base tool above."
                )})
                yield sse("done", {})
                return

            context_text, sources, images = build_context(chunks)

            messages = [{"role": "system", "content": SYSTEM_PROMPT}]

            for msg in req.history[-MAX_HISTORY:]:
                messages.append({"role": msg.role, "content": msg.content})

            messages.append({
                "role": "user",
                "content": (
                    f"CONTEXT FROM INDEXED DOCUMENTS:\n"
                    f"{context_text}\n\n"
                    f"USER QUESTION: {question}\n\n"
                    f"Instructions:\n"
                    f"- Answer directly and confidently from context above\n"
                    f"- If question is yes/no → start with YES or NO\n"
                    f"- Include ALL relevant URLs/filenames from context inline\n"
                    f"- End with source citation — Source URL/Filename is MANDATORY\n"
                    f"- Never show internal labels like [SOURCE:...] in your answer"
                )
            })

            yield sse("status", {"text": "Generating answer..."})

            stream_resp = llm_call(
                messages   = messages,
                max_tokens = 1200,
                stream     = True,
            )

            for chunk in stream_resp:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield sse("token", {"text": delta.content})
                await asyncio.sleep(0)

            yield sse("sources", {"sources": sources})
            if images:
                yield sse("images", {"images": images})
            yield sse("done", {})

        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield sse("error", {"text": str(e)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":        "keep-alive",
        }
    )


# ══════════════════════════════════════════════════════════════════
# HEALTH CHECK
# ══════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    try:
        info      = qdrant.get_collection(QDRANT_COLLECTION)
        qdrant_ok = f"ok — {info.points_count} points"
    except Exception as e:
        qdrant_ok = f"error: {e}"

    return {
        "status":               "ok",
        "llm_provider":         LLM_PROVIDER,
        "llm_model":            LLM_MODEL,
        "embed_model":          EMBEDDING_MODEL,
        "embed_provider":       "gemini",
        "rerank_model":         "none",
        "qdrant":               qdrant_ok,
        "collection":           QDRANT_COLLECTION,
        "hybrid_search":        f"vector({VECTOR_WEIGHT}) + bm25({BM25_WEIGHT})",
        "retrieval_top_k":      RETRIEVAL_TOP_K_EXPANDED,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)