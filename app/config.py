"""
Universal AI Knowledge Assistant — Configuration
=================================================
Centralized settings following 12-Factor principles.
Supports:
  - LLM Cloud: Groq (qwen/qwen3.8-27b, reasoning_effort=none)
  - LLM Local: llama.cpp (bartowski/Ling-3.0-tiny-GGUF:Q4_K_M)
  - Embeddings: 100% Local ONNX all-MiniLM-L6-v2 (384d)
  - Vector DB: ChromaDB (Local persistent)
  - Image OCR: Docling + RapidOCR ONNX (100% Local)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ══════════════════════════════════════════════════════════════════
# PATHS & DIRECTORIES
# ══════════════════════════════════════════════════════════════════

BASE_DIR            = Path(__file__).parent
ROOT_DIR            = BASE_DIR.parent

# Load environment variables (.env in app/ or root)
load_dotenv(dotenv_path=BASE_DIR / ".env")
load_dotenv(dotenv_path=ROOT_DIR / ".env")

UPLOAD_DIR          = BASE_DIR / "uploads"
OUTPUT_DIR          = BASE_DIR / "output"
IMAGES_DIR          = OUTPUT_DIR / "images"
CHROMA_PERSIST_DIR  = BASE_DIR / "chroma_db"
LOCAL_MODELS_DIR    = BASE_DIR / "local_models"

# Ensure runtime directories exist
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS  = {".pdf", ".txt", ".md", ".docx", ".png", ".jpg", ".jpeg"}

# ══════════════════════════════════════════════════════════════════
# LLM CONFIGURATION
# ══════════════════════════════════════════════════════════════════

LLM_PROVIDER        = os.getenv("LLM_PROVIDER", "groq").strip().lower()  # "groq" or "llamacpp"
GROQ_API_KEY        = os.getenv("GROQ_API_KEY", "").strip()
GROQ_MODEL          = "qwen/qwen3.8-27b"

LOCAL_MODEL_REPO    = "bartowski/Ling-3.0-tiny-GGUF"
LOCAL_MODEL_FILE    = "Ling-3.0-tiny-Q4_K_M.gguf"
LOCAL_MODEL_PATH    = LOCAL_MODELS_DIR / LOCAL_MODEL_FILE

LLM_BASE_URL        = "https://api.groq.com/openai/v1"
LLM_API_KEY         = GROQ_API_KEY
LLM_MODEL           = GROQ_MODEL if LLM_PROVIDER == "groq" else f"{LOCAL_MODEL_REPO}:{LOCAL_MODEL_FILE}"

LLM_FALLBACK_MODELS = [
    "llama-3.1-8b-instant",
    "llama3-8b-8192",
    "gemma2-9b-it",
]

# ══════════════════════════════════════════════════════════════════
# EMBEDDING CONFIGURATION — 100% Local (all-MiniLM-L6-v2)
# ══════════════════════════════════════════════════════════════════

EMBEDDING_PROVIDER  = "local"
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"
EMBEDDING_DIM       = 384
EMBEDDING_BATCH     = 32
RERANKER_MODEL      = None
RERANKER_TOP_K      = 5

# ══════════════════════════════════════════════════════════════════
# VECTOR DB & HYBRID RETRIEVAL (ChromaDB + BM25)
# ══════════════════════════════════════════════════════════════════

CHROMA_COLLECTION   = os.getenv("CHROMA_COLLECTION", "knowledge_base").strip() or "knowledge_base"
VECTOR_WEIGHT       = 0.7
BM25_WEIGHT         = 0.3
RETRIEVAL_TOP_K     = 25
RERANK_TOP_K        = 6
CONFIDENCE_THRESHOLD = 0.15
REQUEST_TIMEOUT     = 15

# ══════════════════════════════════════════════════════════════════
# CHUNKER & HYDE CONFIGURATION
# ══════════════════════════════════════════════════════════════════

CHUNK_SIZE          = 512
CHUNK_OVERLAP       = 50
MIN_CHUNK_SIZE      = 15

HYDE_QUESTIONS_PER_CHUNK = 3
HYDE_PROMPT = """You are building a RAG system for document knowledge retrieval.

Given this content from a document:

DOCUMENT: {doc_title}
SECTION: {section}
CONTENT: {content}

Generate exactly {n} questions that a user or student would ask 
whose answer is found directly in this content.

Rules:
- Questions must be answerable ONLY from this content
- ALWAYS include specific entity names, codes, dates, or topics in the question
- Use natural, direct user language
- Cover different aspects of the content
- Do not repeat similar questions
- Return ONLY the questions, one per line, no numbering"""

# ══════════════════════════════════════════════════════════════════
# SYSTEM PROMPT & LOGGING
# ══════════════════════════════════════════════════════════════════

RAG_SYSTEM_PROMPT = """You are an expert AI Document Knowledge Assistant.

Answer questions accurately based ONLY on the provided document context.
Always cite your sources clearly.

Rules:
1. Answer directly and clearly
2. If the answer has a specific number, date, formula, or policy — state it exactly
3. Always end with: Source: [document name, section name]
4. If a relevant link exists in context — always include it inline
5. If you cannot find the answer in context — say exactly:
   "I could not find this information in the indexed documents."
6. Never make up information
7. If answer is in an external link — provide that link directly"""

LOG_LEVEL           = "INFO"
LOG_FORMAT          = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def validate():
    """Verify environment configuration and report status."""
    errors = []
    if LLM_PROVIDER == "groq" and not GROQ_API_KEY:
        errors.append("GROQ_API_KEY environment variable not set")

    if errors:
        print("\n❌ Configuration errors:")
        for e in errors:
            print(f"   → {e}")
        raise SystemExit(1)

    print("\n✅ Config loaded:")
    print(f"   LLM:        {LLM_PROVIDER} → {LLM_MODEL}")
    print(f"   Embeddings: {EMBEDDING_MODEL} (Local ONNX, 384d)")
    print(f"   Vector DB:  ChromaDB (local: {CHROMA_PERSIST_DIR.name}) → Collection: '{CHROMA_COLLECTION}'")
    print(f"   Mode:       {'💻 Local llama.cpp (Ling 3.0 Tiny)' if LLM_PROVIDER == 'llamacpp' else '⚡ Groq Cloud (Qwen 3.8 27B)'}")


if __name__ == "__main__":
    validate()