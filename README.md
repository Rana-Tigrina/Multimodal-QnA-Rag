# Multimodal QnA RAG: Universal Intelligent Document Assistant

![Next.js](https://img.shields.io/badge/Next.js-16.1.6-black?style=for-the-badge&logo=next.js)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![ChromaDB](https://img.shields.io/badge/ChromaDB-local_vector-orange?style=for-the-badge&logo=databricks)
![Docling](https://img.shields.io/badge/Docling-Local_OCR-darkblue?style=for-the-badge)
![llama.cpp](https://img.shields.io/badge/llama.cpp-Local_GGUF-purple?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-Qwen_3.8_27B-orange?style=for-the-badge)

> **"Accurate, fast, and multi-modal knowledge retrieval from documents, web pages, and images."**

A high-performance Question-Answering system leveraging **Advanced Hybrid Retrieval-Augmented Generation** with **Multimodal Intelligence**. This system answers questions from uploaded documents (PDFs, TXT, DOCX, MD), visual content (Images, Scanned PDFs, Tables), and live web URLs using 100% local OCR (**Docling** + **RapidOCR ONNX**), local **all-MiniLM-L6-v2** embeddings, persistent local ChromaDB, and dual LLM inference options (Cloud Groq vs 100% Local llama.cpp).

---

## ✨ Features

- ⚡ **Dual LLM Architecture**:
  - **Cloud:** **Qwen 3.8 27B** via Groq (`reasoning_effort="none"` for instant responses, 4096 tokens).
  - **Local:** **Ling 3.0 Tiny** via llama.cpp (`bartowski/Ling-3.0-tiny-GGUF:Q4_K_M` running offline on CPU).
  - **Dynamic Model Switcher:** Segmented selector directly in the UI TopBar.
- 🧠 **100% Local Image & Document OCR**: Uses **Docling** and **RapidOCR ONNX** for fast, offline text, table, and layout extraction without external cloud vision dependencies.
- 🔍 **Local Hybrid Retrieval**: Combines **Dense Vector Search** (`all-MiniLM-L6-v2`, 384d) + **Sparse BM25 Keyword Search** with Reciprocal Rank Fusion.
- 🗄️ **Local Persistent Vector Store**: Embedded ChromaDB database that survives restarts with zero cloud vector store dependencies.
- 🔗 **Live URL & PDF Ingestion**: Scrapes and parses any web page or web PDF into clean, structured Markdown chunks.
- 💡 **HyDE Enhancement**: Generates hypothetical questions per chunk during ingestion to improve question-to-question vector matching.
- 🎨 **Modern Next.js UI**: Clean dark/light theme, verified source citations, clickable file links, excerpt previews, and knowledge base document management.

---

## 🚀 Quick Start

### Prerequisites
- **Python** 3.11 or higher
- **Node.js** 20.x or higher
- **Groq API Key** (optional, only if using Cloud Qwen 3.8 27B)

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Rana-Tigrina/Multimodal-QnA-Rag.git
cd Multimodal-QnA-Rag

# Install Python dependencies
pip install -r requirements.txt

# For llama-cpp-python CPU prebuilt wheel on Windows (if building from source fails):
# pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --only-binary :all:

# Install Frontend dependencies
cd frontend
npm install
cd ..
```

### 2. Configuration

Create a `.env` file in the project root or `app/`:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
LLM_PROVIDER=groq
CHROMA_COLLECTION=knowledge_base
```

### 3. Run the Application

```bash
# Run Backend (FastAPI on port 8000)
.\run_backend.bat
# or: cd app && uvicorn main:app --reload --port 8000

# Run Frontend (Next.js on port 3000)
cd frontend
npm run dev
```

Open `http://localhost:3000` in your browser.

---

## 💻 Optional: Local Offline Model Setup

To use **Ling 3.0 Tiny** completely offline:
```bash
cd app
python download_local_model.py
```
This downloads `Ling-3.0-tiny-Q4_K_M.gguf` (~4.68 GB) into `app/local_models/`. Once downloaded, select **💻 Ling 3.0 Tiny [Local]** in the TopBar.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────┐
│              FRONTEND (Next.js 16)               │
│  - Chat Interface with Markdown & Citations      │
│  - Model Switcher (Qwen 3.8 27B / Ling 3.0 Tiny) │
│  - Knowledge Base Document Modal (Upload / URL)  │
└──────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────┐
│              BACKEND API (FastAPI)               │
│  - /ask: Streaming SSE with citations            │
│  - /ingest/file: SSE progress document ingest    │
│  - /ingest/url: SSE progress web URL scraper     │
│  - /models: Model readiness & switcher endpoint  │
│  - /documents: Knowledge base management         │
└──────────────────────────────────────────────────┘
        │                        │
        ▼                        ▼
┌──────────────────┐    ┌──────────────────────────┐
│ Local Image OCR  │    │ Ingestion / Chunker      │
│ Docling + Rapid  │    │ Semantic Markdown Parser │
└──────────────────┘    └──────────────────────────┘
        │                        │
        └───────────┬────────────┘
                    ▼
┌──────────────────────────────────────────────────┐
│ CHROMADB (Local Persistent Storage)              │
│  - 100% Local ONNX all-MiniLM-L6-v2 embeddings   │
│  - Hybrid Retrieval: Cosine Vector + BM25        │
└──────────────────────────────────────────────────┘
```

---

## 📄 License
MIT License.
