# Multimodal QnA RAG: Image-Aware Intelligent Assistant

![Next.js](https://img.shields.io/badge/Next.js-16.1.6-black?style=for-the-badge&logo=next.js)
![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge&logo=python)
![Qdrant](https://img.shields.io/badge/Qdrant-vector-purple?style=for-the-badge&logo=qdrant)
![Gemini](https://img.shields.io/badge/Gemini-embedding-blueviolet?style=for-the-badge&logo=google)

> **“One image is worth ten thousand words.”**

A cutting-edge Question-Answering system that leverages advanced **Retrieval-Augmented Generation** with **Multimodal Intelligence**. This system can answer complex questions about uploaded documents (PDFs, TXT) AND visual content (Images, Tables) using the power of Gemini AI.

## ✨ Features

- 🧠 **Multimodal Understanding**: Combines text and image analysis for comprehensive context.
- ⚡ **Hybrid Retrieval**: Uses **Semantic Search** + **Keyword Search** for optimal results.
- 🚀 **Smart Indexing**: Automatic chunking and vector embedding of document content.
- 💡 **HyDE Enhancement**: Generates hypothetical answers to improve retrieval accuracy.
- 🎯 **High Precision**: Optimized with embedding filters and cross-encoder reranking.
- 🎨 **Modern UI**: Clean, responsive chat interface built with Next.js 16.

---

## 🚀 Quick Start

### Prerequisites

- **Node.js** 20.x or higher
- **Python** 3.11 or higher
- **Qdrant** instance (Cloud or Local)
- **Gemini API Key**

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd Multimodal-QnA-Rag
    ```

2.  **Backend Setup (Python):**
    ```bash
    cd app
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Frontend Setup (Next.js):**
    ```bash
    cd frontend
    npm install
    ```

4.  **Configuration:**
    - Create a `.env` file in the `app/` directory (see `.env.example`).
    - Add your API keys and Qdrant credentials.

5.  **Run the Application:**
    ```bash
    # Start Backend
    cd app
    python main.py

    # Start Frontend (in a new terminal)
    cd frontend
    npm run dev
    ```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│          FRONTEND (Next.js)                 │
│  - Chat UI                                  │
│  - Document Upload                          │
│  - Image Preview                            │
└─────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────┐
│           BACKEND API (FastAPI)             │
│  - /chat: Handles Q&A logic                │
│  - /ingest: Handles document upload         │
└─────────────────────────────────────────────┘
                     │
         ┌───────────┴────────────┐
         ▼                        ▼
┌─────────────────┐    ┌────────────────────┐
│ Image Scanner   │    │ PDF/Text Loader    │
└─────────────────┘    └────────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────────────────────────────────┐
│ QDRANT DATABASE                             │
│  - Stores text embeddings                   │
│  - Stores image metadata                    │
└─────────────────────────────────────────────┘
