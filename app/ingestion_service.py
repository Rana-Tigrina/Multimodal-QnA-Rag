"""
Universal Multi-Document Ingestion Service
============================================
Handles multi-modal file parsing (PDF, DOCX, TXT, MD, Images) and URL scraping,
executing the full RAG pipeline with real-time SSE progress updates.
"""

import os
import json
import logging
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any

from config import UPLOAD_DIR, ALLOWED_EXTENSIONS, IMAGES_DIR
from scraper import scrape_url
from image_scanner import scan_single_image
from chunker import chunk_document_content, base_chunk
from hyde_generator import generate_hyde_for_chunks
from embedder import embed_chunk_list
from uploader import upload_chunks_batch

logger = logging.getLogger("ingestion_service")


def sse_event(step: str, detail: str, progress: int = 0) -> str:
    """Format SSE progress event."""
    return f"data: {json.dumps({'type': 'progress', 'step': step, 'detail': detail, 'progress': progress})}\n\n"


def parse_pdf_file(filepath: Path) -> tuple[str, list[Path]]:
    """
    Extract text and embedded images from a PDF file.
    """
    import pypdf

    reader = pypdf.PdfReader(filepath)
    text_parts = []
    extracted_images = []

    for i, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"## Page {i + 1}\n{page_text}")
        except Exception as e:
            logger.warning(f"Error extracting text from page {i + 1}: {e}")

        # Extract embedded image objects safely
        try:
            for img_idx, img_obj in enumerate(page.images):
                try:
                    name = getattr(img_obj, "name", f"img_{img_idx}.png")
                    img_name = f"{filepath.stem}_p{i + 1}_{name}"
                    img_path = IMAGES_DIR / img_name
                    with open(img_path, "wb") as f:
                        f.write(img_obj.data)
                    extracted_images.append(img_path)
                except Exception as e:
                    logger.warning(f"Could not extract image {img_idx} from page {i + 1}: {e}")
        except Exception:
            pass

    return "\n\n".join(text_parts), extracted_images



def parse_docx_file(filepath: Path) -> tuple[str, list[Path]]:
    """
    Extract text, tables, and images from a Microsoft Word .docx file.
    """
    import docx

    doc = docx.Document(filepath)
    text_parts = []
    extracted_images = []

    for p in doc.paragraphs:
        if p.text.strip():
            if p.style and p.style.name.startswith("Heading"):
                level = p.style.name.replace("Heading", "").strip() or "1"
                hashes = "#" * min(int(level), 4) if level.isdigit() else "#"
                text_parts.append(f"\n{hashes} {p.text.strip()}")
            else:
                text_parts.append(p.text.strip())

    for table in doc.tables:
        table_lines = []
        for i, row in enumerate(table.rows):
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            table_lines.append("| " + " | ".join(cells) + " |")
            if i == 0:
                table_lines.append("| " + " | ".join(["---"] * len(cells)) + " |")
        if table_lines:
            text_parts.append("\n" + "\n".join(table_lines) + "\n")

    for rel in doc.part.rels.values():
        target_ref_str = str(getattr(rel, "target_ref", ""))
        if "image" in target_ref_str:
            try:
                img_part = rel.target_part
                img_name = f"{filepath.stem}_{Path(target_ref_str).name}"
                img_path = IMAGES_DIR / img_name

                with open(img_path, "wb") as f:
                    f.write(img_part.blob)
                extracted_images.append(img_path)
            except Exception as e:
                logger.warning(f"Could not extract DOCX inline image: {e}")

    return "\n\n".join(text_parts), extracted_images


def ingest_file_stream(filepath: Path, original_filename: str) -> Generator[str, None, None]:
    """
    Process an uploaded file through full RAG pipeline and yield SSE progress events.
    """
    ext = filepath.suffix.lower()
    doc_title = original_filename
    source_url = f"file://{original_filename}"

    yield sse_event("parsing", f"Reading content from {original_filename}...", 10)

    raw_text = ""
    image_paths = []

    try:
        if ext in (".txt", ".md"):
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                raw_text = f.read()
        elif ext == ".pdf":
            raw_text, image_paths = parse_pdf_file(filepath)
        elif ext == ".docx":
            raw_text, image_paths = parse_docx_file(filepath)
        elif ext in (".png", ".jpg", ".jpeg"):
            image_paths = [filepath]

        yield sse_event("chunking", f"Structuring chunks from {doc_title}...", 30)
        chunks = chunk_document_content(raw_text, doc_title, source_url)

        # Process embedded/uploaded images with OCR
        if image_paths:
            yield sse_event("ocr", f"Running Docling OCR on {len(image_paths)} image(s)...", 45)
            for img_path in image_paths:
                res = scan_single_image(img_path)
                if not res.get("is_decorative") and res.get("image_content"):
                    img_chunk = base_chunk(
                        content=res["image_content"],
                        chunk_type="image",
                        heading=f"Image: {img_path.name}",
                        heading_level=2,
                        doc_title=doc_title,
                        section=f"Image in {doc_title}",
                        breadcrumb=f"{doc_title} > Images",
                        source_url=source_url,
                        parent_doc=doc_title,
                    )
                    img_chunk["image_file"] = img_path.name
                    img_chunk["image_content"] = res["image_content"]
                    chunks.append(img_chunk)

        if not chunks:
            yield sse_event("error", "No readable text or images found in file.", 100)
            return

        yield sse_event("hyde", f"Generating HyDE questions for {len(chunks)} chunk(s)...", 60)
        chunks = generate_hyde_for_chunks(chunks)

        yield sse_event("embedding", f"Vectorizing {len(chunks)} chunk(s) via Gemini...", 80)
        chunks = embed_chunk_list(chunks)

        yield sse_event("indexing", "Upserting points to Qdrant...", 95)
        point_count = upload_chunks_batch(chunks)

        yield sse_event("done", f"Successfully indexed {point_count} chunk(s) for '{doc_title}'", 100)
    except Exception as e:
        logger.error(f"Ingestion failed for file {original_filename}: {e}")
        yield sse_event("error", f"Ingestion error: {str(e)}", 100)


def ingest_url_stream(url: str) -> Generator[str, None, None]:
    """
    Scrape a web page URL through full RAG pipeline and yield SSE progress events.
    """
    yield sse_event("scraping", f"Scraping web page content from {url}...", 15)

    try:
        doc_data = scrape_url(url)
        doc_title = doc_data["doc_title"]
        content = doc_data["content"]
        source_url = doc_data["source_url"]

        yield sse_event("chunking", f"Structuring chunks for '{doc_title}'...", 35)
        chunks = chunk_document_content(content, doc_title, source_url)

        if not chunks:
            yield sse_event("error", f"No readable content found at {url}", 100)
            return

        yield sse_event("hyde", f"Generating HyDE questions for {len(chunks)} chunk(s)...", 60)
        chunks = generate_hyde_for_chunks(chunks)

        yield sse_event("embedding", f"Vectorizing {len(chunks)} chunk(s) via Gemini...", 80)
        chunks = embed_chunk_list(chunks)

        yield sse_event("indexing", "Upserting points to Qdrant...", 95)
        point_count = upload_chunks_batch(chunks)

        yield sse_event("done", f"Successfully indexed {point_count} chunk(s) from '{doc_title}'", 100)
    except Exception as e:
        logger.error(f"Ingestion failed for URL {url}: {e}")
        yield sse_event("error", f"Ingestion error: {str(e)}", 100)
