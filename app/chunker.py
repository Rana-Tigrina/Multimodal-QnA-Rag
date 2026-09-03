"""
Universal AI Knowledge Assistant — Semantic Chunker Module
===========================================================
Splits documents, text, tables, and markdown into semantically rich chunks
with stable IDs, token-aware sliding window overlap, and heading hierarchies.
"""

import re
import hashlib
import logging
from datetime import date

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE, LOG_LEVEL, LOG_FORMAT

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("chunker")

TODAY = date.today().isoformat()
VERSION = "1"


def token_count(text: str) -> int:
    """Approximate token count — 1 token ≈ 4 characters."""
    return max(1, len(text) // 4)


def make_chunk_id(content: str) -> str:
    """Generate a stable 12-char MD5 hash of chunk content."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]


def base_chunk(
    content: str,
    chunk_type: str,
    heading: str,
    heading_level: int,
    doc_title: str,
    section: str,
    breadcrumb: str,
    source_url: str = "",
    parent_doc: str = "",
) -> dict:
    """Build standardized chunk dictionary with enriched embed_text."""
    embed_parts = [
        f"Document: {doc_title}",
        f"Section: {section}",
        content,
    ]
    embed_text = "\n".join(p for p in embed_parts if p.strip())

    return {
        "chunk_id": make_chunk_id(content),
        "chunk_type": chunk_type,
        "content": content,
        "embed_text": embed_text,
        "heading": heading,
        "heading_level": heading_level,
        "doc_title": doc_title,
        "section": section,
        "breadcrumb": breadcrumb,
        "source_url": source_url,
        "parent_doc": parent_doc or doc_title,
        "references": [],
        "hyde_questions": [],
        "created_at": TODAY,
        "version": VERSION,
        "token_count": token_count(content),
    }


def split_into_sections(markdown: str) -> list[dict]:
    """Split markdown into sections at heading boundaries (#, ##, ###)."""
    lines = markdown.splitlines()
    sections = []
    current_lines = []
    current_heading = ""
    current_level = 0

    def flush():
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if content:
            sections.append({
                "heading": current_heading,
                "heading_level": current_level,
                "content": content,
            })
        current_lines = []

    for line in lines:
        match = re.match(r'^(#{1,4})\s+(.+)$', line)
        if match:
            flush()
            current_level = len(match.group(1))
            current_heading = match.group(2).strip()
            current_lines = [line]
            continue
        current_lines.append(line)

    flush()
    return sections


def table_to_plain(table_str: str) -> str:
    """Convert markdown pipe table into readable key-value sentences for embedding."""
    lines = [l.strip() for l in table_str.splitlines() if l.strip()]
    rows = [l for l in lines if l.startswith("|") and "---" not in l]
    if not rows:
        return table_str

    parsed = []
    for row in rows:
        cells = [c.strip() for c in row.split("|") if c.strip()]
        if cells:
            parsed.append(cells)

    if not parsed:
        return table_str

    header = parsed[0]
    out = [" | ".join(header)]
    for row in parsed[1:]:
        pairs = [f"{header[i]}: {val}" for i, val in enumerate(row) if i < len(header)]
        out.append(", ".join(pairs))
    return "\n".join(out)


def split_large_text(text: str, max_tokens: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sentence-aware sliding window splitter for oversized paragraphs."""
    if token_count(text) <= max_tokens:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    chunks = []
    curr = []
    curr_tokens = 0

    for s in sentences:
        s = s.strip()
        if not s:
            continue
        stok = token_count(s)

        if curr_tokens + stok > max_tokens and curr:
            chunks.append(" ".join(curr))
            overlap_curr = []
            overlap_tok = 0
            for prev in reversed(curr):
                ptok = token_count(prev)
                if overlap_tok + ptok <= overlap:
                    overlap_curr.insert(0, prev)
                    overlap_tok += ptok
                else:
                    break
            curr = overlap_curr + [s]
            curr_tokens = overlap_tok + stok
        else:
            curr.append(s)
            curr_tokens += stok

    if curr:
        chunks.append(" ".join(curr))
    return chunks


def chunk_document_content(content: str, doc_title: str, source_url: str = "") -> list[dict]:
    """
    Core entry point: chunk any document, article, or uploaded file into structured chunks.
    Preserves tables, hierarchy, headings, and respects token limits.
    """
    if not content or not content.strip():
        return []

    sections = split_into_sections(content)
    if not sections:
        sections = [{
            "heading": doc_title,
            "heading_level": 1,
            "content": content.strip(),
        }]

    chunks = []
    seen_ids = set()

    for sec in sections:
        heading = sec["heading"] or doc_title
        level = sec["heading_level"] or 1
        sec_content = sec["content"]
        breadcrumb = f"{doc_title} > {heading}" if heading != doc_title else doc_title

        # Check for markdown table
        if "|---" in sec_content or (sec_content.count("|") >= 4 and "\n|" in sec_content):
            plain_table = table_to_plain(sec_content)
            chunk = base_chunk(
                content=plain_table,
                chunk_type="table",
                heading=heading,
                heading_level=level,
                doc_title=doc_title,
                section=heading,
                breadcrumb=breadcrumb,
                source_url=source_url,
            )
            if chunk["chunk_id"] not in seen_ids:
                seen_ids.add(chunk["chunk_id"])
                chunks.append(chunk)
            continue

        # Text paragraphs
        sub_chunks = split_large_text(sec_content, CHUNK_SIZE, CHUNK_OVERLAP)
        for sub in sub_chunks:
            sub = sub.strip()
            if not sub or token_count(sub) < MIN_CHUNK_SIZE:
                continue
            chunk = base_chunk(
                content=sub,
                chunk_type="text",
                heading=heading,
                heading_level=level,
                doc_title=doc_title,
                section=heading,
                breadcrumb=breadcrumb,
                source_url=source_url,
            )
            if chunk["chunk_id"] not in seen_ids:
                seen_ids.add(chunk["chunk_id"])
                chunks.append(chunk)

    # Fallback to keep content if all sub-chunks were under MIN_CHUNK_SIZE
    if not chunks and content.strip():
        chunks.append(base_chunk(
            content=content.strip(),
            chunk_type="text",
            heading=doc_title,
            heading_level=1,
            doc_title=doc_title,
            section=doc_title,
            breadcrumb=doc_title,
            source_url=source_url,
        ))

    return chunks


if __name__ == "__main__":
    sample = "# Test Document\n\nThis is a sample paragraph to test chunking.\n\n## Table Section\n| A | B |\n|---|---|\n| 1 | 2 |"
    out = chunk_document_content(sample, "Test Doc", "http://test.com")
    print(f"Produced {len(out)} chunks:")
    for c in out:
        print(f"- [{c['chunk_type']}] {c['heading']} ({c['token_count']} tokens)")