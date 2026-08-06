"""
IITM BS RAG Pipeline — Stage 3: Chunker
=========================================
INPUT:  output/docs/**/*.md              (from scraper.py)
        output/image_metadata.json       (from image_scanner.py — OCR content)
        output/reference_links.json      (from scraper.py — enriched links)
        output/restricted_links.json     (from scraper.py — blocked docs)

OUTPUT: output/chunks/all_chunks.json

Chunk types produced:
  text           → regular paragraph content
  table          → markdown table content
  image          → OCR'd image content (from image_metadata.json)
  reference_link → enriched external link metadata
  restricted_doc → blocked doc metadata
  section_index  → section navigation index per document

Chunk schema (every chunk):
  chunk_id       → stable MD5 hash of content (not integer — survives re-runs)
  chunk_type     → one of 6 types above
  content        → actual text shown to LLM
  embed_text     → heading + breadcrumb + content (richer retrieval signal)
  heading        → current section heading
  heading_level  → 1-4
  doc_title      → source document title
  section        → section name
  breadcrumb     → full path e.g. "Programme Guide > Fees > 5.1"
  source_url     → exact Google Doc link
  parent_doc     → root document this came from
  references     → list of chunk_ids cross-referenced in this chunk
  hyde_questions → [] — filled in by hyde_generator.py
  created_at     → ISO date string
  version        → "1"
  token_count    → approximate token count

Extra fields per type:
  image          → image_file, image_content, is_decorative, scan_method
  reference_link → link_url, link_text, what_it_contains, when_to_refer,
                   category, access, found_in_doc
  restricted_doc → link_url, skip_reason, note, access, found_in_doc

Run:
  python chunker.py
"""

import json
import re
import hashlib
from pathlib import Path
from datetime import date

from config import (
    DOCS_DIR,
    ALL_CHUNKS_FILE, CHUNKS_DIR,
    IMAGE_METADATA_FILE,
    REFERENCE_LINKS_FILE,
    RESTRICTED_LINKS_FILE,
    CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_SIZE,
    CHUNK_TYPES,
    LOG_LEVEL, LOG_FORMAT,
)


import logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("chunker")

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════

TODAY    = date.today().isoformat()
VERSION  = "1"

def token_count(text: str) -> int:
    """Approximate token count — 1 token ≈ 4 chars."""
    return len(text) // 4


def make_chunk_id(content: str) -> str:
    """
    Stable MD5 hash of content.
    Using full 12-char prefix — stable across re-runs
    unlike sequential integers which break on incremental updates.
    """
    return hashlib.md5(content.encode()).hexdigest()[:12]


def base_chunk(
    content:       str,
    chunk_type:    str,
    heading:       str,
    heading_level: int,
    doc_title:     str,
    section:       str,
    breadcrumb:    str,
    source_url:    str,
    parent_doc:    str,
) -> dict:
    """
    Build the standard chunk dict that every chunk type shares.
    hyde_questions and references are always [] at this stage —
    hyde_generator.py fills hyde_questions later.
    """
    if chunk_type == "table":
        rows = [r for r in content.split(chr(10)) if r.strip().startswith("|") and "---" not in r]
        if len(rows) >= 2:
            hdr = " ".join(c.strip() for c in rows[0].split("|") if c.strip())
            val = " ".join(c.strip() for c in rows[1].split("|") if c.strip())
            table_summary = hdr + ": " + val
        else:
            table_summary = content[:200]
        embed_text = ". ".join(filter(None, [heading, breadcrumb, table_summary]))
    else:
        embed_text = ". ".join(filter(None, [heading, breadcrumb, content]))

    return {
        "chunk_id":      make_chunk_id(content),
        "chunk_type":    chunk_type,
        "content":       content,
        "embed_text":    embed_text,
        "heading":       heading,
        "heading_level": heading_level,
        "doc_title":     doc_title,
        "section":       section,
        "breadcrumb":    breadcrumb,
        "source_url":    source_url,
        "parent_doc":    parent_doc,
        "references":    [],          # cross-reference chunk_ids — detected below
        "hyde_questions": [],         # filled by hyde_generator.py
        "created_at":    TODAY,
        "version":       VERSION,
        "token_count":   token_count(content),
    }


# ══════════════════════════════════════════════════════════════════
# CROSS-REFERENCE DETECTOR
# Finds section references within content and links chunk_ids
# ══════════════════════════════════════════════════════════════════

def detect_references(content: str, all_sections: list[str]) -> list[str]:
    """
    INPUT:  chunk content + list of known section names
    OUTPUT: list of section names mentioned in content

    Looks for patterns like:
      "see section 5.1", "refer to 3.2", "as described in Fees"
    These become the references[] field — used at retrieval time
    to fetch cross-referenced chunks alongside primary chunks.
    """
    refs = []
    content_lower = content.lower()

    # Pattern: "section X.Y" or "refer to X.Y"
    section_refs = re.findall(
        r'(?:section|refer to|see|as per|refer)\s+([\d]+\.[\d]+(?:\.[\d]+)?)',
        content_lower
    )
    refs.extend(section_refs)

    # Match against known section names
    for sec in all_sections:
        if len(sec) > 4 and sec.lower() in content_lower:
            if sec not in refs:
                refs.append(sec)

    return list(set(refs))[:10]  # cap at 10 refs per chunk


def chunk_document_content(content: str, doc_title: str, source_url: str = "") -> list[dict]:
    """
    Modular function to chunk any text/markdown content into structured chunks.
    Used for live document and URL ingestion.
    """
    sections = split_into_sections(content)
    
    # Fallback if markdown had no section headers
    if not sections and content.strip():
        sections = [{
            "content": content.strip(),
            "heading": doc_title,
            "heading_level": 1,
            "meta": {"doc": doc_title, "section": doc_title, "breadcrumb": doc_title, "url": source_url}
        }]

    all_section_names = [s["heading"] for s in sections]
    chunks = build_chunks_from_sections(sections, doc_title, source_url, doc_title, all_section_names)
    
    sec_idx = build_section_index(sections, doc_title, source_url, doc_title)
    if sec_idx:
        chunks.append(sec_idx)

    # Ensure minimum token fallback chunking for very small/single text docs
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
            parent_doc=doc_title,
        ))

    return chunks



# ══════════════════════════════════════════════════════════════════
# FRONTMATTER PARSER
# ══════════════════════════════════════════════════════════════════

def parse_frontmatter(content: str) -> dict:
    """
    INPUT:  full markdown string
    OUTPUT: dict with doc_title, source_url, parent_doc

    Reads the --- frontmatter block written by scraper.py.
    """
    result = {"doc_title": "", "source_url": "", "parent_doc": ""}
    fm_match = re.search(r'^---\n(.*?)\n---', content, re.DOTALL)
    if not fm_match:
        return result
    for line in fm_match.group(1).split("\n"):
        if line.startswith("doc_title:"):
            result["doc_title"] = line.split(":", 1)[1].strip()
        elif line.startswith("source_url:"):
            result["source_url"] = line.split(":", 1)[1].strip()
        elif line.startswith("parent_doc:"):
            result["parent_doc"] = line.split(":", 1)[1].strip()
    return result


# ══════════════════════════════════════════════════════════════════
# META COMMENT PARSER
# ══════════════════════════════════════════════════════════════════

def parse_meta_comment(line: str) -> dict:
    """
    Parse: <!-- meta | doc:X | section:Y | breadcrumb:Z | url:W -->
    Returns dict with keys: doc, section, breadcrumb, url
    """
    line = re.sub(r'<!--\s*meta\s*', '', line)
    line = re.sub(r'\s*-->', '', line)
    meta = {}
    for part in line.split("|"):
        part = part.strip()
        if ":" in part:
            key, _, value = part.partition(":")
            meta[key.strip()] = value.strip()
    return meta


# ══════════════════════════════════════════════════════════════════
# SECTION SPLITTER
# ══════════════════════════════════════════════════════════════════

def split_into_sections(markdown: str) -> list[dict]:
    """
    Split markdown into sections at heading boundaries.
    Meta comment belongs to the NEXT heading (pending_meta pattern).

    INPUT:  full markdown string
    OUTPUT: list of section dicts with content, heading, meta
    """
    lines    = markdown.split("\n")
    sections = []

    pending_meta          = {}
    current_meta          = {}
    current_lines         = []
    current_heading       = ""
    current_heading_level = 0

    def flush_section():
        nonlocal current_lines
        content = "\n".join(current_lines).strip()
        if content and current_heading:
            sections.append({
                "content":       content,
                "heading":       current_heading,
                "heading_level": current_heading_level,
                "meta":          dict(current_meta),
            })
        current_lines = []

    for line in lines:
        if line.strip().startswith("<!-- meta"):
            pending_meta = parse_meta_comment(line)
            continue

        heading_match = re.match(r'^(#{1,4})\s+(.+)$', line)
        if heading_match:
            flush_section()
            current_heading_level = len(heading_match.group(1))
            current_heading       = heading_match.group(2).strip()
            current_meta          = dict(pending_meta)
            pending_meta          = {}
            current_lines         = [line]
            continue

        current_lines.append(line)

    flush_section()
    return sections


# ══════════════════════════════════════════════════════════════════
# SECTION INDEX BUILDER
# One section_index chunk per document — navigation map
# ══════════════════════════════════════════════════════════════════

def build_section_index(sections: list[dict], doc_title: str, source_url: str, parent_doc: str) -> dict | None:
    """
    INPUT:  all sections from one document
    OUTPUT: one section_index chunk listing all headings

    This allows RAG to answer "what topics does X document cover?"
    HyDE skips this chunk type (see config.HYDE_SKIP_TYPES).
    """
    if not sections:
        return None

    lines = [f"# Section Index: {doc_title}\n"]
    for sec in sections:
        indent = "  " * (sec["heading_level"] - 1)
        lines.append(f"{indent}- {sec['heading']}")

    content = "\n".join(lines)

    chunk = base_chunk(
        content       = content,
        chunk_type    = "section_index",
        heading       = f"Section Index: {doc_title}",
        heading_level = 1,
        doc_title     = doc_title,
        section       = "Section Index",
        breadcrumb    = doc_title,
        source_url    = source_url,
        parent_doc    = parent_doc,
    )
    return chunk


# ══════════════════════════════════════════════════════════════════
# TABLE → PLAIN TEXT CONVERTER
# Cross-encoder reranker scores pipe tables very poorly (-4 to -5)
# Converting to readable sentences fixes this
# ══════════════════════════════════════════════════════════════════

def table_to_plain(table: str) -> str:
    """
    Convert markdown pipe table to readable plain text.
    
    Input:
      | Level | Credits | Total |
      | --- | --- | --- |
      | Foundation | 32 | 48000 |
    
    Output:
      Level | Credits | Total
      Foundation | 32 | 48000
      Foundation + One Diploma | 59 | 129000
    """
    lines = [l.strip() for l in table.splitlines() if l.strip()]
    rows  = [l for l in lines if l.startswith("|") and "---" not in l]
    
    if not rows:
        return table
    
    # Parse each row into cells
    parsed = []
    for row in rows:
        cells = [c.strip() for c in row.split("|") if c.strip()]
        if cells:
            parsed.append(cells)
    
    if not parsed:
        return table
    
    # Build readable output: header on first line, then each data row
    result_lines = []
    header = parsed[0]
    result_lines.append(" | ".join(header))
    
    for row in parsed[1:]:
        # Pair header with value for readability
        pairs = []
        for i, val in enumerate(row):
            if i < len(header):
                pairs.append(f"{header[i]}: {val}")
            else:
                pairs.append(val)
        result_lines.append(", ".join(pairs))
    
    return "\n".join(result_lines)


def split_large_text(text: str, max_tokens: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Recursive sentence/line splitter for oversized paragraphs or unformatted documents.
    Preserves context with sentence sliding window overlap.
    """
    if token_count(text) <= max_tokens:
        return [text]

    # Split by sentences or line breaks
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    chunks = []
    current_sentences = []
    current_tokens = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        stokens = token_count(sentence)

        if current_tokens + stokens > max_tokens and current_sentences:
            chunks.append(" ".join(current_sentences))
            # Sliding window overlap: keep last sentences up to overlap token limit
            overlap_sentences = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                s_tok = token_count(s)
                if overlap_tokens + s_tok <= overlap:
                    overlap_sentences.insert(0, s)
                    overlap_tokens += s_tok
                else:
                    break
            current_sentences = overlap_sentences + [sentence]
            current_tokens = overlap_tokens + stokens
        else:
            current_sentences.append(sentence)
            current_tokens += stokens

    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


# ══════════════════════════════════════════════════════════════════
# CORE CHUNK BUILDER
# Processes one section into one or more chunks
# ══════════════════════════════════════════════════════════════════

def build_chunks_from_sections(
    sections:   list[dict],
    doc_title:  str,
    source_url: str,
    parent_doc: str,
    all_section_names: list[str],
) -> list[dict]:
    """
    INPUT:  sections from one document + context
    OUTPUT: list of chunks (text, table, image types)
    """
    chunks      = []
    seen_hashes = set()

    def emit(content: str, chunk_type: str, section: dict, extra: dict = {}) -> dict | None:
        content = content.strip()
        if not content or token_count(content) < MIN_CHUNK_SIZE:
            return None

        chunk_id = make_chunk_id(content)
        if chunk_id in seen_hashes:
            return None
        seen_hashes.add(chunk_id)

        meta    = section.get("meta", {})
        heading = section.get("heading", "")

        chunk_doc        = meta.get("doc", doc_title)
        chunk_section    = meta.get("section", heading)
        chunk_breadcrumb = meta.get("breadcrumb", f"{doc_title} > {heading}")
        chunk_url        = meta.get("url", source_url)

        c = base_chunk(
            content       = content,
            chunk_type    = chunk_type,
            heading       = heading,
            heading_level = section.get("heading_level", 0),
            doc_title     = chunk_doc,
            section       = chunk_section,
            breadcrumb    = chunk_breadcrumb,
            source_url    = chunk_url,
            parent_doc    = parent_doc,
        )
        c.update(extra)

        # Detect cross-references
        c["references"] = detect_references(content, all_section_names)

        return c

    for section in sections:
        content = section["content"]

        # ── IMAGE blocks — placeholder handling ──
        if "[IMAGE:" in content:
            clean = re.sub(
                r'\[IMAGE:[^\]]+\].*?(?=\[IMAGE:|\Z)',
                '', content, flags=re.DOTALL
            ).strip()
            if clean and token_count(clean) >= MIN_CHUNK_SIZE:
                chunk = emit(clean, "text", section)
                if chunk:
                    chunks.append(chunk)
            continue

        # ── TABLE chunks — kept whole with plain-text conversion ──
        has_table = "|---" in content or "|---|" in content or (
            content.count("|") >= 4 and "\n|" in content
        )
        if has_table:
            parts = re.split(r'(\n\|[^\n]+\|(?:\n\|[^\n]+\|)*)', content)
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                is_table  = part.startswith("|") or "|---|" in part or "|---" in part
                chunk_type = "table" if is_table else "text"
                if is_table:
                    plain = table_to_plain(part)
                    chunk = emit(plain, chunk_type, section)
                else:
                    chunk = emit(part, chunk_type, section)
                if chunk:
                    chunks.append(chunk)
            continue

        # ── TEXT chunks — token-aware paragraph & sentence sliding window ──
        tokens = token_count(content)

        if tokens <= CHUNK_SIZE:
            chunk = emit(content, "text", section)
            if chunk:
                chunks.append(chunk)
        else:
            paragraphs     = re.split(r'\n\n+', content)
            current_batch  = []
            current_tokens = 0

            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                para_tokens = token_count(para)

                # Oversized paragraph handling
                if para_tokens > CHUNK_SIZE:
                    if current_batch:
                        batch_content = "\n\n".join(current_batch)
                        chunk = emit(batch_content, "text", section)
                        if chunk:
                            chunks.append(chunk)
                        current_batch  = []
                        current_tokens = 0

                    sub_chunks = split_large_text(para, CHUNK_SIZE, CHUNK_OVERLAP)
                    for sub in sub_chunks:
                        chunk = emit(sub, "text", section)
                        if chunk:
                            chunks.append(chunk)

                elif current_tokens + para_tokens > CHUNK_SIZE and current_batch:
                    batch_content = "\n\n".join(current_batch)
                    chunk = emit(batch_content, "text", section)
                    if chunk:
                        chunks.append(chunk)
                    current_batch  = [para]
                    current_tokens = para_tokens
                else:
                    current_batch.append(para)
                    current_tokens += para_tokens

            if current_batch:
                chunk = emit("\n\n".join(current_batch), "text", section)
                if chunk:
                    chunks.append(chunk)

    return chunks



# ══════════════════════════════════════════════════════════════════
# IMAGE CHUNKS — from image_metadata.json (OCR content)
# This is the key fix: images use Docling OCR content, not markdown placeholder
# ══════════════════════════════════════════════════════════════════

def load_image_chunks() -> list[dict]:
    """
    INPUT:  output/image_metadata.json (written by image_scanner.py)
    OUTPUT: list of image chunks with real OCR content

    Skips decorative images (is_decorative=True).
    Uses image_content from Docling OCR as the chunk content.
    Falls back to surrounding context if OCR content is empty.
    """
    if not IMAGE_METADATA_FILE.exists():
        logger.warning(f"image_metadata.json not found — no image chunks")
        return []

    with open(IMAGE_METADATA_FILE) as f:
        image_records = json.load(f)

    chunks     = []
    seen_hashes = set()
    skipped    = 0

    for rec in image_records:
        # Skip decorative / empty images
        # is_decorative can be True, False, or None (unscanned) — only skip True
        if rec.get("is_decorative") is True:
            skipped += 1
            continue

        image_content = (rec.get("image_content") or "").strip()
        surrounding   = rec.get("surrounding_before", "") or ""
        section       = rec.get("section", "")
        doc_title     = rec.get("doc_title", "")
        source_url    = rec.get("source_url", "")
        filename      = rec.get("filename", "")
        breadcrumb    = rec.get("breadcrumb", doc_title)
        scan_method   = rec.get("scan_method", "")

        # Build content — OCR text is primary, surrounding context is secondary
        if image_content:
            content = f"[Image: {filename}]\nSection: {section}\n\n{image_content}"
        elif surrounding:
            content = f"[Image: {filename}]\nSection: {section}\nContext: {surrounding[:200]}"
        else:
            skipped += 1
            continue

        chunk_id = make_chunk_id(content)
        if chunk_id in seen_hashes:
            continue
        seen_hashes.add(chunk_id)

        embed_text = ". ".join(filter(None, [
            section, breadcrumb, image_content or surrounding[:200]
        ]))

        chunks.append({
            "chunk_id":       chunk_id,
            "chunk_type":     "image",   # always "image" — image_text/table/diagram stored in scan_method
            "content":        content,
            "embed_text":     embed_text,
            "heading":        section,
            "heading_level":  0,
            "doc_title":      doc_title,
            "section":        section,
            "breadcrumb":     breadcrumb,
            "source_url":     source_url,
            "parent_doc":     doc_title,
            "references":     [],
            "hyde_questions": [],
            "created_at":     TODAY,
            "version":        VERSION,
            "token_count":    token_count(content),
            # image-specific fields
            "image_file":     filename,
            "image_content":  image_content,
            "image_type":     rec.get("chunk_type", "image_text"),  # image_text/image_table/image_diagram
            "is_decorative":  False,
            "scan_method":    scan_method,
        })

    logger.info(f"Image chunks: {len(chunks)} built, {skipped} decorative skipped")
    return chunks


# ══════════════════════════════════════════════════════════════════
# REFERENCE LINK CHUNKS — from reference_links.json (enriched)
# Key fix: uses what_it_contains + when_to_refer from classify_link()
# ══════════════════════════════════════════════════════════════════

def load_reference_link_chunks() -> list[dict]:
    """
    INPUT:  output/reference_links.json (written by scraper.py)
    OUTPUT: list of reference_link chunks

    Key fix vs old version:
      - Uses enriched fields: title, what_it_contains, when_to_refer, category
      - embed_text now includes when_to_refer for better retrieval
      - found_in_doc field added
    """
    if not REFERENCE_LINKS_FILE.exists():
        logger.warning("reference_links.json not found")
        return []

    with open(REFERENCE_LINKS_FILE) as f:
        refs = json.load(f)

    chunks     = []
    seen_hashes = set()

    for ref in refs:
        title            = ref.get("title", ref.get("link_text", ""))
        what_it_contains = ref.get("what_it_contains", "")
        when_to_refer    = ref.get("when_to_refer", "")
        url              = ref.get("url", "")
        section          = ref.get("section", "")
        breadcrumb       = ref.get("breadcrumb", "")
        found_in_doc     = ref.get("found_in_doc", "")
        surrounding      = ref.get("surrounding_text", "")
        category         = ref.get("category", "")
        access           = ref.get("access", "public")
        link_text        = ref.get("link_text", "")

        content = (
            f"Reference: {title}\n"
            f"URL: {url}\n"
            f"What it contains: {what_it_contains}\n"
            f"When to refer: {when_to_refer}\n"
            f"Found in: {found_in_doc}, section: {section}"
        )

        # embed_text includes when_to_refer so student questions match
        embed_text = ". ".join(filter(None, [
            title, when_to_refer, what_it_contains, section, breadcrumb
        ]))

        chunk_id = make_chunk_id(url)  # URL is stable identity
        if chunk_id in seen_hashes:
            continue
        seen_hashes.add(chunk_id)

        chunks.append({
            "chunk_id":       chunk_id,
            "chunk_type":     "reference_link",
            "content":        content,
            "embed_text":     embed_text,
            "heading":        section,
            "heading_level":  0,
            "doc_title":      found_in_doc,
            "section":        section,
            "breadcrumb":     breadcrumb,
            "source_url":     url,
            "parent_doc":     found_in_doc,
            "references":     [],
            "hyde_questions": [],
            "created_at":     TODAY,
            "version":        VERSION,
            "token_count":    token_count(content),
            # reference_link-specific fields
            "link_url":          url,
            "link_text":         link_text,
            "what_it_contains":  what_it_contains,
            "when_to_refer":     when_to_refer,
            "category":          category,
            "access":            access,
            "found_in_doc":      found_in_doc,
            "surrounding_text":  surrounding,
        })

    logger.info(f"Reference link chunks: {len(chunks)}")
    return chunks


# ══════════════════════════════════════════════════════════════════
# RESTRICTED DOC CHUNKS — from restricted_links.json
# Key fix: these were completely missing in old chunker
# ══════════════════════════════════════════════════════════════════

def load_restricted_doc_chunks() -> list[dict]:
    """
    INPUT:  output/restricted_links.json (written by scraper.py)
    OUTPUT: list of restricted_doc chunks

    These allow RAG to answer questions about docs it couldn't scrape.
    Instead of silence, it can say "this info is in X, login required".
    """
    if not RESTRICTED_LINKS_FILE.exists():
        logger.warning("restricted_links.json not found")
        return []

    with open(RESTRICTED_LINKS_FILE) as f:
        restricted = json.load(f)

    chunks     = []
    seen_hashes = set()

    for rec in restricted:
        url          = rec.get("url", "")
        section      = rec.get("section", "")
        breadcrumb   = rec.get("breadcrumb", "")
        found_in_doc = rec.get("found_in_doc", "")
        surrounding  = rec.get("surrounding_text", "")
        skip_reason  = rec.get("skip_reason", "restricted")
        note         = rec.get("note", "This document requires IITM login to access.")
        source_url   = rec.get("source_url", "")

        content = (
            f"Restricted document\n"
            f"URL: {url}\n"
            f"Found in: {found_in_doc}, section: {section}\n"
            f"Context: {surrounding[:200]}\n"
            f"Note: {note}"
        )

        # embed_text fallback to URL if all metadata fields are empty
        embed_parts = list(filter(None, [section, breadcrumb, surrounding[:200]]))
        embed_text  = ". ".join(embed_parts) if embed_parts else f"Restricted IITM document: {url}"

        chunk_id = make_chunk_id(url)
        if chunk_id in seen_hashes:
            continue
        seen_hashes.add(chunk_id)

        chunks.append({
            "chunk_id":       chunk_id,
            "chunk_type":     "restricted_doc",
            "content":        content,
            "embed_text":     embed_text,
            "heading":        section,
            "heading_level":  0,
            "doc_title":      found_in_doc,
            "section":        section,
            "breadcrumb":     breadcrumb,
            "source_url":     url,
            "parent_doc":     found_in_doc,
            "references":     [],
            "hyde_questions": [],
            "created_at":     TODAY,
            "version":        VERSION,
            "token_count":    token_count(content),
            # restricted_doc-specific fields
            "link_url":     url,
            "skip_reason":  skip_reason,
            "note":         note,
            "access":       "restricted",
            "found_in_doc": found_in_doc,
        })

    logger.info(f"Restricted doc chunks: {len(chunks)}")
    return chunks


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run():
    print("\n" + "═" * 65)
    print("  IITM BS RAG Pipeline — Stage 3: Chunker")
    print("═" * 65)

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks = []

    # ── Process all markdown docs ─────────────────────────────────
    md_files = list(DOCS_DIR.rglob("*.md"))
    print(f"\n  Found {len(md_files)} markdown files\n")

    # Collect all section names first — used for cross-reference detection
    all_section_names = []
    for md_path in md_files:
        with open(md_path, encoding="utf-8") as f:
            raw = f.read()
        for m in re.finditer(r'^#{1,4}\s+(.+)$', raw, re.MULTILINE):
            all_section_names.append(m.group(1).strip())

    doc_chunks = []

    for md_path in md_files:
        print(f"  📄 {md_path.name}")
        with open(md_path, encoding="utf-8") as f:
            raw = f.read()

        fm        = parse_frontmatter(raw)
        doc_title  = fm["doc_title"] or md_path.stem.replace("_", " ").title()
        source_url = fm["source_url"]
        parent_doc = fm["parent_doc"] or doc_title

        sections = split_into_sections(raw)

        # Section index chunk — one per document
        idx_chunk = build_section_index(sections, doc_title, source_url, parent_doc)
        if idx_chunk:
            doc_chunks.append(idx_chunk)

        # Text/table chunks
        chunks = build_chunks_from_sections(
            sections, doc_title, source_url, parent_doc, all_section_names
        )
        doc_chunks.extend(chunks)

        type_counts = {}
        for c in chunks:
            type_counts[c["chunk_type"]] = type_counts.get(c["chunk_type"], 0) + 1
        print(f"     → {len(chunks)} chunks {type_counts}")

    all_chunks.extend(doc_chunks)

    # ── Image chunks (OCR content from image_scanner.py) ──────────
    image_chunks = load_image_chunks()
    all_chunks.extend(image_chunks)
    print(f"\n  🖼  Image chunks:          {len(image_chunks)}")

    # ── Reference link chunks (enriched) ──────────────────────────
    ref_chunks = load_reference_link_chunks()
    all_chunks.extend(ref_chunks)
    print(f"  🔗 Reference link chunks:  {len(ref_chunks)}")

    # ── Restricted doc chunks (NEW) ───────────────────────────────
    restricted_chunks = load_restricted_doc_chunks()
    all_chunks.extend(restricted_chunks)
    print(f"  🔒 Restricted doc chunks:  {len(restricted_chunks)}")

    # ── Global deduplication ──────────────────────────────────────
    seen_ids    = set()
    deduped     = []
    duplicates  = 0
    for c in all_chunks:
        if c["chunk_id"] not in seen_ids:
            seen_ids.add(c["chunk_id"])
            deduped.append(c)
        else:
            duplicates += 1

    all_chunks = deduped
    if duplicates:
        print(f"  ♻️  Duplicates removed:     {duplicates}")

    # ── Validate all chunks have required fields ──────────────────
    required = ["chunk_id", "chunk_type", "content", "embed_text",
                "source_url", "hyde_questions", "created_at", "version"]
    issues = 0
    for c in all_chunks:
        for field in required:
            if field not in c:
                logger.warning(f"Missing field '{field}' in chunk {c.get('chunk_id','?')}")
                issues += 1
        if not c.get("embed_text", "").strip():
            logger.warning(f"Empty embed_text in chunk {c.get('chunk_id','?')}")
            issues += 1

    # ── Save ──────────────────────────────────────────────────────
    with open(ALL_CHUNKS_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────
    total       = len(all_chunks)
    type_totals = {}
    for c in all_chunks:
        t = c["chunk_type"]
        type_totals[t] = type_totals.get(t, 0) + 1

    avg_tokens  = sum(c["token_count"] for c in all_chunks) // max(total, 1)
    with_refs   = sum(1 for c in all_chunks if c.get("references"))

    print(f"\n  Sample chunk:")
    if all_chunks:
        s = all_chunks[0]
        print(f"    chunk_id:    {s['chunk_id']}")
        print(f"    type:        {s['chunk_type']}")
        print(f"    doc:         {s['doc_title']}")
        print(f"    section:     {s['section']}")
        print(f"    breadcrumb:  {s['breadcrumb']}")
        print(f"    parent_doc:  {s['parent_doc']}")
        print(f"    source_url:  {s['source_url'][:60]}")
        print(f"    embed_text:  {s['embed_text'][:80]}...")
        print(f"    references:  {s['references']}")
        print(f"    created_at:  {s['created_at']}")
        print(f"    version:     {s['version']}")

    print(f"\n  {'═' * 40}")
    print(f"  Total chunks:     {total}")
    for t in CHUNK_TYPES:
        count = type_totals.get(t, 0)
        print(f"    {t:20s}: {count}")
    print(f"  Avg token count:  {avg_tokens}")
    print(f"  With references:  {with_refs}")
    print(f"  Validation issues:{issues}")
    print(f"  Saved to:         {ALL_CHUNKS_FILE}")
    print(f"\n  Next step: python hyde_generator.py")
    print("═" * 65)


if __name__ == "__main__":
    run()