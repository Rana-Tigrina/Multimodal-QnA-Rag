"""
IITM BS RAG Pipeline — Stage 2: Image Scanner
===============================================
INPUT:  output/images/*.png/jpg       (saved by scraper.py)
        output/image_metadata.json    (saved by scraper.py)

OUTPUT: output/image_metadata.json    (UPDATED with image_content)

What it does:
  - Runs Docling OCR on every image (no skipping)
  - Extracts text, tables, diagrams from each image
  - Detects decorative/blank images automatically
  - Updates image_metadata.json with results
  - Tiny images (<1KB) flagged but still scanned

Run:
  python image_scanner.py
"""

import os
import json
import logging
from pathlib import Path
from PIL import Image as PILImage

# Disable Hugging Face Hub symlinks on Windows to prevent WinError 1314
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from config import (
    IMAGE_METADATA_FILE,
    IMAGES_DIR,
    LOG_LEVEL,
    LOG_FORMAT,
)

# ══════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("image_scanner")

# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════

# Images smaller than this are almost certainly blank/spacer pixels
TINY_IMAGE_BYTES = 500

# If Docling returns fewer characters than this → treat as decorative
MIN_CONTENT_CHARS = 10

# ══════════════════════════════════════════════════════════════════
# DOCLING SETUP
# ══════════════════════════════════════════════════════════════════

def load_docling():
    """
    Load Docling pipeline once — reused for all 60 images.
    INPUT:  nothing
    OUTPUT: DocumentConverter instance
    """
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat

    logger.info("Loading Docling pipeline...")

    # Use OCR-optimized pipeline
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True

    converter = DocumentConverter()
    logger.info("Docling pipeline ready")
    return converter


# ══════════════════════════════════════════════════════════════════
# IMAGE CLASSIFIER
# Decides what type of content the image contains
# ══════════════════════════════════════════════════════════════════

def classify_image_content(text: str, file_size_bytes: int) -> dict:
    """
    INPUT:  extracted text from Docling + file size
    OUTPUT: dict with chunk_type, is_decorative

    chunk_type options:
      image_table    → image contains a table
      image_diagram  → image contains a diagram/flowchart
      image_text     → image contains plain text
      image_empty    → nothing useful extracted
    """
    text = text.strip()

    # Blank or near-blank
    if not text or len(text) < MIN_CONTENT_CHARS:
        return {
            "chunk_type":    "image_empty",
            "is_decorative": True,
        }

    # Tiny file size → likely spacer pixel even if text extracted
    if file_size_bytes < TINY_IMAGE_BYTES:
        return {
            "chunk_type":    "image_empty",
            "is_decorative": True,
        }

    text_lower = text.lower()

    # Table indicators
    table_indicators = [
        "|", "---", "course code", "credits", "marks",
        "percentage", "grade", "fee", "term", "week",
        "total", "minimum", "maximum", "sl.no", "s.no",
    ]
    table_score = sum(1 for ind in table_indicators if ind in text_lower)

    if table_score >= 2 or text.count("|") >= 3:
        return {
            "chunk_type":    "image_table",
            "is_decorative": False,
        }

    # Diagram indicators
    diagram_indicators = [
        "→", "←", "↑", "↓", "flow", "process",
        "step", "level", "foundation", "diploma",
        "degree", "pathway", "arrow",
    ]
    diagram_score = sum(1 for ind in diagram_indicators if ind in text_lower)

    if diagram_score >= 2:
        return {
            "chunk_type":    "image_diagram",
            "is_decorative": False,
        }

    # Has meaningful text
    return {
        "chunk_type":    "image_text",
        "is_decorative": False,
    }


# ══════════════════════════════════════════════════════════════════
# SINGLE IMAGE SCANNER
# ══════════════════════════════════════════════════════════════════

def scan_image(filepath: Path, converter) -> dict:
    """
    INPUT:  path to image file + Docling converter
    OUTPUT: dict with image_content, chunk_type, is_decorative

    Tries Docling first.
    Falls back to basic PIL check if Docling fails.
    Never crashes — always returns something.
    """
    file_size = filepath.stat().st_size

    # ── Tiny file fast path ───────────────────────────────────────
    if file_size < TINY_IMAGE_BYTES:
        logger.info(f"  {filepath.name} → tiny ({file_size}B) flagging as decorative")
        return {
            "image_content": "",
            "chunk_type":    "image_empty",
            "is_decorative": True,
            "scan_method":   "size_check",
            "scan_error":    None,
        }

    # ── Docling scan ──────────────────────────────────────────────
    try:
        result   = converter.convert(str(filepath))
        markdown = result.document.export_to_markdown()
        markdown = markdown.strip()

        classification = classify_image_content(markdown, file_size)

        logger.info(
            f"  {filepath.name} → {classification['chunk_type']} "
            f"({'decorative' if classification['is_decorative'] else f'{len(markdown)} chars'})"
        )

        return {
            "image_content": markdown,
            "chunk_type":    classification["chunk_type"],
            "is_decorative": classification["is_decorative"],
            "scan_method":   "docling",
            "scan_error":    None,
        }

    except Exception as e:
        logger.warning(f"  {filepath.name} → Docling failed: {e}")

        # ── PIL fallback — at least get image dimensions ──────────
        try:
            with PILImage.open(filepath) as img:
                w, h = img.size
                mode = img.mode

            # Very small dimensions → decorative
            if w < 10 or h < 10:
                return {
                    "image_content": "",
                    "chunk_type":    "image_empty",
                    "is_decorative": True,
                    "scan_method":   "pil_fallback",
                    "scan_error":    str(e),
                }

            return {
                "image_content": f"[Image scan failed — {w}x{h}px {mode}. Contains visual content that could not be extracted automatically.]",
                "chunk_type":    "image_text",
                "is_decorative": False,
                "scan_method":   "pil_fallback",
                "scan_error":    str(e),
            }

        except Exception as e2:
            logger.error(f"  {filepath.name} → PIL also failed: {e2}")
            return {
                "image_content": "[Image could not be scanned]",
                "chunk_type":    "image_text",
                "is_decorative": False,
                "scan_method":   "failed",
                "scan_error":    f"Docling: {e} | PIL: {e2}",
            }


def scan_with_gemini_vision(filepath: Path) -> dict | None:
    """
    Fallback OCR and image understanding using Gemini Vision.
    Executes if Docling is not installed or encounters an error.
    """
    try:
        import google.generativeai as genai
        from PIL import Image as PILImage
        from config import GEMINI_API_KEY

        if not GEMINI_API_KEY:
            return None

        genai.configure(api_key=GEMINI_API_KEY)
        models_to_try = [
            "gemini-3.5-flash",
            "gemini-flash-latest",
        ]

        
        # Ensure RGBA / CMYK / Palette images convert safely to RGB
        img = PILImage.open(filepath).convert("RGB")
        prompt = (
            "Extract all text, table data, diagram titles, and numbers from this image. "
            "If it contains a table, output it as a markdown table. "
            "If it is a decorative/blank image or logo with no useful information, reply with DECORATIVE."
        )

        response = None
        for m_name in models_to_try:
            try:
                model = genai.GenerativeModel(m_name)
                response = model.generate_content([prompt, img])
                if response:
                    break
            except Exception:
                continue

        text = response.text.strip() if (response and hasattr(response, "text") and response.text) else ""

        if not text or ("DECORATIVE" in text.upper() and len(text) < 30):
            return {
                "image_content": "",
                "chunk_type": "image_text",
                "is_decorative": True,
                "scan_method": "gemini_vision",
            }

        return {
            "image_content": text,
            "chunk_type": "image_text",
            "is_decorative": False,
            "scan_method": "gemini_vision",
        }
    except Exception as e:
        logger.warning(f"Gemini Vision fallback failed for {filepath.name}: {e}")
        return None



def scan_single_image(filepath: Path) -> dict:
    """
    Modular function to scan a single image file via Docling / Gemini Vision / PIL.
    Used for live file uploads and embedded PDF/DOCX images.
    """
    try:
        converter = load_docling()
        res = scan_image(filepath, converter)
        logger.info(f"🛠️ Scanned '{filepath.name}' using DOCLING OCR")
        return res
    except Exception as e:
        logger.info(f"Docling fallback for {filepath.name} ({e}). Attempting Gemini Vision OCR...")

    # Secondary Fallback: Gemini Vision OCR
    gemini_res = scan_with_gemini_vision(filepath)
    if gemini_res:
        logger.info(f"🌐 Scanned '{filepath.name}' using GEMINI VISION API")
        return gemini_res

    # Tertiary Fallback: PIL image specs
    try:
        with PILImage.open(filepath) as img:
            w, h = img.size
            logger.info(f"🖼️ Scanned '{filepath.name}' using PIL specs ({w}x{h} px)")
            return {
                "image_content": f"[Image file: {filepath.name} ({w}x{h} px)]",
                "chunk_type": "image_text",
                "is_decorative": False,
                "scan_method": "pil_fallback",
            }
    except Exception:
        return {
            "image_content": f"[Image file: {filepath.name}]",
            "chunk_type": "image_text",
            "is_decorative": False,
            "scan_method": "fallback",
        }




# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════


def run():
    print("\n" + "═" * 65)
    print("  IITM BS RAG Pipeline — Stage 2: Image Scanner")
    print("═" * 65)

    # Load image metadata from scraper
    if not IMAGE_METADATA_FILE.exists():
        print(f"\n  ❌ {IMAGE_METADATA_FILE} not found")
        print(f"     Run python scraper.py first")
        return

    with open(IMAGE_METADATA_FILE) as f:
        image_records = json.load(f)

    total = len(image_records)
    print(f"\n  Images to scan: {total}")

    # Check which need scanning
    needs_scan = [r for r in image_records if r.get("needs_ocr", True)]
    already_done = total - len(needs_scan)
    if already_done > 0:
        print(f"  Already scanned: {already_done} (skipping)")

    if not needs_scan:
        print(f"\n  ✅ All images already scanned")
        return

    # Load Docling once
    converter = load_docling()

    # Track stats
    stats = {
        "image_table":   0,
        "image_diagram": 0,
        "image_text":    0,
        "image_empty":   0,
        "scan_errors":   0,
    }

    print(f"\n  Scanning {len(needs_scan)} images...\n")

    for i, record in enumerate(image_records):

        # Skip already scanned
        if not record.get("needs_ocr", True):
            continue

        filename = record.get("filename", "")
        filepath = IMAGES_DIR / filename

        print(f"  [{i+1:02d}/{total}] {filename}")
        print(f"         Section: {record.get('section', '')[:60]}")

        if not filepath.exists():
            logger.warning(f"  File not found: {filepath}")
            record.update({
                "image_content": "[File not found]",
                "chunk_type":    "image_empty",
                "is_decorative": True,
                "needs_ocr":     False,
                "scan_error":    "file_not_found",
            })
            stats["image_empty"] += 1
            continue

        # Scan the image
        result = scan_image(filepath, converter)

        # Update record
        record.update({
            "image_content": result["image_content"],
            "chunk_type":    result["chunk_type"],
            "is_decorative": result["is_decorative"],
            "scan_method":   result["scan_method"],
            "scan_error":    result["scan_error"],
            "needs_ocr":     False,  # mark as done
        })

        # Update stats
        chunk_type = result["chunk_type"]
        stats[chunk_type] = stats.get(chunk_type, 0) + 1
        if result["scan_error"]:
            stats["scan_errors"] += 1

        # Show what was extracted
        content_preview = result["image_content"][:100] if result["image_content"] else ""
        if content_preview:
            print(f"         Content: {content_preview}...")
        print()

        # Save after every image — so crash doesn't lose progress
        with open(IMAGE_METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(image_records, f, indent=2, ensure_ascii=False)

    # Final save
    with open(IMAGE_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(image_records, f, indent=2, ensure_ascii=False)

    # Summary
    print("\n" + "═" * 65)
    print("  ✅ IMAGE SCANNING COMPLETE")
    print("═" * 65)
    print(f"  Total images:    {total}")
    print(f"  Tables found:    {stats.get('image_table', 0)}")
    print(f"  Diagrams found:  {stats.get('image_diagram', 0)}")
    print(f"  Text images:     {stats.get('image_text', 0)}")
    print(f"  Empty/Decorative:{stats.get('image_empty', 0)}")
    print(f"  Scan errors:     {stats.get('scan_errors', 0)}")
    print(f"\n  Updated: {IMAGE_METADATA_FILE}")
    print(f"\n  Next step: python chunker.py")
    print("═" * 65)


if __name__ == "__main__":
    run()