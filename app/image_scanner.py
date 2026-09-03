"""
Universal AI Knowledge Assistant — Local Image OCR Scanner
===========================================================
100% Local image text and table recognition.
- Primary: Docling DocumentConverter (layout and markdown table extraction)
- Secondary Fallback: RapidOCR ONNX Engine (PP-OCRv6, ultra-fast CPU inference)
- Tertiary Fallback: PIL metadata specifications
"""

import os
import logging
import concurrent.futures
from pathlib import Path
from typing import Optional, List

# Disable Hugging Face Hub symlinks on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

from config import IMAGES_DIR, LOG_LEVEL, LOG_FORMAT

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("image_scanner")

MIN_CONTENT_CHARS = 10

_docling_converter = None
_rapidocr_engine = None


def load_docling():
    """Load Docling pipeline singleton with OCR enabled."""
    global _docling_converter
    if _docling_converter is not None:
        return _docling_converter

    logger.info("Loading Docling pipeline with OCR...")
    try:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions

        pipeline_options = PdfPipelineOptions(do_ocr=True)
        _docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )
    except Exception as e:
        logger.warning(f"Could not load custom PdfPipelineOptions ({e}), loading standard DocumentConverter...")
        from docling.document_converter import DocumentConverter
        _docling_converter = DocumentConverter()

    logger.info("Docling pipeline ready")
    return _docling_converter


def load_rapidocr():
    """Load RapidOCR singleton engine on ONNXRuntime for fast local OCR fallback."""
    global _rapidocr_engine
    if _rapidocr_engine is None:
        try:
            from rapidocr import RapidOCR
            _rapidocr_engine = RapidOCR()
            logger.info("RapidOCR local ONNX engine ready")
        except Exception as e:
            logger.warning(f"Could not initialize RapidOCR: {e}")
            return None
    return _rapidocr_engine


def scan_with_rapidocr(filepath: Path) -> Optional[dict]:
    """100% Local OCR fallback using RapidOCR (ONNXRuntime PP-OCRv6)."""
    try:
        engine = load_rapidocr()
        if not engine:
            return None

        out = engine(str(filepath))
        txts = getattr(out, "txts", None)
        if txts is None and isinstance(out, (tuple, list)) and len(out) > 0:
            if isinstance(out[0], (list, tuple)):
                txts = [line[1] for line in out[0] if len(line) > 1]

        if not txts:
            return {
                "image_content": "",
                "chunk_type": "image_text",
                "is_decorative": True,
                "scan_method": "rapidocr_onnx",
            }

        text = "\n".join(str(t) for t in txts).strip()
        if len(text) < MIN_CONTENT_CHARS:
            return {
                "image_content": "",
                "chunk_type": "image_text",
                "is_decorative": True,
                "scan_method": "rapidocr_onnx",
            }

        return {
            "image_content": text,
            "chunk_type": "image_text",
            "is_decorative": False,
            "scan_method": "rapidocr_onnx",
        }
    except Exception as e:
        logger.warning(f"RapidOCR local scan failed for {filepath.name}: {e}")
        return None


def scan_image(filepath: Path, converter=None) -> dict:
    """Scan a single image using Docling DocumentConverter."""
    if converter is None:
        converter = load_docling()

    try:
        result = converter.convert(str(filepath))
        markdown = result.document.export_to_markdown().strip()

        # Check for decorative tags
        if not markdown or len(markdown) < MIN_CONTENT_CHARS:
            return {
                "image_content": "",
                "chunk_type": "image_text",
                "is_decorative": True,
                "scan_method": "docling_ocr",
            }

        return {
            "image_content": markdown,
            "chunk_type": "image_table" if ("|---" in markdown or "|---|" in markdown) else "image_text",
            "is_decorative": False,
            "scan_method": "docling_ocr",
        }
    except Exception as e:
        logger.warning(f"Docling conversion failed for {filepath.name}: {e}")
        return {
            "image_content": "",
            "chunk_type": "image_text",
            "is_decorative": False,
            "scan_method": "failed",
        }


def scan_single_image(filepath: Path) -> dict:
    """
    Modular function to scan a single image file via 100% local Docling / RapidOCR / PIL.
    Includes cross-platform timeout protection for both Windows and Linux.
    """
    # 1. Primary: Docling DocumentConverter with 30s timeout
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(scan_image, filepath)
            res = future.result(timeout=30)
            if res and res.get("image_content"):
                logger.info(f"🛠️ Scanned '{filepath.name}' using DOCLING OCR")
                return res
            elif res and res.get("is_decorative"):
                return res
    except Exception as e:
        logger.info(f"Docling timed out or failed for {filepath.name} ({e}). Trying RapidOCR...")

    # 2. Secondary Fallback: Local RapidOCR ONNX engine with 15s timeout
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(scan_with_rapidocr, filepath)
            rapid_res = future.result(timeout=15)
            if rapid_res and (rapid_res.get("image_content") or rapid_res.get("is_decorative")):
                logger.info(f"⚡ Scanned '{filepath.name}' using RAPIDOCR (Local ONNX)")
                return rapid_res
    except Exception as e:
        logger.warning(f"RapidOCR fallback failed for {filepath.name}: {e}")

    # 3. Tertiary Fallback: PIL image specifications
    try:
        from PIL import Image as PILImage
        with PILImage.open(filepath) as img:
            w, h = img.size
            return {
                "image_content": f"[Image: {filepath.name} ({w}x{h} px)]",
                "chunk_type": "image_text",
                "is_decorative": False,
                "scan_method": "pil_fallback",
            }
    except Exception:
        return {
            "image_content": f"[Image: {filepath.name}]",
            "chunk_type": "image_text",
            "is_decorative": False,
            "scan_method": "fallback",
        }


def scan_images(image_paths: List[Path]) -> List[dict]:
    """Batch scan a list of image paths."""
    return [scan_single_image(p) for p in image_paths]