"""
Universal AI Knowledge Assistant — Scraper Module
==================================================
Extracts clean, structured text and markdown from web URLs and direct PDF links.
Follows Clean Code principles: stateless, robust, high-performance.
"""

import os
import re
import logging
import tempfile
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from markdownify import markdownify as md

from config import REQUEST_TIMEOUT, LOG_LEVEL, LOG_FORMAT

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("scraper")

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/pdf,*/*;q=0.8",
}


def clean_markdown(text: str) -> str:
    """Clean noise characters, extra whitespace, and junk links from markdown."""
    if not text:
        return ""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.replace('\u200b', '').replace('\u200c', '').replace('\u200d', '').replace('\u00a0', ' ')
    text = text.replace('\ufeff', '')
    text = re.sub(r'\[]\(#[^)]+\)', '', text)
    text = re.sub(r'\[\s+\]', '', text)
    lines = [line.rstrip() for line in text.split('\n')]
    return '\n'.join(lines).strip()


def scrape_url(url: str) -> dict:
    """
    Fetch any web page or PDF URL, clean HTML to markdown, and return payload.
    
    Returns:
        dict: {
            "doc_title": str,
            "content": str,
            "source_url": str
        }
    """
    url = url.strip()
    if not url.startswith("http"):
        url = f"https://{url}"

    logger.info(f"Scraping content from URL: {url}")
    resp = requests.get(url, headers=DEFAULT_HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    content_type = resp.headers.get("Content-Type", "").lower()
    clean_url_path = url.split("?")[0].lower()

    # 1. Handle direct PDF URL
    if "application/pdf" in content_type or clean_url_path.endswith(".pdf"):
        filename = Path(clean_url_path).name or "document.pdf"
        temp_fd, temp_path = tempfile.mkstemp(suffix=".pdf", prefix="web_pdf_")
        os.close(temp_fd)
        try:
            with open(temp_path, "wb") as f:
                f.write(resp.content)

            from ingestion_service import parse_pdf_file
            text, _ = parse_pdf_file(Path(temp_path))
            return {
                "doc_title": filename[:100],
                "content": text,
                "source_url": url,
            }
        finally:
            if os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception:
                    pass

    # 2. HTML Web Page Parsing
    soup = BeautifulSoup(resp.text, "html.parser")

    # Extract clean title
    title_tag = soup.find("title") or soup.find("h1")
    title = title_tag.get_text().strip() if title_tag else url
    title = re.sub(r'\s*[\-–|].*$', '', title).strip() or url

    # Remove non-content elements
    for element in soup(["script", "style", "nav", "footer", "header", "iframe", "noscript", "svg", "form"]):
        element.decompose()

    # Locate the most specific content container
    main_body = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", {"id": re.compile(r'content|main|article', re.I)})
        or soup.find("body")
        or soup
    )

    # Convert to clean Markdown
    markdown_text = md(str(main_body), heading_style="ATX", bullets="-")
    cleaned_md = clean_markdown(markdown_text)

    # Fallback if markdown conversion is too brief
    if len(cleaned_md.strip()) < 30:
        paragraphs = [
            p.get_text().strip()
            for p in soup.find_all(["p", "h1", "h2", "h3", "h4", "li", "td"])
            if p.get_text().strip()
        ]
        cleaned_md = clean_markdown("\n\n".join(paragraphs))

    return {
        "doc_title": title[:100],
        "content": cleaned_md,
        "source_url": url,
    }


if __name__ == "__main__":
    import sys
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://example.com"
    data = scrape_url(test_url)
    print(f"Title: {data['doc_title']}")
    print(f"Content length: {len(data['content'])} chars")
    print(data["content"][:300])