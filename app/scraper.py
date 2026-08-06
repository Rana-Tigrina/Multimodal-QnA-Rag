"""
IITM BS RAG Pipeline — Stage 1: Scraper
=========================================
INPUT:  config.py (ROOT_DOCS, paths, settings)
OUTPUT: output/docs/*.md
        output/images/*.png
        output/image_metadata.json
        output/reference_links.json  (enriched during scraping)
        output/restricted_links.json
        output/document_registry.json
        output/checkpoint.json
        output/skipped_links.log

KEY POINTS:
  - Breadth-first traversal of all nested Google Docs
  - Link enrichment happens DURING scraping (not separate)
  - Every image saved with full context metadata
  - Restricted/blocked docs stored with best available metadata
  - Checkpoint/resume so crashes don't lose progress
  - Every output carries source_url always

Run:
  python scraper.py
"""

import re
import json
import hashlib
import time
import base64
import logging
import requests

from pathlib import Path
from urllib.parse import unquote
from bs4 import BeautifulSoup, NavigableString
from markdownify import markdownify as md
from tqdm import tqdm
from datetime import datetime, timezone

from config import (
    ROOT_DOCS, FOLLOW_URL_PATTERN, SKIP_DOMAINS,
    REQUEST_TIMEOUT, DELAY_BETWEEN,
    OUTPUT_DIR, DOCS_DIR, LINKED_DIR, IMAGES_DIR,
    CHECKPOINT_FILE, SKIPPED_LOG,
    IMAGE_METADATA_FILE, REFERENCE_LINKS_FILE,
    RESTRICTED_LINKS_FILE, DOC_REGISTRY_FILE,
    LOG_LEVEL, LOG_FORMAT,
)

# ══════════════════════════════════════════════════════════════════
# LOGGING
# ══════════════════════════════════════════════════════════════════

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("scraper")

# ══════════════════════════════════════════════════════════════════
# GLOBAL STATE
# These are reset on each run but restored from checkpoint
# ══════════════════════════════════════════════════════════════════

visited_urls:    set  = set()
seen_img_hashes: set  = set()
reference_links: list = []   # enriched external links
restricted_links: list = []  # blocked/401 docs with metadata
image_metadata:  list = []   # one record per saved image
doc_registry:    list = []   # one record per scraped doc
img_counter:     int  = 0
page_counter:    int  = 0

# ══════════════════════════════════════════════════════════════════
# CHECKPOINT — survive crashes
# ══════════════════════════════════════════════════════════════════

def load_checkpoint() -> dict:
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
            logger.info(f"Resuming — {len(data.get('visited', []))} URLs already done")
            return data
    return {}


def save_checkpoint():
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "visited":      list(visited_urls),
            "img_hashes":   list(seen_img_hashes),
            "img_counter":  img_counter,
            "page_counter": page_counter,
        }, f, indent=2)


def restore_checkpoint(cp: dict):
    global img_counter, page_counter
    visited_urls.update(cp.get("visited", []))
    seen_img_hashes.update(cp.get("img_hashes", []))
    img_counter  = cp.get("img_counter", 0)
    page_counter = cp.get("page_counter", 0)


# ══════════════════════════════════════════════════════════════════
# LOGGING HELPERS
# ══════════════════════════════════════════════════════════════════

def log_skip(url: str, reason: str):
    with open(SKIPPED_LOG, "a") as f:
        f.write(f"SKIP | {reason:35s} | {url}\n")


# ══════════════════════════════════════════════════════════════════
# HTTP HELPERS
# ══════════════════════════════════════════════════════════════════

def fetch(url: str) -> str | None:
    """Fetch HTML from a URL. Returns None if blocked or failed."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; RAG-Scraper/1.0)"}
        r = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)

        if r.status_code == 401:
            log_skip(url, "HTTP 401")
            return None
        if r.status_code != 200:
            log_skip(url, f"HTTP {r.status_code}")
            return None
        if any(p in r.text for p in ["Sign in", "accounts.google.com/signin"]):
            log_skip(url, "Login wall")
            return None

        return r.text

    except requests.exceptions.Timeout:
        log_skip(url, "Timeout")
    except requests.exceptions.ConnectionError:
        log_skip(url, "Connection error")
    except Exception as e:
        log_skip(url, str(e)[:40])
    return None


def download_image(url: str) -> tuple[bytes, str] | None:
    """Download image bytes. Returns (bytes, extension) or None."""
    try:
        r = requests.get(url, timeout=REQUEST_TIMEOUT)
        if r.status_code != 200:
            return None
        ct  = r.headers.get("Content-Type", "image/png")
        ext = ("jpg"  if "jpeg" in ct or "jpg"  in ct else
               "gif"  if "gif"  in ct else
               "webp" if "webp" in ct else "png")
        return r.content, ext
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════
# URL HELPERS
# ══════════════════════════════════════════════════════════════════

def unwrap_google(href: str) -> str:
    """
    Normalize any Google URL variant to a clean canonical form.
    Handles redirect wrappers, /u/N/ paths, edit/view suffixes.
    """
    if not href:
        return ""

    # Unwrap Google redirect: google.com/url?q=REAL_URL
    if "google.com/url" in href and "?q=" in href:
        try:
            href = unquote(href.split("?q=")[1].split("&")[0])
        except Exception:
            pass

    # Remove URL fragment (#h.xxx anchors) — anchors not followable
    href = href.split("#")[0]

    # Remove /u/N/ user prefix: docs.google.com/document/u/2/d/ → /d/
    href = re.sub(r'docs\.google\.com/document/u/\d+/d/',
                  'docs.google.com/document/d/', href)

    # Remove edit/view/copy suffixes but KEEP /pub
    href = re.sub(r'/(edit|view|copy)(\?.*)?$', '', href)

    # Remove query strings from doc URLs (but not /pub URLs)
    if 'docs.google.com/document' in href and not href.endswith('/pub'):
        href = href.split('?')[0]

    return href.strip().rstrip('/')


def is_google_doc(href: str) -> bool:
    """True if this URL is a Google Doc document link."""
    return bool(re.search(FOLLOW_URL_PATTERN, href))


def is_blocked(href: str) -> bool:
    """True if this URL should never be followed (blocked domains)."""
    return any(domain in href for domain in SKIP_DOMAINS)


def should_follow(href: str) -> bool:
    """True if we should scrape this URL as a Google Doc."""
    if not href or href.startswith("#"):
        return False
    if is_blocked(href):
        return False
    real = unwrap_google(href)
    return is_google_doc(real) and real not in visited_urls


def is_external_link(href: str) -> bool:
    """
    True for any link that is not a Google Doc and not blocked.
    These become reference_link chunks — not scraped but enriched.
    """
    if not href or href.startswith("#"):
        return False
    if is_blocked(href):
        return False
    real = unwrap_google(href)
    if is_google_doc(real):
        return False
    return real.startswith("http")


# ══════════════════════════════════════════════════════════════════
# LINK CLASSIFIER
# Runs during scraping — enriches every external link with metadata
# so that reference_link chunks are meaningful and retrievable
# ══════════════════════════════════════════════════════════════════

def classify_link(url: str, section: str, surrounding: str, doc_title: str) -> dict:
    """
    INPUT:  url, section heading, surrounding text, parent doc title
    OUTPUT: dict with title, what_it_contains, when_to_refer, category, access

    This runs during scraping so we never need a separate enricher file.
    """
    from urllib.parse import urlparse
    url_l      = url.lower()
    sec_l      = section.lower()
    sur_l      = surrounding.lower()
    combined   = f"{sec_l} {sur_l}"

    # ── Google Sheets ─────────────────────────────────────────────
    if "spreadsheets" in url_l:
        if "nptel" in combined or "swayam" in combined:
            if "1418834182" in url:
                return {
                    "title":            "NPTEL/SWAYAM Courses for HS/MG Credit Transfer (BS Level)",
                    "what_it_contains": "List of SWAYAM NPTEL courses applicable under HS/MG category for crediting at BS degree level",
                    "when_to_refer":    "student asks about NPTEL credit transfer, SWAYAM courses, HS MG category, BS level electives from NPTEL",
                    "category":         "course_list",
                    "access":           "public",
                }
            return {
                "title":            "NPTEL Courses Eligible for Credit Transfer",
                "what_it_contains": "List of NPTEL courses that can be credited toward the IITM BS degree",
                "when_to_refer":    "student asks about NPTEL credit transfer, which NPTEL courses count, credit transfer eligibility",
                "category":         "course_list",
                "access":           "public",
            }
        if "exam city" in combined or "1vT6KKX" in url:
            return {
                "title":            "Exam City Details",
                "what_it_contains": "List of exam cities available for in-person exams, updated from Sept 2024 term",
                "when_to_refer":    "student asks about exam cities, where to give exam, exam center locations",
                "category":         "exam_info",
                "access":           "public",
            }
        if "bdm" in combined or "project submission timeline" in combined:
            return {
                "title":            "BDM Project Submission Timeline",
                "what_it_contains": "Timeline and deadlines for BDM project submission",
                "when_to_refer":    "student asks about BDM project deadline, BDM submission timeline",
                "category":         "project_timeline",
                "access":           "public",
            }
        if "viva" in combined and ("134548240" in url or "eligib" in combined):
            return {
                "title":            "MLP Viva Eligibility Sheet",
                "what_it_contains": "Sheet showing which students are eligible for the MLP viva exam",
                "when_to_refer":    "student asks about MLP viva eligibility, whether they can attend viva",
                "category":         "project_status",
                "access":           "public",
            }
        if "registration status" in combined or "1287826979" in url:
            return {
                "title":            "MLP Project Registration Status Sheet",
                "what_it_contains": "Sheet showing MLP project registration status for each student",
                "when_to_refer":    "student asks about MLP registration status, MLP project registration confirmation",
                "category":         "project_status",
                "access":           "public",
            }
        if "orientation" in combined or "1vRia7JH" in url:
            return {
                "title":            "Course-wise Orientation YouTube Links",
                "what_it_contains": "YouTube links for orientation videos for each course at all levels",
                "when_to_refer":    "student asks about course orientation video, where to find course intro",
                "category":         "orientation",
                "access":           "public",
            }
        if "degree level" in combined:
            return {
                "title":            "Degree Level Courses List",
                "what_it_contains": "Complete list of courses available at the BS degree level",
                "when_to_refer":    "student asks about degree level courses, course list at degree level",
                "category":         "course_list",
                "access":           "public",
            }
        if "region" in combined or "exam city to region" in combined:
            return {
                "title":            "IITM BS Region Reference (Exam City to Region Mapping)",
                "what_it_contains": "Mapping of exam cities to IITM BS regions and houses",
                "when_to_refer":    "student asks about region, which house they belong to, exam city region mapping",
                "category":         "exam_info",
                "access":           "public",
            }
        if "masters" in combined or "research" in combined:
            return {
                "title":            "Masters/Research Program Reference Sheet",
                "what_it_contains": "Reference document for pathways to Masters or Research programs at IITM",
                "when_to_refer":    "student asks about masters admission, research program, pathway to MTech or MS",
                "category":         "pathway",
                "access":           "public",
            }
        return {
            "title":            f"Reference Sheet — {section[:60]}",
            "what_it_contains": f"Spreadsheet referenced in {doc_title} under: {section}",
            "when_to_refer":    f"student asks about {section[:80]}",
            "category":         "reference_sheet",
            "access":           "public",
        }

    # ── Google Presentations ──────────────────────────────────────
    if "presentation" in url_l:
        if "pgd" in combined or "mtech" in combined:
            return {
                "title":            "PGD/MTech Introduction Presentation",
                "what_it_contains": "Slides introducing the PGD and MTech upgrade pathway from BS program",
                "when_to_refer":    "student asks about PGD, MTech upgrade, postgraduate diploma slides",
                "category":         "presentation",
                "access":           "public",
            }
        if "dl genai" in combined or "registration process" in combined:
            return {
                "title":            "DL GenAI Project Registration Process Slides",
                "what_it_contains": "Step-by-step slides for DL GenAI project registration for Jan 2026",
                "when_to_refer":    "student asks about DL GenAI project registration, how to register for DL project",
                "category":         "project_guide",
                "access":           "public",
            }
        return {
            "title":            f"Presentation — {section[:60]}",
            "what_it_contains": f"Slides referenced in {doc_title} under: {section}",
            "when_to_refer":    f"student asks about {section[:80]}",
            "category":         "presentation",
            "access":           "public",
        }

    # ── External portals and tools ────────────────────────────────
    if "tds.s-anand.net" in url_l:
        return {
            "title":            "Tools in Data Science (TDS) Course Portal",
            "what_it_contains": "Official TDS course portal with graded assignments, projects, ROE links. Seek Portal NOT used for TDS.",
            "when_to_refer":    "student asks about TDS course content, TDS assignments, TDS project, Tools in Data Science portal",
            "category":         "course_portal",
            "access":           "public",
        }

    if "exam.sanand.workers.dev" in url_l:
        return {
            "title":            "TDS Entrance Exam",
            "what_it_contains": "Entrance exam to check prerequisites before registering for Tools in Data Science course",
            "when_to_refer":    "student asks about TDS entrance exam, TDS prerequisites, whether to take TDS",
            "category":         "exam",
            "access":           "public",
        }

    from urllib.parse import urlparse
    parsed = urlparse(url)
    domain = parsed.netloc or "web"

    return {
        "title": f"External Link ({domain})",
        "what_it_contains": f"Reference web link to {domain}",
        "when_to_refer": f"user asks for links or references related to {domain}",
        "category": "external_link",
        "access": "public",
    }


    if "kaggle.com" in url_l:
        course = ("MLP"     if "mlp" in combined or "machine learning" in combined else
                  "DL/GenAI" if "dl"  in combined or "genai"           in combined else
                  "project")
        if "/t/" in url_l or "competition" in url_l:
            return {
                "title":            f"Kaggle Competition — {course} Project",
                "what_it_contains": f"Kaggle competition page for the {course} project submission",
                "when_to_refer":    f"student asks about {course} Kaggle competition, project submission link",
                "category":         "project_submission",
                "access":           "public",
            }
        return {
            "title":            "Kaggle Platform",
            "what_it_contains": "Kaggle platform for ML competitions and notebooks",
            "when_to_refer":    "student asks about Kaggle, ML competition platform",
            "category":         "external_tool",
            "access":           "public",
        }

    if "github.com" in url_l:
        if "sample" in url_l or "dl-genai" in url_l:
            return {
                "title":            "DL GenAI Project Sample GitHub Repository",
                "what_it_contains": "Sample GitHub repo structure and guidelines for DL GenAI project",
                "when_to_refer":    "student asks about DL GenAI GitHub repo, sample repo, GitHub setup for DL project",
                "category":         "project_guide",
                "access":           "public",
            }
        return {
            "title":            "GitHub",
            "what_it_contains": "GitHub platform for code repository and version control",
            "when_to_refer":    "student asks about GitHub, code submission, repository setup",
            "category":         "external_tool",
            "access":           "public",
        }

    if "forms.gle" in url_l or "docs.google.com/forms" in url_l:
        course = ("MLP"     if "mlp" in combined or "machine learning" in combined else
                  "DL"      if "dl"  in combined or "genai"            in combined else
                  "MAD-1"   if "578iUF" in url or "mad 1" in combined  or "mad i" in combined else
                  "MAD-2"   if "zwCzjg" in url or "mad 2" in combined  or "mad ii" in combined else
                  "App Dev")
        if "registration" in combined:
            return {
                "title":            f"{course} Project Registration Form",
                "what_it_contains": f"Google Form to register for the {course} project competition",
                "when_to_refer":    f"student asks about {course} project registration form, how to register",
                "category":         "registration_form",
                "access":           "public",
            }
        return {
            "title":            f"Google Form — {section[:50]}",
            "what_it_contains": f"Form referenced in {doc_title} for: {section}",
            "when_to_refer":    f"student asks about {section[:60]}",
            "category":         "form",
            "access":           "public",
        }

    if "youtube.com" in url_l or "youtu.be" in url_l:
        if "oppe" in combined or "camera" in combined or "mobile" in combined:
            return {
                "title":            "OPPE Mobile Camera Setup Video",
                "what_it_contains": "Video showing how to position mobile phone camera during OPPE and SCT exams",
                "when_to_refer":    "student asks about OPPE camera setup, mobile positioning, SCT camera",
                "category":         "tutorial_video",
                "access":           "public",
            }
        if "submit" in combined or "PCyHKX3" in url:
            return {
                "title":            "App Dev Project Submission Guide Video",
                "what_it_contains": "Video guide on how to submit App Dev 1 and 2 projects including viva workflow",
                "when_to_refer":    "student asks about App Dev submission, how to submit project, submission video",
                "category":         "tutorial_video",
                "access":           "public",
            }
        if "notebook" in combined or "demo" in combined:
            return {
                "title":            "Kaggle Notebook Submission Demo Video",
                "what_it_contains": "Demo video for creating, managing and submitting Kaggle notebook for MLP project",
                "when_to_refer":    "student asks about Kaggle notebook submission, MLP notebook demo",
                "category":         "tutorial_video",
                "access":           "public",
            }
        return {
            "title":            f"Video — {section[:60]}",
            "what_it_contains": f"YouTube video referenced in {doc_title} under: {section}",
            "when_to_refer":    f"student asks about {section[:60]}",
            "category":         "video",
            "access":           "public",
        }

    if "nptel.ac.in" in url_l:
        return {
            "title":            f"NPTEL Course — {surrounding[:60]}",
            "what_it_contains": f"NPTEL course page: {surrounding[:80]}",
            "when_to_refer":    "student asks about this NPTEL course, entrepreneurship NPTEL course link",
            "category":         "nptel_course",
            "access":           "public",
        }

    if "lookerstudio.google.com" in url_l:
        return {
            "title":            "DL GenAI Project Registration Status Dashboard",
            "what_it_contains": "Looker Studio dashboard showing DL GenAI project registration status",
            "when_to_refer":    "student asks about DL GenAI registration status, project registration confirmation",
            "category":         "project_status",
            "access":           "public",
        }

    if any(x in url_l for x in ["iitmbs.org", "bandipur", "corbett", "kanha",
                                  "kaziranga", "nallamala", "namdapha",
                                  "nilgiri", "pichavaram", "saranda",
                                  "sundarbans", "wayanad", "gir."]):
        from urllib.parse import urlparse as _up
        house = _up(url).hostname.split(".")[0].title() if url else "House"
        return {
            "title":            f"{house} House — IITM BS",
            "what_it_contains": f"Portal for {house} House, one of the student houses in IITM BS program",
            "when_to_refer":    f"student asks about {house} house, student house portal",
            "category":         "house_portal",
            "access":           "public",
        }

    # Generic fallback
    from urllib.parse import urlparse as _up
    domain = _up(url).netloc
    return {
        "title":            f"External Reference — {domain}",
        "what_it_contains": f"External link from {doc_title}, section: {section}. Context: {surrounding[:100]}",
        "when_to_refer":    f"student asks about {section[:60]}",
        "category":         "external_reference",
        "access":           "public",
    }


# ══════════════════════════════════════════════════════════════════
# TOC PARSER
# Builds a map of anchor → section info for heading enrichment
# ══════════════════════════════════════════════════════════════════

def parse_toc(soup: BeautifulSoup) -> dict:
    """
    INPUT:  BeautifulSoup of full page HTML
    OUTPUT: dict mapping "#h.xxxx" → {title, number, level}
    """
    toc_map = {}
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if not href.startswith("#h."):
            continue
        text = a.get_text(strip=True)
        if not text:
            continue
        number = ""
        match = re.match(r'^([\d]+(?:\.[\d]+)*\.?)\s+', text)
        if match:
            number = match.group(1).rstrip(".")
        level = number.count(".") + 1 if number else 0
        toc_map[href] = {"title": text, "number": number, "level": level}
    return toc_map


# ══════════════════════════════════════════════════════════════════
# IMAGE HANDLER
# Saves image to disk and records full metadata
# ══════════════════════════════════════════════════════════════════

def process_image(
    img_tag,
    current_heading: str,
    breadcrumb: str,
    doc_title: str,
    source_url: str,
    prev_text: str,
    next_text: str,
) -> str:
    """
    INPUT:  img HTML tag + context from surrounding content
    OUTPUT: markdown string referencing the image
    SIDE EFFECT: saves image to disk, appends to image_metadata list
    """
    global img_counter

    src = img_tag.get("src", "")
    if not src:
        return ""

    image_bytes = None
    ext = "png"

    # base64 embedded image
    if src.startswith("data:image"):
        try:
            header, b64data = src.split(",", 1)
            image_bytes = base64.b64decode(b64data)
            ext = ("jpg"  if "jpeg" in header or "jpg" in header else
                   "gif"  if "gif"  in header else "png")
        except Exception:
            return ""
    else:
        result = download_image(src)
        if not result:
            return f"\n[IMAGE: failed to download | Section: {current_heading}]\n"
        image_bytes, ext = result

    # Deduplicate by content hash
    img_hash = hashlib.md5(image_bytes).hexdigest()
    is_duplicate = img_hash in seen_img_hashes
    seen_img_hashes.add(img_hash)

    if is_duplicate:
        return f"\n[IMAGE: duplicate — same image seen earlier]\n"

    # Save to disk
    img_counter += 1
    filename = f"img_{img_counter:03d}.{ext}"
    filepath = IMAGES_DIR / filename
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    file_size_kb = len(image_bytes) / 1024

    # Store full metadata — used by image_scanner.py later
    image_metadata.append({
        "filename":           filename,
        "filepath":           str(filepath),
        "doc_title":          doc_title,
        "section":            current_heading,
        "breadcrumb":         breadcrumb,
        "source_url":         source_url,
        "surrounding_before": prev_text[:200],
        "surrounding_after":  next_text[:200],
        "file_size_kb":       round(file_size_kb, 2),
        "image_hash":         img_hash,
        # These fields filled in by image_scanner.py
        "image_content":      None,
        "is_decorative":      None,
        "chunk_type":         None,
        "needs_ocr":          True,
    })

    return (
        f"\n[IMAGE:{filename}]\n"
        f"**Image context** — Section: *{current_heading}* | "
        f"Before: {prev_text[:100]} | After: {next_text[:100]}\n"
        f"**Breadcrumb:** {breadcrumb}\n"
    )


# ══════════════════════════════════════════════════════════════════
# TABLE CONVERTER
# Converts HTML table to clean markdown
# ══════════════════════════════════════════════════════════════════

def convert_table(table, heading: str) -> str:
    """
    INPUT:  BeautifulSoup table element + current heading
    OUTPUT: markdown table string with section annotation
    """
    rows = table.find_all("tr")
    if not rows:
        return ""

    data = []
    for row in rows:
        cells = row.find_all(["td", "th"])
        row_data = []
        for cell in cells:
            text = cell.get_text(separator=" ", strip=True)
            text = re.sub(r'\s+', ' ', text).strip()
            text = text.replace("|", "\\|")
            row_data.append(text)
        if any(row_data):
            data.append(row_data)

    if not data:
        return ""

    max_cols = max(len(r) for r in data)
    for r in data:
        while len(r) < max_cols:
            r.append("")

    lines = [f"\n> *Table from section: **{heading}***\n"]
    lines.append("| " + " | ".join(data[0]) + " |")
    lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in data[1:]:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines) + "\n"


# ══════════════════════════════════════════════════════════════════
# REFERENCE LINK COLLECTOR
# Called during HTML walking — enriches every external link
# ══════════════════════════════════════════════════════════════════

def collect_reference_link(
    a_tag,
    doc_title: str,
    breadcrumb: str,
    section: str,
    source_url: str,
):
    """
    INPUT:  anchor tag + context
    OUTPUT: appends enriched record to reference_links list

    Deduplicates by URL so same link from multiple docs
    is stored once but with all source contexts.
    """
    href       = unwrap_google(a_tag.get("href", ""))
    link_text  = a_tag.get_text(strip=True)

    if not href or not href.startswith("http"):
        return

    parent      = a_tag.find_parent(["p", "li", "td"])
    surrounding = parent.get_text(strip=True)[:200] if parent else ""

    # Check if URL already stored — if so add context, don't duplicate
    existing = next((r for r in reference_links if r["url"] == href), None)
    if existing:
        if source_url not in existing.get("also_found_in", []):
            existing.setdefault("also_found_in", []).append(source_url)
        return

    classification = classify_link(href, section, surrounding, doc_title)

    reference_links.append({
        # Identity
        "url":              href,
        "link_text":        link_text or href[:60],

        # Classification (from classify_link)
        "title":            classification["title"],
        "what_it_contains": classification["what_it_contains"],
        "when_to_refer":    classification["when_to_refer"],
        "category":         classification["category"],
        "access":           classification["access"],

        # Source context
        "found_in_doc":     doc_title,
        "section":          section,
        "breadcrumb":       breadcrumb,
        "source_url":       source_url,
        "surrounding_text": surrounding,

        # Flags
        "is_scraped":       False,
        "also_found_in":    [],
    })


# ══════════════════════════════════════════════════════════════════
# RESTRICTED LINK HANDLER
# Called when a Google Doc link returns 401 or login wall
# ══════════════════════════════════════════════════════════════════

def collect_restricted_link(
    url: str,
    reason: str,
    doc_title: str,
    section: str,
    breadcrumb: str,
    source_url: str,
    surrounding: str,
):
    """
    INPUT:  blocked URL + context from where it was found
    OUTPUT: appends to restricted_links list

    These become restricted_doc chunks in chunker.py
    so RAG can still answer "refer this link" for relevant questions.
    """
    # Deduplicate
    if any(r["url"] == url for r in restricted_links):
        return

    restricted_links.append({
        "url":              url,
        "skip_reason":      reason,
        "found_in_doc":     doc_title,
        "section":          section,
        "breadcrumb":       breadcrumb,
        "source_url":       source_url,
        "surrounding_text": surrounding,
        "access":           "restricted",
        "note":             "This document requires IITM login to access. Content not available.",
        "category":         "restricted_doc",
    })


# ══════════════════════════════════════════════════════════════════
# HTML CLEANER
# Removes noise elements before content extraction
# ══════════════════════════════════════════════════════════════════

def clean_soup(soup: BeautifulSoup) -> BeautifulSoup:
    for tag in soup.find_all(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    for div in soup.find_all("div", id=["header", "footer", "banners"]):
        div.decompose()
    for tag in soup.find_all(True):
        tag.attrs.pop("style", None)
    return soup


# ══════════════════════════════════════════════════════════════════
# CORE EXTRACTOR
# Walks HTML body and builds clean markdown with metadata comments
# ══════════════════════════════════════════════════════════════════

def extract_to_markdown(
    html: str,
    source_url: str,
    doc_title: str,
    toc_map: dict,
    parent_doc: str = "",
) -> str:
    """
    INPUT:  raw HTML + context
    OUTPUT: clean markdown string with embedded metadata comments
    SIDE EFFECTS: populates reference_links and image_metadata
    """
    soup = BeautifulSoup(html, "html.parser")
    clean_soup(soup)

    # Build anchor → section info map from TOC
    anchor_map = {aid.lstrip("#"): info for aid, info in toc_map.items()}

    body = soup.find("body") or soup

    # State tracking
    breadcrumb_stack  = [doc_title]
    current_heading   = doc_title
    prev_paragraph    = ""
    output_parts      = []

    # Frontmatter
    output_parts.append(
        f"---\n"
        f"doc_title: {doc_title}\n"
        f"source_url: {source_url}\n"
        f"parent_doc: {parent_doc}\n"
        f"scraped_at: {datetime.now(timezone.utc).isoformat()}\n"
        f"---\n\n"
    )

    def get_breadcrumb() -> str:
        return " > ".join(breadcrumb_stack)

    def get_toc_info(element) -> dict | None:
        el_id = element.get("id", "")
        if el_id and el_id in anchor_map:
            return anchor_map[el_id]
        for child in element.find_all(["span", "a"], id=True):
            if child.get("id", "") in anchor_map:
                return anchor_map[child["id"]]
        return None

    def update_breadcrumb(level: int, title: str):
        nonlocal breadcrumb_stack
        while len(breadcrumb_stack) > level:
            breadcrumb_stack.pop()
        breadcrumb_stack.append(title)

    def emit_meta() -> str:
        return (
            f"\n<!-- meta | doc:{doc_title} | "
            f"section:{current_heading} | "
            f"breadcrumb:{get_breadcrumb()} | "
            f"url:{source_url} -->\n"
        )

    def walk(node):
        nonlocal current_heading, prev_paragraph

        for element in node.children:
            if isinstance(element, NavigableString):
                continue

            tag = getattr(element, "name", None)
            if not tag:
                continue

            # ── HEADINGS ──────────────────────────────────────────
            if tag in ["h1", "h2", "h3", "h4"]:
                text = element.get_text(strip=True)
                if not text:
                    continue

                level           = int(tag[1])
                current_heading = text
                prev_paragraph  = ""

                toc_info = get_toc_info(element)
                display  = text
                if toc_info and toc_info.get("number"):
                    if toc_info["number"] not in text:
                        display = f"{toc_info['number']} {text}"

                update_breadcrumb(level, text)
                output_parts.append(emit_meta())
                output_parts.append(f"\n{'#' * level} {display}\n\n")

            # ── PARAGRAPHS ────────────────────────────────────────
            elif tag == "p":
                img = element.find("img")
                if img:
                    next_sib  = element.find_next_sibling(["p", "li"])
                    next_text = next_sib.get_text(strip=True)[:150] if next_sib else ""
                    img_block = process_image(
                        img,
                        current_heading=current_heading,
                        breadcrumb=get_breadcrumb(),
                        doc_title=doc_title,
                        source_url=source_url,
                        prev_text=prev_paragraph[:150],
                        next_text=next_text,
                    )
                    output_parts.append(img_block)
                    continue

                para_md = md(str(element), heading_style="ATX", bullets="-").strip()
                if not para_md:
                    continue

                # Collect all links in this paragraph
                for a in element.find_all("a", href=True):
                    href = unwrap_google(a["href"])
                    if is_external_link(href):
                        collect_reference_link(
                            a, doc_title, get_breadcrumb(),
                            current_heading, source_url
                        )

                output_parts.append(f"{para_md}\n\n")
                prev_paragraph = element.get_text(strip=True)

            # ── TABLES ────────────────────────────────────────────
            elif tag == "table":
                table_md = convert_table(element, current_heading)
                if table_md:
                    output_parts.append(emit_meta())
                    output_parts.append(table_md + "\n")

            # ── LISTS ─────────────────────────────────────────────
            elif tag in ["ul", "ol"]:
                for a in element.find_all("a", href=True):
                    href = unwrap_google(a["href"])
                    if is_external_link(href):
                        collect_reference_link(
                            a, doc_title, get_breadcrumb(),
                            current_heading, source_url
                        )
                list_md = md(str(element), heading_style="ATX", bullets="-").strip()
                if list_md:
                    output_parts.append(f"{list_md}\n\n")

            # ── DIVS — recurse into them ──────────────────────────
            elif tag in ["div", "section", "article", "main"]:
                walk(element)

    walk(body)

    raw = "".join(output_parts)
    return post_clean(raw)


# ══════════════════════════════════════════════════════════════════
# POST CLEAN
# Removes noise characters and excessive whitespace
# ══════════════════════════════════════════════════════════════════

def post_clean(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.replace('\u200b', '').replace('\u200c', '')
    text = text.replace('\u200d', '').replace('\u00a0', ' ')
    text = text.replace('\ufeff', '')
    text = re.sub(r'Report abuse', '', text)
    text = re.sub(r'Learn more', '', text)
    text = re.sub(r'\[]\(#[^)]+\)', '', text)
    text = re.sub(r'\[\s+\]', '', text)
    lines = [l.rstrip() for l in text.split('\n')]
    return '\n'.join(lines).strip()


def scrape_url(url: str) -> dict:
    """
    Fetch any web page URL, clean HTML to markdown, and return document payload.
    Used for live web URL ingestion.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    
    # Extract title
    title_tag = soup.find("title") or soup.find("h1")
    title = title_tag.get_text().strip() if title_tag else url

    # Remove script, style, nav elements
    for el in soup(["script", "style", "nav", "footer", "iframe", "noscript"]):
        el.decompose()

    # Convert to markdown
    markdown_text = md(str(soup), heading_style="ATX")
    cleaned_md = post_clean(markdown_text)

    return {
        "doc_title": title[:100],
        "content": cleaned_md,
        "source_url": url,
    }



# ══════════════════════════════════════════════════════════════════
# LINK COLLECTOR
# Finds all Google Doc links in a page for BFS queue
# ══════════════════════════════════════════════════════════════════

def collect_doc_links(html: str) -> list[str]:
    """
    INPUT:  raw HTML
    OUTPUT: list of unique Google Doc URLs to follow next
    """
    soup  = BeautifulSoup(html, "html.parser")
    links = []
    seen  = set()
    for a in soup.find_all("a", href=True):
        href = unwrap_google(a["href"].strip())
        if should_follow(href) and href not in seen:
            seen.add(href)
            links.append(href)
    return links


# ══════════════════════════════════════════════════════════════════
# DOCUMENT PROCESSOR
# Processes one Google Doc — fetch, parse, save, return nested links
# ══════════════════════════════════════════════════════════════════

def process_document(
    url: str,
    doc_title: str,
    output_path: Path,
    parent_title: str = "",
) -> list[str]:
    """
    INPUT:  URL, title, where to save, parent doc title
    OUTPUT: list of new nested Google Doc URLs found inside
    SIDE EFFECTS: saves markdown, updates doc_registry,
                  populates reference_links, image_metadata
    """
    global page_counter

    if url in visited_urls:
        return []

    logger.info(f"Processing: {doc_title}")
    logger.info(f"URL: {url[:80]}")
    visited_urls.add(url)
    time.sleep(DELAY_BETWEEN)

    html = fetch(url)
    if not html:
        logger.error(f"Failed to fetch: {url[:60]}")
        # Store as restricted if we got blocked
        collect_restricted_link(
            url=url,
            reason="fetch_failed",
            doc_title=doc_title,
            section="",
            breadcrumb=doc_title,
            source_url=url,
            surrounding="",
        )
        return []

    soup_for_toc = BeautifulSoup(html, "html.parser")
    toc_map      = parse_toc(soup_for_toc)
    logger.info(f"TOC entries: {len(toc_map)}")

    markdown = extract_to_markdown(
        html, url, doc_title, toc_map, parent_title
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    page_counter += 1
    logger.info(f"Saved: {output_path.name} ({len(markdown):,} chars)")

    doc_registry.append({
        "title":        doc_title,
        "url":          url,
        "parent":       parent_title,
        "output_file":  str(output_path),
        "toc_entries":  len(toc_map),
        "char_count":   len(markdown),
        "scraped_at":   datetime.now(timezone.utc).isoformat(),
    })

    nested = collect_doc_links(html)
    logger.info(f"Nested Google Doc links found: {len(nested)}")
    return nested


# ══════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════

def save_outputs():
    """Save all collected data to their output files."""

    with open(REFERENCE_LINKS_FILE, "w", encoding="utf-8") as f:
        json.dump(reference_links, f, indent=2, ensure_ascii=False)

    with open(RESTRICTED_LINKS_FILE, "w", encoding="utf-8") as f:
        json.dump(restricted_links, f, indent=2, ensure_ascii=False)

    with open(IMAGE_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(image_metadata, f, indent=2, ensure_ascii=False)

    with open(DOC_REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(doc_registry, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved reference_links.json  ({len(reference_links)} links)")
    logger.info(f"Saved restricted_links.json ({len(restricted_links)} docs)")
    logger.info(f"Saved image_metadata.json   ({len(image_metadata)} images)")
    logger.info(f"Saved document_registry.json ({len(doc_registry)} docs)")


# ══════════════════════════════════════════════════════════════════
# MAIN ORCHESTRATOR
# Breadth-first traversal of all Google Docs
# ══════════════════════════════════════════════════════════════════

def run():
    print("\n" + "═" * 65)
    print("  IITM BS RAG Pipeline — Stage 1: Scraper")
    print("═" * 65)

    # Create all output directories
    for d in [OUTPUT_DIR, DOCS_DIR, LINKED_DIR, IMAGES_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Clear skipped log for fresh run
    if SKIPPED_LOG.exists():
        SKIPPED_LOG.unlink()

    # Restore checkpoint if exists
    cp = load_checkpoint()
    restore_checkpoint(cp)

    queue = []  # (url, title, parent_title)

    # ── Process root docs first ───────────────────────────────────
    print(f"\n📚 Root documents: {len(ROOT_DOCS)}")
    for doc in ROOT_DOCS:
        if doc["url"] not in visited_urls:
            nested = process_document(
                url=doc["url"],
                doc_title=doc["title"],
                output_path=DOCS_DIR / doc["filename"],
                parent_title="",
            )
            save_checkpoint()
            save_outputs()
            for nested_url in nested:
                if nested_url not in visited_urls:
                    queue.append((nested_url, nested_url[:50], doc["title"]))
        else:
            print(f"  ⏭  Skipping (already done): {doc['title']}")

    # ── Process nested docs (BFS queue) ──────────────────────────
    if queue:
        print(f"\n🔗 Processing nested Google Doc links...")

    processed_nested = 0
    with tqdm(total=len(queue), desc="Nested docs") as pbar:
        while queue:
            url, _, parent_title = queue.pop(0)

            if url in visited_urls:
                pbar.update(1)
                continue

            # Fetch to get title before processing
            html = fetch(url)
            if not html:
                # Store as restricted doc
                collect_restricted_link(
                    url=url,
                    reason="fetch_failed_nested",
                    doc_title="Unknown",
                    section="",
                    breadcrumb="",
                    source_url=url,
                    surrounding="",
                )
                pbar.update(1)
                continue

            # Extract title from <title> tag
            soup_t    = BeautifulSoup(html, "html.parser")
            title_tag = soup_t.find("title")
            doc_title = (title_tag.get_text(strip=True)
                         if title_tag else f"Linked Doc {page_counter + 1}")
            doc_title = re.sub(r'\s*-\s*Google Docs.*$', '', doc_title).strip()
            doc_title = re.sub(r'^Copy of\s+', '', doc_title).strip()

            processed_nested += 1
            filename    = f"page_{processed_nested:03d}.md"
            output_path = LINKED_DIR / filename

            # Process — note: we already fetched html above
            # so we re-use it instead of fetching again
            visited_urls.add(url)

            soup_for_toc = BeautifulSoup(html, "html.parser")
            toc_map      = parse_toc(soup_for_toc)

            markdown = extract_to_markdown(
                html, url, doc_title, toc_map, parent_title
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(markdown)

            page_counter_local = page_counter + 1
            doc_registry.append({
                "title":        doc_title,
                "url":          url,
                "parent":       parent_title,
                "output_file":  str(output_path),
                "toc_entries":  len(toc_map),
                "char_count":   len(markdown),
                "scraped_at":   datetime.now(timezone.utc).isoformat(),
            })

            logger.info(f"Saved: {filename} — {doc_title} ({len(markdown):,} chars)")

            # Find more nested links
            more_nested = collect_doc_links(html)
            for new_url in more_nested:
                if new_url not in visited_urls:
                    queue.append((new_url, new_url[:50], doc_title))
                    pbar.total += 1
                    pbar.refresh()

            save_checkpoint()
            save_outputs()
            time.sleep(DELAY_BETWEEN)
            pbar.update(1)

    # Final save
    save_outputs()

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "═" * 65)
    print("  ✅ SCRAPING COMPLETE")
    print("═" * 65)
    print(f"  Documents scraped:    {len(doc_registry)}")
    print(f"    Root docs:          {len(ROOT_DOCS)}")
    print(f"    Nested docs:        {processed_nested}")
    print(f"  Images saved:         {img_counter}")
    print(f"  Reference links:      {len(reference_links)} (enriched)")
    print(f"  Restricted docs:      {len(restricted_links)}")
    print(f"  Skipped log:          {SKIPPED_LOG}")
    print(f"\n  Output files:")
    print(f"    {REFERENCE_LINKS_FILE}")
    print(f"    {RESTRICTED_LINKS_FILE}")
    print(f"    {IMAGE_METADATA_FILE}")
    print(f"    {DOC_REGISTRY_FILE}")
    print(f"\n  Next step: python image_scanner.py")
    print("═" * 65)


if __name__ == "__main__":
    run()