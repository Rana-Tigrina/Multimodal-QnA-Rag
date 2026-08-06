"""
IITM BS RAG Pipeline — Stage 7: Evaluator
==========================================
INPUT:  Running backend at localhost:8000
OUTPUT: output/eval_report.json

What it does:
  - Runs 50 test questions against the live RAG pipeline
  - Checks retrieval accuracy (did right chunk get retrieved?)
  - Checks answer accuracy (is answer factually correct?)
  - Checks source citation (is source cited?)
  - Breaks down scores by chunk type
  - Saves detailed report for debugging

Run:
  # Make sure backend is running first:
  # uvicorn main:app --reload --port 8000
  
  python evaluator.py
"""

import json
import time
import logging
import requests
from pathlib import Path
from datetime import datetime

from config import (
    ALL_CHUNKS_FILE,
    LOG_LEVEL,
    LOG_FORMAT,
)

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("evaluator")

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════

BACKEND_URL  = "http://localhost:8000"
EVAL_OUTPUT  = Path("output/eval_report.json")
TIMEOUT      = 60   # seconds per question

# ══════════════════════════════════════════════════════════════════
# TEST QUESTIONS
# 50 questions covering all chunk types and topics
# expected_answer: key phrases that MUST appear in answer
# expected_source: doc title or section that must be cited
# chunk_type: what type of chunk should answer this
# ══════════════════════════════════════════════════════════════════

TEST_QUESTIONS = [

    # ── TEXT CHUNKS — policy questions ───────────────────────────

    {
        "id": "T001",
        "question": "What are the eligibility criteria to join the IITM BS programme?",
        "expected_keywords": ["class 10", "english", "math", "JEE", "qualifier"],
        "expected_source": "Student Handbook",
        "chunk_type": "text",
        "topic": "admission",
    },
    {
        "id": "T002",
        "question": "How many credits are required to complete the BS degree?",
        "expected_keywords": ["142", "credits", "BS degree"],
        "expected_source": "Student Handbook",
        "chunk_type": "text",
        "topic": "degree requirements",
    },
    {
        "id": "T003",
        "question": "What happens if a student fails to clear the foundation level in the allowed attempts?",
        "expected_keywords": ["exit", "foundation", "attempts", "deregister"],
        "expected_source": "Student Handbook",
        "chunk_type": "text",
        "topic": "foundation level",
    },
    {
        "id": "T004",
        "question": "Can a student re-enter the programme after exiting at diploma level?",
        "expected_keywords": ["re-entry", "re entry", "diploma", "rejoin"],
        "expected_source": "Student Handbook",
        "chunk_type": "text",
        "topic": "re-entry",
    },
    {
        "id": "T005",
        "question": "What is the minimum score required to pass a course in the foundation level?",
        "expected_keywords": ["40", "pass", "foundation"],
        "expected_source": "Student Handbook",
        "chunk_type": "text",
        "topic": "grading",
    },
    {
        "id": "T006",
        "question": "How many courses are there in the foundation level?",
        "expected_keywords": ["foundation", "courses", "4", "five", "term"],
        "expected_source": "Student Handbook",
        "chunk_type": "text",
        "topic": "foundation level",
    },
    {
        "id": "T007",
        "question": "What is the last date to register for courses in January 2026 term?",
        "expected_keywords": ["register", "January 2026", "deadline", "last date"],
        "expected_source": "BS-DS_ Jan 2026 Grading document (STUDENT)",
        "chunk_type": "text",
        "topic": "deadlines",
    },
    {
        "id": "T008",
        "question": "What is the MLP project and when should it be submitted?",
        "expected_keywords": ["MLP", "machine learning", "project", "submit"],
        "expected_source": "Student Handbook",
        "chunk_type": "text",
        "topic": "projects",
    },
    {
        "id": "T009",
        "question": "How many attempts does a student get for the qualifier exam?",
        "expected_keywords": ["qualifier", "attempts", "foundation"],
        "expected_source": "Student Handbook",
        "chunk_type": "text",
        "topic": "qualifier",
    },
    {
        "id": "T010",
        "question": "What is the process to apply for a hall ticket for end term exam?",
        "expected_keywords": ["hall ticket", "end term", "apply", "exam"],
        "expected_source": "Student Handbook",
        "chunk_type": "text",
        "topic": "exams",
    },

    # ── TABLE CHUNKS — fees, marks, credits ──────────────────────

    {
        "id": "TB001",
        "question": "What is the total fee for foundation level for students joining from January 2026?",
        "expected_keywords": ["48000", "48,000", "foundation"],
        "expected_source": "Student Handbook",
        "chunk_type": "table",
        "topic": "fees",
    },
    {
        "id": "TB002",
        "question": "What is the total fee for Foundation plus one Diploma for Jan 2026 students?",
        "expected_keywords": ["129000", "129,000", "diploma"],
        "expected_source": "Student Handbook",
        "chunk_type": "table",
        "topic": "fees",
    },
    {
        "id": "TB003",
        "question": "What is the fee per credit for foundation level students joining Jan 2026?",
        "expected_keywords": ["1500", "1,500", "credit"],
        "expected_source": "Student Handbook",
        "chunk_type": "table",
        "topic": "fees",
    },
    {
        "id": "TB004",
        "question": "What is the fee waiver for SC/ST students with family income less than 1 LPA?",
        "expected_keywords": ["75%", "SC", "ST", "waiver"],
        "expected_source": "Student Handbook",
        "chunk_type": "table",
        "topic": "fee waiver",
    },
    {
        "id": "TB005",
        "question": "How many credits are required at the diploma level?",
        "expected_keywords": ["diploma", "credits", "54", "59"],
        "expected_source": "Student Handbook",
        "chunk_type": "table",
        "topic": "credits",
    },
    {
        "id": "TB006",
        "question": "What is the assignment deadline for week 4 foundation students in January 2026 term?",
        "expected_keywords": ["week 4", "foundation", "deadline", "january"],
        "expected_source": "BS-DS_ Jan 2026 Grading document (STUDENT)",
        "chunk_type": "table",
        "topic": "deadlines",
    },
    {
        "id": "TB007",
        "question": "What is the grading breakdown for the Python course in terms of assignments and OPPE?",
        "expected_keywords": ["python", "assignment", "OPPE", "grading", "marks"],
        "expected_source": "BS-DS_ Jan 2026 Grading document (STUDENT)",
        "chunk_type": "table",
        "topic": "grading",
    },

    # ── REFERENCE LINK CHUNKS — external resources ───────────────

    {
        "id": "RL001",
        "question": "Where can I find the list of NPTEL courses approved for credit transfer in IITM BS?",
        "expected_keywords": ["docs.google.com", "sheets", "NPTEL", "SWAYAM"],
        "expected_source": "Student Handbook",
        "chunk_type": "reference_link",
        "topic": "NPTEL credit transfer",
    },
    {
        "id": "RL002",
        "question": "Where can I find the OPPE system compatibility test rules?",
        "expected_keywords": ["docs.google.com", "SCT", "compatibility", "OPPE"],
        "expected_source": "BS-DS_ Jan 2026 Grading document (STUDENT)",
        "chunk_type": "reference_link",
        "topic": "OPPE SCT",
    },
    {
        "id": "RL003",
        "question": "Where can I find the MacOS camera and location settings for OPPE?",
        "expected_keywords": ["MacOS", "camera", "location", "settings"],
        "expected_source": "BS-DS_ Jan 2026 Grading document (STUDENT)",
        "chunk_type": "reference_link",
        "topic": "OPPE setup",
    },
    {
        "id": "RL004",
        "question": "Where can I find the BDM project details and submission guidelines?",
        "expected_keywords": ["BDM", "project", "submission", "docs.google.com"],
        "expected_source": "Student Handbook",
        "chunk_type": "reference_link",
        "topic": "BDM project",
    },
    {
        "id": "RL005",
        "question": "Where can I find the App Development project rubric?",
        "expected_keywords": ["App Dev", "rubric", "project", "docs.google.com"],
        "expected_source": "Student Handbook",
        "chunk_type": "reference_link",
        "topic": "App Dev project",
    },

    # ── OPPE SPECIFIC — uses hyde_questions matching ──────────────

    {
        "id": "O001",
        "question": "What are the eligibility criteria for OPPE1 in the Python programming course?",
        "expected_keywords": ["A1", "A2", "A3", "A4", "40", "SCT", "python"],
        "expected_source": "BS-DS_ Jan 2026 Grading document (STUDENT)",
        "chunk_type": "text",
        "topic": "OPPE Python",
    },
    {
        "id": "O002",
        "question": "What are the eligibility criteria for OPPE2 in the Python programming course?",
        "expected_keywords": ["A5", "A6", "A7", "A8", "40", "average", "python"],
        "expected_source": "BS-DS_ Jan 2026 Grading document (STUDENT)",
        "chunk_type": "text",
        "topic": "OPPE Python",
    },
    {
        "id": "O003",
        "question": "What happens if I miss the OPPE system compatibility test?",
        "expected_keywords": ["SCT", "OPPE1", "OPPE2", "not scheduled", "mandatory"],
        "expected_source": "BS-DS_ Jan 2026 Grading document (STUDENT)",
        "chunk_type": "text",
        "topic": "OPPE SCT",
    },
    {
        "id": "O004",
        "question": "Is OPPE mandatory for all courses in IITM BS programme?",
        "expected_keywords": ["OPPE", "mandatory", "courses", "programming"],
        "expected_source": "BS-DS_ Jan 2026 Grading document (STUDENT)",
        "chunk_type": "text",
        "topic": "OPPE mandatory",
    },

    # ── FEES & PAYMENTS ───────────────────────────────────────────

    {
        "id": "F001",
        "question": "What is the total fee for the BS degree for students joining from Jan 2026?",
        "expected_keywords": ["386000", "450000", "386,000", "450,000", "BS degree"],
        "expected_source": "Student Handbook",
        "chunk_type": "table",
        "topic": "BS degree fee",
    },
    {
        "id": "F002",
        "question": "What documents are required for OBC fee waiver in IITM BS?",
        "expected_keywords": ["OBC", "OBC-NCL", "family income", "waiver", "documents"],
        "expected_source": "Student Handbook",
        "chunk_type": "table",
        "topic": "fee waiver OBC",
    },
    {
        "id": "F003",
        "question": "What is the fee for credit transfer from NPTEL courses?",
        "expected_keywords": ["1000", "NPTEL", "credit transfer", "per credit"],
        "expected_source": "Student Handbook",
        "chunk_type": "text",
        "topic": "NPTEL fee",
    },

    # ── GRADING & EXAMS ───────────────────────────────────────────

    {
        "id": "G001",
        "question": "What is the weightage of end term exam in the final score for foundation courses?",
        "expected_keywords": ["end term", "weightage", "foundation", "%"],
        "expected_source": "BS-DS_ Jan 2026 Grading document (STUDENT)",
        "chunk_type": "text",
        "topic": "grading weightage",
    },
    {
        "id": "G002",
        "question": "What is the assignment deadline for week 1 degree students in January 2026?",
        "expected_keywords": ["week 1", "degree", "deadline", "assignment"],
        "expected_source": "BS-DS_ Jan 2026 Grading document (STUDENT)",
        "chunk_type": "table",
        "topic": "assignment deadlines",
    },
    {
        "id": "G003",
        "question": "When is the quiz 1 for January 2026 term?",
        "expected_keywords": ["quiz 1", "january 2026", "date", "week"],
        "expected_source": "BS-DS_ Jan 2026 Grading document (STUDENT)",
        "chunk_type": "text",
        "topic": "quiz schedule",
    },

    # ── COURSES & REGISTRATION ────────────────────────────────────

    {
        "id": "C001",
        "question": "What courses are available at the foundation level in IITM BS?",
        "expected_keywords": ["foundation", "mathematics", "statistics", "english", "computational"],
        "expected_source": "Student Handbook",
        "chunk_type": "text",
        "topic": "foundation courses",
    },
    {
        "id": "C002",
        "question": "Can foundation level students register for diploma level courses?",
        "expected_keywords": ["foundation", "diploma", "register", "eligible"],
        "expected_source": "Student Handbook",
        "chunk_type": "text",
        "topic": "course registration",
    },
    {
        "id": "C003",
        "question": "What is the maximum number of courses a student can register per term?",
        "expected_keywords": ["maximum", "courses", "term", "register"],
        "expected_source": "Student Handbook",
        "chunk_type": "text",
        "topic": "course load",
    },
    {
        "id": "C004",
        "question": "What is the Java programming course code in IITM BS?",
        "expected_keywords": ["java", "BSCS", "code", "programming"],
        "expected_source": "BS-DS_ Jan 2026 Grading document (STUDENT)",
        "chunk_type": "text",
        "topic": "course codes",
    },

    # ── PROJECTS ──────────────────────────────────────────────────

    {
        "id": "P001",
        "question": "What is the BDM project in IITM BS and what level is it offered at?",
        "expected_keywords": ["BDM", "business data management", "diploma", "project"],
        "expected_source": "Student Handbook",
        "chunk_type": "text",
        "topic": "BDM project",
    },
    {
        "id": "P002",
        "question": "What are the prerequisites to register for the App Development 1 project?",
        "expected_keywords": ["App Dev", "prerequisites", "register", "project"],
        "expected_source": "Student Handbook",
        "chunk_type": "text",
        "topic": "App Dev project",
    },

    # ── CREDIT TRANSFER ───────────────────────────────────────────

    {
        "id": "CT001",
        "question": "How many credits can be transferred from NPTEL courses in the BS level?",
        "expected_keywords": ["4 credits", "NPTEL", "BS level", "transfer"],
        "expected_source": "Student Handbook",
        "chunk_type": "text",
        "topic": "NPTEL credits",
    },
    {
        "id": "CT002",
        "question": "Which category of NPTEL courses are eligible for credit transfer in IITM BS?",
        "expected_keywords": ["SWAYAM", "certification", "HS", "MG", "humanities"],
        "expected_source": "Student Handbook",
        "chunk_type": "text",
        "topic": "NPTEL eligibility",
    },

    # ── OFF-TOPIC — should redirect ───────────────────────────────

    {
        "id": "OT001",
        "question": "Who are you?",
        "expected_keywords": ["IITM BS", "programme assistant", "help"],
        "expected_source": None,
        "chunk_type": "off_topic",
        "topic": "off-topic",
    },
    {
        "id": "OT002",
        "question": "What is the capital of France?",
        "expected_keywords": ["IITM BS", "programme", "only"],
        "expected_source": None,
        "chunk_type": "off_topic",
        "topic": "off-topic",
    },

    # ── VAGUE — should ask clarification or give partial answer ──

    {
        "id": "V001",
        "question": "tell me about oppe",
        "expected_keywords": ["OPPE", "exam", "SCT", "eligibility"],
        "expected_source": None,
        "chunk_type": "text",
        "topic": "vague OPPE",
    },
    {
        "id": "V002",
        "question": "what is the fee?",
        "expected_keywords": ["fee", "foundation", "credits"],
        "expected_source": None,
        "chunk_type": "text",
        "topic": "vague fee",
    },

    # ── RESTRICTED DOCS ───────────────────────────────────────────

    {
        "id": "RD001",
        "question": "Is there a document about the grading policy that requires IITM login?",
        "expected_keywords": ["restricted", "login", "IITM", "access", "docs.google.com"],
        "expected_source": None,
        "chunk_type": "restricted_doc",
        "topic": "restricted document",
    },

    # ── CONVERSATION / FOLLOW-UP ──────────────────────────────────

    {
        "id": "FU001",
        "question": "what about diploma level?",
        "history": [
            {"role": "user", "content": "what is the fee for foundation level from Jan 2026?"},
            {"role": "assistant", "content": "The fee for foundation level from Jan 2026 is ₹48,000."},
        ],
        "expected_keywords": ["diploma", "129000", "129,000", "fee"],
        "expected_source": "Student Handbook",
        "chunk_type": "text",
        "topic": "follow-up fee",
    },
    {
        "id": "FU002",
        "question": "what is the eligibility for OPPE2?",
        "history": [
            {"role": "user", "content": "tell me about OPPE for Python course"},
            {"role": "assistant", "content": "OPPE1 eligibility requires A1-A4 >= 40/100 and SCT completion."},
        ],
        "expected_keywords": ["OPPE2", "A5", "A6", "A7", "A8", "average"],
        "expected_source": "BS-DS_ Jan 2026 Grading document (STUDENT)",
        "chunk_type": "text",
        "topic": "follow-up OPPE",
    },
]


# ══════════════════════════════════════════════════════════════════
# RAG CALLER
# ══════════════════════════════════════════════════════════════════

def call_rag(question: str, history: list = []) -> tuple[str, list]:
    """
    INPUT:  question + history
    OUTPUT: (answer_text, sources)

    Calls the live backend SSE endpoint and reconstructs answer.
    """
    try:
        resp = requests.post(
            f"{BACKEND_URL}/ask",
            json={"question": question, "history": history},
            stream=True,
            timeout=TIMEOUT,
        )

        answer  = []
        sources = []

        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            try:
                data = json.loads(line[6:])
                if data.get("type") == "token":
                    answer.append(data.get("text", ""))
                elif data.get("type") == "sources":
                    sources = data.get("sources", [])
            except json.JSONDecodeError:
                continue

        return "".join(answer), sources

    except Exception as e:
        logger.error(f"RAG call failed: {e}")
        return "", []


# ══════════════════════════════════════════════════════════════════
# EVALUATORS
# ══════════════════════════════════════════════════════════════════

def check_answer(answer: str, expected_keywords: list) -> tuple[bool, list, list]:
    """
    INPUT:  answer text + expected keywords
    OUTPUT: (passed, found_keywords, missing_keywords)

    Checks if at least 50% of expected keywords appear in answer.
    Case-insensitive matching.
    """
    answer_lower = answer.lower()
    found   = [kw for kw in expected_keywords if kw.lower() in answer_lower]
    missing = [kw for kw in expected_keywords if kw.lower() not in answer_lower]
    passed  = len(found) >= max(1, len(expected_keywords) // 2)
    return passed, found, missing


def check_source(sources: list, expected_source: str) -> bool:
    """
    INPUT:  sources list + expected source name
    OUTPUT: True if expected source appears in any source

    Returns True if expected_source is None (off-topic questions)
    """
    if expected_source is None:
        return True
    source_str = json.dumps(sources).lower()
    return expected_source.lower() in source_str


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def run():
    print("\n" + "═" * 65)
    print("  IITM BS RAG Pipeline — Stage 7: Evaluator")
    print("═" * 65)

    # Check backend is running
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        info = resp.json()
        print(f"\n  ✅ Backend running")
        print(f"     LLM:     {info.get('llm_model','?')}")
        print(f"     Qdrant:  {info.get('qdrant','?')}")
    except Exception as e:
        print(f"\n  ❌ Backend not running: {e}")
        print(f"     Start with: uvicorn main:app --reload --port 8000")
        return

    total     = len(TEST_QUESTIONS)
    print(f"\n  Total questions: {total}")
    print(f"  Running evaluation...\n")

    # Results storage
    results        = []
    answer_pass    = 0
    source_pass    = 0
    by_type        = {}
    by_topic       = {}

    for i, q in enumerate(TEST_QUESTIONS):
        qid      = q["id"]
        question = q["question"]
        history  = q.get("history", [])
        chunk_type = q.get("chunk_type", "text")
        topic    = q.get("topic", "")

        print(f"  [{i+1:02d}/{total}] {qid} — {question[:55]}...")

        # Call RAG
        start  = time.time()
        answer, sources = call_rag(question, history)
        elapsed = time.time() - start

        # Evaluate
        ans_pass, found, missing = check_answer(answer, q["expected_keywords"])
        src_pass = check_source(sources, q.get("expected_source"))

        if ans_pass:
            answer_pass += 1
        if src_pass:
            source_pass += 1

        # Track by chunk type
        if chunk_type not in by_type:
            by_type[chunk_type] = {"pass": 0, "total": 0}
        by_type[chunk_type]["total"] += 1
        if ans_pass:
            by_type[chunk_type]["pass"] += 1

        # Track by topic
        if topic not in by_topic:
            by_topic[topic] = {"pass": 0, "total": 0}
        by_topic[topic]["total"] += 1
        if ans_pass:
            by_topic[topic]["pass"] += 1

        status = "✅" if ans_pass else "❌"
        print(f"         {status} answer | {'✅' if src_pass else '❌'} source | {elapsed:.1f}s")
        if not ans_pass:
            print(f"         Missing: {missing[:3]}")

        results.append({
            "id":            qid,
            "question":      question,
            "chunk_type":    chunk_type,
            "topic":         topic,
            "answer_pass":   ans_pass,
            "source_pass":   src_pass,
            "found_keywords": found,
            "missing_keywords": missing,
            "answer_preview": answer[:200],
            "sources":       sources,
            "elapsed":       round(elapsed, 2),
        })

        # Small delay to avoid rate limits
        time.sleep(20)

    # ── Final scores ──────────────────────────────────────────────
    answer_score = round(answer_pass / total * 100, 1)
    source_score = round(source_pass / total * 100, 1)

    # Score by chunk type
    type_scores = {}
    for t, d in by_type.items():
        type_scores[t] = round(d["pass"] / d["total"] * 100, 1)

    # Failed questions
    failed = [r for r in results if not r["answer_pass"]]

    # Build report
    report = {
        "run_at":        datetime.now().isoformat(),
        "total":         total,
        "answer_score":  answer_score,
        "source_score":  source_score,
        "by_chunk_type": type_scores,
        "by_topic":      {t: round(d["pass"]/d["total"]*100,1) for t,d in by_topic.items()},
        "failed_count":  len(failed),
        "failed":        failed,
        "all_results":   results,
    }

    # Save
    EVAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_OUTPUT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Print summary ─────────────────────────────────────────────
    print(f"\n  {'═' * 40}")
    print(f"  EVALUATION RESULTS")
    print(f"  {'═' * 40}")
    print(f"  Answer accuracy:  {answer_score}% ({answer_pass}/{total})")
    print(f"  Source accuracy:  {source_score}% ({source_pass}/{total})")

    print(f"\n  By chunk type:")
    for t, score in sorted(type_scores.items()):
        d = by_type[t]
        print(f"    {t:20s}: {score:5.1f}% ({d['pass']}/{d['total']})")

    if failed:
        print(f"\n  Failed questions ({len(failed)}):")
        for f in failed[:5]:
            print(f"    [{f['id']}] {f['question'][:55]}")
            print(f"           Missing: {f['missing_keywords'][:3]}")

    print(f"\n  Report saved: {EVAL_OUTPUT}")
    print(f"\n  Next step: fix weak areas, then rerun to measure improvement")
    print("═" * 65)


if __name__ == "__main__":
    run()