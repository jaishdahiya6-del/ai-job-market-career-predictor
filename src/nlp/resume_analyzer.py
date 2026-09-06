"""Extracts text from an uploaded PDF resume, pulls skills via the same
extraction pipeline used for job postings, and scores/compares against
market demand computed from the processed dataset."""
import ast
import re
from pathlib import Path
from collections import Counter
import pandas as pd
from pypdf import PdfReader
from src.nlp.skill_extraction import extract_skill_names

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = ROOT / "data" / "processed" / "jobs_processed.csv"


def extract_text_from_pdf(file_path_or_buffer) -> str:
    try:
        reader = PdfReader(file_path_or_buffer)
    except Exception as e:
        raise ValueError(f"Could not read PDF: {e}")
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    if not text.strip():
        raise ValueError("No extractable text found in PDF (it may be a scanned image).")
    return text


def _market_skill_demand() -> Counter:
    df = pd.read_csv(PROCESSED_PATH)
    counter = Counter()
    for s in df["skills_extracted"]:
        skills = ast.literal_eval(s) if isinstance(s, str) else []
        counter.update(skills)
    return counter


def guess_name(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    for line in lines[:5]:
        if 2 <= len(line.split()) <= 4 and not any(ch.isdigit() for ch in line) and "@" not in line:
            return line
    return "Not detected"


def guess_education(text: str) -> list:
    keywords = ["btech", "b.tech", "mtech", "m.tech", "bachelor", "master", "b.sc", "m.sc", "mba", "phd"]
    found = []
    lower = text.lower()
    for kw in keywords:
        if kw in lower:
            found.append(kw.upper())
    return sorted(set(found)) or ["Not detected"]


def analyze_resume(text: str) -> dict:
    skills = extract_skill_names(text)
    demand = _market_skill_demand()
    max_demand = max(demand.values()) if demand else 1

    # Score = 70% average relative market-demand of the user's skills
    #       + 30% breadth (number of distinct skills, capped at 15)
    if skills:
        avg_relative_demand = sum(demand.get(s, 0) / max_demand for s in skills) / len(skills)
        resume_score = round(avg_relative_demand * 70 + min(len(skills), 15) / 15 * 30)
    else:
        resume_score = 0
    resume_score = max(0, min(100, resume_score))

    top_market_skills = [s for s, _ in demand.most_common(15)]
    missing_high_demand = [s for s in top_market_skills if s not in skills][:8]

    return {
        "name_guess": guess_name(text),
        "education_guess": guess_education(text),
        "skills_found": skills,
        "resume_score": resume_score,
        "missing_high_demand_skills": missing_high_demand,
        "n_skills_found": len(skills),
    }
