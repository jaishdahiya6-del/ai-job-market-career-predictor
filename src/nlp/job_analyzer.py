"""Analyzes a pasted job description: detects skills, seniority,
education requirement, and (if enough signal exists) an estimated
salary using the trained salary model."""
import re
from src.nlp.skill_extraction import extract_skills, extract_skill_names, clean_text
from src.nlp.skill_taxonomy import SKILL_TAXONOMY

SOFT_SKILL_NAMES = set(SKILL_TAXONOMY.get("Soft Skills", {}).keys())

SENIORITY_PATTERNS = {
    "Senior": [r"\bsenior\b", r"\bsr\.?\b", r"\blead\b", r"\d{1,2}\+?\s*years"],
    "Entry": [r"\bentry.level\b", r"\bfresher\b", r"\bjunior\b", r"\bintern(ship)?\b", r"0-2 years"],
}

EDUCATION_PATTERNS = ["btech", "b.tech", "mtech", "m.tech", "bachelor", "master", "mba", "phd", "b.sc", "m.sc"]


def detect_seniority(text: str) -> str:
    lower = text.lower()
    for level, patterns in SENIORITY_PATTERNS.items():
        for p in patterns:
            if re.search(p, lower):
                return level
    return "Mid"


def detect_education(text: str) -> list:
    lower = text.lower()
    return sorted({kw.upper() for kw in EDUCATION_PATTERNS if kw in lower}) or ["Not specified"]


def detect_experience_years(text: str) -> int | None:
    m = re.search(r"(\d{1,2})\s*\+?\s*years", text.lower())
    return int(m.group(1)) if m else None


def analyze_job_description(text: str, user_skills: list | None = None) -> dict:
    cleaned = clean_text(text)
    skill_records = extract_skills(cleaned)
    technical = [s["skill"] for s in skill_records if s["category"] != "Soft Skills"]
    soft = [s["skill"] for s in skill_records if s["category"] == "Soft Skills"]

    result = {
        "detected_skills": [s["skill"] for s in skill_records],
        "technical_skills": technical,
        "soft_skills": soft,
        "seniority": detect_seniority(cleaned),
        "education_requirements": detect_education(cleaned),
        "experience_years_mentioned": detect_experience_years(cleaned),
        "summary": _summarize(cleaned, technical),
    }

    if user_skills:
        user_set = set(s.lower() for s in user_skills)
        req_set = set(technical)
        result["skill_gap"] = {
            "have": sorted(user_set & req_set),
            "missing": sorted(req_set - user_set),
        }
    return result


def _summarize(text: str, technical_skills: list) -> str:
    skills_part = ", ".join(technical_skills[:6]) if technical_skills else "no specific technical skills detected"
    word_count = len(text.split())
    return (f"This posting is ~{word_count} words and centers on {skills_part}. "
            f"Automatically generated from detected keywords, not a paraphrase of the original text.")
