"""Analyzes a pasted job description: detects skills, seniority,
education requirements, experience years, and generates a summary and skill gap analysis.
"""
import re
from typing import Any, Dict, List, Optional
from src.nlp.skill_extraction import extract_skills, clean_text
from src.nlp.skill_taxonomy import SKILL_TAXONOMY
from src.utils.logger import get_logger

logger = get_logger(__name__)

SOFT_SKILL_NAMES = set(SKILL_TAXONOMY.get("Soft Skills", {}).keys())

SENIORITY_PATTERNS = {
    "Senior": [r"\bsenior\b", r"\bsr\.?\b", r"\blead\b", r"\bprincipal\b", r"\d{1,2}\+?\s*years"],
    "Entry": [r"\bentry.level\b", r"\bfresher\b", r"\bjunior\b", r"\bintern(ship)?\b", r"0-2 years"],
}

EDUCATION_PATTERNS = ["btech", "b.tech", "mtech", "m.tech", "bachelor", "master", "mba", "phd", "b.sc", "m.sc"]


def detect_seniority(text: str) -> str:
    """Detects seniority level from text using keyword patterns.

    Args:
        text: Input text string.

    Returns:
        str: Seniority level ("Senior", "Entry", or "Mid").
    """
    lower = text.lower()
    for level, patterns in SENIORITY_PATTERNS.items():
        for p in patterns:
            if re.search(p, lower):
                return level
    return "Mid"


def detect_education(text: str) -> List[str]:
    """Detects educational requirements mentioned in text.

    Args:
        text: Input text string.

    Returns:
        List[str]: List of upper-case education keywords detected or ["Not specified"].
    """
    lower = text.lower()
    found = sorted({kw.upper() for kw in EDUCATION_PATTERNS if kw in lower})
    return found if found else ["Not specified"]


def detect_experience_years(text: str) -> Optional[int]:
    """Detects required experience in years mentioned in text.

    Args:
        text: Input text string.

    Returns:
        Optional[int]: Number of experience years if detected, else None.
    """
    m = re.search(r"(\d{1,2})\s*\+?\s*years", text.lower())
    return int(m.group(1)) if m else None


def analyze_job_description(text: str, user_skills: Optional[List[str]] = None) -> Dict[str, Any]:
    """Analyzes job description to extract structured insight and optional skill gap.

    Args:
        text: Job description text.
        user_skills: Optional list of user's current skills.

    Returns:
        Dict[str, Any]: Extracted attributes and optional skill gap analysis.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "detected_skills": [],
            "technical_skills": [],
            "soft_skills": [],
            "seniority": "Mid",
            "education_requirements": ["Not specified"],
            "experience_years_mentioned": None,
            "summary": "Empty job description provided.",
        }

    skill_records = extract_skills(cleaned)
    technical = [s["skill"] for s in skill_records if s["category"] != "Soft Skills"]
    soft = [s["skill"] for s in skill_records if s["category"] == "Soft Skills"]

    result: Dict[str, Any] = {
        "detected_skills": [s["skill"] for s in skill_records],
        "technical_skills": technical,
        "soft_skills": soft,
        "seniority": detect_seniority(cleaned),
        "education_requirements": detect_education(cleaned),
        "experience_years_mentioned": detect_experience_years(cleaned),
        "summary": _summarize(cleaned, technical),
    }

    if user_skills is not None:
        user_set = set(s.lower() for s in user_skills)
        req_set = set(technical)
        result["skill_gap"] = {
            "have": sorted(user_set & req_set),
            "missing": sorted(req_set - user_set),
        }

    logger.info("Analyzed job description (~%d words, %d skills detected)", len(cleaned.split()), len(skill_records))
    return result


def _summarize(text: str, technical_skills: List[str]) -> str:
    skills_part = ", ".join(technical_skills[:6]) if technical_skills else "no specific technical skills detected"
    word_count = len(text.split())
    return (f"This posting is ~{word_count} words and centers on {skills_part}. "
            f"Automatically generated from detected keywords, not a paraphrase of the original text.")
