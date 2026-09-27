"""Skill extraction: rule-based alias matching over cleaned text.
Scans text for taxonomy aliases using word-boundary regex matching.
"""
import re
from typing import Dict, List
from src.nlp.skill_taxonomy import flat_alias_map
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ALIAS_MAP = flat_alias_map()
# Sort longest-alias-first so multi-word aliases match before substring aliases
_SORTED_ALIASES = sorted(_ALIAS_MAP.keys(), key=len, reverse=True)


def _compile_pattern(alias: str) -> re.Pattern:
    escaped = re.escape(alias.strip())
    return re.compile(rf"(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9])", re.IGNORECASE)


_PATTERNS: Dict[str, re.Pattern] = {alias: _compile_pattern(alias) for alias in _SORTED_ALIASES}


def clean_text(text: str) -> str:
    """Strips HTML tags, normalizes whitespace, and cleans raw string.

    Args:
        text: Raw input string.

    Returns:
        str: Cleaned text string.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_skills(text: str) -> List[Dict[str, str]]:
    """Returns a list of {skill, category} dicts found in the text, deduped.

    Args:
        text: Input job description or resume text.

    Returns:
        List[Dict[str, str]]: List of dictionaries containing skill and category.
    """
    cleaned = clean_text(text).lower()
    if not cleaned:
        return []

    found: Dict[str, str] = {}
    for alias, pattern in _PATTERNS.items():
        if pattern.search(cleaned):
            canonical, category = _ALIAS_MAP[alias]
            found[canonical] = category

    logger.debug("Extracted %d skills from text", len(found))
    return [{"skill": k, "category": v} for k, v in sorted(found.items())]


def extract_skill_names(text: str) -> List[str]:
    """Returns a list of unique canonical skill names found in text.

    Args:
        text: Input text.

    Returns:
        List[str]: List of canonical skill names.
    """
    return [d["skill"] for d in extract_skills(text)]


if __name__ == "__main__":
    sample = "Looking for a Python Developer with SQL, Power BI and strong communication skills."
    skills = extract_skills(sample)
    logger.info("Extracted skills from sample: %s", skills)
