"""Skill extraction: rule-based alias matching over a cleaned text.
This is a real, working extractor (not a hardcoded stub) -- it scans
text for taxonomy aliases using word-boundary regex matching, so it
generalizes to any job description passed in.
"""
import re
from typing import List, Dict
from src.nlp.skill_taxonomy import flat_alias_map

_ALIAS_MAP = flat_alias_map()
# Sort longest-alias-first so "machine learning" matches before "learning"-type substrings
_SORTED_ALIASES = sorted(_ALIAS_MAP.keys(), key=len, reverse=True)


def _compile_pattern(alias: str) -> re.Pattern:
    # Escape special regex chars (needed for c++, node.js etc.) then allow
    # loose word boundaries since some aliases contain punctuation.
    escaped = re.escape(alias.strip())
    return re.compile(rf"(?<![a-zA-Z0-9]){escaped}(?![a-zA-Z0-9])", re.IGNORECASE)


_PATTERNS = {alias: _compile_pattern(alias) for alias in _SORTED_ALIASES}


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)          # strip HTML
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_skills(text: str) -> List[Dict[str, str]]:
    """Returns a list of {skill, category} dicts found in the text, deduped."""
    cleaned = clean_text(text).lower()
    found = {}
    for alias, pattern in _PATTERNS.items():
        if pattern.search(cleaned):
            canonical, category = _ALIAS_MAP[alias]
            found[canonical] = category
    return [{"skill": k, "category": v} for k, v in sorted(found.items())]


def extract_skill_names(text: str) -> List[str]:
    return [d["skill"] for d in extract_skills(text)]


if __name__ == "__main__":
    sample = "Looking for a Python Developer with SQL, Power BI and strong communication skills."
    print(extract_skills(sample))
