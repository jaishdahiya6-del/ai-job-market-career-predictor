"""Cleans raw job-posting data into a processed dataset ready for
analytics, salary modeling, and recommendation. Reusable functions,
not notebook-only logic.
"""
import re
from pathlib import Path
from typing import Dict, Optional
import pandas as pd

from src.nlp.skill_extraction import extract_skill_names, clean_text
from src.utils.logger import get_logger

logger = get_logger(__name__)

TITLE_ALIASES: Dict[str, str] = {
    "python dev": "Python Developer",
    "python developer": "Python Developer",
    "python software developer": "Python Developer",
    "sr data scientist": "Data Scientist",
    "senior data scientist": "Data Scientist",
}


def normalize_title(title: str) -> str:
    """Normalizes job titles using defined aliases.

    Args:
        title: Raw job title.

    Returns:
        str: Normalized job title string.
    """
    if not isinstance(title, str) or not title.strip():
        return "Unknown"
    key = title.strip().lower()
    return TITLE_ALIASES.get(key, title.strip())


def normalize_location(city: Optional[str], state: Optional[str], country: Optional[str]) -> str:
    """Formats city, state, and country into a normalized string.

    Args:
        city: City name.
        state: State name.
        country: Country name.

    Returns:
        str: Formatted location string or "Unknown".
    """
    parts = [p.strip() for p in [city, state, country] if isinstance(p, str) and p.strip()]
    return ", ".join(parts) if parts else "Unknown"


def parse_experience_level(level: str) -> str:
    """Standardizes experience level to Entry, Mid, or Senior.

    Args:
        level: Raw experience level string.

    Returns:
        str: Standardized experience level.
    """
    if not isinstance(level, str) or not level.strip():
        return "Unknown"
    level_lower = level.strip().lower()
    if level_lower in ("entry", "junior", "fresher"):
        return "Entry"
    if level_lower in ("mid", "mid-level", "intermediate"):
        return "Mid"
    if level_lower in ("senior", "sr", "lead"):
        return "Senior"
    return level.strip().title()


def load_and_clean(raw_csv_path: str) -> pd.DataFrame:
    """Loads raw job dataset, validates required columns, cleans fields, and extracts skills.

    Args:
        raw_csv_path: Path to the raw CSV dataset.

    Returns:
        pd.DataFrame: Processed pandas DataFrame.
    """
    path = Path(raw_csv_path)
    if not path.exists():
        logger.error("Raw CSV file not found: %s", raw_csv_path)
        raise FileNotFoundError(f"Raw CSV file not found: {raw_csv_path}")

    logger.info("Loading raw dataset from %s", raw_csv_path)
    df = pd.read_csv(raw_csv_path)

    # Required columns check
    required_cols = {"title", "salary"}
    if not required_cols.issubset(set(df.columns)):
        missing = required_cols - set(df.columns)
        logger.error("Missing required columns in dataset: %s", missing)
        raise ValueError(f"Missing required columns in dataset: {missing}")

    # Drop exact duplicate postings
    initial_len = len(df)
    subset_cols = [c for c in ["title", "company", "city", "description"] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols)

    # Missing-value handling
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df = df.dropna(subset=["salary", "title"])
    df = df[df["salary"] > 0]  # Ensure positive salary

    if "description" not in df.columns:
        df["description"] = ""
    else:
        df["description"] = df["description"].fillna("")

    df["industry"] = df["industry"].fillna("Unknown") if "industry" in df.columns else "Unknown"
    df["education"] = df["education"].fillna("Unknown") if "education" in df.columns else "Unknown"
    df["employment_type"] = df["employment_type"].fillna("Full-time") if "employment_type" in df.columns else "Full-time"

    # Normalize fields
    df["title_clean"] = df["title"].apply(normalize_title)
    df["location_clean"] = df.apply(
        lambda r: normalize_location(r.get("city"), r.get("state"), r.get("country")), axis=1
    )
    if "experience_level" in df.columns:
        df["experience_level"] = df["experience_level"].apply(parse_experience_level)
    else:
        df["experience_level"] = "Mid"

    df["description_clean"] = df["description"].apply(clean_text)

    # Skill extraction from description
    def extract_row_skills(row):
        text = row["description_clean"]
        if isinstance(row.get("skills_raw"), str) and row["skills_raw"].strip():
            text = text + " " + row["skills_raw"]
        return extract_skill_names(text)

    df["skills_extracted"] = df.apply(extract_row_skills, axis=1)
    df["skills_str"] = df["skills_extracted"].apply(lambda s: ", ".join(s))
    df["n_skills"] = df["skills_extracted"].apply(len)

    if "date_posted" in df.columns:
        df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")

    logger.info("Successfully cleaned dataset. %d -> %d valid records", initial_len, len(df))
    return df.reset_index(drop=True)


if __name__ == "__main__":
    out = load_and_clean("data/raw/jobs_demo.csv")
    out_path = Path("data/processed/jobs_processed.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    logger.info("Processed %d rows -> %s", len(out), out_path)
