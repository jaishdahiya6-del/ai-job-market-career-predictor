"""Cleans raw job-posting data into a processed dataset ready for
analytics, salary modeling, and recommendation. Reusable functions,
not notebook-only logic.
"""
import re
import pandas as pd
from src.nlp.skill_extraction import extract_skill_names, clean_text

TITLE_ALIASES = {
    "python dev": "Python Developer",
    "python developer": "Python Developer",
    "python software developer": "Python Developer",
    "sr data scientist": "Data Scientist",
    "senior data scientist": "Data Scientist",
}


def normalize_title(title: str) -> str:
    if not isinstance(title, str):
        return "Unknown"
    key = title.strip().lower()
    return TITLE_ALIASES.get(key, title.strip())


def normalize_location(city: str, state: str, country: str) -> str:
    parts = [p for p in [city, state, country] if isinstance(p, str) and p.strip()]
    return ", ".join(parts) if parts else "Unknown"


def parse_experience_level(level: str) -> str:
    if not isinstance(level, str):
        return "Unknown"
    level = level.strip().lower()
    if level in ("entry", "junior", "fresher"):
        return "Entry"
    if level in ("mid", "mid-level", "intermediate"):
        return "Mid"
    if level in ("senior", "sr"):
        return "Senior"
    return level.title()


def load_and_clean(raw_csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_csv_path)

    # Drop exact duplicate postings
    df = df.drop_duplicates(subset=["title", "company", "city", "description"])

    # Missing-value handling
    df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
    df = df.dropna(subset=["salary", "title"])
    df["description"] = df["description"].fillna("")
    df["industry"] = df["industry"].fillna("Unknown")
    df["education"] = df["education"].fillna("Unknown")

    # Normalize fields
    df["title_clean"] = df["title"].apply(normalize_title)
    df["location_clean"] = df.apply(
        lambda r: normalize_location(r.get("city"), r.get("state"), r.get("country")), axis=1
    )
    if "experience_level" in df.columns:
        df["experience_level"] = df["experience_level"].apply(parse_experience_level)

    df["description_clean"] = df["description"].apply(clean_text)

    # Skill extraction from description (falls back to skills_raw column if present)
    def extract_row_skills(row):
        text = row["description_clean"]
        if isinstance(row.get("skills_raw"), str):
            text = text + " " + row["skills_raw"]
        return extract_skill_names(text)

    df["skills_extracted"] = df.apply(extract_row_skills, axis=1)
    df["skills_str"] = df["skills_extracted"].apply(lambda s: ", ".join(s))
    df["n_skills"] = df["skills_extracted"].apply(len)

    if "date_posted" in df.columns:
        df["date_posted"] = pd.to_datetime(df["date_posted"], errors="coerce")

    return df.reset_index(drop=True)


if __name__ == "__main__":
    out = load_and_clean("data/raw/jobs_demo.csv")
    out.to_csv("data/processed/jobs_processed.csv", index=False)
    print(f"Processed {len(out)} rows -> data/processed/jobs_processed.csv")
