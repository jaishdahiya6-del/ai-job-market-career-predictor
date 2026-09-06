import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.nlp.skill_extraction import extract_skills, extract_skill_names, clean_text
from src.preprocessing.clean import normalize_title, parse_experience_level, normalize_location
from src.recommendation.recommend import recommend_roles, skill_gap, build_roadmap
from src.models.predict_salary import predict_salary
from src.nlp.job_analyzer import analyze_job_description


# ---------- Preprocessing ----------

def test_normalize_title_alias():
    assert normalize_title("Python Dev") == "Python Developer"


def test_normalize_title_passthrough():
    assert normalize_title("Data Scientist") == "Data Scientist"


def test_parse_experience_level():
    assert parse_experience_level("Junior") == "Entry"
    assert parse_experience_level("sr") == "Senior"


def test_normalize_location():
    assert normalize_location("Pune", "Maharashtra", "India") == "Pune, Maharashtra, India"
    assert normalize_location(None, None, None) == "Unknown"


def test_clean_text_strips_html():
    assert "<p>" not in clean_text("<p>Hello</p>")


# ---------- Skill extraction ----------

def test_extract_skills_basic():
    names = extract_skill_names("Looking for a Python developer with SQL and Power BI skills.")
    assert "python" in names
    assert "sql" in names
    assert "power bi" in names


def test_extract_skills_empty_text():
    assert extract_skills("") == []


def test_extract_skills_no_false_positive_substring():
    names = extract_skill_names("We use Docker and Kubernetes.")
    assert "c" not in names
    assert "docker" in names


# ---------- Recommendation ----------

def test_recommend_roles_returns_ranked_list():
    recs = recommend_roles(["python", "sql", "pandas", "power bi"], top_k=3)
    assert len(recs) == 3
    scores = [r["match_score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_skill_gap_unknown_role_raises():
    with pytest.raises(ValueError):
        skill_gap(["python"], "Nonexistent Role")


def test_build_roadmap_structure():
    roadmap = build_roadmap(["python"])
    assert len(roadmap) == 4
    assert "python" in roadmap[0]["already_have"]


# ---------- Salary prediction ----------

def test_predict_salary_returns_bounds():
    result = predict_salary(
        title="Data Analyst", location="Bangalore, Karnataka, India",
        experience_level="Mid", industry="IT Services", education="Bachelors",
        employment_type="Full-time", experience_years=3, skills=["python", "sql"],
    )
    assert result["lower_bound"] <= result["predicted_salary"] <= result["upper_bound"]


# ---------- Job description analyzer ----------

def test_analyze_job_description_detects_seniority():
    result = analyze_job_description("Senior Python developer needed, 8+ years experience.")
    assert result["seniority"] == "Senior"
    assert "python" in result["technical_skills"]
