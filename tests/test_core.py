"""Comprehensive test suite covering preprocessing, NLP, models, recommendation, database, and REST API.
"""
import io
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from src.preprocessing.clean import (
    normalize_title,
    parse_experience_level,
    normalize_location,
    load_and_clean,
)
from src.nlp.skill_extraction import extract_skills, extract_skill_names, clean_text
from src.nlp.job_analyzer import analyze_job_description
from src.nlp.resume_analyzer import analyze_resume
from src.recommendation.recommend import recommend_roles, skill_gap, build_roadmap
from src.models.predict_salary import predict_salary
from src.utils.db import init_db, seed_from_processed_csv, get_session, Job, Skill
from app.api import app


# --- Preprocessing Tests ---

def test_normalize_title_alias():
    assert normalize_title("python dev") == "Python Developer"
    assert normalize_title(" senior data scientist ") == "Data Scientist"


def test_normalize_title_passthrough():
    assert normalize_title("Quantum Engineer") == "Quantum Engineer"
    assert normalize_title("") == "Unknown"
    assert normalize_title(None) == "Unknown"


def test_parse_experience_level():
    assert parse_experience_level("junior") == "Entry"
    assert parse_experience_level("mid-level") == "Mid"
    assert parse_experience_level("sr") == "Senior"
    assert parse_experience_level("Lead") == "Senior"
    assert parse_experience_level(None) == "Unknown"


def test_normalize_location():
    assert normalize_location("Bangalore", "Karnataka", "India") == "Bangalore, Karnataka, India"
    assert normalize_location("Remote", "", "India") == "Remote, India"
    assert normalize_location(None, None, None) == "Unknown"


def test_clean_text_strips_html():
    raw = "<p>We need a <b>Python</b> dev.</p>"
    cleaned = clean_text(raw)
    assert "<" not in cleaned
    assert "Python" in cleaned


def test_load_and_clean_raises_missing_file():
    with pytest.raises(FileNotFoundError):
        load_and_clean("non_existent_file.csv")


# --- NLP Skill Extraction Tests ---

def test_extract_skills_basic():
    text = "Hiring a Python and SQL engineer with Docker skills."
    skills = extract_skill_names(text)
    assert "python" in skills
    assert "sql" in skills
    assert "docker" in skills


def test_extract_skills_empty_text():
    assert extract_skills("") == []
    assert extract_skill_names(None) == []


def test_extract_skills_no_false_positive_substring():
    # 'c' as a programming language alias should not match inside 'docker'
    text = "Experience with docker and python"
    skills = extract_skill_names(text)
    assert "c" not in skills
    assert "docker" in skills


def test_analyze_job_description_detects_seniority():
    res = analyze_job_description("Looking for a Senior ML Engineer with 5+ years experience in Python and AWS.")
    assert res["seniority"] == "Senior"
    assert "python" in res["technical_skills"]
    assert "aws" in res["technical_skills"]
    assert res["experience_years_mentioned"] == 5


def test_analyze_job_description_empty():
    res = analyze_job_description("")
    assert res["detected_skills"] == []
    assert res["seniority"] == "Mid"


def test_analyze_resume_scoring():
    res = analyze_resume("John Doe\nExperienced Python, SQL, and Machine Learning engineer with Pandas expertise.")
    assert res["resume_score"] > 0
    assert "python" in res["skills_found"]
    assert "sql" in res["skills_found"]


# --- Recommendation Tests ---

def test_recommend_roles_returns_ranked_list():
    recs = recommend_roles(["python", "sql", "pandas", "machine learning"], top_k=3)
    assert len(recs) == 3
    assert "role" in recs[0]
    assert "match_score" in recs[0]
    assert recs[0]["match_score"] >= recs[1]["match_score"]


def test_skill_gap_unknown_role_raises():
    with pytest.raises(ValueError):
        skill_gap(["python"], target_role="NonExistentRole123")


def test_build_roadmap_structure():
    roadmap = build_roadmap(["python"])
    assert len(roadmap) == 4
    assert roadmap[0]["phase"] == "Foundation"
    assert "python" in roadmap[0]["already_have"]


# --- Model Tests ---

def test_predict_salary_returns_bounds():
    result = predict_salary(
        title="Data Analyst", location="Bangalore, Karnataka, India",
        experience_level="Mid", industry="IT Services", education="Bachelors",
        employment_type="Full-time", experience_years=3, skills=["python", "sql"],
    )
    assert result["predicted_salary"] > 0
    assert result["lower_bound"] <= result["predicted_salary"]
    assert result["upper_bound"] >= result["predicted_salary"]


# --- Database Tests ---

def test_db_seeding_and_session():
    init_db()
    seed_from_processed_csv("data/processed/jobs_processed.csv", limit=10)
    session = next(get_session())
    try:
        job_count = session.query(Job).count()
        assert job_count == 10
        skill_count = session.query(Skill).count()
        assert skill_count > 0
    finally:
        session.close()


# --- REST API Tests ---

client = TestClient(app)


def test_api_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True


def test_api_predict_salary():
    payload = {
        "title": "Data Scientist",
        "location": "Bangalore, Karnataka, India",
        "experience_level": "Mid",
        "industry": "IT Services",
        "education": "Bachelors",
        "employment_type": "Full-time",
        "experience_years": 3,
        "skills": ["python", "sql", "machine learning"]
    }
    response = client.post("/api/v1/predict-salary", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_salary" in data
    assert data["predicted_salary"] > 0


def test_api_recommend_roles():
    payload = {"skills": ["python", "sql"], "top_k": 3}
    response = client.post("/api/v1/recommend-roles", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 3


def test_api_skill_gap():
    payload = {"skills": ["python", "sql"], "target_role": "Data Scientist"}
    response = client.post("/api/v1/skill-gap", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["target_role"] == "Data Scientist"
    assert "have" in data
    assert "missing" in data


def test_api_analyze_job():
    payload = {"text": "Senior Python Developer with FastAPI and Docker experience required."}
    response = client.post("/api/v1/analyze-job", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "fastapi" in data["technical_skills"]


def test_api_analyze_job_empty_text():
    payload = {"text": "   "}
    response = client.post("/api/v1/analyze-job", json=payload)
    assert response.status_code == 400
