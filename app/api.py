"""FastAPI application serving RESTful endpoints for Career IQ platform.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, File, UploadFile, status
from pydantic import BaseModel, Field

from src.models.predict_salary import predict_salary
from src.recommendation.recommend import recommend_roles, skill_gap, build_roadmap, _load_role_profiles
from src.nlp.job_analyzer import analyze_job_description
from src.nlp.resume_analyzer import extract_text_from_pdf, analyze_resume
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Career IQ API",
    description="AI Job Market Intelligence & Career Prediction Platform API",
    version="1.0.0",
)


# --- Schemas ---

class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    model_loaded: bool


class SalaryPredictionRequest(BaseModel):
    title: str = Field(..., json_schema_extra={"example": "Data Scientist"})
    location: str = Field(..., json_schema_extra={"example": "Bangalore, Karnataka, India"})
    experience_level: str = Field(..., json_schema_extra={"example": "Mid"})
    industry: str = Field(default="IT Services", json_schema_extra={"example": "IT Services"})
    education: str = Field(default="Bachelors", json_schema_extra={"example": "Bachelors"})
    employment_type: str = Field(default="Full-time", json_schema_extra={"example": "Full-time"})
    experience_years: int = Field(default=3, ge=0, le=50, json_schema_extra={"example": 3})
    skills: List[str] = Field(..., json_schema_extra={"example": ["python", "sql", "machine learning"]})


class SalaryPredictionResponse(BaseModel):
    predicted_salary: int
    lower_bound: int
    upper_bound: int
    unmatched_skills: List[str]
    disclaimer: str


class RecommendRolesRequest(BaseModel):
    skills: List[str] = Field(..., json_schema_extra={"example": ["python", "sql"]})
    top_k: int = Field(default=5, ge=1, le=20)


class RoleRecommendation(BaseModel):
    role: str
    match_score: float
    matched_skills: List[str]
    avg_salary: int
    n_postings: int


class SkillGapRequest(BaseModel):
    skills: List[str] = Field(..., json_schema_extra={"example": ["python", "sql"]})
    target_role: str = Field(..., json_schema_extra={"example": "Data Scientist"})


class SkillGapResponse(BaseModel):
    target_role: str
    have: List[str]
    missing: List[str]
    priority_to_learn: List[str]


class JobAnalyzerRequest(BaseModel):
    text: str = Field(..., json_schema_extra={"example": "We are looking for a Senior Data Scientist with Python and SQL experience."})
    user_skills: Optional[List[str]] = Field(default=None, json_schema_extra={"example": ["python"]})


class JobAnalyzerResponse(BaseModel):
    detected_skills: List[str]
    technical_skills: List[str]
    soft_skills: List[str]
    seniority: str
    education_requirements: List[str]
    experience_years_mentioned: Optional[int]
    summary: str
    skill_gap: Optional[Dict[str, List[str]]] = None


class ResumeAnalyzerResponse(BaseModel):
    name_guess: str
    education_guess: List[str]
    skills_found: List[str]
    resume_score: int
    missing_high_demand_skills: List[str]
    n_skills_found: int


# --- Endpoints ---

@app.get("/health", response_model=HealthResponse)
def health_check():
    model_path = Path(__file__).resolve().parents[1] / "models" / "salary_model.joblib"
    return HealthResponse(status="ok", version="1.0.0", model_loaded=model_path.exists())


@app.post("/api/v1/predict-salary", response_model=SalaryPredictionResponse)
def api_predict_salary(req: SalaryPredictionRequest):
    try:
        res = predict_salary(
            title=req.title,
            location=req.location,
            experience_level=req.experience_level,
            industry=req.industry,
            education=req.education,
            employment_type=req.employment_type,
            experience_years=req.experience_years,
            skills=req.skills,
        )
        return SalaryPredictionResponse(**res)
    except FileNotFoundError as e:
        logger.error("Salary model not found: %s", e)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)) from e
    except Exception as e:
        logger.error("Salary prediction error: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction error: {e}") from e


@app.post("/api/v1/recommend-roles", response_model=List[RoleRecommendation])
def api_recommend_roles(req: RecommendRolesRequest):
    try:
        recs = recommend_roles(user_skills=req.skills, top_k=req.top_k)
        return [RoleRecommendation(**r) for r in recs]
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)) from e


@app.post("/api/v1/skill-gap", response_model=SkillGapResponse)
def api_skill_gap(req: SkillGapRequest):
    try:
        gap = skill_gap(user_skills=req.skills, target_role=req.target_role)
        return SkillGapResponse(**gap)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except FileNotFoundError as e:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e)) from e


@app.post("/api/v1/analyze-job", response_model=JobAnalyzerResponse)
def api_analyze_job(req: JobAnalyzerRequest):
    if not req.text.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Job description text cannot be empty.")
    res = analyze_job_description(text=req.text, user_skills=req.user_skills)
    return JobAnalyzerResponse(**res)


@app.post("/api/v1/analyze-resume", response_model=ResumeAnalyzerResponse)
async def api_analyze_resume(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only PDF files are supported.")
    try:
        contents = await file.read()
        text = extract_text_from_pdf(contents)
        res = analyze_resume(text)
        return ResumeAnalyzerResponse(**res)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to analyze resume: {e}") from e
