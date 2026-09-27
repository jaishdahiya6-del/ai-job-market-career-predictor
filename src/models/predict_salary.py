"""Loads trained salary model and serves salary predictions with confidence bounds.
"""
import json
from pathlib import Path
from typing import Any, Dict, List
import joblib
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"

_model = None
_mlb = None
_feature_columns = None
_residual_std = 0.0


def _load() -> None:
    global _model, _mlb, _feature_columns, _residual_std
    if _model is not None:
        return

    model_path = MODEL_DIR / "salary_model.joblib"
    binarizer_path = MODEL_DIR / "skill_binarizer.joblib"
    features_path = MODEL_DIR / "feature_columns.joblib"
    metrics_path = MODEL_DIR / "salary_model_metrics.json"

    if not model_path.exists():
        logger.error("Salary model file not found at %s", model_path)
        raise FileNotFoundError(
            "No trained salary model found. Run: python scripts/train_salary_model.py"
        )

    try:
        _model = joblib.load(model_path)
        _mlb = joblib.load(binarizer_path)
        _feature_columns = joblib.load(features_path)

        if metrics_path.exists():
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
            _residual_std = metrics.get("residual_std", 0.0)
        else:
            _residual_std = 0.0

        logger.info("Successfully loaded salary model and artifacts from %s", MODEL_DIR)
    except Exception as e:
        logger.error("Failed to load model artifacts: %s", e)
        raise RuntimeError(f"Error loading model artifacts: {e}") from e


def predict_salary(
    title: str,
    location: str,
    experience_level: str,
    industry: str,
    education: str,
    employment_type: str,
    experience_years: int,
    skills: List[str],
) -> Dict[str, Any]:
    """Predicts salary and returns range estimation based on input attributes.

    Args:
        title: Job title.
        location: Location string.
        experience_level: Seniority/experience level.
        industry: Industry name.
        education: Minimum required education.
        employment_type: Type of employment.
        experience_years: Years of experience.
        skills: List of candidate/required skills.

    Returns:
        Dict[str, Any]: Predicted salary point estimate, lower and upper bounds.
    """
    _load()

    known_skills = [s for s in skills if s in getattr(_mlb, "classes_", [])]
    skill_vec = _mlb.transform([known_skills])[0]

    row: Dict[str, Any] = {
        "title_clean": title,
        "location_clean": location,
        "experience_level": experience_level,
        "industry": industry,
        "education": education,
        "employment_type": employment_type,
        "experience_years": experience_years,
        "n_skills": len(skills),
    }

    for i, cls in enumerate(_mlb.classes_):
        row[f"skill_{cls}"] = skill_vec[i]

    X = pd.DataFrame([row])
    X = X.reindex(columns=_feature_columns, fill_value=0)

    pred = float(_model.predict(X)[0])
    band = 1.28 * _residual_std  # ~80% interval assuming normal residuals

    result = {
        "predicted_salary": round(pred),
        "lower_bound": round(max(0, pred - band)),
        "upper_bound": round(pred + band),
        "unmatched_skills": [s for s in skills if s not in getattr(_mlb, "classes_", [])],
        "disclaimer": "Estimate only, not a guaranteed salary offer.",
    }

    logger.info("Predicted salary for %s: %d (range: %d - %d)", title, result["predicted_salary"], result["lower_bound"], result["upper_bound"])
    return result
