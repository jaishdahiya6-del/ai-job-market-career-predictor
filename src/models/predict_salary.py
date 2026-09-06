"""Loads the trained salary model and exposes predict_salary(). Raises
a clear FileNotFoundError if the model hasn't been trained yet."""
import joblib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"

_model = None
_mlb = None
_feature_columns = None
_residual_std = 0.0


def _load():
    global _model, _mlb, _feature_columns, _residual_std
    if _model is not None:
        return
    model_path = MODEL_DIR / "salary_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            "No trained salary model found. Run: python scripts/train_salary_model.py"
        )
    _model = joblib.load(model_path)
    _mlb = joblib.load(MODEL_DIR / "skill_binarizer.joblib")
    _feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")
    import json
    with open(MODEL_DIR / "salary_model_metrics.json") as f:
        metrics = json.load(f)
    _residual_std = metrics.get("residual_std", 0.0)


def predict_salary(title: str, location: str, experience_level: str, industry: str,
                    education: str, employment_type: str, experience_years: int,
                    skills: list) -> dict:
    _load()
    known_skills = [s for s in skills if s in _mlb.classes_]
    skill_vec = _mlb.transform([known_skills])[0]
    row = {
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
    return {
        "predicted_salary": round(pred),
        "lower_bound": round(max(0, pred - band)),
        "upper_bound": round(pred + band),
        "unmatched_skills": [s for s in skills if s not in _mlb.classes_],
        "disclaimer": "Estimate only, not a guaranteed salary offer.",
    }
