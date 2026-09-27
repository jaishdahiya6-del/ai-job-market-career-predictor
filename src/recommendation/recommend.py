"""Role recommendation, skill-gap analysis, and roadmap generation.
Uses actual Jaccard-similarity scoring against skill profiles derived
from the processed job dataset.
"""
import ast
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = ROOT / "data" / "processed" / "jobs_processed.csv"

ROADMAP_PHASES = [
    ("Foundation", ["python", "sql", "statistics", "excel"]),
    ("Data & Analysis", ["pandas", "numpy", "power bi", "tableau"]),
    ("Machine Learning", ["scikit-learn", "machine learning", "xgboost"]),
    ("Advanced / Specialization", ["deep learning", "nlp", "computer vision", "mlops", "aws", "docker"]),
]


def _load_role_profiles() -> Dict[str, Dict[str, Any]]:
    """Builds {role: {skill_freq, n_postings, avg_salary}} from processed dataset."""
    if not PROCESSED_PATH.exists():
        logger.error("Processed job data not found at %s", PROCESSED_PATH)
        raise FileNotFoundError(f"Processed dataset not found at {PROCESSED_PATH}. Run scripts/process_dataset.py first.")

    df = pd.read_csv(PROCESSED_PATH)
    if "skills_extracted" not in df.columns:
        logger.error("'skills_extracted' column missing in %s", PROCESSED_PATH)
        raise ValueError("Processed dataset missing 'skills_extracted' column.")

    df["skills_extracted"] = df["skills_extracted"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) and s.startswith("[") else []
    )

    profiles = {}
    for role, group in df.groupby("title_clean"):
        counter = Counter()
        for skills in group["skills_extracted"]:
            counter.update(skills)
        total = len(group)
        avg_sal = float(group["salary"].mean()) if "salary" in group.columns and not group["salary"].isnull().all() else 0.0
        profiles[role] = {
            "skill_freq": dict(counter),
            "n_postings": total,
            "avg_salary": avg_sal,
        }

    return profiles


def recommend_roles(user_skills: List[str], top_k: int = 5) -> List[Dict[str, Any]]:
    """Ranks roles by composite Jaccard + skill centrality matching score.

    Args:
        user_skills: List of candidate skills.
        top_k: Number of top roles to return.

    Returns:
        List[Dict[str, Any]]: Top matching role recommendations with match score.
    """
    profiles = _load_role_profiles()
    user_set = set(s.lower() for s in user_skills if isinstance(s, str))
    results = []

    for role, profile in profiles.items():
        role_skills = set(profile["skill_freq"].keys())
        if not role_skills:
            continue

        overlap = user_set & role_skills
        union = user_set | role_skills
        jaccard = len(overlap) / len(union) if union else 0.0

        # Skill centrality weighting
        weighted_overlap = sum(profile["skill_freq"][s] for s in overlap)
        total_weight = sum(profile["skill_freq"].values()) or 1
        centrality = weighted_overlap / total_weight

        score = 0.5 * jaccard + 0.5 * centrality
        results.append({
            "role": role,
            "match_score": round(score * 100, 1),
            "matched_skills": sorted(overlap),
            "avg_salary": round(profile["avg_salary"]),
            "n_postings": profile["n_postings"],
        })

    results.sort(key=lambda r: r["match_score"], reverse=True)
    logger.info("Generated %d role recommendations for %d user skills", min(top_k, len(results)), len(user_skills))
    return results[:top_k]


def skill_gap(user_skills: List[str], target_role: str, top_n_required: int = 8) -> Dict[str, Any]:
    """Computes skill gap for a target role against user's current skills.

    Args:
        user_skills: List of candidate skills.
        target_role: Name of target job role.
        top_n_required: Number of top required skills to evaluate.

    Returns:
        Dict[str, Any]: Detailed skill gap results.
    """
    profiles = _load_role_profiles()
    if target_role not in profiles:
        logger.error("Target role '%s' not found in profiles", target_role)
        raise ValueError(f"Unknown role '{target_role}'. Known roles: {list(profiles.keys())}")

    freq = profiles[target_role]["skill_freq"]
    required_ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n_required]
    required_skills = [s for s, _ in required_ranked]
    user_set = set(s.lower() for s in user_skills if isinstance(s, str))

    have = [s for s in required_skills if s in user_set]
    missing = [s for s in required_skills if s not in user_set]
    priority = sorted(missing, key=lambda s: freq.get(s, 0), reverse=True)

    return {
        "target_role": target_role,
        "have": have,
        "missing": missing,
        "priority_to_learn": priority[:5],
    }


def build_roadmap(user_skills: List[str]) -> List[Dict[str, Any]]:
    """Builds a 4-phase career learning roadmap based on missing skills.

    Args:
        user_skills: List of current skills.

    Returns:
        List[Dict[str, Any]]: Learning roadmap split into phases.
    """
    user_set = set(s.lower() for s in user_skills if isinstance(s, str))
    roadmap = []
    for phase_name, phase_skills in ROADMAP_PHASES:
        missing = [s for s in phase_skills if s not in user_set]
        roadmap.append({
            "phase": phase_name,
            "skills_needed": phase_skills,
            "already_have": [s for s in phase_skills if s in user_set],
            "to_learn": missing,
        })
    return roadmap
