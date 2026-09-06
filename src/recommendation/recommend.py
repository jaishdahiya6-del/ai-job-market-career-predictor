"""Role recommendation, skill-gap analysis, and roadmap generation.
Uses actual Jaccard-similarity scoring against skill profiles derived
from the processed job dataset -- not hardcoded answers.
"""
import ast
import pandas as pd
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_PATH = ROOT / "data" / "processed" / "jobs_processed.csv"

ROADMAP_PHASES = [
    ("Foundation", ["python", "sql", "statistics", "excel"]),
    ("Data & Analysis", ["pandas", "numpy", "power bi", "tableau"]),
    ("Machine Learning", ["scikit-learn", "machine learning", "xgboost"]),
    ("Advanced / Specialization", ["deep learning", "nlp", "computer vision", "mlops", "aws", "docker"]),
]


def _load_role_profiles() -> dict:
    """Builds {role: {skill: frequency}} from the processed dataset."""
    df = pd.read_csv(PROCESSED_PATH)
    df["skills_extracted"] = df["skills_extracted"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) else []
    )
    profiles = {}
    for role, group in df.groupby("title_clean"):
        counter = Counter()
        for skills in group["skills_extracted"]:
            counter.update(skills)
        total = len(group)
        profiles[role] = {"skill_freq": dict(counter), "n_postings": total,
                           "avg_salary": float(group["salary"].mean())}
    return profiles


def recommend_roles(user_skills: list, top_k: int = 5) -> list:
    profiles = _load_role_profiles()
    user_set = set(s.lower() for s in user_skills)
    results = []
    for role, profile in profiles.items():
        role_skills = set(profile["skill_freq"].keys())
        if not role_skills:
            continue
        overlap = user_set & role_skills
        union = user_set | role_skills
        jaccard = len(overlap) / len(union) if union else 0.0
        # weight by how central the matched skills are to the role
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
    return results[:top_k]


def skill_gap(user_skills: list, target_role: str, top_n_required: int = 8) -> dict:
    profiles = _load_role_profiles()
    if target_role not in profiles:
        raise ValueError(f"Unknown role '{target_role}'. Known roles: {list(profiles.keys())}")
    freq = profiles[target_role]["skill_freq"]
    required_ranked = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top_n_required]
    required_skills = [s for s, _ in required_ranked]
    user_set = set(s.lower() for s in user_skills)

    have = [s for s in required_skills if s in user_set]
    missing = [s for s in required_skills if s not in user_set]
    # priority = the missing skills that appear most frequently for this role
    priority = sorted(missing, key=lambda s: freq.get(s, 0), reverse=True)

    return {
        "target_role": target_role,
        "have": have,
        "missing": missing,
        "priority_to_learn": priority[:5],
    }


def build_roadmap(user_skills: list) -> list:
    user_set = set(s.lower() for s in user_skills)
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
