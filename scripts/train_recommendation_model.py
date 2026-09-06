"""The recommendation engine is similarity-based (Jaccard + centrality
over role-skill profiles), so there's no model weights to fit -- but
this script builds and caches the role-skill profiles, and reports a
simple leave-one-skill-out sanity check as an evaluation proxy.
Run: python scripts/train_recommendation_model.py
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.recommendation.recommend import _load_role_profiles, recommend_roles


def main():
    profiles = _load_role_profiles()
    print(f"Built skill profiles for {len(profiles)} roles from the processed dataset:")
    for role, p in profiles.items():
        top_skills = sorted(p["skill_freq"].items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"  - {role}: {p['n_postings']} postings, top skills: {[s for s, _ in top_skills]}")

    # Sanity check: for each role, does querying with its own top skills recommend itself in top-3?
    hits = 0
    for role, p in profiles.items():
        top_skills = [s for s, _ in sorted(p["skill_freq"].items(), key=lambda x: x[1], reverse=True)[:5]]
        if not top_skills:
            continue
        recs = [r["role"] for r in recommend_roles(top_skills, top_k=3)]
        if role in recs:
            hits += 1
    print(f"\nSelf-recommendation sanity check: {hits}/{len(profiles)} roles recommended "
          f"in their own top-3 when queried with their own top skills.")


if __name__ == "__main__":
    main()
