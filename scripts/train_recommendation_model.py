"""Evaluates role recommendation skill profiles and runs a self-recommendation sanity check.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.recommendation.recommend import _load_role_profiles, recommend_roles
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main():
    try:
        profiles = _load_role_profiles()
        logger.info("Built skill profiles for %d roles from processed dataset:", len(profiles))
        for role, p in profiles.items():
            top_skills = sorted(p["skill_freq"].items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info("  - %s: %d postings, top skills: %s", role, p["n_postings"], [s for s, _ in top_skills])

        hits = 0
        for role, p in profiles.items():
            top_skills = [s for s, _ in sorted(p["skill_freq"].items(), key=lambda x: x[1], reverse=True)[:5]]
            if not top_skills:
                continue
            recs = [r["role"] for r in recommend_roles(top_skills, top_k=3)]
            if role in recs:
                hits += 1

        logger.info("Self-recommendation sanity check: %d/%d roles recommended in top-3 when queried with top skills.", hits, len(profiles))
    except Exception as e:
        logger.error("Error during recommendation evaluation: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
