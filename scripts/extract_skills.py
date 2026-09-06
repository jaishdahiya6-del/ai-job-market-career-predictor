"""Standalone CLI to run skill extraction on arbitrary text, useful for
quick testing or piping in a single job description.
Run: python scripts/extract_skills.py "some job description text"
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.nlp.skill_extraction import extract_skills

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    for s in extract_skills(text):
        print(f"{s['skill']} ({s['category']})")
