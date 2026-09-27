"""Standalone CLI script to run skill extraction on text from command line argument or stdin.
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.nlp.skill_extraction import extract_skills
from src.utils.logger import get_logger

logger = get_logger(__name__)

if __name__ == "__main__":
    text = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    skills = extract_skills(text)
    for s in skills:
        print(f"{s['skill']} ({s['category']})")
