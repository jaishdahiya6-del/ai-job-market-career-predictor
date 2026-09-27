"""Expandable skill taxonomy used by the rule-based skill extractor.
Add new skills/categories here -- everything else picks them up automatically.
Keys are canonical skill names; values are lists of surface-form aliases
(lowercase) that should normalize to the canonical name.
"""
from typing import Dict, List, Tuple

SKILL_TAXONOMY: Dict[str, Dict[str, List[str]]] = {
    "Programming": {
        "python": ["python", "python3"],
        "java": ["java"],
        "c++": ["c++", "cpp"],
        "c": [" c ", "c programming"],
        "javascript": ["javascript", "js"],
        "typescript": ["typescript", "ts"],
    },
    "Data": {
        "sql": ["sql", "mysql", "postgresql", "postgres"],
        "excel": ["excel", "ms excel"],
        "power bi": ["power bi", "powerbi"],
        "tableau": ["tableau"],
        "pandas": ["pandas"],
        "numpy": ["numpy"],
        "statistics": ["statistics", "statistical analysis"],
    },
    "Machine Learning": {
        "scikit-learn": ["scikit-learn", "sklearn", "scikit learn"],
        "tensorflow": ["tensorflow"],
        "pytorch": ["pytorch", "torch"],
        "xgboost": ["xgboost"],
        "nlp": ["nlp", "natural language processing"],
        "computer vision": ["computer vision", "cv"],
        "machine learning": ["machine learning", "ml"],
        "deep learning": ["deep learning", "dl"],
        "mlops": ["mlops"],
    },
    "Cloud": {
        "aws": ["aws", "amazon web services"],
        "azure": ["azure"],
        "gcp": ["gcp", "google cloud"],
    },
    "DevOps": {
        "docker": ["docker"],
        "kubernetes": ["kubernetes", "k8s"],
        "jenkins": ["jenkins"],
        "github actions": ["github actions"],
        "linux": ["linux"],
        "etl": ["etl"],
        "airflow": ["airflow"],
        "spark": ["spark", "pyspark"],
    },
    "Web": {
        "react": ["react", "react.js", "reactjs"],
        "node.js": ["node.js", "nodejs", "node"],
        "django": ["django"],
        "flask": ["flask"],
        "fastapi": ["fastapi"],
        "html": ["html"],
        "css": ["css"],
    },
    "Soft Skills": {
        "communication": ["communication"],
        "teamwork": ["teamwork", "team work"],
        "problem solving": ["problem solving", "problem-solving"],
        "leadership": ["leadership"],
        "adaptability": ["adaptability"],
    },
}


def flat_alias_map() -> Dict[str, Tuple[str, str]]:
    """Returns {alias: (canonical_skill, category)} for fast lookup.

    Returns:
        Dict[str, Tuple[str, str]]: Mapping from alias string to (canonical_skill, category).
    """
    mapping: Dict[str, Tuple[str, str]] = {}
    for category, skills in SKILL_TAXONOMY.items():
        for canonical, aliases in skills.items():
            for alias in aliases:
                mapping[alias.strip().lower()] = (canonical, category)
    return mapping


def all_skills() -> List[Tuple[str, str]]:
    """Returns a list of all (canonical_skill, category) pairs.

    Returns:
        List[Tuple[str, str]]: All canonical skills with their respective category.
    """
    out: List[Tuple[str, str]] = []
    for category, skills in SKILL_TAXONOMY.items():
        for canonical in skills:
            out.append((canonical, category))
    return out
