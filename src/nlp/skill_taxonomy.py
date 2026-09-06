"""Expandable skill taxonomy used by the rule-based skill extractor.
Add new skills/categories here -- everything else picks them up automatically.
Keys are canonical skill names; values are lists of surface-form aliases
(lowercase) that should normalize to the canonical name.
"""

SKILL_TAXONOMY = {
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


def flat_alias_map():
    """Returns {alias: (canonical_skill, category)} for fast lookup."""
    mapping = {}
    for category, skills in SKILL_TAXONOMY.items():
        for canonical, aliases in skills.items():
            for alias in aliases:
                mapping[alias.strip().lower()] = (canonical, category)
    return mapping


def all_skills():
    out = []
    for category, skills in SKILL_TAXONOMY.items():
        for canonical in skills:
            out.append((canonical, category))
    return out
