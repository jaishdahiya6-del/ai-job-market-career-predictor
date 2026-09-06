"""
Generates a labeled DEMO dataset (NOT real market data) so the app is
fully runnable out of the box. Replace data/raw/jobs.csv with a real
dataset (e.g. a Kaggle job-postings CSV) with the same column names
to use real data -- see README for the column schema.
"""
import random
import pandas as pd
from pathlib import Path

random.seed(42)

ROLES = {
    "Data Analyst": ["python", "sql", "excel", "power bi", "tableau", "pandas", "statistics"],
    "Data Scientist": ["python", "pandas", "numpy", "scikit-learn", "statistics", "machine learning", "sql"],
    "ML Engineer": ["python", "pytorch", "tensorflow", "docker", "aws", "scikit-learn", "mlops"],
    "Data Engineer": ["python", "sql", "spark", "airflow", "aws", "docker", "etl"],
    "AI Engineer": ["python", "pytorch", "nlp", "computer vision", "deep learning", "aws"],
    "Software Engineer": ["python", "java", "javascript", "react", "node.js", "git", "docker"],
    "BI Analyst": ["sql", "power bi", "tableau", "excel", "communication"],
    "Frontend Developer": ["javascript", "react", "typescript", "css", "html"],
    "Backend Developer": ["python", "django", "flask", "fastapi", "sql", "docker"],
    "DevOps Engineer": ["docker", "kubernetes", "aws", "jenkins", "github actions", "linux"],
}

COMPANIES = ["Acme Corp", "DataWorks", "NimbusTech", "Vertex Labs", "BrightSoft",
             "Cloudify", "PixelForge", "Quantify Inc", "NextGen AI", "StackWave"]

LOCATIONS = [("Bangalore", "Karnataka", "India"), ("Hyderabad", "Telangana", "India"),
             ("Pune", "Maharashtra", "India"), ("Delhi", "Delhi", "India"),
             ("Remote", "Remote", "India"), ("Mumbai", "Maharashtra", "India"),
             ("Chennai", "Tamil Nadu", "India")]

EXPERIENCE_LEVELS = ["Entry", "Mid", "Senior"]
EDUCATION = ["Bachelors", "Masters", "BTech", "MTech"]
INDUSTRY = ["IT Services", "Fintech", "E-commerce", "Healthcare", "EdTech", "SaaS"]
EMPLOYMENT_TYPE = ["Full-time", "Contract", "Internship"]

SOFT_SKILLS = ["communication", "teamwork", "problem solving", "leadership", "adaptability"]

BASE_SALARY = {"Entry": 500000, "Mid": 1100000, "Senior": 2200000}


def make_description(role, skills):
    soft = random.sample(SOFT_SKILLS, k=2)
    skill_text = ", ".join(skills)
    return (
        f"We are hiring a {role}. Required skills: {skill_text}. "
        f"Strong {soft[0]} and {soft[1]} expected. "
        f"<p>Work with cross-functional teams to deliver impact.</p>"
    )


def generate(n=800):
    rows = []
    for i in range(n):
        role = random.choice(list(ROLES.keys()))
        base_skills = ROLES[role]
        k = random.randint(3, len(base_skills))
        skills = random.sample(base_skills, k=k)
        exp = random.choice(EXPERIENCE_LEVELS)
        years = {"Entry": random.randint(0, 2), "Mid": random.randint(2, 6), "Senior": random.randint(6, 15)}[exp]
        loc = random.choice(LOCATIONS)
        salary = BASE_SALARY[exp] + random.randint(-150000, 400000) + len(skills) * 20000
        salary = max(300000, salary)
        row = {
            "job_id": i + 1,
            "title": role,
            "company": random.choice(COMPANIES),
            "city": loc[0],
            "state": loc[1],
            "country": loc[2],
            "remote": loc[0] == "Remote",
            "salary": salary,
            "experience_years": years,
            "experience_level": exp,
            "description": make_description(role, skills),
            "skills_raw": ", ".join(skills),
            "employment_type": random.choice(EMPLOYMENT_TYPE),
            "industry": random.choice(INDUSTRY),
            "education": random.choice(EDUCATION),
            "date_posted": pd.Timestamp("2025-01-01") + pd.Timedelta(days=random.randint(0, 240)),
        }
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = generate(800)
    out_path = out_dir / "jobs_demo.csv"
    df.to_csv(out_path, index=False)
    print(f"DEMO DATA — NOT REAL MARKET DATA. Wrote {len(df)} rows to {out_path}")
