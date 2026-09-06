"""SQLite database via SQLAlchemy. Swap DATABASE_URL for a Postgres
DSN later (e.g. postgresql://user:pass@host/db) with no code changes
elsewhere, since all access goes through this module's Session."""
import os
import datetime as dt
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Job(Base):
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    company = Column(String)
    location = Column(String)
    salary = Column(Float)
    experience_level = Column(String)
    description = Column(Text)
    industry = Column(String)
    employment_type = Column(String)
    date_posted = Column(DateTime, nullable=True)
    skills = relationship("JobSkill", back_populates="job")


class Skill(Base):
    __tablename__ = "skills"
    id = Column(Integer, primary_key=True, index=True)
    skill_name = Column(String, unique=True, index=True)
    category = Column(String)


class JobSkill(Base):
    __tablename__ = "job_skills"
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"))
    skill_id = Column(Integer, ForeignKey("skills.id"))
    job = relationship("Job", back_populates="skills")


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    skills = Column(Text)
    experience = Column(String)
    education = Column(String)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    predicted_salary = Column(Float)
    created_at = Column(DateTime, default=dt.datetime.utcnow)


def init_db():
    os.makedirs("data", exist_ok=True)
    Base.metadata.create_all(bind=engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed_from_processed_csv(csv_path: str = "data/processed/jobs_processed.csv", limit: int = 800):
    """Populates jobs/skills/job_skills tables from the processed CSV.
    Safe to re-run: clears existing rows first."""
    import ast
    import pandas as pd

    init_db()
    db = SessionLocal()
    try:
        db.query(JobSkill).delete()
        db.query(Job).delete()
        db.query(Skill).delete()
        db.commit()

        df = pd.read_csv(csv_path).head(limit)
        skill_cache = {}

        for _, row in df.iterrows():
            job = Job(
                title=row.get("title_clean", row.get("title")),
                company=row.get("company"),
                location=row.get("location_clean"),
                salary=row.get("salary"),
                experience_level=row.get("experience_level"),
                description=row.get("description_clean", ""),
                industry=row.get("industry"),
                employment_type=row.get("employment_type"),
                date_posted=pd.to_datetime(row.get("date_posted"), errors="coerce"),
            )
            db.add(job)
            db.flush()  # get job.id

            skills = row.get("skills_extracted")
            skills = ast.literal_eval(skills) if isinstance(skills, str) else []
            for s in skills:
                if s not in skill_cache:
                    existing = db.query(Skill).filter_by(skill_name=s).first()
                    if not existing:
                        existing = Skill(skill_name=s, category="Unknown")
                        db.add(existing)
                        db.flush()
                    skill_cache[s] = existing.id
                db.add(JobSkill(job_id=job.id, skill_id=skill_cache[s]))
        db.commit()
        print(f"Seeded {len(df)} jobs and {len(skill_cache)} distinct skills into the database.")
    finally:
        db.close()


if __name__ == "__main__":
    seed_from_processed_csv()
