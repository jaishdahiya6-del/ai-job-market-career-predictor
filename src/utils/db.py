"""Database models and utilities using SQLAlchemy ORM.
Supports SQLite by default and PostgreSQL via DATABASE_URL environment variable.
"""
import ast
import os
import datetime as dt
from pathlib import Path
from typing import Generator, Optional
import pandas as pd
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.orm import declarative_base, sessionmaker, relationship, Session

from src.utils.logger import get_logger

logger = get_logger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./data/app.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
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
    skills = relationship("JobSkill", back_populates="job", cascade="all, delete-orphan")


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


def init_db() -> None:
    """Creates database directories and tables if they do not exist."""
    db_path = Path("data")
    db_path.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    logger.info("Database initialized successfully at %s", DATABASE_URL)


def get_session() -> Generator[Session, None, None]:
    """Yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed_from_processed_csv(csv_path: str = "data/processed/jobs_processed.csv", limit: int = 800) -> None:
    """Populates jobs/skills/job_skills tables from processed CSV file.

    Args:
        csv_path: Path to processed CSV file.
        limit: Max number of rows to seed.
    """
    init_db()
    db = SessionLocal()
    try:
        if not Path(csv_path).exists():
            logger.error("Processed CSV path '%s' does not exist for seeding", csv_path)
            return

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
            db.flush()

            skills_raw = row.get("skills_extracted")
            skills = ast.literal_eval(skills_raw) if isinstance(skills_raw, str) and skills_raw.startswith("[") else []
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
        logger.info("Seeded %d jobs and %d distinct skills into the database.", len(df), len(skill_cache))
    except Exception as e:
        db.rollback()
        logger.error("Failed to seed database from CSV: %s", e)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_from_processed_csv()
