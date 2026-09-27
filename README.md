# Career IQ — AI Job Market Intelligence & Career Predictor

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.112%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37%2B-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end, production-ready AI job market intelligence platform and career predictor. Career IQ analyzes job posting datasets, extracts technical and soft skills using NLP, predicts salary ranges using ensemble ML models, evaluates skill gaps, and recommends career pathways — all served via a RESTful FastAPI service and an interactive Streamlit dashboard.

---

## Key Features

- **Rule-Based NLP Skill Extraction**: Word-boundary regex extraction over an expandable 40+ skill taxonomy spanning 7 categories.
- **Robust Data Pipeline**: Cleaning, deduplication, title/location normalization, and missing value imputation with schema validation.
- **Multi-Model Salary Prediction**: Evaluates 4 regression models (Linear Regression, Random Forest, Gradient Boosting, XGBoost), selects the top-performing architecture on validation $R^2$, and calculates prediction confidence intervals (~80% bounds).
- **Role Recommendation Engine**: Ranks roles using composite Jaccard similarity and skill-centrality scoring against role-skill frequency profiles.
- **Skill Gap & Learning Roadmap**: Highlights required vs. missing skills for target roles and builds a 4-phase learning roadmap.
- **Resume & Job Description Analyzers**: Automated PDF resume scoring against market demand and job posting parsing.
- **Dual Application Layer**:
  - **RESTful API** (FastAPI) with Pydantic request/response validation and interactive Swagger UI (`/docs`).
  - **Interactive Dashboard** (Streamlit + Plotly) with custom dark theme, KPI cards, and custom charts.
- **Database & Persistence**: SQLite / PostgreSQL support via SQLAlchemy ORM.

---

## Tech Stack

- **Core & Data Processing**: Python 3.11+, Pandas, NumPy
- **Machine Learning**: Scikit-Learn, XGBoost, Joblib
- **NLP**: Rule-based regex extraction, PyPDF
- **API & Web Dashboard**: FastAPI, Pydantic, Streamlit, Plotly
- **Database & Infrastructure**: SQLAlchemy, SQLite / PostgreSQL, Docker, Docker Compose
- **Testing**: PyTest, TestClient, HTTPX

---

## Project Architecture

```
.
├── app/
│   ├── api.py               # RESTful FastAPI service & Pydantic schemas
│   └── dashboard.py         # Streamlit interactive UI dashboard
├── src/
│   ├── models/
│   │   └── predict_salary.py  # Model inference and salary range calculation
│   ├── nlp/
│   │   ├── job_analyzer.py    # Job posting parser and seniority detection
│   │   ├── resume_analyzer.py # PDF resume extractor & market score
│   │   ├── skill_extraction.py# NLP regex skill extraction engine
│   │   └── skill_taxonomy.py  # Taxonomy dictionary and alias map
│   ├── preprocessing/
│   │   └── clean.py           # Data cleaning and normalization pipeline
│   ├── recommendation/
│   │   └── recommend.py       # Jaccard + centrality role recommender
│   └── utils/
│       ├── db.py              # SQLAlchemy database models and seed script
│       └── logger.py          # Centralized logging configuration
├── scripts/
│   ├── extract_skills.py            # CLI skill extraction utility
│   ├── generate_sample_data.py      # Synthetic dataset generator
│   ├── process_dataset.py           # Dataset processing runner
│   ├── train_recommendation_model.py# Recommendation profile builder & evaluator
│   └── train_salary_model.py        # ML model training & comparison
├── tests/
│   └── test_core.py                 # Comprehensive unit and integration test suite
├── Dockerfile                       # Production multi-service Docker container
├── docker-compose.yml               # Docker Compose orchestration
└── requirements.txt                 # Pinned Python dependencies
```

---

## Quick Start (Local Setup)

### 1. Prerequisites
Ensure Python 3.11+ is installed.

### 2. Environment Setup
```bash
# Clone the repository
git clone https://github.com/jaishdahiya6-del/ai-job-market-career-predictor.git
cd ai-job-market-career-predictor

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Pipeline Execution & Training
```bash
# Generate sample demo data
python scripts/generate_sample_data.py

# Clean and process dataset
python scripts/process_dataset.py

# Train salary ML models
python scripts/train_salary_model.py

# Build recommendation profiles
python scripts/train_recommendation_model.py

# Seed database
python -m src.utils.db
```

### 4. Running the Applications

#### Option A: Interactive Streamlit Dashboard
```bash
streamlit run app/dashboard.py
```
Access dashboard at `http://localhost:8501`.

#### Option B: RESTful FastAPI Service
```bash
uvicorn app.api:app --reload --port 8000
```
Access API documentation (Swagger) at `http://localhost:8000/docs`.

---

## Docker Deployment

Build and run both the API service and Streamlit dashboard using Docker Compose:

```bash
docker compose up --build
```
- **Streamlit Dashboard**: `http://localhost:8501`
- **FastAPI Documentation**: `http://localhost:8000/docs`

---

## API Usage Examples

### Health Check
```bash
curl -X GET "http://localhost:8000/health"
```

### Predict Salary
```bash
curl -X POST "http://localhost:8000/api/v1/predict-salary" \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Data Scientist",
       "location": "Bangalore, Karnataka, India",
       "experience_level": "Mid",
       "industry": "IT Services",
       "education": "Bachelors",
       "employment_type": "Full-time",
       "experience_years": 3,
       "skills": ["python", "sql", "machine learning"]
     }'
```

### Recommend Roles
```bash
curl -X POST "http://localhost:8000/api/v1/recommend-roles" \
     -H "Content-Type: application/json" \
     -d '{
       "skills": ["python", "sql", "pandas"],
       "top_k": 3
     }'
```

---

## Testing

Run the full automated test suite (23 tests covering preprocessing, NLP, ML, database, and API endpoints):

```bash
python -m pytest tests/ -v
```

---

## License

This project is licensed under the MIT License.
