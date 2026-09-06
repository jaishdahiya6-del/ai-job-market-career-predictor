# Career IQ — AI Job Market Intelligence & Career Predictor

A self-contained, end-to-end system that analyzes job postings, extracts
skills with NLP, predicts salaries with trained ML models, recommends
career roles, finds skill gaps, and builds a personalized learning
roadmap — all inside one polished Streamlit dashboard, backed by SQLite.

No separate backend/API process to run — the dashboard talks directly
to the ML/NLP modules, so there's one command to start the whole app.

> ⚠️ **Runs on demo/synthetic data by default** (clearly labeled). See
> `data/raw/README.md` to plug in a real job-postings dataset.

## Features

- Rule-based NLP skill extraction over an expandable skill taxonomy (40+ skills, 7 categories)
- Data cleaning pipeline: dedup, salary parsing, title/location normalization, HTML stripping
- Salary prediction: 4 models compared (Linear Regression, Random Forest, Gradient Boosting, XGBoost), best one auto-selected on validation R²
- Role recommendation engine using Jaccard similarity + skill-centrality scoring against real role-skill profiles built from the dataset
- Skill-gap analysis and a 4-phase personalized career roadmap
- Resume analyzer (PDF upload → skill extraction → market-demand score)
- Job description analyzer (paste text → skills, seniority, education, summary)
- Polished, custom-themed Streamlit dashboard: gradient hero banners, KPI cards, skill badges, dark theme with a branded color palette across all Plotly charts
- 10-page navigation with icons
- SQLite + SQLAlchemy ORM (swappable for Postgres via `DATABASE_URL`)
- 13 automated pytest tests covering preprocessing, NLP, recommendation, and salary prediction

## Tech Stack

Python 3.11 · pandas/numpy · scikit-learn · XGBoost · Streamlit · Plotly · SQLAlchemy · pypdf · pytest · Docker

## Architecture

```
proj/
├── data/{raw,processed}/       # demo dataset + cleaned output
├── src/
│   ├── preprocessing/clean.py       # cleaning & normalization
│   ├── nlp/                          # skill taxonomy, extraction, resume/job analyzers
│   ├── models/predict_salary.py      # loads trained model, serves predictions
│   ├── recommendation/recommend.py   # role recommender, skill gap, roadmap
│   └── utils/db.py                   # SQLAlchemy models + seeding
├── app/dashboard.py             # Streamlit UI (10 pages, custom theme)
├── scripts/                     # CLI entry points (data gen, training, seeding)
├── models/                      # saved joblib artifacts (generated)
├── tests/test_core.py           # pytest suite (13 tests)
├── Dockerfile / docker-compose.yml
└── requirements.txt
```

## Installation (Windows)

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running the Project (Windows / any OS)

```
:: 1. Generate demo data (skip if you've added a real dataset)
python scripts/generate_sample_data.py

:: 2. Clean + extract skills
python scripts/process_dataset.py

:: 3. Train the salary model
python scripts/train_salary_model.py

:: 4. (optional) validate the recommender
python scripts/train_recommendation_model.py

:: 5. Seed the SQLite database
python -m src.utils.db

:: 6. Run the dashboard
streamlit run app/dashboard.py
:: opens automatically at http://localhost:8501
```

## Running Tests

```
pytest tests/ -v
```

## Docker

```
docker compose up --build
```
Dashboard → http://localhost:8501

## ML Methodology

**Salary prediction**: features = one-hot encoded categoricals (title,
location, experience level, industry, education, employment type) +
numeric (years of experience, skill count) + multi-hot skill vector.
Four regressors are trained on an 80/20 split and compared on MAE,
RMSE, R², and MAPE; the highest-R² model is saved with joblib along
with the skill binarizer and feature-column order, so inference always
reindexes to the training schema. A prediction range is derived from
the residual standard deviation (~80% interval).

**Recommendation engine**: for each role, a skill-frequency profile is
built from the processed dataset. A candidate's skills are scored
against every role using `0.5 × Jaccard(user, role) + 0.5 × centrality`,
where centrality is the fraction of the role's total skill-mentions
covered by the overlap — this rewards matching a role's *most common*
skills, not just any overlap. Roles are ranked by this composite score.

**Skill extraction**: word-boundary regex matching against an alias
map built from an expandable skill taxonomy (`src/nlp/skill_taxonomy.py`) — add
a new skill/alias there and every downstream feature (dashboard,
resume/job analyzers) picks it up automatically.

## Model Performance

(from a run against the included demo dataset; regenerate with
`python scripts/train_salary_model.py` — see `models/salary_model_metrics.json`)

| Model             | MAE      | RMSE     | R²    | MAPE  |
|-------------------|----------|----------|-------|-------|
| Linear Regression | ~145,000 | ~173,000 | 0.936 | 12.7% |
| Random Forest     | ~145,000 | ~170,000 | 0.939 | 12.6% |
| Gradient Boosting | ~144,000 | ~171,000 | 0.938 | 12.4% |
| XGBoost           | ~152,000 | ~183,000 | 0.929 | 13.4% |

*(Numbers are from synthetic demo data with a fairly linear salary
formula, so scores are high; expect lower, more realistic R² on real
market data.)*

## Known Limitations

- Ships with synthetic demo data, not real job postings
- Rule-based skill extraction (no embeddings/semantic matching), so it will miss skills phrased very differently from the taxonomy aliases
- Resume parsing only supports text-based PDFs, not scanned images (no OCR)
- Recommendation engine is similarity-based, not a learned/collaborative-filtering model
- No authentication/user accounts — `users`/`predictions` tables exist but aren't wired into a login flow
- No standalone REST API — all functionality is inside the Streamlit app (fine for a demo/portfolio project; a FastAPI layer could be reintroduced later if a separate frontend is needed)

## Future Improvements

- Add spaCy NER + sentence embeddings for semantic skill matching
- OCR fallback for scanned resumes
- SHAP-based explainability for salary predictions
- User accounts + saved prediction history
- Postgres deployment + CI pipeline

## Disclaimer

Salary predictions are statistical estimates from a model trained on
the currently loaded dataset — not a guaranteed offer or professional
career/financial advice.

## Author

Built as a portfolio/college project demonstrating full-stack ML engineering.

---

## Resume Description

> Built an end-to-end AI Job Market Intelligence platform that analyzes
> job-posting data, extracts skills via NLP, and serves salary
> predictions and personalized career recommendations through a
> custom-themed, interactive Streamlit dashboard, backed by a
> trained/compared ensemble of regression models and a similarity-based
> recommendation engine.

## Resume Bullet Points

- Designed and trained a salary-prediction pipeline comparing 4 regression models (Linear, Random Forest, Gradient Boosting, XGBoost), auto-selecting the best on validation R² and serving predictions in real time through the app.
- Built a rule-based NLP skill-extraction engine over a 40+ skill taxonomy across 7 categories, applied to 800+ synthetic job postings for market analytics.
- Implemented a role-recommendation engine using Jaccard similarity and skill-centrality scoring, validated with a 100% self-recommendation sanity check across 10 roles.
- Designed and built a 10-page, custom-themed Streamlit dashboard (gradient hero banners, KPI cards, skill badges) backed by SQLAlchemy/SQLite ORM, with Plotly visualizations and PDF resume uploads.
- Wrote a 13-test pytest suite covering preprocessing, NLP extraction, recommendation ranking, and salary prediction; containerized the app with Docker Compose.

## Interview Questions & Answers

**1. Why did you compare multiple regression models instead of picking one?**
Different algorithms capture different relationships (linear vs.
non-linear, interactions between skills/location/experience); comparing
MAE/RMSE/R²/MAPE on a held-out split lets me pick the best-generalizing
model objectively rather than assuming one architecture is always best.

**2. How does your skill extractor avoid matching substrings like "c" inside other words?**
Each alias is compiled into a regex with negative lookaround
assertions (`(?<![a-zA-Z0-9])...(?![a-zA-Z0-9])`) enforcing word
boundaries, so "c" won't match inside "docker" or "communication".

**3. How does the recommendation engine score a role match?**
`0.5 × Jaccard(user_skills, role_skills) + 0.5 × centrality`, where
centrality is the share of the role's total skill-mentions covered by
the user's matching skills — so matching a role's most in-demand
skills counts more than matching rare ones.

**4. How do you generate the salary prediction's range, not just a point estimate?**
I compute the residual standard deviation on the held-out test set for
the chosen model and use ±1.28×σ as an approximate 80% interval around
the point prediction.

**5. What happens if a user enters a skill the model has never seen?**
`predict_salary()` filters skills to only those in the trained
`MultiLabelBinarizer` classes and separately returns any
`unmatched_skills` so the UI can surface that transparently instead of
silently ignoring them.

**6. Why SQLite now but designed for Postgres later?**
All DB access goes through SQLAlchemy's engine/session created from a
single `DATABASE_URL` env var — swapping to Postgres is a connection-string
change, no code changes, since no raw SQLite-specific SQL is used.

**7. How do you prevent the app from crashing on bad input?**
Each page wraps its risky operations (missing model file, malformed
PDF, empty job description, unknown target role) in explicit checks or
`try/except` blocks and surfaces a clear `st.error()` message instead
of letting an exception crash the whole Streamlit session.

**8. How would you extend skill extraction to catch skills not in the taxonomy?**
Add TF-IDF or embedding-based similarity as a secondary pass: score
n-grams from the description against known skill embeddings, and flag
high-similarity, low-frequency terms as taxonomy candidates for human
review before adding them as new aliases.

**9. How did you validate the recommendation engine without labeled ground truth?**
A self-recommendation sanity check: querying with each role's own
top-5 most frequent skills and checking whether that role appears in
its own top-3 recommendations (10/10 roles passed on the demo data) —
a proxy for internal consistency in absence of labeled train/test pairs.

**10. What's the biggest limitation of using synthetic data here?**
The synthetic salary formula is close to linear by construction, so
all models score unrealistically high R²; real job-market data has
much noisier, non-linear salary relationships, so I'd expect lower R²
and more benefit from non-linear models (RF/XGBoost) on real data.
