"""Main Streamlit dashboard for Career IQ platform.
Run: streamlit run app/dashboard.py
"""
import ast
import sys
from pathlib import Path
from collections import Counter
from typing import Any, List, Optional

import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.models.predict_salary import predict_salary
from src.recommendation.recommend import recommend_roles, skill_gap, build_roadmap, _load_role_profiles
from src.nlp.job_analyzer import analyze_job_description
from src.nlp.resume_analyzer import extract_text_from_pdf, analyze_resume
from src.nlp.skill_taxonomy import all_skills
from src.utils.logger import get_logger

logger = get_logger(__name__)

PROCESSED_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "jobs_processed.csv"

st.set_page_config(page_title="Career IQ — AI Job Market Intelligence", layout="wide", page_icon="🧠")

# ----------------------------------------------------------------------
# Theme: brand colors + plotly template shared across every chart
# ----------------------------------------------------------------------
PRIMARY = "#6C5CE7"      # indigo/violet
PRIMARY_DARK = "#4834D4"
ACCENT = "#00CEC9"       # teal
WARN = "#FDCB6E"
DANGER = "#FF7675"
BG_CARD = "#12131A"
TEXT_MUTED = "#A0A3B1"

CHART_COLORWAY = [PRIMARY, ACCENT, WARN, "#74B9FF", "#FD79A8", "#55EFC4", DANGER, "#A29BFE"]
pio.templates["career_iq"] = pio.templates["plotly_dark"]
pio.templates["career_iq"].layout.colorway = CHART_COLORWAY
pio.templates["career_iq"].layout.paper_bgcolor = "rgba(0,0,0,0)"
pio.templates["career_iq"].layout.plot_bgcolor = "rgba(0,0,0,0)"
pio.templates["career_iq"].layout.font = dict(family="Inter, sans-serif", size=13)
pio.templates.default = "career_iq"

st.markdown(f"""
<style>
    .stApp {{
        background: radial-gradient(1200px circle at 10% -10%, rgba(108,92,231,0.14), transparent 40%),
                    radial-gradient(1000px circle at 100% 0%, rgba(0,206,201,0.10), transparent 45%),
                    #0B0C10;
    }}
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #14151E 0%, #0B0C10 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }}
    h1, h2, h3 {{ font-family: 'Inter', sans-serif; letter-spacing: -0.02em; }}
    .hero {{
        padding: 28px 32px; border-radius: 20px; margin-bottom: 22px;
        background: linear-gradient(120deg, {PRIMARY} 0%, {PRIMARY_DARK} 55%, #1B1D2A 100%);
        box-shadow: 0 12px 40px rgba(108,92,231,0.25);
    }}
    .hero h1 {{ color: white; margin: 0; font-size: 2.0rem; }}
    .hero p {{ color: rgba(255,255,255,0.85); margin-top: 6px; font-size: 0.95rem; }}
    .badge {{
        display:inline-block; padding: 3px 10px; margin: 3px 4px 3px 0; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600;
    }}
    .badge-have {{ background: rgba(0,206,201,0.15); color: {ACCENT}; border: 1px solid rgba(0,206,201,0.4); }}
    .badge-missing {{ background: rgba(255,118,117,0.15); color: {DANGER}; border: 1px solid rgba(255,118,117,0.4); }}
    .badge-priority {{ background: rgba(253,203,110,0.15); color: {WARN}; border: 1px solid rgba(253,203,110,0.4); }}
    .badge-neutral {{ background: rgba(108,92,231,0.15); color: #B9AEFF; border: 1px solid rgba(108,92,231,0.4); }}
    .kpi-card {{
        background: {BG_CARD}; border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px; padding: 16px 18px; text-align: left;
    }}
    .kpi-label {{ color: {TEXT_MUTED}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.06em; }}
    .kpi-value {{ color: white; font-size: 1.6rem; font-weight: 700; margin-top: 2px; }}
    .role-card {{
        background: {BG_CARD}; border: 1px solid rgba(255,255,255,0.07); border-radius: 16px;
        padding: 16px 20px; margin-bottom: 14px;
    }}
    .role-title {{ color: white; font-size: 1.05rem; font-weight: 700; }}
    .role-score {{ color: {ACCENT}; font-weight: 700; }}
    .insight-row {{
        background: {BG_CARD}; border-left: 3px solid {PRIMARY}; border-radius: 8px;
        padding: 10px 14px; margin-bottom: 8px; color: #E6E6F0; font-size: 0.92rem;
    }}
    div.stButton > button {{
        background: linear-gradient(120deg, {PRIMARY}, {PRIMARY_DARK});
        color: white; border: none; border-radius: 10px; font-weight: 600; padding: 0.5rem 1.2rem;
    }}
    div.stButton > button:hover {{ opacity: 0.9; color: white; }}
</style>
""", unsafe_allow_html=True)


def kpi_card(col: Any, label: str, value: Any) -> None:
    col.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)


def badges(items: List[str], kind: str = "neutral") -> str:
    if not items:
        return "<span style='color:#666'>none</span>"
    return "".join(f"<span class='badge badge-{kind}'>{s}</span>" for s in items)


def page_header(icon: str, title: str, subtitle: str) -> None:
    st.markdown(f"""
    <div class="hero">
        <h1>{icon} {title}</h1>
        <p>{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


@st.cache_data
def load_data() -> Optional[pd.DataFrame]:
    if not PROCESSED_PATH.exists():
        return None
    try:
        df = pd.read_csv(PROCESSED_PATH)
        df["skills_extracted"] = df["skills_extracted"].apply(
            lambda s: ast.literal_eval(s) if isinstance(s, str) and s.startswith("[") else []
        )
        return df
    except Exception as e:
        logger.error("Error loading processed dataset: %s", e)
        return None


df = load_data()

PAGES = {
    "Overview": "🏠", "Job Market": "📈", "Skills Intelligence": "🧩",
    "Salary Predictor": "💰", "Career Predictor": "🎯", "Skill Gap": "🧭",
    "Resume Analyzer": "📄", "Job Analyzer": "🔍", "Career Roadmap": "🗺️", "About": "ℹ️",
}

st.sidebar.markdown(f"""
<div style="text-align:center; padding: 8px 0 18px 0;">
    <div style="font-size:2.2rem;">🧠</div>
    <div style="color:white; font-weight:800; font-size:1.15rem;">Career IQ</div>
    <div style="color:{TEXT_MUTED}; font-size:0.78rem;">AI Job Market Intelligence</div>
</div>
""", unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", list(PAGES.keys()), format_func=lambda p: f"{PAGES[p]}  {p}")
st.sidebar.markdown("---")
st.sidebar.caption("⚠️ Demo dataset — see `data/raw/README.md` to plug in real job-market data.")

if df is None:
    st.error("No processed dataset found. Run: `python scripts/process_dataset.py`")
    st.stop()

ALL_SKILL_NAMES = sorted({s for s, _ in all_skills()})

# ---------------- Overview ----------------
if page == "Overview":
    page_header("🧠", "Career IQ", "AI-powered job market intelligence & career predictor — built on your skills, not guesswork.")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpi_card(c1, "Total Jobs", f"{len(df):,}")
    kpi_card(c2, "Unique Roles", df["title_clean"].nunique())
    top_skill = Counter([s for row in df["skills_extracted"] for s in row]).most_common(1)
    kpi_card(c3, "Top Skill", top_skill[0][0].title() if top_skill else "-")
    kpi_card(c4, "Avg Salary", f"₹{df['salary'].mean():,.0f}")
    remote_pct = df["remote"].mean() * 100 if "remote" in df.columns else 0
    kpi_card(c5, "Remote %", f"{remote_pct:.1f}%")
    top_loc = df["location_clean"].value_counts().idxmax() if "location_clean" in df.columns else "-"
    kpi_card(c6, "Top Location", top_loc.split(",")[0])

    st.write("")
    st.subheader("✨ AI Career Insights")
    counter = Counter([s for row in df["skills_extracted"] for s in row])
    for skill, count in counter.most_common(5):
        st.markdown(f"<div class='insight-row'>📌 <b>{skill.title()}</b> appears in "
                     f"{count/len(df)*100:.1f}% of analyzed postings.</div>", unsafe_allow_html=True)

# ---------------- Job Market ----------------
elif page == "Job Market":
    page_header("📈", "Job Market Analysis", "Roles, locations, salary spread, and experience levels across the dataset.")
    colA, colB = st.columns(2)
    with colA:
        role_counts = df["title_clean"].value_counts().reset_index()
        role_counts.columns = ["role", "count"]
        st.plotly_chart(px.bar(role_counts, x="role", y="count", title="Jobs by Role", color="role"), use_container_width=True)
    with colB:
        loc_counts = df["location_clean"].value_counts().reset_index().head(10)
        loc_counts.columns = ["location", "count"]
        st.plotly_chart(px.bar(loc_counts, x="location", y="count", title="Jobs by Location (Top 10)", color="location"), use_container_width=True)

    colC, colD = st.columns(2)
    with colC:
        st.plotly_chart(px.histogram(df, x="salary", nbins=30, title="Salary Distribution"), use_container_width=True)
    with colD:
        exp_counts = df["experience_level"].value_counts().reset_index()
        exp_counts.columns = ["level", "count"]
        st.plotly_chart(px.pie(exp_counts, names="level", values="count", title="Experience Distribution", hole=0.45), use_container_width=True)

    st.plotly_chart(px.box(df, x="title_clean", y="salary", title="Salary by Role", color="title_clean"), use_container_width=True)
    st.plotly_chart(px.box(df, x="industry", y="salary", title="Salary by Industry", color="industry"), use_container_width=True)

# ---------------- Skills Intelligence ----------------
elif page == "Skills Intelligence":
    page_header("🧩", "Skills Intelligence", "Which skills matter most, and to which roles.")
    counter = Counter([s for row in df["skills_extracted"] for s in row])
    top20 = pd.DataFrame(counter.most_common(20), columns=["skill", "count"])
    st.plotly_chart(px.bar(top20, x="skill", y="count", title="Top 20 Skills", color="count",
                            color_continuous_scale=[PRIMARY, ACCENT]), use_container_width=True)

    st.subheader("🔥 Skill Demand by Role")
    roles = df["title_clean"].unique()
    top_skills = [s for s, _ in counter.most_common(15)]
    heat_data = []
    for role in roles:
        sub = df[df["title_clean"] == role]
        c = Counter([s for row in sub["skills_extracted"] for s in row])
        heat_data.append([c.get(s, 0) for s in top_skills])
    heat_df = pd.DataFrame(heat_data, index=roles, columns=top_skills)
    st.plotly_chart(px.imshow(heat_df, aspect="auto", title="Skill × Role Heatmap",
                               color_continuous_scale=[BG_CARD, PRIMARY, ACCENT]), use_container_width=True)

    if "date_posted" in df.columns:
        df["month"] = pd.to_datetime(df["date_posted"], errors="coerce").dt.to_period("M").astype(str)
        trend_rows = []
        for skill in top_skills[:5]:
            sub = df[df["skills_extracted"].apply(lambda s: skill in s)]
            counts = sub.groupby("month").size().reset_index(name="count")
            counts["skill"] = skill
            trend_rows.append(counts)
        if trend_rows:
            trend_df = pd.concat(trend_rows)
            st.plotly_chart(px.line(trend_df, x="month", y="count", color="skill", markers=True,
                                     title="Skill Demand Trend Over Time"), use_container_width=True)

# ---------------- Salary Predictor ----------------
elif page == "Salary Predictor":
    page_header("💰", "Salary Predictor", "ML-estimated salary range based on role, location, experience, and skills.")
    with st.form("salary_form"):
        col1, col2 = st.columns(2)
        with col1:
            title = st.selectbox("Job Role", sorted(df["title_clean"].unique()))
            location = st.selectbox("Location", sorted(df["location_clean"].unique()))
            experience_level = st.selectbox("Experience Level", ["Entry", "Mid", "Senior"])
            experience_years = st.slider("Years of Experience", 0, 20, 2)
        with col2:
            industry = st.selectbox("Industry", sorted(df["industry"].unique()))
            education = st.selectbox("Education", sorted(df["education"].unique()))
            employment_type = st.selectbox("Employment Type", sorted(df["employment_type"].unique()))
            skills = st.multiselect("Skills", ALL_SKILL_NAMES, default=["python", "sql"])
        submitted = st.form_submit_button("🔮 Predict Salary")

    if submitted:
        try:
            result = predict_salary(title, location, experience_level, industry, education,
                                     employment_type, experience_years, skills)
            st.markdown(f"""
            <div class="hero" style="background:linear-gradient(120deg,{ACCENT}22,{PRIMARY}33); margin-top:16px;">
                <div style="color:{TEXT_MUTED}; font-size:0.85rem;">ESTIMATED ANNUAL SALARY</div>
                <div style="color:white; font-size:2.4rem; font-weight:800;">₹{result['predicted_salary']:,}</div>
                <div style="color:{TEXT_MUTED};">Range: ₹{result['lower_bound']:,} — ₹{result['upper_bound']:,}</div>
            </div>
            """, unsafe_allow_html=True)
            st.caption(result["disclaimer"])
        except FileNotFoundError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error executing salary prediction: {e}")

# ---------------- Career Predictor ----------------
elif page == "Career Predictor":
    page_header("🎯", "Career / Role Recommender", "Tell us your skills — we'll rank the roles that fit best.")
    skills = st.multiselect("Your Skills", ALL_SKILL_NAMES, default=["python", "sql"])
    if st.button("🎯 Recommend Roles") and skills:
        try:
            recs = recommend_roles(skills, top_k=6)
            for r in recs:
                st.markdown(f"""
                <div class="role-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <span class="role-title">{r['role']}</span>
                        <span class="role-score">{r['match_score']}% match</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.progress(min(r["match_score"] / 100, 1.0))
                st.markdown(f"Matched skills: {badges(r['matched_skills'], 'have')}", unsafe_allow_html=True)
                st.caption(f"Avg salary in dataset: ₹{r['avg_salary']:,} · {r['n_postings']} postings")
                st.write("")
        except Exception as e:
            st.error(f"Error generating role recommendations: {e}")

# ---------------- Skill Gap ----------------
elif page == "Skill Gap":
    page_header("🧭", "Skill Gap Analysis", "See exactly what separates you from your target role.")
    try:
        profiles = _load_role_profiles()
        roles = sorted(profiles.keys())
        col1, col2 = st.columns(2)
        with col1:
            skills = st.multiselect("Your Current Skills", ALL_SKILL_NAMES, default=["python", "sql"])
        with col2:
            target = st.selectbox("Target Role", roles)
        if st.button("🧭 Analyze Gap"):
            gap = skill_gap(skills, target)
            st.markdown("**✓ Already have:**", unsafe_allow_html=True)
            st.markdown(badges(gap["have"], "have"), unsafe_allow_html=True)
            st.markdown("**⚠ Missing:**", unsafe_allow_html=True)
            st.markdown(badges(gap["missing"], "missing"), unsafe_allow_html=True)
            st.markdown("**⭐ Priority to learn:**", unsafe_allow_html=True)
            st.markdown(badges(gap["priority_to_learn"], "priority"), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error in skill gap analysis: {e}")

# ---------------- Resume Analyzer ----------------
elif page == "Resume Analyzer":
    page_header("📄", "Resume Analyzer", "Upload a PDF resume for an instant skill & market-fit score.")
    uploaded = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    if uploaded is not None:
        try:
            text = extract_text_from_pdf(uploaded)
            result = analyze_resume(text)
            col1, col2 = st.columns([1, 2])
            with col1:
                kpi_card(st, "Resume Skill Score", f"{result['resume_score']} / 100")
            with col2:
                st.write("**Name (guess):**", result["name_guess"])
                st.write("**Education (guess):**", ", ".join(result["education_guess"]))
            st.markdown("**Skills found:**", unsafe_allow_html=True)
            st.markdown(badges(result["skills_found"], "have"), unsafe_allow_html=True)
            st.markdown("**Missing high-demand skills:**", unsafe_allow_html=True)
            st.markdown(badges(result["missing_high_demand_skills"], "missing"), unsafe_allow_html=True)
            if result["skills_found"]:
                recs = recommend_roles(result["skills_found"], top_k=3)
                st.write("**Recommended roles:**", ", ".join(r["role"] for r in recs))
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Failed to analyze resume PDF: {e}")

# ---------------- Job Analyzer ----------------
elif page == "Job Analyzer":
    page_header("🔍", "Job Description Analyzer", "Paste any posting to extract skills, seniority, and requirements.")
    text = st.text_area("Paste a job description", height=200)
    user_skills = st.multiselect("Your skills (optional, for gap analysis)", ALL_SKILL_NAMES)
    if st.button("🔍 Analyze") and text.strip():
        result = analyze_job_description(text, user_skills or None)
        st.markdown("**Technical skills:**", unsafe_allow_html=True)
        st.markdown(badges(result["technical_skills"], "neutral"), unsafe_allow_html=True)
        st.markdown("**Soft skills:**", unsafe_allow_html=True)
        st.markdown(badges(result["soft_skills"], "priority"), unsafe_allow_html=True)
        st.write("**Seniority:**", result["seniority"])
        st.write("**Education requirements:**", ", ".join(result["education_requirements"]))
        st.info(result["summary"])
        if "skill_gap" in result:
            st.markdown("✓ Have:", unsafe_allow_html=True)
            st.markdown(badges(result["skill_gap"]["have"], "have"), unsafe_allow_html=True)
            st.markdown("⚠ Missing:", unsafe_allow_html=True)
            st.markdown(badges(result["skill_gap"]["missing"], "missing"), unsafe_allow_html=True)

# ---------------- Career Roadmap ----------------
elif page == "Career Roadmap":
    page_header("🗺️", "Career Roadmap", "A phase-by-phase learning plan generated from your current skills.")
    skills = st.multiselect("Your Current Skills", ALL_SKILL_NAMES, default=["python"])
    if st.button("🗺️ Generate Roadmap"):
        roadmap = build_roadmap(skills)
        for i, phase in enumerate(roadmap, 1):
            with st.expander(f"Phase {i} — {phase['phase']}", expanded=True):
                st.markdown("✓ Already have:", unsafe_allow_html=True)
                st.markdown(badges(phase["already_have"], "have"), unsafe_allow_html=True)
                st.markdown("📚 To learn:", unsafe_allow_html=True)
                st.markdown(badges(phase["to_learn"], "priority") if phase["to_learn"]
                            else "<span style='color:#55EFC4'>Phase complete! 🎉</span>", unsafe_allow_html=True)

# ---------------- About ----------------
else:
    page_header("ℹ️", "About Career IQ", "How this project works, under the hood.")
    st.markdown(f"""
    **Career IQ** analyzes job postings, extracts skills via NLP, predicts salaries with a
    trained ML model, and recommends career roles + learning roadmaps based on a user's
    skill profile — all in one self-contained Streamlit app and FastAPI service.

    Built with Python, scikit-learn/XGBoost, rule-based NLP skill extraction,
    SQLAlchemy, FastAPI, and Streamlit + Plotly for visualization.

    ⚠️ Currently running on demo/synthetic data — see `data/raw/README.md` for how to
    swap in a real job-postings dataset.
    """)
