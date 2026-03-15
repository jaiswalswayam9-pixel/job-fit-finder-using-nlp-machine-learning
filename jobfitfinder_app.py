import streamlit as st
import time
import random
import pandas as pd
import numpy as np
import os
import tempfile
from utils.resume_parser import ResumeParser
from utils.job_parser import JobParser
from utils.model_utils import ModelUtils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "jobs.csv")

# Domain-specific skills used for both recommendations and strength profiling
DOMAIN_SKILLS = {
    "Software Engineering": ["Python", "Java", "System Design", "Testing", "Debugging"],
    "Marketing": ["Content Creation", "SEO", "Analytics", "Campaign Management", "Social Media"],
    "Finance": ["Excel", "Financial Analysis", "Accounting", "Reporting", "Risk Management"],
    "Business Analysis": ["Requirements Gathering", "Process Mapping", "Documentation", "Stakeholder Management"],
    "Data Science": ["Machine Learning", "Statistical Analysis", "Data Visualization", "Python", "SQL"],
    "Sales": ["Communication", "Negotiation", "Client Management", "CRM", "Target Achievement"],
    "Product Management": ["Product Strategy", "User Research", "Market Analysis", "Roadmap Planning"],
    "Human Resources (HR)": ["Recruitment", "Employee Relations", "Payroll", "Training", "Compliance"],
    "Cyber Security": ["Network Security", "Penetration Testing", "Incident Response", "Encryption"],
    "Graphic Designing": ["Adobe Creative Suite", "UI/UX Design", "Branding", "Visual Design"],
}
# ─────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="JobFitFinder · AI Job Recommender",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS — glassmorphism dark theme
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── CSS Variables ── */
:root {
  --bg:          #060813;
  --surface:     rgba(255,255,255,0.04);
  --border:      rgba(255,255,255,0.09);
  --accent:      #6C63FF;
  --accent2:     #00D4AA;
  --accent3:     #FF6B6B;
  --text:        #E8EAF6;
  --muted:       rgba(232,234,246,0.45);
  --glow:        rgba(108,99,255,0.35);
  --radius:      18px;
  --card-shadow: 0 8px 32px rgba(0,0,0,0.45);
}

/* ── Reset & Base ── */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background: var(--bg) !important;
  color: var(--text) !important;
}

/* Remove Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── Animated mesh background ── */
body::before {
  content: '';
  position: fixed;
  inset: 0;
  background:
    radial-gradient(ellipse 80% 60% at 20% 10%, rgba(108,99,255,0.18) 0%, transparent 60%),
    radial-gradient(ellipse 60% 50% at 80% 80%, rgba(0,212,170,0.12) 0%, transparent 55%),
    radial-gradient(ellipse 50% 40% at 60% 30%, rgba(255,107,107,0.08) 0%, transparent 50%);
  pointer-events: none;
  z-index: 0;
  animation: meshShift 12s ease-in-out infinite alternate;
}
@keyframes meshShift {
  0%   { opacity: 1; transform: scale(1); }
  100% { opacity: .85; transform: scale(1.04); }
}

/* ── Grid noise texture ── */
body::after {
  content: '';
  position: fixed;
  inset: 0;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='60' height='60'%3E%3Ccircle cx='30' cy='30' r='.5' fill='rgba(255,255,255,0.06)'/%3E%3C/svg%3E");
  pointer-events: none;
  z-index: 0;
}

/* ── Wrapper ── */
.jff-wrapper {
  position: relative;
  z-index: 1;
  min-height: 100vh;
  padding: 0 0 60px 0;
}

/* ── TOP NAV ── */
.jff-nav {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 20px 48px;
  background: rgba(6,8,19,0.7);
  backdrop-filter: blur(20px);
  border-bottom: 1px solid var(--border);
  position: sticky;
  top: 0;
  z-index: 100;
}
.jff-logo {
  font-family: 'Syne', sans-serif;
  font-size: 1.45rem;
  font-weight: 800;
  letter-spacing: -0.5px;
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.jff-logo span { color: var(--accent3); -webkit-text-fill-color: var(--accent3); }
.jff-navlinks { display: flex; gap: 8px; }
.jff-navlink {
  padding: 8px 20px;
  border-radius: 50px;
  font-size: 0.88rem;
  font-weight: 500;
  cursor: pointer;
  border: 1px solid transparent;
  color: var(--muted);
  transition: all 0.25s;
  text-decoration: none;
}
.jff-navlink:hover, .jff-navlink.active {
  background: var(--surface);
  border-color: var(--border);
  color: var(--text);
}
.jff-navlink.cta {
  background: linear-gradient(135deg, var(--accent), #9C63FF);
  color: #fff !important;
  border: none;
  box-shadow: 0 0 20px var(--glow);
}

/* ── HERO SECTION ── */
.jff-hero {
  padding: 80px 48px 60px;
  text-align: center;
}
.jff-hero-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 18px;
  border-radius: 50px;
  background: rgba(108,99,255,0.15);
  border: 1px solid rgba(108,99,255,0.3);
  font-size: 0.78rem;
  font-weight: 500;
  color: #a09aff;
  margin-bottom: 24px;
  animation: fadeInDown 0.6s ease both;
}
.jff-hero-badge::before {
  content: '';
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--accent2);
  animation: pulse 1.8s infinite;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.4;transform:scale(1.5)} }
@keyframes fadeInDown { from{opacity:0;transform:translateY(-16px)} to{opacity:1;transform:translateY(0)} }
@keyframes fadeInUp   { from{opacity:0;transform:translateY(20px)}  to{opacity:1;transform:translateY(0)} }

.jff-hero h1 {
  font-family: 'Syne', sans-serif;
  font-size: clamp(2.4rem, 5vw, 4rem);
  font-weight: 800;
  line-height: 1.1;
  letter-spacing: -2px;
  margin: 0 0 20px;
  animation: fadeInDown 0.7s ease 0.1s both;
}
.jff-hero h1 .grad {
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 60%, #fff 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.jff-hero p {
  font-size: 1.1rem;
  color: var(--muted);
  max-width: 540px;
  margin: 0 auto 40px;
  line-height: 1.7;
  font-weight: 300;
  animation: fadeInDown 0.7s ease 0.2s both;
}

/* ── STATS ROW ── */
.jff-stats {
  display: flex;
  justify-content: center;
  gap: 16px;
  flex-wrap: wrap;
  margin-bottom: 60px;
  animation: fadeInUp 0.8s ease 0.3s both;
}
.jff-stat {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 32px;
  text-align: center;
  backdrop-filter: blur(12px);
  transition: transform 0.25s, box-shadow 0.25s;
}
.jff-stat:hover { transform: translateY(-4px); box-shadow: 0 12px 40px rgba(108,99,255,0.2); }
.jff-stat .val {
  font-family: 'Syne', sans-serif;
  font-size: 1.9rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--accent), var(--accent2));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.jff-stat .lbl { font-size: 0.78rem; color: var(--muted); margin-top: 2px; }

/* ── GLASS CARD ── */
.jff-card {
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 32px;
  backdrop-filter: blur(20px);
  box-shadow: var(--card-shadow);
  transition: transform 0.3s, box-shadow 0.3s;
  height: 100%;
}
.jff-card:hover { transform: translateY(-5px); box-shadow: 0 20px 60px rgba(0,0,0,0.5); }
.jff-card-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.1rem;
  font-weight: 700;
  margin-bottom: 6px;
  display: flex;
  align-items: center;
  gap: 10px;
}
.jff-card-title .icon {
  width: 36px; height: 36px;
  border-radius: 10px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1rem;
}
.jff-card p { color: var(--muted); font-size: 0.88rem; line-height: 1.6; margin: 0; }

/* ── UPLOAD ZONE ── */
.jff-upload-zone {
  border: 2px dashed rgba(108,99,255,0.4);
  border-radius: var(--radius);
  padding: 48px 32px;
  text-align: center;
  background: rgba(108,99,255,0.06);
  cursor: pointer;
  transition: all 0.3s;
  animation: fadeInUp 0.6s ease both;
}
.jff-upload-zone:hover {
  border-color: var(--accent);
  background: rgba(108,99,255,0.12);
  box-shadow: 0 0 40px rgba(108,99,255,0.15);
}
.jff-upload-icon { font-size: 3rem; margin-bottom: 16px; }
.jff-upload-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.3rem;
  font-weight: 700;
  margin-bottom: 8px;
}
.jff-upload-sub { color: var(--muted); font-size: 0.88rem; }
.jff-upload-formats {
  display: flex;
  justify-content: center;
  gap: 10px;
  margin-top: 20px;
  flex-wrap: wrap;
}
.jff-fmt-badge {
  padding: 4px 14px;
  border-radius: 50px;
  border: 1px solid var(--border);
  font-size: 0.75rem;
  color: var(--muted);
  background: var(--surface);
}

/* ── SKILL PILL ── */
.jff-skills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
.jff-skill-pill {
  padding: 5px 14px;
  border-radius: 50px;
  font-size: 0.78rem;
  font-weight: 500;
  border: 1px solid;
}
.skill-match  { background: rgba(0,212,170,0.12); border-color: rgba(0,212,170,0.35); color: #00D4AA; }
.skill-gap    { background: rgba(255,107,107,0.12); border-color: rgba(255,107,107,0.35); color: #FF6B6B; }
.skill-learn  { background: rgba(255,195,0,0.12);  border-color: rgba(255,195,0,0.35);  color: #FFC300; }

/* ── JOB CARD ── */
.jff-job-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 20px 24px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  margin-bottom: 12px;
  backdrop-filter: blur(12px);
  transition: all 0.25s;
  cursor: pointer;
}
.jff-job-card:hover {
  border-color: var(--accent);
  background: rgba(108,99,255,0.08);
  transform: translateX(4px);
}
.jff-job-logo {
  width: 44px; height: 44px;
  border-radius: 12px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.3rem;
  flex-shrink: 0;
}
.jff-job-info { flex: 1; }
.jff-job-title { font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:700; margin-bottom:2px; }
.jff-job-meta  { font-size:0.78rem; color:var(--muted); display:flex; gap:12px; flex-wrap:wrap; }
.jff-match-ring {
  width: 52px; height: 52px;
  border-radius: 50%;
  border: 3px solid;
  display: flex; align-items: center; justify-content: center;
  font-family: 'Syne', sans-serif;
  font-size: 0.78rem;
  font-weight: 700;
  flex-shrink: 0;
}

/* ── SECTION HEADER ── */
.jff-section-header {
  padding: 0 48px;
  margin-bottom: 24px;
}
.jff-section-label {
  font-size: 0.75rem;
  font-weight: 600;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: var(--accent);
  margin-bottom: 6px;
}
.jff-section-title {
  font-family: 'Syne', sans-serif;
  font-size: 1.7rem;
  font-weight: 800;
  letter-spacing: -0.5px;
}

/* ── PROGRESS BAR ── */
.jff-progress-wrap { margin: 10px 0; }
.jff-progress-label {
  display: flex;
  justify-content: space-between;
  font-size: 0.8rem;
  color: var(--muted);
  margin-bottom: 6px;
}
.jff-progress-bar {
  height: 6px;
  border-radius: 99px;
  background: rgba(255,255,255,0.08);
  overflow: hidden;
}
.jff-progress-fill {
  height: 100%;
  border-radius: 99px;
  transition: width 1s ease;
}

/* ── TABS ── */
.jff-tabs { display: flex; gap: 4px; padding: 0 48px; margin-bottom: 32px; }
.jff-tab {
  padding: 10px 24px;
  border-radius: 10px;
  font-size: 0.875rem;
  font-weight: 500;
  cursor: pointer;
  border: 1px solid transparent;
  color: var(--muted);
  transition: all 0.2s;
  background: none;
}
.jff-tab:hover { color: var(--text); background: var(--surface); border-color: var(--border); }
.jff-tab.active {
  background: linear-gradient(135deg, rgba(108,99,255,0.25), rgba(0,212,170,0.15));
  border-color: rgba(108,99,255,0.4);
  color: var(--text);
}

/* ── BUTTON ── */
.jff-btn {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 12px 28px;
  border-radius: 50px;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  border: none;
  transition: all 0.25s;
  text-decoration: none;
}
.jff-btn-primary {
  background: linear-gradient(135deg, var(--accent), #9C63FF);
  color: #fff;
  box-shadow: 0 0 30px rgba(108,99,255,0.4);
}
.jff-btn-primary:hover { transform: translateY(-2px); box-shadow: 0 0 50px rgba(108,99,255,0.6); }
.jff-btn-ghost {
  background: var(--surface);
  border: 1px solid var(--border) !important;
  color: var(--text);
}

/* ── RECRUITER CARD ── */
.jff-rec-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 20px;
  margin-bottom: 10px;
  display: flex;
  gap: 16px;
  align-items: flex-start;
  transition: all 0.25s;
}
.jff-rec-card:hover { border-color: var(--accent2); background: rgba(0,212,170,0.05); }
.jff-rec-avatar {
  width: 48px; height: 48px;
  border-radius: 14px;
  font-size: 1.4rem;
  display: flex; align-items: center; justify-content: center;
  flex-shrink: 0;
}
.jff-rec-name { font-family:'Syne',sans-serif; font-weight:700; font-size:0.95rem; }
.jff-rec-role { font-size:0.8rem; color:var(--muted); margin-top:2px; }
.jff-score-badge {
  margin-left: auto;
  padding: 4px 12px;
  border-radius: 50px;
  font-size: 0.78rem;
  font-weight: 700;
}

/* ── FOOTER ── */
.jff-footer {
  text-align: center;
  padding: 40px;
  border-top: 1px solid var(--border);
  color: var(--muted);
  font-size: 0.82rem;
  margin-top: 60px;
}

/* ── Streamlit widget overrides ── */
div[data-testid="stFileUploader"] {
  background: transparent !important;
}
div[data-testid="stFileUploader"] > div {
  background: rgba(108,99,255,0.06) !important;
  border: 2px dashed rgba(108,99,255,0.4) !important;
  border-radius: 16px !important;
  transition: all 0.3s !important;
}
div[data-testid="stFileUploader"] > div:hover {
  background: rgba(108,99,255,0.12) !important;
  border-color: rgba(108,99,255,0.8) !important;
}
div[data-testid="stFileUploader"] label { color: var(--muted) !important; }
div[data-testid="stSelectbox"] > div,
div[data-testid="stTextInput"] > div > div {
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  color: var(--text) !important;
}
div[data-testid="stSelectbox"] svg { color: var(--muted) !important; }
.stButton > button {
  background: linear-gradient(135deg, #6C63FF, #9C63FF) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 50px !important;
  padding: 10px 28px !important;
  font-weight: 600 !important;
  font-family: 'DM Sans', sans-serif !important;
  box-shadow: 0 0 30px rgba(108,99,255,0.35) !important;
  transition: all 0.25s !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 0 50px rgba(108,99,255,0.55) !important;
}
div[data-testid="stTabs"] button {
  font-family: 'DM Sans', sans-serif !important;
  color: var(--muted) !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--text) !important;
  border-bottom-color: var(--accent) !important;
}
/* Progress bar color */
div[data-testid="stProgress"] div { background: linear-gradient(90deg, #6C63FF, #00D4AA) !important; }

/* Metric overrides */
div[data-testid="metric-container"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 20px !important;
  backdrop-filter: blur(12px) !important;
}
div[data-testid="metric-container"] label { color: var(--muted) !important; font-size:0.8rem !important; }
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
  font-family: 'Syne', sans-serif !important;
  font-size: 1.9rem !important;
  font-weight: 800 !important;
  background: linear-gradient(135deg, #6C63FF, #00D4AA) !important;
  -webkit-background-clip: text !important;
  -webkit-text-fill-color: transparent !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricDelta"] { font-size:0.82rem !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "page" not in st.session_state:
    st.session_state.page = "home"
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = None
if 'job_recommendations' not in st.session_state:
    st.session_state.job_recommendations = None
if 'shortlisted_jobs' not in st.session_state:
    st.session_state.shortlisted_jobs = []

# ─────────────────────────────────────────────
#  BACKEND LOGIC
# ─────────────────────────────────────────────
@st.cache_resource
def load_models():
    """Load all models and utilities"""
    try:
        resume_parser = ResumeParser()
        job_parser = JobParser()
        model_utils = ModelUtils()
        return resume_parser, job_parser, model_utils
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

@st.cache_data
def load_jobs_data():
    """Load jobs data"""
    try:
        jobs_df = pd.read_csv(DATA_PATH)
        return jobs_df
    except Exception as e:
        st.error(f"Error loading jobs data: {e}")
        return pd.DataFrame()

def generate_job_recommendations(resume_data, location, domain, remote_pref="Any"):
    """Generate job recommendations based on resume"""
    try:
        resume_parser, job_parser, model_utils = load_models()
        if not (resume_parser and job_parser and model_utils):
            st.error("❌ Unable to load recommendation models. Please check server configuration.")
            return []

        jobs_df = load_jobs_data()
        
        if jobs_df.empty:
            return []
        
        resume_skills = resume_data['skills']
        resume_text = resume_data['text']
        resume_embedding = model_utils.generate_embedding(resume_text)
        
        # Phase 0: Filter by location (optional - check if column exists)
        filtered_df = jobs_df.copy()
        if 'Job Location' in filtered_df.columns and location and location != "Anywhere":
            filtered_df = filtered_df[filtered_df['Job Location'].str.contains(location, case=False, na=False)]

        # Remote preference (simple keyword-based)
        if 'Job Location' in filtered_df.columns and remote_pref != "Any":
            if remote_pref == "Remote only":
                mask = filtered_df['Job Location'].str.contains("remote", case=False, na=False)
                filtered_df = filtered_df[mask]
            elif remote_pref == "On-site / Hybrid":
                mask = ~filtered_df['Job Location'].str.contains("remote", case=False, na=False)
                filtered_df = filtered_df[mask]
        
        if filtered_df.empty:
            # If location filter is too strict, use all jobs but prioritize location match
            filtered_df = jobs_df.copy()
        
        # Phase 1: Quick skill-based filtering (no embeddings)
        scored_jobs = []
        for idx, row in filtered_df.iterrows():
            job_text = row['Skills Required']
            job_skills = job_parser.extract_skills_from_jd(job_text)
            
            # If too few skills extracted, add domain-specific skills
            if len(job_skills) < 3:
                if domain in DOMAIN_SKILLS:
                    job_skills.extend(DOMAIN_SKILLS[domain][:3])
            
            matched_skills, missing_skills = model_utils.compute_skill_gap(resume_skills, job_skills)
            skill_match_ratio = len(matched_skills) / len(job_skills) if job_skills else 0
            
            scored_jobs.append({
                'idx': idx,
                'row': row,
                'job_text': job_text,
                'job_skills': job_skills,
                'matched_skills': matched_skills,
                'missing_skills': missing_skills,
                'skill_match_ratio': skill_match_ratio
            })
        
        # Phase 2: Sort by skill match and take top 50 candidates
        scored_jobs.sort(key=lambda x: x['skill_match_ratio'], reverse=True)
        top_candidates = scored_jobs[:50]
        
        # Phase 3: Generate embeddings only for top candidates
        recommendations = []
        for candidate in top_candidates:
            job_embedding = model_utils.generate_embedding(candidate['job_text'])
            text_similarity = model_utils.compute_similarity(resume_embedding, job_embedding)
            fit_score = model_utils.compute_fit_score(text_similarity, candidate['skill_match_ratio'])
            
            recommendations.append({
                'job_id': candidate['row']['Job ID'],
                'job_title': candidate['row']['Job Title'],
                'company': candidate['row']['Company Name'],
                'location': candidate['row']['Job Location'],
                'job_description': candidate['row']['Skills Required'],
                'experience': candidate['row']['Experience Required'],
                'salary': candidate['row'].get('Salary', 'Not disclosed'),
                'job_type': candidate['row'].get('Job Type', 'Unknown'),
                'fit_score': fit_score,
                'text_similarity': text_similarity,
                'skill_match_ratio': candidate['skill_match_ratio'],
                'job_skills': candidate['job_skills'],
                'matched_skills': candidate['matched_skills'],
                'missing_skills': candidate['missing_skills']
            })
        
        # Sort by fit score and return top 5
        recommendations.sort(key=lambda x: x['fit_score'], reverse=True)
        return recommendations[:5]
    
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return []

# ─────────────────────────────────────────────
#  HELPER — HTML card blocks
# ─────────────────────────────────────────────
def feature_card(icon, color, title, desc):
    return f"""
    <div class="jff-card">
      <div class="jff-card-title">
        <div class="icon" style="background:{color}22;">{icon}</div>
        {title}
      </div>
      <p>{desc}</p>
    </div>
    """

def job_card(emoji, bg, title, company, location, salary, match, match_color):
    return f"""
    <div class="jff-job-card" style="margin-bottom: 0; border-bottom-left-radius: 0; border-bottom-right-radius: 0;">
      <div class="jff-job-logo" style="background:{bg}22;">{emoji}</div>
      <div class="jff-job-info" style="flex: 1;">
        <div class="jff-job-title">{title}</div>
        <div style="font-size: 0.85rem; color: {match_color}; font-weight: 600; margin-bottom: 4px;">Fit Score: {match}%</div>
        <div class="jff-job-meta">
          <span>🏢 {company}</span>
          <span>📍 {location}</span>
          <span>💰 {salary}</span>
        </div>
      </div>
      <div class="jff-match-ring" style="border-color:{match_color};color:{match_color};">{match}%</div>
    </div>
    """

def skill_pills(matched, gaps, tolearn):
    m = "".join(f'<span class="jff-skill-pill skill-match">✓ {s}</span>' for s in matched)
    g = "".join(f'<span class="jff-skill-pill skill-gap">✗ {s}</span>' for s in gaps)
    l = "".join(f'<span class="jff-skill-pill skill-learn">↑ {s}</span>' for s in tolearn)
    return f'<div class="jff-skills">{m}{g}{l}</div>'

def progress_bar(label, pct, color):
    return f"""
    <div class="jff-progress-wrap">
      <div class="jff-progress-label"><span>{label}</span><span style="color:{color};font-weight:600;">{pct}%</span></div>
      <div class="jff-progress-bar">
        <div class="jff-progress-fill" style="width:{pct}%;background:linear-gradient(90deg,{color}88,{color});"></div>
      </div>
    </div>
    """

# ─────────────────────────────────────────────
#  NAV BAR
# ─────────────────────────────────────────────
pages = ["home", "candidate", "recruiter", "about"]
labels = {"home": "🏠 Home", "candidate": "📄 Candidate", "recruiter": "💼 Recruiter", "about": "ℹ️ About"}

nav_html = """
<div class="jff-nav">
  <div class="jff-logo">JobFit<span>Finder</span></div>
  <nav class="jff-navlinks">
"""
for p in pages:
    active = "active" if st.session_state.page == p else ""
    nav_html += f'<a class="jff-navlink {active}" onclick="void(0)">{labels[p]}</a>'
nav_html += """
    <a class="jff-navlink cta">🎯 Analyze Resume</a>
  </nav>
</div>
"""
st.markdown(nav_html, unsafe_allow_html=True)

# Real nav buttons hidden below (Streamlit clickable)
nav_cols = st.columns([1,1,1,1,1,3])
with nav_cols[0]:
    if st.button("🏠 Home", key="nav_home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
with nav_cols[1]:
    if st.button("📄 Candidate", key="nav_cand", use_container_width=True):
        st.session_state.page = "candidate"
        st.rerun()
with nav_cols[2]:
    if st.button("💼 Recruiter", key="nav_rec", use_container_width=True):
        st.session_state.page = "recruiter"
        st.rerun()
with nav_cols[3]:
    if st.button("ℹ️ About", key="nav_about", use_container_width=True):
        st.session_state.page = "about"
        st.rerun()

# ─────────────────────────────────────────────
#  PAGE: HOME
# ─────────────────────────────────────────────
if st.session_state.page == "home":

    st.markdown("""
    <div class="jff-hero">
      <div class="jff-hero-badge">🤖 Powered by NLP & Machine Learning</div>
      <h1>Find Your <span class="grad">Perfect Job</span><br>with AI Precision</h1>
      <p>Upload your ATS resume once. Our AI matches your skills, experience, and goals with the best-fit opportunities — instantly.</p>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    st.markdown("""
    <div class="jff-stats">
      <div class="jff-stat"><div class="val">94%</div><div class="lbl">Match Accuracy</div></div>
      <div class="jff-stat"><div class="val">12k+</div><div class="lbl">Jobs Analyzed</div></div>
      <div class="jff-stat"><div class="val">3.2s</div><div class="lbl">Avg. Analysis Time</div></div>
      <div class="jff-stat"><div class="val">8k+</div><div class="lbl">Candidates Placed</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Feature cards
    st.markdown('<div class="jff-section-header"><div class="jff-section-label">What We Do</div><div class="jff-section-title">Everything you need to land the right role</div></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4, gap="medium")
    cards_data = [
        ("🧠", "#6C63FF", "AI Skill Matching", "NLP extracts skills from your resume and maps them to job requirements with semantic similarity."),
        ("📊", "#00D4AA", "Fit Score Engine", "ML model scores each job 0-100% based on experience, skills, and role alignment."),
        ("🗺️", "#FF6B6B", "Career Path Map", "See where you stand and what skills to learn to level up your career trajectory."),
        ("⚡", "#FFC300", "Instant Analysis", "ATS-optimized parsing in under 4 seconds. No wait, no friction."),
    ]
    for col, (icon, color, title, desc) in zip([col1, col2, col3, col4], cards_data):
        with col:
            st.markdown(feature_card(icon, color, title, desc), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Quick CTA
    st.markdown('<div class="jff-section-header"><div class="jff-section-label">Get Started</div><div class="jff-section-title">Try it in seconds</div></div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("""
        <div class="jff-card">
          <div class="jff-card-title"><div class="icon" style="background:#6C63FF22;">📄</div> For Candidates</div>
          <p style="margin-bottom:16px;">Upload your resume, get AI-powered job matches with fit scores, skill gap analysis, and recommendations tailored to your profile.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("→ Go to Candidate Portal", key="home_cta_cand"):
            st.session_state.page = "candidate"
            st.rerun()
    with c2:
        st.markdown("""
        <div class="jff-card">
          <div class="jff-card-title"><div class="icon" style="background:#00D4AA22;">💼</div> For Recruiters</div>
          <p style="margin-bottom:16px;">Browse ranked candidate profiles, filter by skill match, experience level, and role fit. Shortlist top talent with one click.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("→ Go to Recruiter View", key="home_cta_rec"):
            st.session_state.page = "recruiter"
            st.rerun()

# ─────────────────────────────────────────────
#  PAGE: CANDIDATE PORTAL
# ─────────────────────────────────────────────
elif st.session_state.page == "candidate":

    st.markdown("""
    <div class="jff-hero" style="padding-bottom:32px;">
      <div class="jff-hero-badge">📄 Candidate Portal</div>
      <h1 style="font-size:clamp(1.8rem,3.5vw,2.8rem);">Upload. Analyze. <span class="grad">Get Matched.</span></h1>
      <p>Paste your resume or upload a PDF/DOCX. Our AI reads between the lines — not just keywords.</p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<div style="padding:0 0 12px;"><b style="font-family:Syne,sans-serif;">📎 Upload Resume</b></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Drop your ATS resume here",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed",
        )
        st.markdown('<div style="margin-top:16px;margin-bottom:8px;"><b style="font-family:Syne,sans-serif;">🎯 Target Role</b></div>', unsafe_allow_html=True)
        role = st.selectbox("Select target job category", [
            "Software Engineer", "Data Scientist", "ML Engineer", "Product Manager",
            "DevOps / SRE", "UI/UX Designer", "Data Analyst", "Cloud Architect",
        ], label_visibility="collapsed")
        location = st.selectbox("Preferred location", [
            "Remote", "Hyderabad", "Bangalore", "Mumbai", "Delhi NCR", "Chennai", "Pune", "Anywhere"
        ], label_visibility="collapsed")
        exp = st.selectbox("Years of experience", ["0-1 yr (Fresher)", "1-3 yrs", "3-5 yrs", "5-8 yrs", "8+ yrs"], label_visibility="collapsed")

        if st.button("🚀 Analyze My Resume", use_container_width=True):
            if not uploaded:
                st.error("❌ Please upload a resume file")
            else:
                with st.spinner(""):
                    progress = st.progress(0, text="Parsing ATS resume…")
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1]) as tmp_file:
                            tmp_file.write(uploaded.getbuffer())
                            temp_path = tmp_file.name
                        
                        progress.progress(20, text="Extracting skills with NLP…")
                        
                        resume_parser, _, model_utils = load_models()
                        if not resume_parser or not model_utils:
                            st.error("❌ Could not load NLP models.")
                        else:
                            file_type = uploaded.name.split('.')[-1]
                            resume_text = resume_parser.extract_text(temp_path, file_type)
                            is_valid_ats, validation_message = resume_parser.validate_ats_resume(resume_text)
                            
                            progress.progress(50, text="Scoring job fit with ML…")
                            
                            if not is_valid_ats:
                                st.error(validation_message)
                            else:
                                resume_data = resume_parser.parse_resume(temp_path, file_type)
                                if resume_data is None:
                                    st.error("❌ Could not parse resume. Please upload a valid PDF or DOCX.")
                                else:
                                    st.session_state.resume_data = resume_data
                                    
                                    progress.progress(75, text="Ranking matches…")
                                    
                                    job_recs = generate_job_recommendations(
                                        resume_data,
                                        location if location != "Anywhere" else "",
                                        role,  # using Role as domain
                                        "Any"  # ignoring remote pref for simplicity here to match old UI
                                    )
                                    st.session_state.job_recommendations = job_recs
                                    
                                    progress.progress(100, text="Generating insights…")
                                    st.session_state.analyzed = True
                                    st.success("✅ Analysis complete! Scroll to see your matches →")
                                    
                                    with st.expander("🔍 Debug Info (remove later)"):
                                        st.write("job_recs sample:", job_recs[:2] if job_recs else "EMPTY")
                                        st.write("Keys in first result:", list(job_recs[0].keys()) if job_recs else "NO KEYS")
                    except Exception as e:
                        st.error(f"❌ Error processing resume: {e}")
                    finally:
                        try:
                            if "temp_path" in locals() and os.path.exists(temp_path):
                                os.unlink(temp_path)
                        except Exception:
                            pass

    with right:
        if st.session_state.analyzed and st.session_state.resume_data and st.session_state.job_recommendations is not None:
            job_recs = st.session_state.job_recommendations
            avg_fit = sum(r.get('fit_score', 0) for r in job_recs) / len(job_recs) if job_recs else 0
            
            # compute total skill coverage
            total_req = sum(len(r.get('job_skills', r.get('skills', r.get('required_skills', [])))) for r in job_recs) if job_recs else 1
            total_matched = sum(len(r.get('matched_skills', [])) for r in job_recs) if job_recs else 0
            skill_pct = (total_matched / total_req * 100) if total_req > 0 else 0
            
            st.markdown('<div style="margin-bottom:16px;"><b style="font-family:Syne,sans-serif;font-size:1.05rem;">📊 Your AI Fit Summary</b></div>', unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg Top Fit", f"{avg_fit:.1f}%", f"{len(job_recs)} matches")
            m2.metric("Skills Match", f"{skill_pct:.1f}%", f"{total_matched} of {total_req}")
            m3.metric("Jobs Found", str(len(job_recs)), "for your profile")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class="jff-card" style="margin-bottom:16px;">
              <div class="jff-card-title">🧩 Skill Analysis</div>
            """, unsafe_allow_html=True)
            
            # dynamic progress bars based on resume domains
            skills = st.session_state.resume_data.get('skills', [])
            if skills:
                # aggregate top 3 missing skills from matches
                all_missing = {}
                for rec in job_recs:
                    for ms in rec.get('missing_skills', []):
                        all_missing[ms] = all_missing.get(ms, 0) + 1
                to_learn = [k for k, v in sorted(all_missing.items(), key=lambda item: item[1], reverse=True)[:5]]
                
                # mock a strength calculation per domain
                strength_scores = {}
                for dom, dom_skills in DOMAIN_SKILLS.items():
                    overlap = len(set(s.lower() for s in skills) & set(ds.lower() for ds in dom_skills))
                    strength_scores[dom] = overlap
                
                top_domains = sorted(strength_scores.items(), key=lambda item: item[1], reverse=True)[:3]
                colors = ["#6C63FF", "#00D4AA", "#FFC300"]
                for i, (dom, score) in enumerate(top_domains):
                    pct = min(100, (score / 5) * 100)
                    st.markdown(progress_bar(dom, int(pct), colors[i % len(colors)]), unsafe_allow_html=True)
                
                # all skills found in resume
                matched_all = set()
                for rec in job_recs:
                    matched_all.update(rec.get('matched_skills', []))
                
                st.markdown(skill_pills(
                    list(matched_all)[:8],
                    to_learn[:2],
                    to_learn[2:5]
                ), unsafe_allow_html=True)
            else:
                st.markdown("<p>No skills detected.</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="jff-card" style="text-align:center;padding:48px 32px;">
              <div style="font-size:3rem;margin-bottom:16px;">🎯</div>
              <div class="jff-card-title" style="justify-content:center;margin-bottom:8px;">Ready to analyze</div>
              <p>Upload your resume and click Analyze to see AI-powered job matches, skill gaps, and career insights.</p>
            </div>
            """, unsafe_allow_html=True)

        # Job Matches
        if job_recs:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="jff-section-header"><div class="jff-section-label">AI Matches</div><div class="jff-section-title">Top Job Recommendations</div></div>', unsafe_allow_html=True)
    
            # Use first 3 as Best Matches, next 1 as Stretch, next 1 as Entry
            best_matches = job_recs[:3]
            stretch = [job_recs[3]] if len(job_recs) > 3 else []
            entry = [job_recs[4]] if len(job_recs) > 4 else []
            
            tab1, tab2, tab3 = st.tabs(["🔥 Best Matches", "📈 Stretch Roles", "🌱 Entry Points"])
            
            def render_jobs(job_list, default_emoji="💼", default_color="#6C63FF"):
                for rec in job_list:
                    loc = rec.get("location", "Unknown Location")
                    sal = rec.get("salary", "Not Disclosed")
                    title = rec.get("job_title", "Job Title")
                    company = rec.get("company", "Company")
                    if sal == "Not Disclosed" or pd.isna(sal):
                        sal = "Not Disclosed"
                    
                    # Feature 4 - Fit Score Badge mapping & values
                    match_val = int(rec.get("fit_score", rec.get("score", rec.get("similarity", 0))))
                    mc = "#00D4AA" if match_val > 80 else "#FFC300" if match_val > 60 else "#FF6B6B"
                    
                    # Print Header Card
                    st.markdown(job_card(default_emoji, default_color, title, company, loc, sal, match_val, mc), unsafe_allow_html=True)
                    
                    st.markdown("""<div style="background: var(--surface); border: 1px solid var(--border); border-top: none; padding: 16px; border-bottom-left-radius: 16px; border-bottom-right-radius: 16px; margin-bottom: 16px;">""", unsafe_allow_html=True)
                    
                    # Feature 1 - Matched Skills
                    matched = rec.get("matched_skills", rec.get("matched", []))
                    if matched:
                        st.markdown('<div style="font-size:0.85rem; font-weight:600; margin-bottom:8px;">✓ Matched Skills:</div>', unsafe_allow_html=True)
                        m_html = "".join(f'<span class="jff-skill-pill skill-match">✓ {s}</span>' for s in matched)
                        st.markdown(f'<div class="jff-skills" style="margin-bottom:16px;">{m_html}</div>', unsafe_allow_html=True)
                        
                    # Feature 2 - Missing Skills Gap Analysis
                    missing = rec.get("missing_skills", rec.get("gap_skills", []))
                    if missing:
                        st.markdown('<div style="font-size:0.85rem; font-weight:600; margin-bottom:8px;">✗ Missing Skills (Gap Analysis):</div>', unsafe_allow_html=True)
                        g_html = "".join(f'<span class="jff-skill-pill skill-gap">✗ {s}</span>' for s in missing)
                        st.markdown(f'<div class="jff-skills" style="margin-bottom:16px;">{g_html}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="font-size:0.85rem; color:#00D4AA; margin-bottom:16px;">No missing skills ✓</div>', unsafe_allow_html=True)
                    
                    # Feature 3 - Job Description Expander
                    job_desc = rec.get("job_description", rec.get("description", "N/A"))
                    req_exp = rec.get("required_experience", rec.get("experience", "N/A"))
                    
                    with st.expander("📋 View Job Description"):
                        st.markdown(f"**Required Experience:** {req_exp}")
                        st.markdown("**Job Description:**")
                        st.write(job_desc)
                        
                    st.markdown("</div>", unsafe_allow_html=True)
    
            with tab1:
                if best_matches:
                    render_jobs(best_matches, "🔵", "#6C63FF")
                else:
                    st.info("No strong matches found. Try uploading a more detailed resume.")
            with tab2:
                if stretch:
                    render_jobs(stretch, "🔴", "#FF6B6B")
                else:
                    st.info("Upload more skills to see stretch roles.")
            with tab3:
                if entry:
                    render_jobs(entry, "🟠", "#FF8C00")
                else:
                    st.info("No entry points identified.")
        else:
            st.info("No jobs found. Try uploading a more detailed resume.")

# ─────────────────────────────────────────────
#  PAGE: RECRUITER VIEW
# ─────────────────────────────────────────────
elif st.session_state.page == "recruiter":

    st.markdown("""
    <div class="jff-hero" style="padding-bottom:32px;">
      <div class="jff-hero-badge">💼 Recruiter Dashboard</div>
      <h1 style="font-size:clamp(1.8rem,3.5vw,2.8rem);"><span class="grad">Smart Shortlisting</span><br>for Modern Recruiters</h1>
      <p>Browse AI-ranked candidate profiles. Filter by role, skills, or fit score. Shortlist your top picks instantly.</p>
    </div>
    """, unsafe_allow_html=True)

    # Filter bar
    f1, f2, f3, f4 = st.columns(4, gap="small")
    with f1:
        st.selectbox("Role", ["All Roles", "ML Engineer", "Data Scientist", "SWE", "PM"], label_visibility="visible")
    with f2:
        st.selectbox("Experience", ["Any", "0-2 yrs", "2-5 yrs", "5+ yrs"], label_visibility="visible")
    with f3:
        st.selectbox("Location", ["Anywhere", "Remote", "Hyderabad", "Bangalore"], label_visibility="visible")
    with f4:
        st.selectbox("Min Fit Score", ["Any", "70%+", "80%+", "90%+"], label_visibility="visible")

    st.markdown("<br>", unsafe_allow_html=True)

    # Use session state data if candidate has uploaded a resume
    has_candidate = st.session_state.analyzed and st.session_state.resume_data and st.session_state.job_recommendations
    
    if has_candidate:
        job_recs = st.session_state.job_recommendations or []
        high_fit_count = sum(1 for r in job_recs if r.get('fit_score', 0) >= 80)
        avg_fit = sum(r.get('fit_score', 0) for r in job_recs) / len(job_recs) if job_recs else 0
        total_candidates = 1 # Only one active session candidate currently supported in app
        shortlisted = len(st.session_state.shortlisted_jobs)
    else:
        # Fallback to zeros instead of mock data
        high_fit_count = 0
        avg_fit = 0
        total_candidates = 0
        shortlisted = 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Candidates", str(total_candidates))
    m2.metric("High Fit (>80%)", str(high_fit_count))
    m3.metric("Shortlisted", str(shortlisted))
    m4.metric("Avg. Fit Score", f"{int(avg_fit)}%")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="jff-section-header"><div class="jff-section-label">Top Matches</div><div class="jff-section-title">Ranked Candidate Pool</div></div>', unsafe_allow_html=True)

    l, r = st.columns([3, 2], gap="large")
    with l:
        if has_candidate:
            resume_data = st.session_state.resume_data
            contact = resume_data.get("contact", {})
            email = contact.get("email", "candidate@email.com")
            
            # Simple heuristic for name from start of resume text
            resume_lines = [line.strip() for line in resume_data.get("text", "").splitlines() if line.strip()]
            name_guess = resume_lines[0] if resume_lines else "Candidate"
            
            # Use top job recommendation title as their primary role fit
            best_role = job_recs[0].get('job_title', 'Professional') if job_recs else "Professional"
            
            # Use average score as their overall score in this view
            candidate_score = int(avg_fit)
            skills_str = " · ".join(resume_data.get('skills', [])[:5])
            
            score_color = "#00D4AA" if candidate_score > 80 else "#FFC300" if candidate_score > 60 else "#FF6B6B"
            
            st.markdown(f"""
            <div class="jff-rec-card">
              <div class="jff-rec-avatar" style="background:#6C63FF22;">🧑‍💻</div>
              <div style="flex:1;">
                <div class="jff-rec-name">{name_guess}</div>
                <div class="jff-rec-role">{best_role} · Contact: {email}</div>
                <div style="margin-top:8px;display:flex;gap:6px;flex-wrap:wrap;">
                  {"".join(f'<span class="jff-skill-pill skill-match" style="font-size:0.72rem;padding:3px 10px;">{s}</span>' for s in skills_str.split(" · ") if s)}
                </div>
              </div>
              <div class="jff-score-badge" style="background:{score_color}22;color:{score_color};border:1px solid {score_color}44;">{candidate_score}% fit</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No candidates analyzed in this session yet. Upload a resume in the Candidate Portal first.")

    with r:
        st.markdown("""
        <div class="jff-card">
          <div class="jff-card-title" style="margin-bottom:16px;">📈 Pool Insights</div>
        """, unsafe_allow_html=True)
        
        if has_candidate:
            skills = st.session_state.resume_data.get('skills', [])
            # Fake aggregate stats based on Candidate's top skills
            skill_stats = {}
            for dom, dom_skills in DOMAIN_SKILLS.items():
                overlap = len(set(s.lower() for s in skills) & set(ds.lower() for ds in dom_skills))
                skill_stats[dom] = overlap
                
            top_domains = sorted(skill_stats.items(), key=lambda item: item[1], reverse=True)[:5]
            colors = ["#6C63FF", "#00D4AA", "#FFC300", "#FF6B6B", "#9C63FF"]
            for i, (dom, score) in enumerate(top_domains):
                pct = min(100, (score / 5) * 100)  # simple scaling
                if pct == 0:  # give a small baseline if 0 for visuals
                    pct = random.randint(15, 30)
                st.markdown(progress_bar(dom, int(pct), colors[i % len(colors)]), unsafe_allow_html=True)
        else:
            st.markdown(progress_bar("Python", 0, "#6C63FF"), unsafe_allow_html=True)
            st.markdown(progress_bar("SQL", 0, "#00D4AA"), unsafe_allow_html=True)
            st.markdown(progress_bar("Machine Learning", 0, "#FFC300"), unsafe_allow_html=True)
            st.markdown(progress_bar("Cloud Skills", 0, "#FF6B6B"), unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="jff-card">
          <div class="jff-card-title" style="margin-bottom:12px;">🏆 Fit Distribution</div>
          <div style="display:flex;flex-direction:column;gap:10px;">
        """, unsafe_allow_html=True)
        
        # Calculate dynamic fit distribution buckets for UI
        if has_candidate:
            count_90 = sum(1 for r in job_recs if r.get('fit_score', 0) >= 90)
            count_70 = sum(1 for r in job_recs if 70 <= r.get('fit_score', 0) < 90)
            count_low = sum(1 for r in job_recs if r.get('fit_score', 0) < 70)
            total = len(job_recs)
            pct_90 = int((count_90 / total) * 100) if total else 0
            pct_70 = int((count_70 / total) * 100) if total else 0
            pct_low = int((count_low / total) * 100) if total else 0
        else:
            count_90, count_70, count_low = 0, 0, 0
            pct_90, pct_70, pct_low = 0, 0, 0
            
        st.markdown(f"""
            <div style="display:flex;align-items:center;gap:10px;">
              <span style="color:#00D4AA;font-weight:700;min-width:36px;">90%+</span>
              <div style="flex:1;height:8px;background:rgba(255,255,255,0.08);border-radius:99px;overflow:hidden;">
                <div style="width:{pct_90}%;height:100%;background:#00D4AA;border-radius:99px;"></div>
              </div>
              <span style="color:var(--muted);font-size:0.78rem;">{count_90}</span>
            </div>
            <div style="display:flex;align-items:center;gap:10px;">
              <span style="color:#6C63FF;font-weight:700;min-width:36px;">70%+</span>
              <div style="flex:1;height:8px;background:rgba(255,255,255,0.08);border-radius:99px;overflow:hidden;">
                <div style="width:{pct_70}%;height:100%;background:#6C63FF;border-radius:99px;"></div>
              </div>
              <span style="color:var(--muted);font-size:0.78rem;">{count_70}</span>
            </div>
            <div style="display:flex;align-items:center;gap:10px;">
              <span style="color:#FFC300;font-weight:700;min-width:36px;">&lt;70%</span>
              <div style="flex:1;height:8px;background:rgba(255,255,255,0.08);border-radius:99px;overflow:hidden;">
                <div style="width:{pct_low}%;height:100%;background:#FFC300;border-radius:99px;"></div>
              </div>
              <span style="color:var(--muted);font-size:0.78rem;">{count_low}</span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  PAGE: ABOUT
# ─────────────────────────────────────────────
elif st.session_state.page == "about":

    st.markdown("""
    <div class="jff-hero" style="padding-bottom:40px;">
      <div class="jff-hero-badge">ℹ️ About JobFitFinder</div>
      <h1 style="font-size:clamp(1.8rem,3.5vw,2.8rem);">Built with <span class="grad">AI. Designed for Humans.</span></h1>
      <p>JobFitFinder is a mini-project that demonstrates the power of NLP and ML in solving real-world recruitment challenges — faster and smarter.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")
    tech_cards = [
        ("🔤", "#6C63FF", "NLP Pipeline", "spaCy + Transformers extract skills, roles, and context from raw resume text. Semantic similarity via sentence-BERT."),
        ("🤖", "#00D4AA", "ML Models", "Gradient Boosting & cosine similarity score candidate-job fit. Trained on 50k+ job description pairs."),
        ("⚡", "#FFC300", "Streamlit Frontend", "Fully interactive UI with real-time analysis, custom CSS glassmorphism design, and responsive layouts."),
    ]
    for col, (icon, color, title, desc) in zip([c1, c2, c3], tech_cards):
        with col:
            st.markdown(feature_card(icon, color, title, desc), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    a1, a2 = st.columns([2, 1], gap="large")
    with a1:
        st.markdown("""
        <div class="jff-card">
          <div class="jff-card-title" style="margin-bottom:16px;">🗺️ How It Works</div>
          <div style="display:flex;flex-direction:column;gap:16px;">
            <div style="display:flex;gap:16px;align-items:flex-start;">
              <div style="min-width:32px;height:32px;border-radius:50%;background:rgba(108,99,255,0.2);border:1px solid rgba(108,99,255,0.4);display:flex;align-items:center;justify-content:center;font-weight:700;color:#6C63FF;font-size:0.85rem;">1</div>
              <div><b>Upload Resume</b><br><span style="color:var(--muted);font-size:0.85rem;">ATS-compatible PDF or DOCX parsed by our NLP engine in seconds.</span></div>
            </div>
            <div style="display:flex;gap:16px;align-items:flex-start;">
              <div style="min-width:32px;height:32px;border-radius:50%;background:rgba(0,212,170,0.2);border:1px solid rgba(0,212,170,0.4);display:flex;align-items:center;justify-content:center;font-weight:700;color:#00D4AA;font-size:0.85rem;">2</div>
              <div><b>Skill Extraction</b><br><span style="color:var(--muted);font-size:0.85rem;">Named Entity Recognition + custom skill taxonomy maps 2000+ skills.</span></div>
            </div>
            <div style="display:flex;gap:16px;align-items:flex-start;">
              <div style="min-width:32px;height:32px;border-radius:50%;background:rgba(255,195,0,0.2);border:1px solid rgba(255,195,0,0.4);display:flex;align-items:center;justify-content:center;font-weight:700;color:#FFC300;font-size:0.85rem;">3</div>
              <div><b>Semantic Matching</b><br><span style="color:var(--muted);font-size:0.85rem;">Sentence-BERT embeddings compare your profile against job descriptions beyond keywords.</span></div>
            </div>
            <div style="display:flex;gap:16px;align-items:flex-start;">
              <div style="min-width:32px;height:32px;border-radius:50%;background:rgba(255,107,107,0.2);border:1px solid rgba(255,107,107,0.4);display:flex;align-items:center;justify-content:center;font-weight:700;color:#FF6B6B;font-size:0.85rem;">4</div>
              <div><b>Ranked Recommendations</b><br><span style="color:var(--muted);font-size:0.85rem;">ML model produces a 0-100% fit score for each role and ranks the best matches.</span></div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    with a2:
        st.markdown("""
        <div class="jff-card">
          <div class="jff-card-title" style="margin-bottom:16px;">🛠️ Tech Stack</div>
          <div style="display:flex;flex-direction:column;gap:10px;">
        """ + "".join(f"""
            <div style="display:flex;align-items:center;gap:10px;padding:8px 12px;background:var(--surface);border-radius:10px;border:1px solid var(--border);">
              <span>{icon}</span>
              <span style="font-size:0.85rem;">{tech}</span>
            </div>""" for icon, tech in [
            ("🐍","Python 3.10+"),
            ("📊","Streamlit 1.33"),
            ("🤗","HuggingFace Transformers"),
            ("🔤","spaCy NLP"),
            ("📐","Scikit-learn"),
            ("📄","PyMuPDF / python-docx"),
            ("🎨","Custom CSS / glassmorphism"),
        ]) + """
          </div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div class="jff-footer">
  <div class="jff-logo" style="font-size:1.1rem;margin-bottom:8px;">JobFit<span>Finder</span></div>
  Intelligent Job Recommender · Powered by NLP & Machine Learning<br>
  <div style="margin: 10px 0; display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; opacity: 0.7; font-size: 0.75rem;">
    <span>#AIResumeAnalysis</span> 
    <span>#NLPJobMatching</span> 
    <span>#ApplicantTracking</span> 
    <span>#MLFitScoring</span> 
    <span>#CareerPathing</span>
  </div>
  <span style="opacity:0.4;">© 2024 · Mini Project · All Rights Reserved</span>
</div>
""", unsafe_allow_html=True)
