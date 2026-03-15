"""
JobFitFinder: Intelligent Job Recommender Using NLP & ML
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import tempfile
from utils.resume_parser import ResumeParser
from utils.job_parser import JobParser
from utils.model_utils import ModelUtils
# (DB integration removed for candidate-only deployment)


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


# Page Configuration
st.set_page_config(
    page_title="JobFitFinder - Job Recommender",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern, glassmorphic UI
st.markdown("""
<style>
    :root {
        --primary: #6366f1;
        --primary-soft: rgba(99, 102, 241, 0.12);
        --accent: #22c55e;
        --bg-dark: #020617;
        --card-bg: rgba(15, 23, 42, 0.96);
        --border-soft: rgba(148, 163, 184, 0.35);
        --text-muted: #9ca3af;
    }

    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }

    body {
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background:
            radial-gradient(circle at 0% 0%, #1d4ed8 0, transparent 40%),
            radial-gradient(circle at 100% 0%, #4c1d95 0, transparent 45%),
            radial-gradient(circle at 50% 100%, #0f766e 0, transparent 50%),
            var(--bg-dark);
        color: #e5e7eb;
    }

    .main {
        background: transparent;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        max-width: 1180px;
    }

    .main-header {
        position: relative;
        text-align: left;
        padding: 1.8rem 1.6rem;
        background: radial-gradient(circle at 0 0, rgba(248, 250, 252, 0.18), transparent 60%),
                    linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 64, 175, 0.96));
        color: white;
        border-radius: 1.6rem;
        margin-bottom: 1.6rem;
        border: 1px solid rgba(148, 163, 184, 0.45);
        box-shadow:
            0 24px 80px rgba(15, 23, 42, 0.9),
            0 0 0 1px rgba(15, 23, 42, 0.9);
        overflow: hidden;
    }

    .main-header::after {
        content: "";
        position: absolute;
        inset: 0;
        background:
            radial-gradient(circle at 20% 0%, rgba(96, 165, 250, 0.2), transparent 55%),
            radial-gradient(circle at 80% 0%, rgba(129, 140, 248, 0.25), transparent 55%);
        opacity: 0.9;
        pointer-events: none;
    }

    .main-header-inner {
        position: relative;
        z-index: 1;
        display: flex;
        flex-direction: column;
        gap: 0.75rem;
    }

    .brand-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        border-radius: 999px;
        padding: 0.2rem 0.7rem;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(148, 163, 184, 0.6);
        color: var(--text-muted);
    }

    .brand-pill-dot {
        width: 7px;
        height: 7px;
        border-radius: 999px;
        background: radial-gradient(circle at 30% 0, #22c55e, #16a34a);
        box-shadow: 0 0 0 4px rgba(34, 197, 94, 0.3);
    }

    .main-header h1 {
        font-size: clamp(2rem, 3vw, 2.4rem);
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .gradient-text {
        background: linear-gradient(120deg, #a5b4fc, #38bdf8, #4ade80);
        -webkit-background-clip: text;
        color: transparent;
    }

    .main-header p {
        font-size: 0.95rem;
        opacity: 0.9;
        max-width: 32rem;
    }

    .header-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.6rem;
        margin-top: 0.35rem;
    }

    .meta-chip {
        border-radius: 999px;
        padding: 0.25rem 0.7rem;
        font-size: 0.75rem;
        border: 1px solid rgba(148, 163, 184, 0.5);
        background: rgba(15, 23, 42, 0.85);
        color: var(--text-muted);
    }

    .section-header {
        font-size: 1.7rem;
        font-weight: 700;
        color: #e5e7eb;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.45rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.45);
    }

    .section-header span {
        background: linear-gradient(120deg, #a5b4fc, #38bdf8);
        -webkit-background-clip: text;
        color: transparent;
    }

    .job-card {
        background: var(--card-bg);
        border-radius: 18px;
        padding: 1.35rem 1.4rem;
        margin-bottom: 1.2rem;
        border: 1px solid var(--border-soft);
        box-shadow:
            0 14px 45px rgba(15, 23, 42, 0.95),
            0 0 0 1px rgba(15, 23, 42, 0.9);
        transition: transform 0.2s ease-out, box-shadow 0.2s ease-out, border-color 0.2s ease-out;
        position: relative;
        overflow: hidden;
    }

    .job-card::before {
        content: "";
        position: absolute;
        top: -40%;
        right: -20%;
        width: 200px;
        height: 200px;
        border-radius: 999px;
        background: radial-gradient(circle at 30% 0, rgba(129, 140, 248, 0.25), transparent 55%);
        opacity: 0.8;
        pointer-events: none;
    }

    .job-card:hover {
        transform: translateY(-4px);
        border-color: rgba(129, 140, 248, 0.85);
        box-shadow:
            0 22px 60px rgba(15, 23, 42, 0.95),
            0 0 0 1px rgba(59, 130, 246, 0.8);
    }

    .job-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e5e7eb;
        margin-bottom: 0.2rem;
    }

    .job-company {
        font-size: 0.95rem;
        color: var(--text-muted);
        margin-bottom: 0.15rem;
    }

    .job-location {
        font-size: 0.85rem;
        color: #6b7280;
        margin-bottom: 0.9rem;
    }

    .fit-score {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: radial-gradient(circle at 0 0, rgba(34, 197, 94, 0.25), transparent 60%),
                    linear-gradient(135deg, #4ade80, #22c55e, #16a34a);
        color: #052e16;
        padding: 0.4rem 0.85rem;
        border-radius: 999px;
        font-weight: 600;
        margin-bottom: 0.6rem;
        font-size: 0.9rem;
        box-shadow: 0 12px 30px rgba(34, 197, 94, 0.4);
    }

    .fit-score span {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.85;
    }

    .skills-section {
        margin: 0.85rem 0;
    }

    .skill-badge {
        display: inline-block;
        background-color: rgba(59, 130, 246, 0.18);
        color: #bfdbfe;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
        font-weight: 500;
        border: 1px solid rgba(59, 130, 246, 0.6);
    }

    .skill-badge-missing {
        display: inline-block;
        background-color: rgba(248, 113, 113, 0.1);
        color: #fecaca;
        padding: 0.3rem 0.7rem;
        border-radius: 999px;
        font-size: 0.78rem;
        margin-right: 0.35rem;
        margin-bottom: 0.35rem;
        font-weight: 500;
        border: 1px solid rgba(248, 113, 113, 0.65);
    }

    .upload-box {
        border-radius: 18px;
        padding: 1.6rem 1.4rem;
        text-align: left;
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(15, 23, 42, 0.94));
        border: 1px dashed rgba(129, 140, 248, 0.7);
        margin: 1.3rem 0 0.8rem 0;
        box-shadow:
            0 18px 45px rgba(15, 23, 42, 0.95),
            0 0 0 1px rgba(15, 23, 42, 0.9);
    }

    .upload-box-title {
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }

    .upload-box-subtitle {
        font-size: 0.85rem;
        color: var(--text-muted);
    }

    .button-container {
        display: flex;
        gap: 0.8rem;
        margin-top: 1.2rem;
        justify-content: center;
    }

    .candidate-card {
        background: var(--card-bg);
        border-radius: 18px;
        padding: 1.35rem 1.4rem;
        margin-bottom: 1.2rem;
        border: 1px solid var(--border-soft);
        box-shadow:
            0 14px 45px rgba(15, 23, 42, 0.95),
            0 0 0 1px rgba(15, 23, 42, 0.9);
    }

    .candidate-name {
        font-size: 1.15rem;
        font-weight: 600;
        color: #e5e7eb;
    }

    .candidate-card p {
        font-size: 0.88rem;
        color: var(--text-muted);
    }

    .about-section {
        background: var(--card-bg);
        border-radius: 18px;
        padding: 1.4rem 1.5rem;
        margin-bottom: 1.2rem;
        border: 1px solid var(--border-soft);
        box-shadow:
            0 16px 48px rgba(15, 23, 42, 0.96),
            0 0 0 1px rgba(15, 23, 42, 0.9);
    }

    .about-section h3 {
        margin-bottom: 0.6rem;
        font-size: 1.1rem;
        color: #e5e7eb;
    }

    .about-section p {
        font-size: 0.9rem;
        color: var(--text-muted);
    }

    .feature-list {
        list-style: none;
        padding: 0;
        margin: 0.2rem 0 0 0;
    }

    .feature-list li {
        padding: 0.45rem 0;
        padding-left: 1.55rem;
        position: relative;
        font-size: 0.9rem;
        color: var(--text-muted);
    }

    .feature-list li:before {
        content: "✓";
        position: absolute;
        left: 0;
        color: #4ade80;
        font-weight: 600;
        font-size: 1rem;
    }

    .sidebar-header {
        text-align: center;
        font-size: 1.3rem;
        font-weight: 700;
        color: #e5e7eb;
        margin-bottom: 0.9rem;
        padding-bottom: 0.9rem;
        border-bottom: 1px solid rgba(148, 163, 184, 0.4);
    }

    .sidebar-header span {
        background: linear-gradient(120deg, #a5b4fc, #38bdf8);
        -webkit-background-clip: text;
        color: transparent;
    }

    [data-testid="stSidebar"] {
        background: radial-gradient(circle at 0 0, rgba(30, 64, 175, 0.35), transparent 55%),
                    radial-gradient(circle at 100% 100%, rgba(8, 47, 73, 0.4), transparent 55%),
                    #020617;
        border-right: 1px solid rgba(15, 23, 42, 0.95);
    }

    .stButton>button {
        border-radius: 999px;
        border: none;
        padding: 0.55rem 0.9rem;
        font-weight: 500;
        font-size: 0.9rem;
        background-image: linear-gradient(135deg, #4f46e5, #22c55e);
        color: #f9fafb;
        box-shadow: 0 12px 30px rgba(79, 70, 229, 0.55);
        transition: transform 0.08s ease-out, box-shadow 0.08s ease-out, filter 0.08s ease-out;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        filter: brightness(1.04);
        box-shadow: 0 16px 40px rgba(79, 70, 229, 0.7);
    }

    .stButton>button:focus {
        outline: none;
        box-shadow:
            0 0 0 1px rgba(129, 140, 248, 0.85),
            0 0 0 4px rgba(129, 140, 248, 0.4);
    }

    .stFileUploader label {
        font-size: 0.9rem;
        color: var(--text-muted);
    }

    .stFileUploader div[data-testid="stFileUploaderDropzone"] {
        border-radius: 14px;
        background: rgba(15, 23, 42, 0.96);
        border: 1px dashed rgba(148, 163, 184, 0.8);
    }

    .stSelectbox>div>div {
        border-radius: 999px !important;
        border-color: rgba(148, 163, 184, 0.6) !important;
        background: rgba(15, 23, 42, 0.96) !important;
        color: #e5e7eb !important;
    }

    .stSelectbox svg {
        color: #64748b;
    }

    .stAlert {
        border-radius: 14px;
    }

    .stExpander {
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.4);
        background: rgba(15, 23, 42, 0.96);
    }

    .loading-spinner {
        display: inline-block;
        animation: spin 1s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
        }

        .main-header {
            padding: 1.4rem 1.1rem;
        }

        .section-header {
            font-size: 1.4rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'Home'
if 'resume_data' not in st.session_state:
    st.session_state.resume_data = None
if 'job_recommendations' not in st.session_state:
    st.session_state.job_recommendations = None
if 'shortlisted_jobs' not in st.session_state:
    st.session_state.shortlisted_jobs = []
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []


# Initialize models and utilities
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


def render_home_page():
    """Render home page"""
    st.markdown("""
    <div class="main-header">
        <h1>💼 JobFitFinder</h1>
        <p>Intelligent Job Recommender Using NLP & ML</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 12px; padding: 2rem; text-align: center; color: white; cursor: pointer;">
        <h2>👤 Candidate Portal</h2>
        <p>Upload your resume and discover your perfect job matches</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🔍 Go to Candidate Portal", key="btn_candidate", use_container_width=True):
        st.session_state.page = 'Candidate Portal'
        st.rerun()


def render_candidate_portal():
    """Render candidate portal"""
    st.markdown("""
    <div class="section-header">👤 Candidate Portal</div>
    """, unsafe_allow_html=True)

    st.markdown("### 📄 Upload Your ATS Resume")
    st.write("📋 **ATS-Compliant Formats:** PDF, DOCX, TXT")
    st.info("ℹ️ **ATS Requirements:** Your resume must include contact information (email/phone), standard sections (Experience, Education, Skills), and be in text-based format (not scanned images).")

    # File upload
    uploaded_file = st.file_uploader("Choose a resume file", type=["pdf", "docx", "txt"])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        location = st.selectbox(
            "Preferred Job Location",
            ["Any"] + ["Ahmedabad", "Jaipur", "Delhi", "Pune", "Noida",
                       "Mumbai", "Hyderabad", "Kolkata", "Bangalore", "Chennai"],
            index=0,
        )
    with col2:
        domain = st.selectbox(
            "Preferred Job Domain",
            list(DOMAIN_SKILLS.keys()),
            index=list(DOMAIN_SKILLS.keys()).index("Software Engineering"),
        )
    with col3:
        remote_pref = st.radio(
            "Remote preference",
            ["Any", "Remote only", "On-site / Hybrid"],
            index=0,
            horizontal=False,
        )
    
    # Submit button
    if st.button("🔍 Get Job Recommendations", use_container_width=True, key="submit_resume"):
        if not uploaded_file:
            st.error("❌ Please upload a resume file")
        else:
            with st.spinner("🔄 Analyzing your resume..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getbuffer())
                        temp_path = tmp_file.name
                    
                    # Parse resume
                    resume_parser, _, model_utils = load_models()
                    if not resume_parser or not model_utils:
                        st.error("❌ Could not load NLP models. Please try again later or contact the administrator.")
                        return

                    file_type = uploaded_file.name.split('.')[-1]
                    resume_text = resume_parser.extract_text(temp_path, file_type)
                    
                    # Validate ATS format
                    is_valid_ats, validation_message = resume_parser.validate_ats_resume(resume_text)
                    
                    if not is_valid_ats:
                        st.error(validation_message)
                    else:
                        st.success(validation_message)
                        
                        # Parse complete resume data
                        resume_data = resume_parser.parse_resume(temp_path, file_type)

                        st.session_state.resume_data = resume_data

                        # Store recommendations for display
                        job_recs = generate_job_recommendations(
                            resume_data,
                            location if location != "Any" else "",
                            domain,
                            remote_pref,
                        )
                        st.session_state.job_recommendations = job_recs

                        # Track simple history (within this session)
                        st.session_state.analysis_history.append(
                            {
                                "location": location,
                                "domain": domain,
                                "remote_pref": remote_pref,
                                "num_recs": len(job_recs),
                            }
                        )
                    
                except Exception as e:
                    st.error(f"❌ Error processing resume: {e}")
                finally:
                    try:
                        if "temp_path" in locals() and os.path.exists(temp_path):
                            os.unlink(temp_path)
                    except Exception:
                        # Best-effort cleanup; ignore OS errors
                        pass
    
    # Display recommendations and enhancements
    job_recs = st.session_state.job_recommendations
    if job_recs:
        st.markdown("---")

        # Compact candidate summary + resume quality card
        resume_data = st.session_state.get("resume_data", {}) or {}
        if resume_data:
            contact = resume_data.get("contact", {})
            email = contact.get("email", "Not found")
            phone = contact.get("phone", "Not found")
            skills = resume_data.get("skills", [])
            skills_preview = ", ".join(skills[:8]) if skills else "Not detected"
            word_count = resume_data.get("word_count", 0)
            resume_lines = [
                line.strip()
                for line in resume_data.get("text", "").splitlines()
                if line.strip()
            ]
            name_guess = resume_lines[0] if resume_lines else "Candidate"

            left, right = st.columns([2, 1])
            with left:
                st.markdown(
                    f"""
                    <div class="candidate-card">
                        <div class="candidate-name">{name_guess}</div>
                        <p style="margin-top: 0.5rem;">
                            <strong>Contact:</strong> {email} | {phone}
                        </p>
                        <p style="margin: 0.25rem 0;">
                            <strong>Resume length:</strong> {word_count} words
                            &nbsp;•&nbsp;
                            <strong>Detected skills:</strong> {len(skills)}
                        </p>
                        <p style="margin-top: 0.5rem;">
                            <strong>Skill snapshot:</strong> {skills_preview}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with right:
                # Simple resume quality score based on ATS-friendly features
                score = 60
                tips = []
                if word_count < 200:
                    score -= 10
                    tips.append("Increase overall resume content above 200 words.")
                if len(skills) < 5:
                    score -= 10
                    tips.append("Add more explicit skills to your Skills section.")
                if "@" not in email:
                    score -= 5
                    tips.append("Make sure your email address is clearly visible.")
                score = max(0, min(100, score))

                st.markdown(
                    f"""
                    <div class="about-section">
                        <h3>📊 Resume Quality Score</h3>
                        <p><strong>Score:</strong> {score}/100</p>
                        <p style="font-size: 0.85rem;">
                            This score is based on content length, skills coverage and contact information.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if tips:
                    st.markdown("**Recommended improvements:**")
                    for t in tips[:4]:
                        st.markdown(f"- {t}")

        # Filters, export and strength profile
        st.markdown("""
        <div class="section-header">🎯 Top Job Recommendations</div>
        """, unsafe_allow_html=True)
        st.caption(
            "Use filters below to refine your matches. Fit score combines text similarity and skill matching."
        )

        filter_col1, filter_col2, filter_col3 = st.columns([1, 1, 1])
        with filter_col1:
            min_fit = st.slider("Minimum fit score", 0, 100, 70, 5)
        with filter_col2:
            min_skill_match = st.slider("Minimum skill match %", 0, 100, 40, 5)
        with filter_col3:
            search_text = st.text_input("Search by title/company keyword", "")

        # Skill roadmap based on missing skills across recommendations
        all_missing = {}
        for rec in job_recs:
            for s in rec.get("missing_skills", []):
                all_missing[s] = all_missing.get(s, 0) + 1
        if all_missing:
            st.markdown("#### 🧭 Skill Roadmap (based on missing skills)")
            sorted_missing = sorted(all_missing.items(), key=lambda x: x[1], reverse=True)[:8]
            for skill, freq in sorted_missing:
                st.markdown(f"- **{skill}** – appears in {freq} of your top matches")

        # Domain strength profile (simple frequency-based view)
        if resume_data:
            skills = resume_data.get("skills", []) or []
            if skills:
                strength_scores = {}
                for dom, dom_skills in DOMAIN_SKILLS.items():
                    overlap = len(set(s.lower() for s in skills) & set(ds.lower() for ds in dom_skills))
                    strength_scores[dom] = overlap
                st.markdown("#### 📈 Domain strength profile")
                st.bar_chart(strength_scores)

        # Build filtered list for display and export
        filtered_recs = []
        for rec in job_recs:
            if rec["fit_score"] < min_fit:
                continue
            if rec.get("skill_match_ratio", 0.0) * 100 < min_skill_match:
                continue
            if search_text:
                blob = f"{rec['job_title']} {rec['company']}".lower()
                if search_text.lower() not in blob:
                    continue
            filtered_recs.append(rec)

        # Export filtered recommendations
        if filtered_recs:
            export_df = pd.DataFrame(
                [
                    {
                        "Job ID": r["job_id"],
                        "Job Title": r["job_title"],
                        "Company": r["company"],
                        "Location": r["location"],
                        "Fit Score": round(r["fit_score"], 1),
                        "Text Similarity": round(r.get("text_similarity", 0.0) * 100, 1),
                        "Skill Match %": round(r.get("skill_match_ratio", 0.0) * 100, 1),
                    }
                    for r in filtered_recs
                ]
            )
            st.download_button(
                label="⬇️ Download recommendations (CSV)",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="jobfitfinder_recommendations.csv",
                mime="text/csv",
            )

        # Shortlisted jobs panel
        if st.session_state.shortlisted_jobs:
            st.markdown("#### ⭐ Shortlisted jobs (this session)")
            for rec in st.session_state.shortlisted_jobs:
                st.markdown(
                    f"- **{rec['job_title']}** at {rec['company']} – {rec['fit_score']:.1f}% fit"
                )

        # Render each recommendation with ability to shortlist and see checklist
        for idx, rec in enumerate(filtered_recs or job_recs, 1):
            with st.container():
                st.markdown(f"""
                <div class="job-card">
                    <div class="job-title">#{idx} {rec['job_title']}</div>
                    <div class="job-company">🏢 {rec['company']}</div>
                    <div class="job-location">📍 {rec['location']}</div>
                    <div class="fit-score">
                        Fit Score: {rec['fit_score']:.1f}%
                        <span>overall match</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Explain why this job is a good match
                text_match_pct = rec.get("text_similarity", 0.0) * 100
                skill_match_pct = rec.get("skill_match_ratio", 0.0) * 100
                st.markdown(
                    f"<p style='font-size: 0.9rem; color: #555;'>"
                    f"Text match: <strong>{text_match_pct:.1f}%</strong> &nbsp;•&nbsp; "
                    f"Skill match: <strong>{skill_match_pct:.1f}%</strong>"
                    f"</p>",
                    unsafe_allow_html=True,
                )
                
                # Matched skills
                matched_text = " ".join([f'<span class="skill-badge">{s}</span>' for s in rec['matched_skills']])
                st.markdown(f"""
                <div class="skills-section">
                    <strong>✓ Matched Skills:</strong><br>
                    {matched_text if matched_text else '<span style="color: #999;">No matched skills</span>'}
                </div>
                """, unsafe_allow_html=True)
                
                # Missing skills
                missing_text = " ".join([f'<span class="skill-badge-missing">{s}</span>' for s in rec['missing_skills']])
                st.markdown(f"""
                <div class="skills-section">
                    <strong>✗ Missing Skills (Gap Analysis):</strong><br>
                    {missing_text if missing_text else '<span style="color: #999;">No missing skills</span>'}
                </div>
                """, unsafe_allow_html=True)

                # Application readiness checklist
                with st.expander("✅ Application readiness checklist"):
                    st.markdown(
                        "- Customised resume for this role\n"
                        "- Cover letter mentions top matched skills\n"
                        "- Portfolio / GitHub / LinkedIn link included\n"
                        "- Prepared examples for 2–3 key skills"
                    )

                # Shortlist button
                if st.button("⭐ Save this job", key=f"shortlist_{rec['job_id']}"):
                    if rec not in st.session_state.shortlisted_jobs:
                        st.session_state.shortlisted_jobs.append(rec)
                        st.success("Added to shortlisted jobs for this session.")
                
                # Job description in expander
                with st.expander("📝 View Job Description"):
                    st.write(rec['job_description'][:500] + "..." if len(rec['job_description']) > 500 else rec['job_description'])
                    st.write(f"**Required Experience:** {rec['experience']}")
                
                st.markdown("---")



def render_about_page():
    """Render about page"""
    st.markdown("""
    <div class="section-header">ℹ️ About JobFitFinder</div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="about-section">
        <h3>How JobFitFinder Works</h3>
        <p>
        JobFitFinder is an intelligent job recommender system that uses Natural Language Processing (NLP) and Machine Learning 
        to match job seekers with perfect job opportunities. The system analyzes ATS-compliant resumes and uses semantic similarity 
        to find the best fitting jobs.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Process Steps
    st.markdown("""
    <div class="about-section">
        <h3>📋 The JobFitFinder Process</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; padding: 1.5rem; color: white; text-align: center;">
            <h4>Step 1: Upload Resume</h4>
            <p style="font-size: 0.9rem;">Upload your ATS-compliant resume in PDF, DOCX, or TXT format</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 10px; padding: 1.5rem; color: white; text-align: center;">
            <h4>Step 2: Validate & Extract</h4>
            <p style="font-size: 0.9rem;">System validates ATS format and extracts skills, experience, and contact info</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 10px; padding: 1.5rem; color: white; text-align: center;">
            <h4>Step 3: Get Recommendations</h4>
            <p style="font-size: 0.9rem;">Receive top 5 job matches with fit scores and skill gap analysis</p>
        </div>
        """, unsafe_allow_html=True)

    # ATS Resume Requirements
    st.markdown("""
    <div class="about-section">
        <h3>✅ ATS Resume Requirements</h3>
        <p>JobFitFinder only accepts <strong>Application Tracking System (ATS) compliant resumes</strong>. Your resume must meet these criteria:</p>
        <ul class="feature-list">
            <li><strong>Minimum 200 words</strong> - Sufficient content for analysis</li>
            <li><strong>Contact Information</strong> - Email address and/or phone number</li>
            <li><strong>Standard Sections</strong> - At least 2 sections (e.g., Education, Experience, Skills)</li>
            <li><strong>Text-based Format</strong> - Not scanned/image-based PDFs</li>
            <li><strong>Clear Structure</strong> - Standard resume format without excessive special characters</li>
            <li><strong>Supported Formats</strong> - PDF, DOCX, or TXT files</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Functionality
    st.markdown("""
    <div class="about-section">
        <h3>🎯 Key Functionality</h3>
        <ul class="feature-list">
            <li><strong>Resume Parsing:</strong> Extracts text from PDF, DOCX, and TXT files</li>
            <li><strong>ATS Validation:</strong> Ensures resume meets industry standards for parsing</li>
            <li><strong>Skill Extraction:</strong> Identifies 40+ technical and soft skills using NLP</li>
            <li><strong>Semantic Matching:</strong> Uses Sentence-BERT embeddings for intelligent job matching</li>
            <li><strong>Fit Score Calculation:</strong> Combines text similarity (60%) with skill matching (40%)</li>
            <li><strong>Skill Gap Analysis:</strong> Shows matched skills and missing skills for each job</li>
            <li><strong>Fast Recommendations:</strong> Processes 20,000+ jobs to find top 5 matches</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Technology Stack
    st.markdown("""
    <div class="about-section">
        <h3>🛠️ Technology Stack</h3>
        <ul class="feature-list">
            <li><strong>Streamlit</strong> - Interactive web application framework</li>
            <li><strong>Sentence-BERT (all-MiniLM-L6-v2)</strong> - Semantic embeddings (384-dimensional)</li>
            <li><strong>spaCy</strong> - Natural Language Processing for skill extraction</li>
            <li><strong>scikit-learn</strong> - Cosine similarity computation</li>
            <li><strong>PyPDF2</strong> - PDF document parsing</li>
            <li><strong>python-docx</strong> - DOCX document parsing</li>
            <li><strong>pandas & numpy</strong> - Data processing and analysis</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="about-section">
        <p style="text-align: center; color: #667eea; font-style: italic;"><strong>Version 1.0.0</strong> | Powered by ML & NLP</p>
    </div>
    """, unsafe_allow_html=True)




def render_recruiter_view():
    """Simple recruiter-facing tools for writing JDs and screening"""
    st.markdown("""
    <div class="section-header">🧑‍💼 Recruiter View</div>
    """, unsafe_allow_html=True)

    st.markdown(
        "Paste a job description to extract key skills and auto-generate a quick screening checklist."
    )

    jd_text = st.text_area("Paste job description here", height=220)
    domain = st.selectbox("Primary domain for this role", list(DOMAIN_SKILLS.keys()))

    if st.button("✨ Analyze job description", use_container_width=True):
        if not jd_text.strip():
            st.error("Please paste a job description first.")
        else:
            resume_parser, job_parser, model_utils = load_models()
            if not job_parser:
                st.error("Unable to load NLP components for recruiter view.")
                return

            with st.spinner("Extracting skills and generating suggestions..."):
                try:
                    jd_skills = job_parser.extract_skills_from_jd(jd_text)
                except Exception as e:
                    st.error(f"Error while extracting skills from JD: {e}")
                    return

            col_left, col_right = st.columns(2)
            with col_left:
                st.markdown("#### 🔑 Detected core skills")
                if jd_skills:
                    for s in jd_skills:
                        st.markdown(f"- **{s}**")
                else:
                    st.markdown("_No clear skills detected. Try simplifying the JD language._")

            with col_right:
                st.markdown("#### 📝 Suggested screening questions")
                top_for_questions = jd_skills[:5] if jd_skills else DOMAIN_SKILLS.get(domain, [])[:5]
                for s in top_for_questions:
                    st.markdown(
                        f"- Can you describe a recent project where you used **{s}**?\n"
                        f"- How would you rate your proficiency in **{s}** and why?"
                    )

            st.markdown("#### 📋 Ideal candidate profile (summary)")
            dom_skills = DOMAIN_SKILLS.get(domain, [])
            dom_snippet = ", ".join(dom_skills[:5]) if dom_skills else "domain-relevant skills"
            st.markdown(
                f"- Strong hands-on experience with: **{dom_snippet}**\n"
                "- Clear, outcome-focused project examples\n"
                "- Ability to collaborate across teams and communicate findings"
            )


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
        if 'Job Location' in filtered_df.columns and location:
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
                'job_type': candidate['row'].get('Job Type', 'Unknown'),
                'fit_score': fit_score,
                'text_similarity': text_similarity,
                'skill_match_ratio': candidate['skill_match_ratio'],
                'matched_skills': candidate['matched_skills'],
                'missing_skills': candidate['missing_skills']
            })
        
        # Sort by fit score and return top 5
        recommendations.sort(key=lambda x: x['fit_score'], reverse=True)
        return recommendations[:5]
    
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return []



# Sidebar Navigation
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        💼 JobFitFinder
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Navigation
    page_options = {
        "🏠 Home": "Home",
        "👤 Candidate Portal": "Candidate Portal",
        "🧑‍💼 Recruiter View": "Recruiter",
        "ℹ️ About": "About"
    }
    
    for label, page in page_options.items():
        if st.button(label, use_container_width=True, key=f"nav_{page}"):
            st.session_state.page = page
            st.rerun()
    
    st.markdown("---")


# Main Content Router
if st.session_state.page == "Home":
    render_home_page()
elif st.session_state.page == "Candidate Portal":
    render_candidate_portal()
elif st.session_state.page == "Recruiter":
    render_recruiter_view()
elif st.session_state.page == "About":
    render_about_page()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 0.9rem; padding: 1rem 0;">
    <p>💼 JobFitFinder - Intelligent Job Recommender | Powered by NLP & ML</p>
    <p>© 2024 | All Rights Reserved</p>
</div>
""", unsafe_allow_html=True)
