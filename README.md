# 🎯 JobFitFinder — AI Resume to Job Matcher

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://job-fit-finder.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![NLP](https://img.shields.io/badge/NLP-spaCy-green.svg)](https://spacy.io)
[![ML](https://img.shields.io/badge/ML-Scikit--learn-orange.svg)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

> Upload your ATS resume once. Our AI matches your skills and 
> experience with the best-fit jobs — with real fit scores, 
> skill gap analysis, and job descriptions. Instantly.

---


---

## ✨ Features
- 📄 **ATS Resume Parsing** — Supports PDF and DOCX formats
- 🧠 **NLP Skill Extraction** — spaCy extracts skills semantically
- 📊 **ML Fit Scoring** — Scikit-learn scores every job 0–100%
- 🗺️ **Skill Gap Analysis** — See matched vs missing skills per job
- 💼 **Top 5 Job Recommendations** — Ranked by AI fit score
- 📋 **Job Description Viewer** — Full JD with required experience
- 🎨 **Glassmorphism Dark UI** — Beautiful modern Streamlit interface
- ⚡ **3 second analysis** — Fast end-to-end pipeline

---

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| 🎨 Frontend | Streamlit + Custom CSS Glassmorphism |
| 🔤 NLP | spaCy, HuggingFace Transformers |
| 🤖 ML Model | Scikit-learn Gradient Boosting |
| 📄 Resume Parser | PyMuPDF, python-docx |
| 📊 Analysis | Pandas, NumPy |
| 🐍 Language | Python 3.10+ |

---

## 📁 Project Structure

```text
FitFinder/
├── data/
│   └── jobs.csv                # Dataset of jobs 
├── utils/
│   ├── db_utils.py             # Database utility
│   ├── job_parser.py           # Job description text parser
│   ├── model_utils.py          # Machine learning embeddings and scoring logic
│   └── resume_parser.py        # ATS resume parsing
├── analysis.py                 # ML data exploration & pipeline testing
├── app.py                      # Old backend UI processing
├── jobfitfinder_app.py         # 🚀 Main entry point - Streamlit Integrated dark-mode UI
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git tracking exclusions
└── README.md                   # Project documentation
```

---

## 🚀 How to Run Locally

1. **Clone the repository:**
```bash
git clone https://github.com/jaiswalswayam9-pixel/job-fit-finder-using-nlp-machine-learning.git
cd job-fit-finder-using-nlp-machine-learning
```

2. **Create a virtual environment and install dependencies:**
```bash
python -m venv .venv

# On Windows:
.venv\Scripts\activate

# On Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

3. **Run the Streamlit application:**
```bash
streamlit run jobfitfinder_app.py
```
