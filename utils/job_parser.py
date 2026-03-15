"""Job description parsing utilities"""
import re
from typing import Dict, List
import spacy


class JobParser:
    """Parse job descriptions and extract key information"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            print("Downloading spaCy model...")
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def extract_skills_from_jd(self, job_description: str) -> List[str]:
        """Extract skills required from job description"""
        skills_list = []
        
        common_skills = {
            "python", "java", "javascript", "c++", "c#", "ruby", "php", "swift",
            "golang", "rust", "typescript", "kotlin", "scala", "r", "matlab",
            "sql", "mysql", "postgresql", "mongodb", "cassandra", "redis",
            "react", "angular", "vue", "nodejs", "express", "django", "flask",
            "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "gitlab",
            "git", "linux", "windows", "macos", "agile", "scrum", "jira",
            "machine learning", "deep learning", "nlp", "computer vision",
            "tensorflow", "keras", "pytorch", "scikit-learn", "pandas", "numpy",
            "data analysis", "data science", "business intelligence", "tableau",
            "power bi", "excel", "rest", "graphql", "microservices", "oop",
            "api development", "web development", "mobile development",
            "html", "css", "responsive design", "ui/ux", "figma", "adobe xd",
            "communication", "leadership", "problem solving", "teamwork",
            "sales", "marketing", "finance", "accounting", "hr", "recruitment",
            "graphic design", "photoshop", "illustrator", "design", "ui", "ux",
            "cybersecurity", "cyber security", "security", "networking", "devops",
            "nosql", "oracle", "salesforce", "sap", "hadoop", "spark",
            "communication skills", "presentation skills", "project management"
        }
        
        jd_lower = job_description.lower()
        
        # First, try to split by comma and extract individual skills
        potential_skills = [s.strip() for s in jd_lower.split(',')]
        
        for potential_skill in potential_skills:
            potential_skill = potential_skill.strip()
            # Check if any of our common skills are in this potential skill
            for skill in common_skills:
                if skill in potential_skill:
                    skills_list.append(skill.title())
                    break
        
        # Also check for skills mentioned directly in the text
        for skill in common_skills:
            if skill in jd_lower and skill.title() not in skills_list:
                skills_list.append(skill.title())
        
        return list(set(skills_list)) if skills_list else ["Not Specified"]
    
    def extract_experience(self, job_description: str) -> str:
        """Extract years of experience required"""
        patterns = [
            r'(\d+)\+?\s*years?\s*of\s*experience',
            r'(\d+)\s*-\s*(\d+)\s*years?',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, job_description, re.IGNORECASE)
            if match:
                return match.group()
        
        return "Not specified"
    
    def extract_job_title(self, job_description: str) -> str:
        """Extract job title from description"""
        # Try to find common job titles
        job_titles = [
            "software engineer", "data scientist", "product manager",
            "devops engineer", "ui/ux designer", "frontend developer",
            "backend developer", "full stack developer", "qa engineer",
            "machine learning engineer", "business analyst"
        ]
        
        jd_lower = job_description.lower()
        for title in job_titles:
            if title in jd_lower:
                return title.title()
        
        return "Software Engineer"
    
    def parse_job_description(self, job_description: str) -> Dict:
        """Complete job description parsing"""
        skills = self.extract_skills_from_jd(job_description)
        experience = self.extract_experience(job_description)
        job_title = self.extract_job_title(job_description)
        
        return {
            "text": job_description,
            "required_skills": skills,
            "experience": experience,
            "job_title": job_title,
            "word_count": len(job_description.split())
        }

    def validate_job_description(self, job_description: str) -> Dict:
        """
        Validate that the job description contains the required structured parts.

        Returns a dict with:
        - is_valid: bool
        - missing: list of missing section names
        - summary: one-paragraph short summary when valid
        """
        jd = job_description or ""
        jd_lower = jd.lower()

        # Define checks for each required section
        checks = {
            "job_title": lambda t: bool(re.search(r"\b(title|job title)\b", t)) or len(t.split()) >= 2,
            "company_overview": lambda t: bool(re.search(r"\b(company|about us|about the company|company overview|about)\b", t)),
            "job_summary": lambda t: bool(re.search(r"\b(summary|overview|role is about|role summary|job summary)\b", t)),
            "roles_responsibilities": lambda t: bool(re.search(r"\b(responsibilit|responsibilities|tasks|duties|kpi|kpIs|daily|daily tasks)\b", t)),
            "required_qualifications": lambda t: bool(re.search(r"\b(requirements|qualifications|required experience|must have|education|experience)\b", t)),
            "preferred_skills": lambda t: bool(re.search(r"\b(preferred|nice to have|good to have|preferred skills|nice-to-have)\b", t)),
            "compensation": lambda t: bool(re.search(r"\b(salary|compensation|benefits|bonus|pay range)\b", t)),
            "work_environment": lambda t: bool(re.search(r"\b(location|remote|hybrid|onsite|work environment|shift|travel)\b", t)),
            "equal_opportunity": lambda t: bool(re.search(r"\b(equal opportunity|equal-opportunity|we are an equal|inclusive)", t)),
            "application_process": lambda t: bool(re.search(r"\b(apply|application process|how to apply|selection|interview rounds|apply online)\b", t)),
        }

        # Required sections according to user: mandate most sections but make compensation optional
        required_for_validation = [
            "job_title",
            "company_overview",
            "job_summary",
            "roles_responsibilities",
            "required_qualifications",
            "preferred_skills",
            "work_environment",
            "equal_opportunity",
            "application_process",
        ]

        missing = []
        for key in required_for_validation:
            check_fn = checks.get(key)
            if check_fn and not check_fn(jd_lower):
                missing.append(key)

        is_valid = len(missing) == 0

        # Generate a short summary (one paragraph) if valid
        summary = ""
        if is_valid:
            # Try to synthesize a one-paragraph summary using first few sentences.
            # Note: use a real whitespace split pattern here (no escaped backslash).
            sentences = re.split(r"(?<=[.!?])\s+", jd.strip())
            summary_sentences = []
            for s in sentences:
                if len(summary_sentences) >= 3:
                    break
                if len(s.split()) > 4:
                    summary_sentences.append(s.strip())
            summary = " ".join(summary_sentences).strip()
            if not summary:
                summary = jd.strip()[:400]

        return {
            "is_valid": is_valid,
            "missing": missing,
            "summary": summary
        }
