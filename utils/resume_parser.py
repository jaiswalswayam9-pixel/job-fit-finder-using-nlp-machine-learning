"""Resume parsing and text extraction utilities"""
import PyPDF2
from docx import Document
import re
import spacy
from typing import Dict, List, Tuple


class ResumeParser:
    """Parse resume and extract key information"""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
    
    def validate_ats_resume(self, text: str) -> Tuple[bool, str]:
        """
        Validate if the resume is ATS-compatible (Application Tracking System format)
        
        ATS resumes should contain:
        - Contact information (email, phone)
        - Clear section headers (Experience, Education, Skills, etc.)
        - Structured format with identifiable sections
        - Professional content (not scanned images, graphics-heavy, etc.)
        """
        # Convert to lowercase for checking
        text_lower = text.lower()
        
        # Check for minimum content length (ATS resumes are typically 300+ words)
        word_count = len(text.split())
        if word_count < 200:
            return False, "❌ Resume is too short. Please upload a complete ATS-format resume with at least 200 words."
        
        # Check for contact information (email or phone)
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        phone_pattern = r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b'
        
        has_email = bool(re.search(email_pattern, text))
        has_phone = bool(re.search(phone_pattern, text))
        
        if not (has_email or has_phone):
            return False, "❌ No contact information found. ATS resumes must include an email address or phone number."
        
        # Check for standard resume sections
        section_keywords = [
            'experience', 'employment', 'work history',
            'education', 'skills', 'qualifications',
            'summary', 'objective', 'profile'
        ]
        
        found_sections = sum(1 for keyword in section_keywords if keyword in text_lower)
        
        if found_sections < 2:
            return False, "❌ Missing standard resume sections. ATS resumes should include sections like Experience, Education, Skills, etc."
        
        # Check for common resume keywords to ensure it's actually a resume
        resume_keywords = [
            'experience', 'education', 'skills', 'employment',
            'work', 'project', 'achievement', 'responsibility',
            'developed', 'managed', 'implemented', 'designed'
        ]
        
        found_keywords = sum(1 for keyword in resume_keywords if keyword in text_lower)
        
        if found_keywords < 3:
            return False, "❌ This doesn't appear to be a valid resume. Please upload an ATS-compatible professional resume."
        
        # Check for mostly readable text (not image-heavy or corrupted)
        # Calculate the ratio of special characters - high ratio indicates corrupted/image text
        special_chars = sum(1 for char in text if ord(char) > 127 or char in '©®™§¶')
        special_char_ratio = special_chars / max(len(text), 1)
        
        if special_char_ratio > 0.3:  # More than 30% special characters suggests image-based PDF
            return False, "❌ This appears to be a scanned/image-based PDF. Please upload a text-based ATS resume."
        
        # If all checks pass
        return True, "✅ Valid ATS resume detected"
    
    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        text = ""
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error reading DOCX: {e}")
        return text
    
    def extract_text(self, file_path: str, file_type: str) -> str:
        """Extract text based on file type"""
        if file_type.lower() == "pdf":
            return self.extract_text_from_pdf(file_path)
        elif file_type.lower() in ["docx", "doc"]:
            return self.extract_text_from_docx(file_path)
        elif file_type.lower() == "txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        return ""
    
    def extract_skills(self, text: str) -> List[str]:
        """Extract skills from resume text using NLP"""
        skills_list = []
        
        # Common skills dictionary - expanded
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
            "communication skills", "presentation skills", "project management",
            "spring", "hibernate", "junit", "maven", "gradle", "junit",
            "jquery", "bootstrap", "webpack", "npm", "yarn", "gulp",
            "junit", "mockito", "selenium", "postman", "insomnia",
            "swagger", "jira", "confluence", "bitbucket", "github", "gitlab"
        }
        
        text_lower = text.lower()
        doc = self.nlp(text_lower)
        
        # Extract skills using token matching
        for skill in common_skills:
            if skill in text_lower:
                skills_list.append(skill.title())
        
        # If no skills found, try to extract from common patterns
        if not skills_list:
            skills_list.append("General Skills")
        
        return list(set(skills_list))  # Remove duplicates
    
    def extract_contact_info(self, text: str) -> Dict[str, str]:
        """Extract contact information from resume"""
        contact = {}
        
        # Email regex
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            contact['email'] = email_match.group()
        
        # Phone regex
        phone_match = re.search(r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b', text)
        if phone_match:
            contact['phone'] = phone_match.group()
        
        return contact
    
    def parse_resume(self, file_path: str, file_type: str) -> Dict:
        """Complete resume parsing"""
        text = self.extract_text(file_path, file_type)
        skills = self.extract_skills(text)
        contact = self.extract_contact_info(text)
        
        return {
            "text": text,
            "skills": skills,
            "contact": contact,
            "word_count": len(text.split())
        }
