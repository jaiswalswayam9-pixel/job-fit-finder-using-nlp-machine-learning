"""Model utilities for embedding generation and similarity computation"""
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class ModelUtils:
    """Handle embeddings and similarity computations"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the sentence transformer model"""
        try:
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a given text"""
        if not self.model:
            return np.zeros(384)
        
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return np.zeros(384)
    
    def generate_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for multiple texts"""
        if not self.model:
            return np.zeros((len(texts), 384))
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        except Exception as e:
            print(f"Error generating batch embeddings: {e}")
            return np.zeros((len(texts), 384))
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
        except Exception as e:
            print(f"Error computing similarity: {e}")
            return 0.0
    
    def compute_similarities_batch(self, embedding1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """Compute similarity between one embedding and multiple embeddings"""
        try:
            similarities = cosine_similarity([embedding1], embeddings2)[0]
            return similarities
        except Exception as e:
            print(f"Error computing batch similarities: {e}")
            return np.zeros(len(embeddings2))
    
    def compute_skill_gap(self, candidate_skills: List[str], required_skills: List[str]) -> Tuple[List[str], List[str]]:
        """Compute skill gap between candidate and job"""
        candidate_skills_lower = [s.lower() for s in candidate_skills]
        required_skills_lower = [s.lower() for s in required_skills]
        
        matched_skills = [skill for skill in required_skills if skill.lower() in candidate_skills_lower]
        missing_skills = [skill for skill in required_skills if skill.lower() not in candidate_skills_lower]
        
        return matched_skills, missing_skills
    
    def compute_fit_score(self, 
                         text_similarity: float, 
                         skill_match_ratio: float,
                         weight_text: float = 0.6,
                         weight_skills: float = 0.4) -> float:
        """Compute overall fit score based on text similarity and skill matching"""
        fit_score = (text_similarity * weight_text + skill_match_ratio * weight_skills) * 100
        return min(100, max(0, fit_score))  # Clamp between 0 and 100
