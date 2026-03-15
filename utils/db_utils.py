"""Database utilities for FitFinder.
Provides MongoDB helpers for realtime job postings.
"""
import os
from typing import Dict, List, Any
from datetime import datetime

try:
    from pymongo import MongoClient
except Exception:
    MongoClient = None


MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017')
DB_NAME = os.environ.get('FITFINDER_DB', 'fitfinder')


def get_mongo_client() -> Any:
    """Return a MongoClient or None if pymongo not installed."""
    if MongoClient is None:
        return None
    try:
        client = MongoClient(MONGO_URI)
        # quick ping
        client.admin.command('ping')
        return client
    except Exception:
        return None


def save_job_posting(job_doc: Dict) -> Any:
    """Save a job posting document to MongoDB and return the inserted id.

    job_doc will be augmented with `created_at`.
    """
    client = get_mongo_client()
    if client is None:
        return None
    db = client[DB_NAME]
    jobs = db.jobs
    job_doc = job_doc.copy()
    job_doc.setdefault('created_at', datetime.utcnow())
    res = jobs.insert_one(job_doc)
    return res.inserted_id


def fetch_recent_jobs(limit: int = 10) -> List[Dict]:
    """Fetch recent job postings from MongoDB (most recent first)."""
    client = get_mongo_client()
    if client is None:
        return []
    db = client[DB_NAME]
    jobs = db.jobs
    docs = list(jobs.find().sort('created_at', -1).limit(limit))
    for d in docs:
        # stringify ObjectId and datetimes for Streamlit display
        d['_id'] = str(d.get('_id'))
        if isinstance(d.get('created_at'), datetime):
            d['created_at'] = d['created_at'].strftime('%Y-%m-%d %H:%M:%S')
    return docs
