from pydantic import BaseModel
from typing import List, Optional, Dict

class ResumeUploadResponse(BaseModel):
    skills: List[str]
    roles: List[str]
    score: int = 0
    session_id: Optional[str] = None
    # Per-dimension 0-100 sub-scores so the overall score is explainable.
    breakdown: Dict[str, int] = {}
    # Resume-specific improvement tips (weakest areas first).
    suggestions: List[str] = []
    # What's already working in the resume.
    strengths: List[str] = []

class JobResponse(BaseModel):
    title: str
    company: str
    location: str
    link: str
    source: str = "LinkedIn"

class JobScrapeResponse(BaseModel):
    jobs: List[JobResponse]

class ChatResponse(BaseModel):
    reply: str

class ResumeStatusResponse(BaseModel):
    # ready  -> the vectorstore background build has finished
    # exists -> the backend still holds this session at all (resume text, skills).
    #           The match/email flow only needs `exists`; sessions live in memory
    #           and are wiped on backend restart, so this tells the frontend when
    #           a cached resume analysis is no longer backed by the server.
    ready: bool
    exists: bool = False

class MatchResult(BaseModel):
    score: int
    matched_skills: List[str] = []
    missing_skills: List[str] = []
    summary: str = ""
    jd_skill_count: int = 0
    # 'good fit' | 'under-qualified' | 'over-qualified' | 'unknown'
    seniority_fit: str = "unknown"

class ContactInfo(BaseModel):
    emails: List[str] = []
    phones: List[str] = []
    source: str = "none"
    confidence: str = "none"
    site: Optional[str] = None

class EmailDraft(BaseModel):
    subject: str = ""
    body: str = ""
    to: str = ""

class JobMatchResponse(BaseModel):
    match: MatchResult
    contact: ContactInfo
    email: EmailDraft
    description_excerpt: str = ""
    has_description: bool = False
