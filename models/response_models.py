from pydantic import BaseModel
from typing import List, Optional

class ResumeUploadResponse(BaseModel):
    skills: List[str]
    roles: List[str]
    score: int = 0
    session_id: Optional[str] = None

class JobResponse(BaseModel):
    title: str
    company: str
    location: str
    link: str

class JobScrapeResponse(BaseModel):
    jobs: List[JobResponse]

class ChatResponse(BaseModel):
    reply: str

class ResumeStatusResponse(BaseModel):
    ready: bool

class MatchResult(BaseModel):
    score: int
    matched_skills: List[str] = []
    missing_skills: List[str] = []
    summary: str = ""
    jd_skill_count: int = 0

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
