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
