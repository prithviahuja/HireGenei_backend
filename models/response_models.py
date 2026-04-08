from pydantic import BaseModel
from typing import List

class ResumeUploadResponse(BaseModel):
    skills: List[str]
    roles: List[str]

class JobResponse(BaseModel):
    title: str
    company: str
    location: str
    link: str

class JobScrapeResponse(BaseModel):
    jobs: List[JobResponse]

class ChatResponse(BaseModel):
    reply: str
