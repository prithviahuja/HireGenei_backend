from pydantic import BaseModel
from typing import List, Optional

class JobScrapeRequest(BaseModel):
    roles: List[str]
    cities: str
    country: str
    work_types: List[str]
    exp_levels: List[str]
    time_filter: str
    # Which engines to run. None => all available. Values: "linkedin", "jsearch".
    sources: Optional[List[str]] = None


class ResumeUpdateRequest(BaseModel):
    session_id: str
    skills: List[str]
    roles: Optional[List[str]] = None

class BreakdownDetailRequest(BaseModel):
    session_id: str
    # One of the six score dimensions, e.g. "Action verbs", "Impact & metrics".
    metric: str

class ChatMessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class JobMatchRequest(BaseModel):
    session_id: str
    title: str
    company: str
    location: Optional[str] = ""
    link: str
    # Optional user-supplied base email. When provided, the cold-email generator
    # personalizes this draft for the job instead of writing one from scratch.
    email_template: Optional[str] = ""
