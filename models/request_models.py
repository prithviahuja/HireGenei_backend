from pydantic import BaseModel
from typing import List, Optional

class JobScrapeRequest(BaseModel):
    roles: List[str]
    cities: str
    country: str
    work_types: List[str]
    exp_levels: List[str]
    time_filter: str

class ChatMessageRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class JobMatchRequest(BaseModel):
    session_id: str
    title: str
    company: str
    location: Optional[str] = ""
    link: str
