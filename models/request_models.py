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
