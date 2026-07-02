import os
import logging
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Load .env if present in the backend working directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

class Settings(BaseSettings):
    HF_TOKEN: str | None = None
    GROQ_API_KEY: str | None = None

    # RapidAPI key for the JSearch job aggregator (Google for Jobs → Naukri,
    # Indeed, LinkedIn, foundit, …). Optional: when unset, the JSearch source is
    # simply inactive and only LinkedIn scraping runs. Get a free key at
    # https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch
    JSEARCH_API_KEY: str | None = None
    JSEARCH_API_HOST: str = "jsearch.p.rapidapi.com"
    # JSearch v5 serves search results from /search-v2 (the old /search 404s).
    # Kept configurable so a future endpoint rename needs no code change.
    JSEARCH_SEARCH_PATH: str = "/search-v2"

    # Groq model used for domain-agnostic resume extraction (skills/roles for any
    # field, not just tech). Falls back to the keyword matcher if the LLM fails.
    GROQ_EXTRACT_MODEL: str = "llama-3.3-70b-versatile"

    # Job-match scoring. When MATCH_USE_LLM is true, an LLM reads the resume + JD
    # for a domain-agnostic, seniority-aware score (one extra fast Groq call per
    # job opened); set false to use the deterministic matcher only (no LLM cost).
    MATCH_USE_LLM: bool = True
    GROQ_MATCH_MODEL: str = "llama-3.1-8b-instant"

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
        extra = "allow"

settings = Settings()

if not settings.HF_TOKEN or not settings.GROQ_API_KEY:
    logging.warning("HF_TOKEN and/or GROQ_API_KEY are not set in backend environment. Make sure they are provided in backend/.env or through OS environment variables.")
