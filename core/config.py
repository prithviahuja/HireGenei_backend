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

    class Config:
        env_file = os.path.join(os.path.dirname(__file__), "..", ".env")
        extra = "allow"

settings = Settings()

if not settings.HF_TOKEN or not settings.GROQ_API_KEY:
    logging.warning("HF_TOKEN and/or GROQ_API_KEY are not set in backend environment. Make sure they are provided in backend/.env or through OS environment variables.")
