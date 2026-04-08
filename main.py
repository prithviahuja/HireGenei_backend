from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from routes import resume, jobs, chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Starting HireGenei FastAPI Backend")

app = FastAPI(title="HireGenei API", version="1.0.0")

# CORS Configuration
# allowing all origins for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routers
app.include_router(resume.router, prefix="/api/resume", tags=["Resume"])
app.include_router(jobs.router, prefix="/api/jobs", tags=["Jobs"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])

@app.get("/health")
def health_check():
    return {"status": "ok"}
