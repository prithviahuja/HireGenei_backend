from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
import os

from routes import resume, jobs, chat

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

logger.info("Starting HireGenei FastAPI Backend")

app = FastAPI(title="HireGenei API", version="1.0.0")

# CORS Configuration
# CORS: set ALLOWED_ORIGINS (comma-separated) to lock to your frontend URL.
# Unset -> falls back to "*" so existing deploys keep working.
_origins_env = os.environ.get("ALLOWED_ORIGINS", "").strip()
if _origins_env:
    _allowed = [o.strip() for o in _origins_env.split(",") if o.strip()]
else:
    _allowed = ["*"]
_allow_credentials = _allowed != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed,
    allow_credentials=_allow_credentials,
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

@app.get("/")
def root():
    return {"status": "running"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render injects $PORT
    uvicorn.run("main:app", host="0.0.0.0", port=port)