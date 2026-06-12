# from fastapi import FastAPI
# from fastapi.middleware.cors import CORSMiddleware
# import logging

# from routes import resume, jobs, chat

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# )
# logger = logging.getLogger(__name__)

# logger.info("Starting HireGenei FastAPI Backend")

# app = FastAPI(title="HireGenei API", version="1.0.0")

# # CORS Configuration
# # allowing all origins for dev
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"], 
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # API Routers
# app.include_router(resume.router, prefix="/api/resume", tags=["Resume"])
# app.include_router(jobs.router, prefix="/api/jobs", tags=["Jobs"])
# app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])

# @app.get("/health")
# def health_check():
#     return {"status": "ok"}

# @app.get("/")
# def root():
#     return {"status": "running"}

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

# CORS Configuration (#2)
# Lock to your frontend by setting ALLOWED_ORIGINS (comma-separated) in the
# environment, e.g. ALLOWED_ORIGINS="https://your-site.vercel.app".
# If unset, we fall back to "*" so existing deploys keep working — but we warn.
_origins_env = os.environ.get("ALLOWED_ORIGINS", "").strip()
if _origins_env:
    allowed_origins = [o.strip() for o in _origins_env.split(",") if o.strip()]
    logger.info(f"CORS locked to: {allowed_origins}")
else:
    allowed_origins = ["*"]
    logger.warning("ALLOWED_ORIGINS not set; CORS is open to all origins. "
                   "Set ALLOWED_ORIGINS to your frontend URL to lock this down.")

# Credentials cannot be combined with a wildcard origin per the CORS spec.
_allow_credentials = allowed_origins != ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
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