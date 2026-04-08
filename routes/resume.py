import shutil
import tempfile
import os
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from backend.models.response_models import ResumeUploadResponse
from backend.services.resume_service import extract_resume_details, build_vectorstore_bg

import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    logger.info(f"Received resume upload request: {file.filename}")
    
    if not file.filename.endswith('.pdf'):
        logger.warning(f"Rejected unsupported file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
        
    try:
        # Save temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        logger.info(f"Successfully saved temp file to {tmp_path}")

        logger.info("Starting skill extraction...")
        skills, roles = extract_resume_details(tmp_path)
        logger.info(f"Extraction complete. Found {len(skills)} skills and {len(roles)} roles.")
        
        # Build giant ML embeddings in the background safely without lagging UI!
        logger.info(f"Offloading FAISS vectorstore build to background tasks for {tmp_path}")
        background_tasks.add_task(build_vectorstore_bg, tmp_path)

        return ResumeUploadResponse(skills=skills, roles=roles)
    except Exception as e:
        logger.error(f"Error during resume upload processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
