import shutil
import tempfile
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from models.response_models import ResumeUploadResponse, ResumeStatusResponse
from services.resume_service import extract_resume_details, build_vectorstore_bg
from utils.vectorstore import create_session, is_ready

import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    logger.info(f"Received resume upload request: {file.filename}")

    if not file.filename or not file.filename.lower().endswith('.pdf'):
        logger.warning(f"Rejected unsupported file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        logger.info(f"Successfully saved temp file to {tmp_path}")

        logger.info("Starting skill extraction...")
        skills, roles, score = extract_resume_details(tmp_path)
        logger.info(f"Extraction complete. {len(skills)} skills, {len(roles)} roles, score {score}.")

        # One session per upload so concurrent users never clobber each other.
        session_id = str(uuid.uuid4())
        create_session(session_id, skills, roles)

        logger.info(f"Offloading vectorstore build for session {session_id}")
        background_tasks.add_task(build_vectorstore_bg, tmp_path, session_id)

        return ResumeUploadResponse(skills=skills, roles=roles, score=score, session_id=session_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during resume upload processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{session_id}", response_model=ResumeStatusResponse)
async def resume_status(session_id: str):
    return ResumeStatusResponse(ready=is_ready(session_id))
