import shutil
import tempfile
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from models.request_models import ResumeUpdateRequest, BreakdownDetailRequest
from models.response_models import ResumeUploadResponse, ResumeStatusResponse
from services.resume_service import analyze_resume, build_vectorstore_bg
from utils.vectorstore import create_session, is_ready, get_session, update_profile

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

        logger.info("Starting domain-agnostic resume analysis...")
        # LLM-first extraction (works for any field) with keyword fallback, plus a
        # content-aware readiness score. Also returns the readable full text,
        # extracted BEFORE the background task deletes the PDF — needed later by the
        # tailored-email generator.
        analysis = analyze_resume(tmp_path)
        skills = analysis["skills"]
        roles = analysis["roles"]
        resume_text = analysis["resume_text"]
        logger.info(
            f"Analysis complete. {len(skills)} skills, {len(roles)} roles, "
            f"score {analysis['score']}, domain='{analysis['domain']}'."
        )

        # One session per upload so concurrent users never clobber each other.
        session_id = str(uuid.uuid4())
        create_session(
            session_id, skills, roles, resume_text=resume_text,
            domain=analysis["domain"], seniority=analysis["seniority"],
        )

        logger.info(f"Offloading vectorstore build for session {session_id}")
        background_tasks.add_task(build_vectorstore_bg, tmp_path, session_id)

        return ResumeUploadResponse(
            skills=skills,
            roles=roles,
            score=analysis["score"],
            session_id=session_id,
            breakdown=analysis["breakdown"],
            suggestions=analysis["suggestions"],
            strengths=analysis["strengths"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during resume upload processing: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{session_id}", response_model=ResumeStatusResponse)
async def resume_status(session_id: str):
    return ResumeStatusResponse(
        ready=is_ready(session_id),
        exists=get_session(session_id) is not None,
    )


@router.post("/breakdown/detail")
async def breakdown_detail(request: BreakdownDetailRequest):
    """Drill into ONE score dimension: pinpoint the actual resume lines that hurt
    it and how to rewrite them. Used by the clickable score-breakdown rows."""
    session = get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please re-upload your resume.")
    try:
        from services.breakdown_service import explain_metric
        return explain_metric(
            session.get("resume_text", ""),
            session.get("skills", []),
            session.get("roles", []),
            request.metric,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Breakdown detail failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Couldn't generate the breakdown detail. Please try again.")


@router.post("/profile", response_model=ResumeUploadResponse)
async def update_resume_profile(request: ResumeUpdateRequest):
    """Let the user correct the detected skills/roles (the manual-override the UI
    exposes). The corrected values are stored on the session so the match score and
    cold-email generator use them too — not just the displayed chips."""
    session = get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please re-upload your resume.")

    skills = [s.strip() for s in (request.skills or []) if s and s.strip()]
    roles = None
    if request.roles is not None:
        roles = [r.strip() for r in request.roles if r and r.strip()]

    update_profile(request.session_id, skills=skills, roles=roles)

    from services.resume_service import score_resume
    final_roles = roles if roles is not None else session.get("roles", [])
    resume_text = session.get("resume_text", "")
    scored = score_resume(resume_text, skills, final_roles)
    return ResumeUploadResponse(
        skills=skills,
        roles=final_roles,
        score=scored["score"],
        session_id=request.session_id,
        breakdown=scored["breakdown"],
        suggestions=scored["suggestions"],
        strengths=scored["strengths"],
    )
