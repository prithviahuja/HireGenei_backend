import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.concurrency import run_in_threadpool
from models.request_models import JobScrapeRequest, JobMatchRequest
from models.response_models import (
    JobScrapeResponse, JobMatchResponse, MatchResult, ContactInfo, EmailDraft,
)
from services.job_service import scrape_jobs_async, scrape_jobs_stream
from services.job_detail_service import fetch_job_description
from services.match_service import compute_match
from services.contact_service import find_contacts
from services.email_service import generate_cold_email
from utils.vectorstore import get_session, get_resume_text, get_skills
import logging

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/scrape", response_model=JobScrapeResponse)
async def scrape_jobs(request: JobScrapeRequest):
    logger.info(f"Scrape request roles={request.roles[:3]} cities={request.cities}")
    try:
        jobs = await scrape_jobs_async(
            roles=request.roles, cities=request.cities, country=request.country,
            work_types=request.work_types, exp_levels=request.exp_levels, time_filter=request.time_filter,
        )
        return JobScrapeResponse(jobs=jobs)
    except Exception as e:
        logger.error(f"Failed job scraping: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/scrape/stream")
async def scrape_jobs_stream_route(request: JobScrapeRequest):
    """Streaming (NDJSON): emits each job the moment it's found."""
    logger.info(f"STREAM scrape roles={request.roles[:3]} cities={request.cities}")

    async def generate():
        count = 0
        try:
            async for job in scrape_jobs_stream(
                roles=request.roles, cities=request.cities, country=request.country,
                work_types=request.work_types, exp_levels=request.exp_levels, time_filter=request.time_filter,
            ):
                count += 1
                yield json.dumps({"type": "job", "job": job}) + "\n"
            if count == 0:
                yield json.dumps({"type": "warning", "detail": "No jobs found. Try widening your filters or try again in a moment — LinkedIn may be rate-limiting."}) + "\n"
            yield json.dumps({"type": "done", "total": count}) + "\n"
        except Exception as e:
            logger.error(f"Streaming scrape failed: {str(e)}", exc_info=True)
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _run_match_pipeline(request: JobMatchRequest):
    """Shared orchestration for the on-click flow: fetch JD -> score ->
    find contacts -> draft email. Yields (stage_type, payload) tuples so both
    the streaming and non-streaming endpoints can reuse it."""
    session = get_session(request.session_id)
    if not session:
        raise ValueError("Upload your resume first so we can score the match and draft an email.")

    resume_text = get_resume_text(request.session_id)
    resume_skills = get_skills(request.session_id)
    if not resume_text:
        raise ValueError("We couldn't read your resume text for this session. Please re-upload your resume.")

    # 1) Job description
    jd_text = await run_in_threadpool(fetch_job_description, request.link)
    yield "jd", {
        "has_description": bool(jd_text),
        "description_excerpt": (jd_text[:800] if jd_text else ""),
    }

    # 2) Match score + breakdown
    match = await run_in_threadpool(compute_match, resume_text, resume_skills, jd_text)
    yield "score", match

    # 3) Company contacts (JD regex first, then keyless web search)
    contact = await run_in_threadpool(find_contacts, jd_text, request.company)
    yield "contact", contact

    # 4) Tailored cold email
    job_payload = {
        "title": request.title,
        "company": request.company,
        "location": request.location,
        "link": request.link,
        "description": jd_text,
        "_contacts": contact,
    }
    email = await run_in_threadpool(generate_cold_email, resume_text, resume_skills, job_payload)
    yield "email", email


@router.post("/match/stream")
async def match_job_stream(request: JobMatchRequest):
    """Streaming (NDJSON): emits each stage (jd, score, contact, email) as it
    completes so the UI fills progressively."""
    logger.info(f"MATCH stream session={request.session_id} company={request.company}")

    async def generate():
        try:
            async for stage, payload in _run_match_pipeline(request):
                yield json.dumps({"type": stage, **payload}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        except ValueError as e:
            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"
        except Exception as e:
            logger.error(f"Match pipeline failed: {str(e)}", exc_info=True)
            yield json.dumps({"type": "error", "detail": "Something went wrong while analyzing this job."}) + "\n"

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/match", response_model=JobMatchResponse)
async def match_job(request: JobMatchRequest):
    """Non-streaming fallback: runs the full pipeline and returns one object."""
    logger.info(f"MATCH (non-stream) session={request.session_id} company={request.company}")
    try:
        result = {"description_excerpt": "", "has_description": False}
        match = contact = email = None
        async for stage, payload in _run_match_pipeline(request):
            if stage == "jd":
                result["has_description"] = payload.get("has_description", False)
                result["description_excerpt"] = payload.get("description_excerpt", "")
            elif stage == "score":
                match = payload
            elif stage == "contact":
                contact = payload
            elif stage == "email":
                email = payload
        return JobMatchResponse(
            match=MatchResult(**(match or {"score": 0})),
            contact=ContactInfo(**(contact or {})),
            email=EmailDraft(**(email or {})),
            description_excerpt=result["description_excerpt"],
            has_description=result["has_description"],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Match pipeline failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
