import json
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from models.request_models import JobScrapeRequest
from models.response_models import JobScrapeResponse
from services.job_service import scrape_jobs_async, scrape_jobs_stream
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
