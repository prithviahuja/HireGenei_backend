from fastapi import APIRouter, HTTPException
from models.request_models import JobScrapeRequest
from models.response_models import JobScrapeResponse
from services.job_service import scrape_jobs_async
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/scrape", response_model=JobScrapeResponse)
async def scrape_jobs(request: JobScrapeRequest):
    logger.info(f"Received job scrape request with roles: {request.roles[:3]} and cities: {request.cities}")
    try:
        jobs = await scrape_jobs_async(
            roles=request.roles,
            cities=request.cities,
            country=request.country,
            work_types=request.work_types,
            exp_levels=request.exp_levels,
            time_filter=request.time_filter
        )
        logger.info(f"Successfully returning {len(jobs)} total jobs from scrape router.")
        return JobScrapeResponse(jobs=jobs)
    except Exception as e:
        logger.error(f"Failed job scraping at router level: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
