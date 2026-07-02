"""Second job source: JSearch (RapidAPI / Google for Jobs).

LinkedIn scraping (job_service.py) stays as the primary source. This module adds
aggregated listings from the popular Indian job sites that Google for Jobs indexes
— Naukri, Indeed, foundit (Monster India), LinkedIn, etc. — via the keyless-to-us
JSearch API. We only ever hit a clean JSON API here (no HTML scraping, no ban
risk).

Activation is opt-in: if JSEARCH_API_KEY is unset, every function here no-ops and
returns [] so the app keeps working LinkedIn-only. Calls are kept frugal (one
request per role, capped) to respect RapidAPI's free-tier rate limits.

Normalized output matches the LinkedIn scraper's shape so the rest of the pipeline
is source-agnostic:
    {"title", "company", "location", "link", "source"}
"""

import time
import logging

import requests

from core.config import settings

logger = logging.getLogger(__name__)

# Keep free-tier-friendly: at most this many roles trigger a JSearch call per scrape.
MAX_JSEARCH_ROLES = 3

# Map the UI's "Posted" filter to JSearch's date_posted enum.
_DATE_POSTED = {
    "Past 24 hours": "today",
    "Past week": "week",
    "Past month": "month",
}


def is_enabled() -> bool:
    """True only when a RapidAPI key is present — otherwise the source is inactive."""
    return bool(settings.JSEARCH_API_KEY)


def _primary_location(cities: str, country: str) -> str:
    """JSearch takes a single free-text location; use the first city if given,
    else fall back to the country (we're India-focused, so country defaults there)."""
    first_city = next((c.strip() for c in (cities or "").split(",") if c.strip()), "")
    return first_city or (country or "India")


def _normalize(item: dict) -> dict | None:
    title = (item.get("job_title") or "").strip()
    company = (item.get("employer_name") or "").strip()
    link = (item.get("job_apply_link") or item.get("job_google_link") or "").strip()
    if not title or not link:
        return None

    loc_parts = [item.get("job_city"), item.get("job_state"), item.get("job_country")]
    location = ", ".join(p for p in loc_parts if p) or ("Remote" if item.get("job_is_remote") else "")

    # job_publisher tells us which real board this came from (Naukri / Indeed / …).
    source = (item.get("job_publisher") or "Aggregator").strip()

    return {
        "title": title,
        "company": company or "—",
        "location": location,
        "link": link,
        "source": source,
    }


def _search_one(query: str, country: str, date_posted: str, remote_only: bool) -> list[dict]:
    params = {
        "query": query,
        "page": "1",
        "num_pages": "1",
        # JSearch expects a 2-letter country code; we are India-focused.
        "country": "in",
        "date_posted": date_posted,
    }
    if remote_only:
        params["work_from_home"] = "true"

    headers = {
        "X-RapidAPI-Key": settings.JSEARCH_API_KEY,
        "X-RapidAPI-Host": settings.JSEARCH_API_HOST,
    }
    # JSearch v5 serves results from /search-v2 (the old /search 404s).
    url = f"https://{settings.JSEARCH_API_HOST}{settings.JSEARCH_SEARCH_PATH}"

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=15)
    except Exception as e:
        logger.warning(f"JSearch request error for '{query}': {str(e)}")
        return []

    if resp.status_code == 429:
        logger.warning("JSearch rate-limited (429) — free-tier quota likely exhausted.")
        return []
    if resp.status_code == 403:
        logger.warning("JSearch 403 — key not subscribed, or BASIC rate limit ('Too many requests'). Check the RapidAPI key/subscription.")
        return []
    if resp.status_code == 404:
        logger.warning(f"JSearch 404 — endpoint '{settings.JSEARCH_SEARCH_PATH}' not found; the API version may have changed.")
        return []
    if resp.status_code != 200:
        logger.warning(f"JSearch HTTP {resp.status_code} for '{query}'.")
        return []

    try:
        payload = resp.json().get("data")
        # v2 nests jobs under data.jobs; the older /search returned a bare list.
        if isinstance(payload, dict):
            data = payload.get("jobs") or []
        else:
            data = payload or []
    except Exception as e:
        logger.warning(f"JSearch JSON parse failed: {str(e)}")
        return []

    out = []
    for item in data:
        job = _normalize(item)
        if job:
            out.append(job)
    logger.info(f"JSearch '{query}' → {len(out)} jobs.")
    return out


def search_jobs(roles, cities, country, work_types, time_filter) -> list[dict]:
    """Synchronous (called inside a threadpool like the LinkedIn scraper). Returns a
    de-duped list of normalized jobs across the (capped) roles. Returns [] when the
    source is disabled or the key is missing."""
    if not is_enabled():
        logger.info("JSearch disabled (no JSEARCH_API_KEY) — skipping aggregator source.")
        return []

    positions = [str(p).strip() for p in (roles or []) if isinstance(p, str) and p.strip()]
    if not positions:
        return []

    if len(positions) > MAX_JSEARCH_ROLES:
        logger.info(
            f"JSearch: capping {len(positions)} roles to {MAX_JSEARCH_ROLES} to stay within free-tier limits."
        )
        positions = positions[:MAX_JSEARCH_ROLES]

    location = _primary_location(cities, country)
    date_posted = _DATE_POSTED.get(time_filter, "month")
    # Only force remote when the user picked remote exclusively.
    wt = [w for w in (work_types or [])]
    remote_only = wt == ["Remote"]

    results = []
    for i, role in enumerate(positions):
        # Space out calls — the JSearch free/BASIC plan rate-limits bursts
        # ("Too many requests"), so don't fire role queries back-to-back.
        if i > 0:
            time.sleep(1.2)
        query = f"{role} in {location}".strip()
        results.extend(_search_one(query, country, date_posted, remote_only))

    # De-dupe by link.
    unique = {j["link"]: j for j in results}
    return list(unique.values())
