"""Fetch the full job description for a single scraped LinkedIn job.

The job-search results page only yields title/company/location/link. The full
description lives on the individual job page, so we fetch it on demand (when the
user opens a job) rather than during the bulk scrape — keeps scraping fast and
avoids extra LinkedIn requests per card.
"""

import re
import time
import random
import logging

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


def _ua() -> str:
    return (
        f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        f'(KHTML, like Gecko) Chrome/{random.randint(115, 122)}.0.0.0 Safari/537.36'
    )


def _headers() -> dict:
    return {
        'User-Agent': _ua(),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Connection': 'keep-alive',
    }


def extract_job_id(link: str) -> str | None:
    """LinkedIn job links look like
    https://www.linkedin.com/jobs/view/some-title-at-company-3812345678
    The trailing 10-ish digit number is the job id."""
    if not link:
        return None
    m = re.search(r'/jobs/view/(?:[^/?]*?-)?(\d{6,})', link)
    if m:
        return m.group(1)
    m = re.search(r'(\d{8,})', link)
    return m.group(1) if m else None


def _parse_description(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    node = (
        soup.find('div', class_='show-more-less-html__markup')
        or soup.find('div', class_='description__text')
        or soup.find('section', class_='description')
    )
    if node:
        text = node.get_text(separator='\n')
    else:
        # Last resort: whole body text (the guest endpoint returns a JD fragment).
        text = soup.get_text(separator='\n')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def fetch_job_description(link: str) -> str:
    """Return the plain-text job description for a LinkedIn job link, or ''."""
    job_id = extract_job_id(link)
    candidates = []
    if job_id:
        # Guest endpoint returns a clean JD fragment without auth — most reliable.
        candidates.append(
            f"https://www.linkedin.com/jobs-guest/jobs/api/jobPosting/{job_id}"
        )
    if link:
        candidates.append(link)

    for url in candidates:
        try:
            time.sleep(random.uniform(0.4, 1.0))
            resp = requests.get(url, headers=_headers(), timeout=12)
            if resp.status_code == 429:
                logger.warning("LinkedIn 429 while fetching JD.")
                continue
            if resp.status_code != 200 or not resp.text:
                continue
            desc = _parse_description(resp.text)
            if desc and len(desc) > 60:
                logger.info(f"Fetched JD ({len(desc)} chars) from {url[:60]}...")
                return desc
        except Exception as e:
            logger.warning(f"JD fetch failed for {url[:60]}: {str(e)}")
            continue

    logger.info("No job description could be fetched.")
    return ""
