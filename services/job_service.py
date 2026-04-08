import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import logging
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from fastapi.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

experience_level_mapping = {
    "Internship": "f_E=1",
    "Intership": "f_E=1", # Fallback for backwards compatibility with original Streamlit script
    "Entry level": "f_E=2",
    "Associate": "f_E=3",
    "Mid-senior level": "f_E=4"
}

work_type_mapping = {
    "On-site": "f_WT=1",
    "Hybrid": "f_WT=2",
    "Remote": "f_WT=3",
}

time_filter_mapping = {
    "Past 24 hours": "f_TPR=r86400",
    "Past week": "f_TPR=r604800",
    "Past month": "f_TPR=r2592000",
}

def get_skills(text):
    sentence = Sentence(text)
    flair_model.predict(sentence)
    return [entity.text for entity in sentence.get_spans("ner")]

def process_job(job, work_type, exp_level, position):
    try:
        title_element = job.find('h3', class_='base-search-card__title')
        
        company_element = job.find('h4', class_='base-search-card__subtitle')
        
        loc_element = job.find('span', class_='job-search-card__location')
        link_element = job.find('a', class_='base-card__full-link')

        if not all([title_element, company_element, loc_element, link_element]):
            return None

        title = title_element.text.strip()
        company = company_element.text.strip()
        location = loc_element.text.strip()
        link = link_element['href'].split('?')[0]

        return {
            "title": title,
            "company": company,
            "location": location,
            "link": link
        }
    except Exception as e:
        logger.error(f"Error processing job card: {str(e)}")
        return None

def scrape_jobs_sync(location, position, work_types, exp_levels, time_filter):
    results = []

    # Collapse arrays directly into supported LinkedIn comma-separated query values
    wt_val = "%2C".join([work_type_mapping[wt].split('=')[1] for wt in work_types if wt in work_type_mapping])
    exp_val = "%2C".join([experience_level_mapping[el].split('=')[1] for el in exp_levels if el in experience_level_mapping])
    
    ua = f'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{random.randint(115, 122)}.0.0.0 Safari/537.36'
    
    try:
        base_url = (
            f"https://www.linkedin.com/jobs/search/?keywords={position}&location={location}"
            f"&f_WT={wt_val}"
            f"&f_E={exp_val}"
            f"&{time_filter_mapping[time_filter]}"
            f"&radius=0"
        )
        try:
            headers = {
                'User-Agent': ua,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
                'Connection': 'keep-alive',
            }
            time.sleep(random.uniform(1.0, 2.0))
            response = requests.get(base_url, headers=headers, timeout=10)
            if response.status_code == 429:
                logger.warning("LinkedIn 429 block detected.")
                return []
            
            soup = BeautifulSoup(response.text, 'html.parser')
            count_text = soup.find('span', class_='results-context-header__job-count')
            total_jobs = int(count_text.text.replace(',', '').replace('.', '').replace('+', '').strip()) if count_text else 25
        except Exception:
            total_jobs = 25

        total_jobs = min(total_jobs, 25) # strictly limit to 1 page

        for start in range(0, total_jobs, 25):
            url = f"{base_url}&start={start}"

            try:
                if start > 0:
                    time.sleep(1.0)
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 429:
                    continue
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                jobs = soup.find_all('div', class_='base-card')
            except Exception as e:
                continue

            for job in jobs:
                job_data = process_job(job, "Mixed", "Mixed", position)
                if job_data:
                    results.append(job_data)
    except Exception as e:
        logger.error(f"Scraping error: {str(e)}")
        
    return results

import concurrent.futures

def run_scrapper_logic(cities, states, positions, work_types, exp_levels, time_filter):
    cities_list = [c.strip() for c in cities.split(',') if c.strip()]
    states_list = [s.strip() for s in states.split(',') if s.strip()]
    
    location = []
    if states_list:
        location = [f"{city},{state}" for city in cities_list for state in states_list]
    else:
        location = cities_list

    positions_list = [str(p).strip().replace(' ', '%20') for p in positions if isinstance(p, str) and p.strip()]
    
    all_jobs = []
    
    logger.info(f"Starting ThreadPoolExecutor scraping for {len(location)} locations and {len(positions_list)} roles.")

    # Run remaining lightweight configurations concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for loc in location:
            for pos in positions_list:
                logger.info(f"Dispatching thread worker for [City: {loc}] | [Role: {pos}]")
                futures.append(executor.submit(scrape_jobs_sync, loc, pos, work_types, exp_levels, time_filter))
        
        for future in concurrent.futures.as_completed(futures):
            try:
                jobs = future.result()
                all_jobs.extend(jobs)
            except Exception as e:
                logger.error(f"Thread worker failed during combination execution: {str(e)}", exc_info=True)
            
    # Remove duplicates
    unique_jobs = {job["link"]: job for job in all_jobs}.values()
    logger.info(f"Threadpool completed. Collapsed {len(all_jobs)} total parsed cards into {len(unique_jobs)} distinct jobs.")
    df = pd.DataFrame(list(unique_jobs))
    if df.empty:
        return []
    return df.to_dict(orient="records")

async def scrape_jobs_async(roles, cities, country, work_types, exp_levels, time_filter):
    # Offload the synchronous scraping logic to a threadpool to avoid blocking FastAPI
    return await run_in_threadpool(
        run_scrapper_logic,
        cities=cities,
        states=country, # map country to states
        positions=roles,
        work_types=work_types,
        exp_levels=exp_levels,
        time_filter=time_filter
    )
