"""Domain-agnostic resume extraction via the Groq LLM.

The legacy extractor (resume_service.SKILLS_LIST / JOB_ROLES) only understands a
fixed tech taxonomy, so a nurse, accountant, marketer, teacher or designer gets
zero skills and zero roles — which then breaks job search and matching.

This module asks the LLM to read the resume text and return structured skills +
target job titles for ANY field. It is strictly grounded: "only extract what is
literally in the resume; never invent." If the LLM is unavailable or returns junk,
the caller falls back to the keyword matcher, so behaviour never regresses.

Returns a dict on success, or None so the caller can fall back:
    {"skills": [...], "roles": [...], "domain": str, "seniority": str}
"""

import re
import json
import logging

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise resume parser. You read a candidate's resume and \
extract a structured profile that works for ANY profession (tech, healthcare, finance, \
law, marketing, design, education, skilled trades, etc.) — never assume software.

Rules:
1. GROUNDING: Only extract things that are actually supported by the resume text. \
NEVER invent skills, titles, tools, or domains that are not present or clearly implied.
2. skills: 8-25 concrete, specific skills/tools/competencies the candidate demonstrates \
(e.g. "Patient triage", "Financial modeling", "Adobe Illustrator", "React", "Tax filing"). \
Prefer specific over generic; avoid vague soft-skill filler unless the resume emphasizes it.
3. roles: 3-6 realistic job TITLES this candidate should search and apply for, phrased \
the way they appear on job boards (e.g. "Staff Nurse", "Financial Analyst", "Graphic \
Designer", "Backend Developer"). These are search queries, so keep them standard and concise.
4. domain: one short label for the candidate's primary field (e.g. "Healthcare", \
"Finance", "Software", "Marketing").
5. seniority: one of "intern", "entry", "mid", "senior", "lead", or "unknown".

OUTPUT: Return ONLY minified JSON, no markdown, no commentary, exactly:
{"skills":["..."],"roles":["..."],"domain":"...","seniority":"..."}"""


def _get_llm(model_name: str):
    from langchain_groq import ChatGroq
    from core.config import settings

    if not settings.GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY is not configured for the backend.")
    return ChatGroq(api_key=settings.GROQ_API_KEY, model_name=model_name, temperature=0.1)


def _parse_json(raw: str) -> dict | None:
    if not raw:
        return None
    raw = re.sub(r'^```(?:json)?|```$', '', raw.strip(), flags=re.MULTILINE).strip()
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    candidate = m.group(0) if m else raw
    try:
        obj = json.loads(candidate)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    return obj


def _clean_list(values, limit: int) -> list[str]:
    """Dedupe (case-insensitively), trim, drop empties, cap length."""
    out, seen = [], set()
    for v in values or []:
        if not isinstance(v, str):
            continue
        s = v.strip().strip(".,;")
        key = s.lower()
        if not s or key in seen:
            continue
        seen.add(key)
        out.append(s)
        if len(out) >= limit:
            break
    return out


def extract_profile_llm(resume_text: str) -> dict | None:
    """Extract a domain-agnostic profile from resume text, or None to fall back."""
    if not resume_text or len(resume_text.strip()) < 40:
        logger.info("LLM extraction skipped: resume text too short/empty.")
        return None

    from core.config import settings

    user_msg = (
        "Extract the structured profile from this resume.\n\nRESUME:\n"
        + resume_text[:8000]
        + "\n\nReturn JSON only."
    )
    messages = [("system", SYSTEM_PROMPT), ("human", user_msg)]

    for model_name in (settings.GROQ_EXTRACT_MODEL, "llama-3.1-8b-instant"):
        try:
            llm = _get_llm(model_name)
            resp = llm.invoke(messages)
            raw = getattr(resp, "content", str(resp))
            obj = _parse_json(raw)
            if not obj:
                logger.warning(f"LLM extraction returned unparseable output on '{model_name}'.")
                continue

            skills = _clean_list(obj.get("skills"), 25)
            roles = _clean_list(obj.get("roles"), 6)
            if not skills and not roles:
                logger.warning(f"LLM extraction empty on '{model_name}'.")
                continue

            domain = (obj.get("domain") or "").strip() if isinstance(obj.get("domain"), str) else ""
            seniority = (obj.get("seniority") or "").strip().lower() if isinstance(obj.get("seniority"), str) else ""
            logger.info(
                f"LLM extraction ok via '{model_name}': {len(skills)} skills, {len(roles)} roles, domain='{domain}'."
            )
            return {"skills": skills, "roles": roles, "domain": domain, "seniority": seniority}
        except Exception as e:
            logger.warning(f"LLM extraction failed on '{model_name}': {str(e)}")
            continue

    logger.info("LLM extraction unavailable — caller will fall back to keyword matcher.")
    return None
