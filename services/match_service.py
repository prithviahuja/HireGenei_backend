"""Score how well a resume matches a specific job description.

Two paths, LLM-first with a deterministic fallback (same pattern as resume
extraction):

  • LLM match (MATCH_USE_LLM, default on): an LLM reads the resume + JD and scores
    fit for ANY field — weighing required vs. nice-to-have requirements and
    seniority/experience fit, not just keyword overlap. One fast Groq call per job.

  • Deterministic fallback: domain-agnostic skill overlap (any resume skill phrase
    that appears in the JD counts, so it works beyond the tech taxonomy) blended
    with resume↔JD embedding similarity, then nudged by a regex-based seniority-fit
    check. Used whenever the LLM is disabled or unavailable, so behaviour never
    regresses.

Both return the same shape:
    {score, matched_skills, missing_skills, summary, jd_skill_count, seniority_fit}
"""

import re
import json
import logging

from services.resume_service import (
    clean_resume_text,
    match_skills_in_text,
    prettify_skill,
    get_sentence_model,
    _cos_sim,
)

logger = logging.getLogger(__name__)

# Seniority ladder used to compare a JD's level with the candidate's.
_RANK = {"intern": 0, "entry": 1, "mid": 2, "senior": 3, "lead": 4}


# ---------------------------------------------------------------- embeddings
def _embedding_similarity(resume_repr: str, jd_repr: str):
    """Cosine similarity in [0, 1] between resume and JD text, or None if the
    embedding service is unavailable/slow (so the caller can fall back instead
    of blocking)."""
    try:
        model = get_sentence_model()
        # MiniLM only uses the first ~256 tokens, so long inputs are wasteful and
        # slow the HF call; cap them.
        emb = model.encode([resume_repr[:1200], jd_repr[:1200]], convert_to_tensor=True)
        sim = float(_cos_sim(emb[0], emb[1:2])[0][0])
        return max(0.0, min(sim, 1.0))
    except Exception as e:
        logger.warning(f"Embedding similarity unavailable, using skill coverage only: {str(e)}")
        return None


# ------------------------------------------------------------ seniority fit
def _detect_jd_level(jd_low: str):
    """Best-effort read of the seniority a JD targets, from explicit level words
    or a years-of-experience requirement."""
    if re.search(r'\b(intern|internship)\b', jd_low):
        return "intern"
    if re.search(r'\b(entry[\s-]?level|fresher|trainee|graduate trainee|0[\s-]?1\s*years?)\b', jd_low):
        return "entry"
    if re.search(r'\b(senior|sr\.?|lead|principal|staff|head of|architect|manager)\b', jd_low):
        return "senior"
    m = re.search(r'(\d{1,2})\s*\+?\s*(?:to|-|–)?\s*\d{0,2}\s*years?', jd_low)
    if m:
        try:
            years = int(m.group(1))
        except ValueError:
            years = None
        if years is not None:
            return "senior" if years >= 5 else ("mid" if years >= 2 else "entry")
    return ""


def _seniority_fit(jd_text: str, resume_seniority: str) -> str:
    """Returns 'good fit' | 'under-qualified' | 'over-qualified' | 'unknown'."""
    rs = (resume_seniority or "").strip().lower()
    jd_level = _detect_jd_level((jd_text or "").lower())
    if not jd_level or rs not in _RANK or jd_level not in _RANK:
        return "unknown"
    diff = _RANK[rs] - _RANK[jd_level]
    if diff <= -2:
        return "under-qualified"
    if diff >= 2:
        return "over-qualified"
    return "good fit"


def _seniority_delta(fit: str) -> int:
    """Small score nudge so seniority fit actually moves the number."""
    return {"good fit": 5, "over-qualified": -3, "under-qualified": -10, "unknown": 0}.get(fit, 0)


def _clamp_score(v) -> int:
    return int(round(max(5, min(v, 99))))


# ------------------------------------------------------------------- LLM path
_MATCH_SYSTEM = """You are a calibrated recruiter scoring how well a candidate's resume \
matches a specific job. The job can be in ANY field (tech, healthcare, finance, design, \
etc.) — never assume software.

Judge fit on: (1) how many of the job's REQUIRED/must-have requirements the candidate \
meets (weigh these much higher than nice-to-have/preferred ones), (2) seniority and \
years-of-experience fit, (3) overall domain relevance. Be realistic and calibrated:
85-100 = strong match, 65-84 = good, 45-64 = partial, below 45 = stretch.

Ground everything strictly in the two texts; never invent skills the resume doesn't show.

Return ONLY minified JSON, no markdown, exactly:
{"score":<int 0-100>,"matched_skills":["..."],"missing_skills":["..."],\
"seniority_fit":"good fit|under-qualified|over-qualified|unknown","summary":"<1-2 sentences>"}
- matched_skills: up to 10 requirements the candidate HAS (use the candidate's wording).
- missing_skills: up to 8 important requirements the candidate LACKS.
- summary: concise, mention the main reason and the seniority fit."""


def _parse_match_json(raw: str):
    if not raw:
        return None
    raw = re.sub(r'^```(?:json)?|```$', '', raw.strip(), flags=re.MULTILINE).strip()
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    try:
        obj = json.loads(m.group(0) if m else raw)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _clean_list(values, limit):
    out, seen = [], set()
    for v in values or []:
        if not isinstance(v, str):
            continue
        s = v.strip().strip(".,;")
        k = s.lower()
        if s and k not in seen:
            seen.add(k)
            out.append(s)
        if len(out) >= limit:
            break
    return out


def _match_llm(resume_text: str, resume_skills, jd_text: str):
    """LLM-scored, domain-agnostic match. Returns the result dict or None to fall back."""
    from core.config import settings
    if not settings.GROQ_API_KEY:
        return None

    from langchain_groq import ChatGroq

    skills = ", ".join(resume_skills or []) or "(see resume)"
    user_msg = (
        f"JOB DESCRIPTION:\n{(jd_text or '')[:4000]}\n\n"
        f"CANDIDATE DETECTED SKILLS: {skills}\n\n"
        f"CANDIDATE RESUME:\n{(resume_text or '')[:4000]}\n\n"
        "Score the match and return JSON only."
    )
    try:
        llm = ChatGroq(api_key=settings.GROQ_API_KEY, model_name=settings.GROQ_MATCH_MODEL, temperature=0.1)
        resp = llm.invoke([("system", _MATCH_SYSTEM), ("human", user_msg)])
        obj = _parse_match_json(getattr(resp, "content", str(resp)))
        if not obj or "score" not in obj:
            logger.warning("LLM match returned no usable score; falling back.")
            return None

        score = _clamp_score(float(obj.get("score", 0)))
        matched = _clean_list(obj.get("matched_skills"), 10)
        missing = _clean_list(obj.get("missing_skills"), 8)
        fit = (obj.get("seniority_fit") or "unknown").strip().lower()
        if fit not in ("good fit", "under-qualified", "over-qualified", "unknown"):
            fit = "unknown"
        summary = (obj.get("summary") or "").strip() or "Scored your fit against this job description."
        logger.info(f"LLM match ok: score={score}, fit={fit}, matched={len(matched)}, missing={len(missing)}.")
        return {
            "score": score,
            "matched_skills": matched,
            "missing_skills": missing,
            "summary": summary,
            "jd_skill_count": len(matched) + len(missing),
            "seniority_fit": fit,
        }
    except Exception as e:
        logger.warning(f"LLM match failed, falling back to deterministic: {str(e)}")
        return None


# ----------------------------------------------------------- deterministic path
def _skill_in_jd(skill: str, jd_low: str) -> bool:
    """Domain-agnostic presence check: does this resume skill phrase appear in the
    JD as a whole word/phrase? Works for 'React' and for 'Patient triage' alike."""
    s = (skill or "").lower().strip()
    if len(s) < 2:
        return False
    return re.search(r'\b' + re.escape(s) + r'\b', jd_low) is not None


def compute_match(resume_text: str, resume_skills: list, jd_text: str, resume_seniority: str = "") -> dict:
    """Deterministic match. Domain-agnostic skill overlap blended with embedding
    similarity, nudged by seniority fit."""
    resume_skills = resume_skills or []
    resume_lower = {s.lower() for s in resume_skills}
    jd_low = (jd_text or "").lower()

    # JD skills via the tech taxonomy (gives us a 'missing' list for tech roles).
    jd_skill_keys = match_skills_in_text(clean_resume_text(jd_text), use_semantic=False)
    jd_pretty = [prettify_skill(k) for k in jd_skill_keys]

    matched_tech = [p for p in jd_pretty if p.lower() in resume_lower]
    missing = [p for p in jd_pretty if p.lower() not in resume_lower]

    # Domain-agnostic: any resume skill phrase that literally appears in the JD.
    matched_generic = [s for s in resume_skills if _skill_in_jd(s, jd_low)]
    matched = sorted({*matched_tech, *matched_generic}, key=str.lower)

    # Coverage: prefer the tech-skill ratio when the JD has detectable tech skills,
    # otherwise fall back to how many of the candidate's skills the JD mentions.
    if jd_pretty:
        coverage = len(matched_tech) / len(jd_pretty)
    elif resume_skills:
        coverage = len(matched_generic) / len(resume_skills)
    else:
        coverage = 0.0

    # Semantic similarity (best-effort; never blocks the score).
    resume_repr = " ".join(resume_skills) + "\n" + (resume_text or "")
    sim = _embedding_similarity(resume_repr, jd_text or "")

    if jd_pretty or matched_generic:
        raw = (0.55 * coverage + 0.45 * sim) if sim is not None else coverage
    else:
        raw = sim if sim is not None else 0.4  # nothing concrete -> neutral
    score = _clamp_score(raw * 100)

    # Seniority fit nudge.
    fit = _seniority_fit(jd_text, resume_seniority)
    score = _clamp_score(score + _seniority_delta(fit))

    if score >= 75:
        band = "Strong match"
    elif score >= 55:
        band = "Good match"
    elif score >= 40:
        band = "Partial match"
    else:
        band = "Stretch role"

    fit_note = {
        "good fit": " Your seniority looks right for this role.",
        "under-qualified": " It may want more experience than your resume shows.",
        "over-qualified": " You may be over-qualified for this level.",
        "unknown": "",
    }[fit]

    if jd_pretty:
        summary = f"{band} — you have {len(matched_tech)} of {len(jd_pretty)} key skills this role mentions.{fit_note}"
    elif matched:
        summary = f"{band} — {len(matched)} of your skills appear in this job description.{fit_note}"
    else:
        summary = f"{band} based on overall resume–description similarity.{fit_note}"

    return {
        "score": score,
        "matched_skills": matched[:15],
        "missing_skills": missing[:12],
        "summary": summary,
        "jd_skill_count": len(jd_pretty),
        "seniority_fit": fit,
    }


# --------------------------------------------------------------- public entry
def score_match(resume_text: str, resume_skills: list, jd_text: str, resume_seniority: str = "") -> dict:
    """LLM-first match with deterministic fallback. Always returns the full shape."""
    from core.config import settings
    if settings.MATCH_USE_LLM:
        result = _match_llm(resume_text, resume_skills, jd_text)
        if result:
            # The LLM may not know the candidate's stored seniority label; if it
            # punted to 'unknown', fill from our regex check.
            if result.get("seniority_fit") in (None, "", "unknown"):
                result["seniority_fit"] = _seniority_fit(jd_text, resume_seniority)
            return result
    return compute_match(resume_text, resume_skills, jd_text, resume_seniority)
