"""Score how well a resume matches a specific job description.

Deterministic (no LLM) so it's fast and consistent: blends semantic similarity
(resume vs JD embeddings) with concrete skill overlap, and reports which of the
JD's skills the candidate already has vs is missing.
"""

import logging

from services.resume_service import (
    clean_resume_text,
    match_skills_in_text,
    prettify_skill,
    get_sentence_model,
    _cos_sim,
)

logger = logging.getLogger(__name__)


def _embedding_similarity(resume_repr: str, jd_repr: str) -> float:
    """Cosine similarity in [0, 1] between resume and JD text."""
    try:
        model = get_sentence_model()
        emb = model.encode([resume_repr, jd_repr], convert_to_tensor=True)
        sim = float(_cos_sim(emb[0], emb[1:2])[0][0])
        return max(0.0, min(sim, 1.0))
    except Exception as e:
        logger.warning(f"Embedding similarity failed, falling back to 0: {str(e)}")
        return 0.0


def compute_match(resume_text: str, resume_skills: list[str], jd_text: str) -> dict:
    """Returns {score, matched_skills, missing_skills, summary, jd_skill_count}."""
    resume_skill_set = {s.lower() for s in (resume_skills or [])}

    # ---- Skill overlap against skills detected in the JD ----
    jd_skill_keys = match_skills_in_text(clean_resume_text(jd_text), use_semantic=False)
    jd_pretty = [prettify_skill(k) for k in jd_skill_keys]

    matched, missing = [], []
    for pretty in jd_pretty:
        if pretty.lower() in resume_skill_set:
            matched.append(pretty)
        else:
            missing.append(pretty)

    coverage = len(matched) / len(jd_pretty) if jd_pretty else 0.0

    # ---- Semantic similarity (handles JDs whose skills aren't in our list) ----
    resume_repr = (" ".join(resume_skills or []) + "\n" + (resume_text or ""))[:2500]
    sim = _embedding_similarity(resume_repr, (jd_text or "")[:2500])

    # ---- Blend ----
    if jd_pretty:
        raw = 0.55 * coverage + 0.45 * sim
    else:
        raw = sim  # no detectable JD skills -> lean entirely on semantics
    score = int(round(max(5, min(raw * 100, 99))))

    # ---- Human-readable summary ----
    if score >= 75:
        band = "Strong match"
    elif score >= 55:
        band = "Good match"
    elif score >= 40:
        band = "Partial match"
    else:
        band = "Stretch role"

    if jd_pretty:
        summary = f"{band} — you have {len(matched)} of {len(jd_pretty)} key skills this role mentions."
    else:
        summary = f"{band} based on overall resume–description similarity."

    return {
        "score": score,
        "matched_skills": matched[:15],
        "missing_skills": missing[:12],
        "summary": summary,
        "jd_skill_count": len(jd_pretty),
    }
