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

    # ---- Semantic similarity (best-effort; never blocks the score) ----
    logger.info("Computing resume-JD embedding similarity...")
    resume_repr = " ".join(resume_skills or []) + "\n" + (resume_text or "")
    sim = _embedding_similarity(resume_repr, jd_text or "")

    # ---- Blend (gracefully degrade to coverage if embedding is unavailable) ----
    if jd_pretty:
        raw = (0.55 * coverage + 0.45 * sim) if sim is not None else coverage
    else:
        raw = sim if sim is not None else 0.4  # nothing concrete to go on -> neutral
    score = int(round(max(5, min(raw * 100, 99))))
    logger.info(f"Match computed: score={score}, matched={len(matched)}/{len(jd_pretty)}, sim={sim}")

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
