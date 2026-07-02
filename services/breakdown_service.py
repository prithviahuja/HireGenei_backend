"""Per-metric, resume-specific breakdown coaching.

Given a session's resume text and ONE of the six score dimensions, this:
  1) deterministically pinpoints the ACTUAL lines from the resume that dragged
     that dimension down (so the quoted excerpts are always real, never made up),
  2) asks the LLM to rewrite each offending line + give targeted, specific tips.

If the LLM is unavailable it still returns useful, deterministic guidance — so
the feature degrades gracefully instead of erroring.
"""
import re
import json
import logging

from services.resume_service import (
    score_resume,
    _IMPACT_HINT_RE, _ACTION_VERBS, _WEAK_PHRASES, _SECTION_PATTERNS,
    _EMAIL_RE, _PHONE_RE, _LINKEDIN_RE,
)

logger = logging.getLogger(__name__)

DETAIL_MODEL = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "llama-3.1-8b-instant"

METRIC_KEYS = {
    "Impact & metrics", "Action verbs", "Completeness",
    "Skills breadth", "Length", "Role fit",
}

# What each dimension actually measures — fed to the LLM so its advice is on-topic.
_METRIC_DEF = {
    "Impact & metrics": "whether bullet points quantify results with numbers, %, $, time saved, scale, or other measurable outcomes",
    "Action verbs": "whether bullets open with strong action verbs (Led, Built, Designed, Shipped) instead of weak/passive phrasing like 'responsible for' or 'worked on'",
    "Completeness": "whether the resume has all the expected sections (summary, experience, education, skills, projects) and reachable contact details (email, phone, LinkedIn)",
    "Skills breadth": "whether the resume lists a broad, concrete set of relevant tools, technologies and skills",
    "Length": "whether the resume length (word count) is in the healthy 1–2 page range — not too thin, not bloated",
    "Role fit": "whether the resume points clearly at well-defined target roles",
}


def _substantive_lines(text):
    return [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 25]


def _verdict(score):
    return "high" if score >= 75 else "medium" if score >= 50 else "low"


def _norm(s):
    return re.sub(r'\s+', ' ', (s or '').lower()).strip()


def _candidates(metric, text, skills, roles):
    """Return (facts, problem_lines) for a metric.

    problem_lines are verbatim lines from the resume that hurt this dimension
    (empty for structural metrics like Length/Completeness where there's no
    single 'bad line' to quote)."""
    lines = _substantive_lines(text)
    low = text.lower()

    if metric == "Impact & metrics":
        bad = [ln for ln in lines if not _IMPACT_HINT_RE.search(ln)]
        facts = {
            "bullets_total": len(lines),
            "bullets_with_a_metric": sum(1 for ln in lines if _IMPACT_HINT_RE.search(ln)),
        }
        return facts, bad[:6]

    if metric == "Action verbs":
        bad = []
        for ln in lines:
            head = re.sub(r'^[\-•‣◦\*·\s]+', '', ln).split()
            first = head[0].lower().rstrip(",.:;") if head else ""
            has_weak = any(p in ln.lower() for p in _WEAK_PHRASES)
            weak_start = bool(first) and first not in _ACTION_VERBS
            if has_weak or weak_start:
                bad.append(ln)
        facts = {"weak_phrases_found": sorted({p for p in _WEAK_PHRASES if p in low})}
        return facts, bad[:6]

    if metric == "Completeness":
        missing = [name for name, pat in _SECTION_PATTERNS.items() if not re.search(pat, low)]
        has_email = bool(_EMAIL_RE.search(text))
        has_phone = any(8 <= len(re.sub(r'\D', '', c)) <= 15 for c in _PHONE_RE.findall(text))
        has_linkedin = bool(_LINKEDIN_RE.search(text))
        need = [n for n, ok in (("email", has_email), ("phone", has_phone), ("LinkedIn URL", has_linkedin)) if not ok]
        return {"missing_sections": missing, "missing_contact": need}, []

    if metric == "Skills breadth":
        return {"skills_detected": len(skills), "skills": skills[:40]}, []

    if metric == "Length":
        return {"word_count": len(text.split())}, []

    if metric == "Role fit":
        return {"roles_detected": len(roles), "roles": roles}, []

    return {}, []


# ---------------------------------------------------------------- LLM

def _get_llm(model_name: str, temperature: float = 0.3):
    from langchain_groq import ChatGroq
    from core.config import settings
    if not settings.GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY is not configured for the backend.")
    return ChatGroq(api_key=settings.GROQ_API_KEY, model_name=model_name, temperature=temperature)


_SYSTEM = (
    "You are a precise, brutally helpful resume reviewer. You explain exactly WHY one "
    "scoring dimension of a resume scored the way it did, quoting the user's OWN lines, "
    "and you rewrite the weak ones into strong replacements. "
    "Be specific and concrete — never generic. "
    "Return ONLY valid minified JSON, no markdown, with this shape: "
    '{"summary": string, "issues": [{"excerpt": string, "problem": string, "fix": string}], "tips": [string]}. '
    "Rules: "
    "- 'excerpt' MUST be copied verbatim from the PROBLEM LINES provided (do not invent lines). "
    "- 'problem' = one short sentence on why that line hurts THIS dimension. "
    "- 'fix' = a rewritten version of that exact line that would score well (keep it truthful/plausible; "
    "use a clear placeholder like [X%] or [N] where the user must drop in a real number). "
    "- If no problem lines are given, return an empty 'issues' array and put the guidance in 'tips'. "
    "- 'tips' = 2-4 sharp, resume-specific actions. "
    "- 'summary' = 1-2 sentences explaining the score in plain English."
)


def _build_human(metric, score, facts, problem_lines, text):
    excerpt_block = "\n".join(f"- {ln}" for ln in problem_lines) if problem_lines else "(none — this is a structural dimension)"
    resume_slice = text if len(text) <= 6000 else text[:6000]
    verdict = _verdict(score)

    if verdict == "high":
        guidance = (
            "This dimension ALREADY scores well — do NOT invent problems or improvements. "
            "Return an EMPTY 'issues' array and an EMPTY 'tips' array. "
            "'summary' = 1-2 sentences affirming what the resume is doing right here."
        )
    else:
        guidance = (
            "Identify the weakest lines and rewrite them. "
            "'tips' = 2-4 sharp, specific actions to RAISE this score."
        )

    return (
        f"DIMENSION: {metric}\n"
        f"WHAT IT MEASURES: {_METRIC_DEF.get(metric, '')}\n"
        f"SCORE: {score}/100 ({verdict})\n"
        f"DETERMINISTIC FACTS: {json.dumps(facts, ensure_ascii=False)}\n\n"
        f"PROBLEM LINES (quote these verbatim as 'excerpt' and rewrite them in 'fix'):\n{excerpt_block}\n\n"
        f"RESUME (for context):\n\"\"\"\n{resume_slice}\n\"\"\"\n\n"
        f"INSTRUCTION: {guidance}\n"
        "Now return the JSON."
    )


def _parse_json(raw):
    if not raw:
        return None
    raw = re.sub(r'^```(?:json)?|```$', '', raw.strip(), flags=re.MULTILINE).strip()
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    candidate = m.group(0) if m else raw
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _llm_explain(metric, score, facts, problem_lines, text):
    messages = [("system", _SYSTEM), ("human", _build_human(metric, score, facts, problem_lines, text))]
    for model_name in (DETAIL_MODEL, FALLBACK_MODEL):
        try:
            resp = _get_llm(model_name).invoke(messages)
            raw = getattr(resp, "content", str(resp))
            obj = _parse_json(raw)
            if obj and ("issues" in obj or "tips" in obj):
                logger.info(f"Breakdown detail for '{metric}' generated with '{model_name}'.")
                return obj
        except Exception as e:
            logger.warning(f"Breakdown detail failed on '{model_name}': {e}")
            continue
    return None


# ---------------------------------------------------------------- fallback

def _fallback(metric, facts, problem_lines):
    """Deterministic guidance when the LLM is unavailable."""
    issues, tips, summary = [], [], ""

    if metric == "Impact & metrics":
        summary = (f"Only {facts.get('bullets_with_a_metric', 0)} of {facts.get('bullets_total', 0)} "
                   "bullet points include a measurable result. Recruiters skim for numbers.")
        for ln in problem_lines:
            issues.append({
                "excerpt": ln,
                "problem": "No number, %, $, scale or time — the impact is invisible.",
                "fix": ln.rstrip(". ") + " — quantify it, e.g. add “…, cutting processing time by [X%]” or “…for [N]+ users”.",
            })
        tips = ["End each bullet with a measurable outcome (%, #, $, time).",
                "Lead with the result, then the action: “Cut load time 40% by…”."]

    elif metric == "Action verbs":
        weak = facts.get("weak_phrases_found", [])
        summary = ("Several bullets use weak or passive openings"
                   + (f" ({', '.join(weak)})" if weak else "") + " instead of strong action verbs.")
        for ln in problem_lines:
            issues.append({
                "excerpt": ln,
                "problem": "Doesn't open with a strong action verb.",
                "fix": "Led / Built / Designed / Shipped / Automated " + re.sub(r'^(responsible for|worked on|helped (?:with|to)?|involved in|duties included)\s*', '', ln, flags=re.I).lstrip('-•* ').strip(),
            })
        tips = ["Start every bullet with a past-tense action verb.",
                "Replace “responsible for / worked on” with Led, Owned, Built, Drove."]

    elif metric == "Completeness":
        miss_s = facts.get("missing_sections", [])
        miss_c = facts.get("missing_contact", [])
        summary = "Some expected sections or contact details are missing."
        if miss_s:
            tips.append(f"Add the missing section(s): {', '.join(miss_s)}.")
        if miss_c:
            tips.append(f"Add your {', '.join(miss_c)} so recruiters can reach you.")
        if not tips:
            tips = ["All key sections look present — keep them clearly labelled."]

    elif metric == "Skills breadth":
        summary = f"{facts.get('skills_detected', 0)} skills detected — add more concrete, relevant tools."
        tips = ["List specific tools/technologies, not vague terms (e.g. ‘PostgreSQL, Docker, PyTorch’).",
                "Group skills by category so they’re easy to scan."]

    elif metric == "Length":
        wc = facts.get("word_count", 0)
        if wc < 250:
            summary = f"Your resume is thin (~{wc} words)."
            tips = ["Expand on projects, responsibilities and outcomes.", "Aim for ~400–800 words (1–2 pages)."]
        elif wc > 1100:
            summary = f"Your resume is long (~{wc} words)."
            tips = ["Trim to the most relevant, recent and impactful points.", "Target 1–2 pages."]
        else:
            summary = f"Length looks healthy (~{wc} words)."
            tips = ["Keep it tight — every line should earn its place."]

    elif metric == "Role fit":
        summary = f"{facts.get('roles_detected', 0)} clear target role(s) detected."
        tips = ["Add a one-line headline naming your target role.",
                "Mirror the keywords from job descriptions you’re targeting."]

    return {"summary": summary, "issues": issues, "tips": tips}


# ---------------------------------------------------------------- entrypoint

def explain_metric(resume_text: str, skills, roles, metric: str) -> dict:
    metric = (metric or "").strip()
    if metric not in METRIC_KEYS:
        raise ValueError(f"Unknown metric '{metric}'.")

    text = resume_text or ""
    skills = skills or []
    roles = roles or []

    scored = score_resume(text, skills, roles)
    score = scored["breakdown"].get(metric, 0)

    facts, problem_lines = _candidates(metric, text, skills, roles)

    result = _llm_explain(metric, score, facts, problem_lines, text)
    if not result:
        result = _fallback(metric, facts, problem_lines)

    # Keep the model honest: drop any 'issue' whose excerpt isn't actually in the
    # resume (guards against fabricated quotes). Structural metrics have no lines.
    norm_resume = _norm(text)
    issues = []
    for it in (result.get("issues") or []):
        if not isinstance(it, dict):
            continue
        ex = (it.get("excerpt") or "").strip()
        fix = (it.get("fix") or "").strip()
        prob = (it.get("problem") or "").strip()
        if not fix:
            continue
        # accept the excerpt if a meaningful slice of it appears in the resume
        head = _norm(ex)
        head_slice = " ".join(head.split()[:6])
        if ex and head_slice and head_slice in norm_resume:
            issues.append({"excerpt": ex, "problem": prob, "fix": fix})
        elif not problem_lines:
            # structural metric: no verbatim requirement
            issues.append({"excerpt": ex, "problem": prob, "fix": fix})

    tips = [t for t in (result.get("tips") or []) if isinstance(t, str) and t.strip()][:4]
    summary = (result.get("summary") or "").strip()

    verdict = _verdict(score)

    if verdict == "high":
        # A strong dimension needs no "fixes" or "how to improve" — just affirm it.
        issues, tips = [], []
        if not summary:
            summary = f"This is a strength — {metric} scores {score}/100. Nothing to fix here."
    elif not issues and not tips:
        # Nothing usable from the model — fall back deterministically.
        fb = _fallback(metric, facts, problem_lines)
        issues, tips, summary = fb["issues"], fb["tips"], summary or fb["summary"]

    return {
        "metric": metric,
        "score": score,
        "verdict": verdict,
        "summary": summary,
        "issues": issues,
        "tips": tips,
    }
