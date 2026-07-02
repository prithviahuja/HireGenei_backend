"""Generate a short, tailored cold email for a specific job/company.

The email is written *as the candidate*, highlighting 1–2 of their real projects
or experiences (pulled from the uploaded resume) that fit the role. It is never a
fixed template — tone and framing adapt to the candidate's domain and the target
role. The signature uses contact details extracted from the resume.
"""

import os
import re
import json
import logging

from services.resume_service import extract_applicant_contacts

logger = logging.getLogger(__name__)

EMAIL_MODEL = os.environ.get("GROQ_EMAIL_MODEL", "llama-3.3-70b-versatile")
FALLBACK_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT = """You are an expert career copywriter. You write SHORT, sharp, \
personalized cold emails that a candidate sends directly to a company to express \
interest in a job or internship. You always write in the FIRST PERSON as the candidate.

Follow these rules strictly:

1. LENGTH: Keep it short and crisp — roughly 110–170 words, at most 3 short \
paragraphs. No padding, no generic filler, no over-the-top flattery.

2. PERSONALIZE to two things at once:
   - the candidate's ACTUAL background and domain (infer it from their resume), and
   - the SPECIFIC company and role from the job description.
   Adapt tone and framing to the candidate's field — a GenAI/ML candidate, a \
frontend developer, and a data analyst should not sound the same. Do NOT reuse a \
rigid template.

3. HIGHLIGHT 1–2 concrete PROJECTS or EXPERIENCES taken from the candidate's resume \
that are most relevant to this role/company. Name them specifically and include a \
concrete outcome/impact ONLY if it appears in the resume. NEVER invent projects, \
employers, metrics, skills, degrees, or facts that are not in the resume.

4. Briefly and genuinely connect the candidate's interest/skills to the company's \
domain or this role — one sentence, not a paragraph of praise.

5. Include one line indicating the resume is attached for their review.

6. GREETING: Use the recipient's name if provided; otherwise a respectful generic \
salutation (e.g. "Dear Hiring Manager," or "Greetings,").

7. SIGN-OFF: End with a warm sign-off ("Warm regards," on its own line) then the \
candidate's name, and then each provided contact detail (phone, LinkedIn, GitHub) \
on its own line. Use ONLY the contact details provided in CANDIDATE CONTACT DETAILS. \
Omit any that are missing. Never fabricate contact details.

8. SUBJECT: Write a concise, specific subject line (no clickbait, no emojis).

OUTPUT FORMAT: Return ONLY valid minified JSON, nothing else, in exactly this shape:
{"subject": "<subject line>", "body": "<full email body with \\n for line breaks>"}
Do not wrap it in markdown code fences. Do not add commentary before or after."""


# Used instead of SYSTEM_PROMPT when the candidate supplied their own draft. The
# model must keep the user's wording/tone/structure and only adapt it to the
# specific job — NOT rewrite it into a brand-new email.
TEMPLATE_SYSTEM_PROMPT = """You are an expert career copywriter. The candidate has \
written their OWN base email draft. Your job is to ADAPT that draft to one specific \
job/company — not to rewrite it from scratch. Always write in the FIRST PERSON as \
the candidate.

Follow these rules strictly:

1. PRESERVE the candidate's draft as the foundation: keep their overall wording, \
tone, voice, structure, and paragraph order. Make the SMALLEST changes needed to \
fit this specific role and company.

2. FILL IN any placeholders in the draft (e.g. [Role], [Company], [Hiring Manager], \
{role}, {company}, <role>, blanks like "the ___ position") using the target role, \
company, and recipient name. If a recipient name isn't provided, use a respectful \
generic salutation.

3. PERSONALIZE lightly: where the draft refers to the role/company or to why the \
candidate is a fit, tailor that wording to THIS job and, if helpful, weave in 1–2 \
concrete projects/experiences from the resume that match the role. Only use real \
facts from the resume — NEVER invent projects, employers, metrics, skills, degrees, \
or contact details. Do not pad the draft with generic filler or flattery.

4. Keep it roughly the same length as the candidate's draft (do not balloon it).

5. SIGNATURE: If the draft already ends with a sign-off and contact details, keep \
them. If it does not, append a warm sign-off ("Warm regards," on its own line), the \
candidate's name, then each provided contact detail (phone, LinkedIn, GitHub, email) \
on its own line — using ONLY the details in CANDIDATE CONTACT DETAILS, omitting any \
that are missing. Never fabricate contact details.

6. SUBJECT: If the draft includes a subject line, adapt it to this role/company. \
Otherwise write a concise, specific subject line (no clickbait, no emojis).

OUTPUT FORMAT: Return ONLY valid minified JSON, nothing else, in exactly this shape:
{"subject": "<subject line>", "body": "<full email body with \\n for line breaks>"}
Do not wrap it in markdown code fences. Do not add commentary before or after."""


def _build_user_message(resume_text, resume_skills, job, contacts, email_template=""):
    role = job.get("title", "")
    company = job.get("company", "")
    jd = (job.get("description") or "").strip()
    jd_section = jd[:3500] if jd else "(No job description text was available — rely on the role title and company.)"

    contact_lines = []
    if contacts.get("name"):
        contact_lines.append(f"Name: {contacts['name']}")
    if contacts.get("phone"):
        contact_lines.append(f"Phone: {contacts['phone']}")
    if contacts.get("linkedin"):
        contact_lines.append(f"LinkedIn: {contacts['linkedin']}")
    if contacts.get("github"):
        contact_lines.append(f"GitHub: {contacts['github']}")
    if contacts.get("email"):
        contact_lines.append(f"Email: {contacts['email']}")
    contact_block = "\n".join(contact_lines) if contact_lines else "(none detected — sign off with just the name inferred from the resume)"

    skills = ", ".join(resume_skills or []) or "(see resume)"

    template_section = ""
    closing = "Write the tailored cold email now as JSON only."
    if email_template:
        template_section = (
            f"\nCANDIDATE'S OWN EMAIL DRAFT (adapt THIS to the job above — keep its "
            f"wording, tone and structure; fill placeholders; do not rewrite from scratch):\n"
            f"{email_template[:4000]}\n"
        )
        closing = "Adapt the candidate's draft to this specific job now, as JSON only."

    return f"""TARGET ROLE: {role}
TARGET COMPANY: {company}

JOB DESCRIPTION:
{jd_section}

CANDIDATE SKILLS (detected): {skills}

CANDIDATE RESUME (use real projects/experience from here; infer the candidate's name from it):
{(resume_text or '')[:6000]}

CANDIDATE CONTACT DETAILS (use these verbatim in the signature; omit any not listed):
{contact_block}
{template_section}
{closing}"""


def _get_llm(model_name: str):
    from langchain_groq import ChatGroq
    from core.config import settings

    if not settings.GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY is not configured for the backend.")
    return ChatGroq(api_key=settings.GROQ_API_KEY, model_name=model_name, temperature=0.6)


def _unescape(s: str) -> str:
    return (s.replace('\\n', '\n').replace('\\t', '\t')
             .replace('\\"', '"').replace('\\/', '/').replace('\\\\', '\\'))


def _parse_email_json(raw: str) -> dict | None:
    if not raw:
        return None
    # Strip accidental code fences.
    raw = re.sub(r'^```(?:json)?|```$', '', raw.strip(), flags=re.MULTILINE).strip()

    candidate = raw
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        candidate = m.group(0)

    # 1) Strict parse (works when the model escapes newlines correctly).
    try:
        obj = json.loads(candidate)
        if isinstance(obj, dict) and obj.get("body"):
            return {"subject": (obj.get("subject") or "").strip(), "body": str(obj["body"]).strip()}
    except Exception:
        pass

    # 2) Tolerant field extraction — handles bodies with LITERAL newlines that
    #    break json.loads (a common LLM failure mode).
    subj_m = re.search(r'"subject"\s*:\s*"((?:[^"\\]|\\.)*)"', candidate, re.DOTALL)
    body_m = (re.search(r'"body"\s*:\s*"(.*?)"\s*\}\s*$', candidate, re.DOTALL)
              or re.search(r'"body"\s*:\s*"(.*)"', candidate, re.DOTALL))
    if body_m:
        subject = _unescape(subj_m.group(1)) if subj_m else ""
        body = _unescape(body_m.group(1))
        return {"subject": subject.strip(), "body": body.strip()}
    return None


def generate_cold_email(resume_text: str, resume_skills: list[str], job: dict,
                        email_template: str = "") -> dict:
    """Returns {subject, body, to}. `to` is the best discovered recipient email
    (may be empty — the frontend lets the user fill it in).

    If `email_template` is provided, it's the candidate's own base draft: the
    model personalizes that draft for this job rather than writing one from
    scratch. When empty, the default auto-writer is used."""
    email_template = (email_template or "").strip()
    applicant = extract_applicant_contacts(resume_text)
    user_msg = _build_user_message(resume_text, resume_skills, job, applicant, email_template)

    system_prompt = TEMPLATE_SYSTEM_PROMPT if email_template else SYSTEM_PROMPT
    messages = [("system", system_prompt), ("human", user_msg)]

    result = None
    for model_name in (EMAIL_MODEL, FALLBACK_MODEL):
        try:
            llm = _get_llm(model_name)
            resp = llm.invoke(messages)
            raw = getattr(resp, "content", str(resp))
            result = _parse_email_json(raw)
            if result:
                logger.info(f"Cold email generated with model '{model_name}'.")
                break
            # Model didn't return clean JSON — wrap whatever text we got.
            if raw and model_name == FALLBACK_MODEL:
                result = {"subject": f"Application for {job.get('title', 'the role')}", "body": raw.strip()}
        except Exception as e:
            logger.warning(f"Email generation failed on '{model_name}': {str(e)}")
            continue

    if not result:
        result = {
            "subject": f"Application for {job.get('title', 'the role')} at {job.get('company', 'your company')}",
            "body": "Could not generate the email automatically. Please try again.",
        }

    recipient = ""
    contacts = job.get("_contacts") or {}
    if contacts.get("emails"):
        recipient = contacts["emails"][0]
    result["to"] = recipient
    return result
