# Per-session store (keeps your memory-light design; only changes how state is keyed).
# Previously a single global vectorstore/skills/roles was shared across ALL users,
# so concurrent uploads overwrote each other. We now key state by a session_id
# returned at resume-upload time, with an LRU cap so memory stays bounded.

import time
import threading
from collections import OrderedDict

_LOCK = threading.Lock()
_SESSIONS: "OrderedDict[str, dict]" = OrderedDict()
_MAX_SESSIONS = 10


def _evict_locked():
    while len(_SESSIONS) > _MAX_SESSIONS:
        _SESSIONS.popitem(last=False)


def create_session(session_id: str, skills, roles, resume_text: str = "", domain: str = "", seniority: str = ""):
    with _LOCK:
        _SESSIONS[session_id] = {
            "vectorstore": None,
            "skills": skills,
            "roles": roles,
            # Raw resume text kept so the tailored-email generator can reference
            # the user's real projects/experience (the PDF itself is deleted once
            # the vectorstore is built).
            "resume_text": resume_text or "",
            # From the LLM extractor — used to judge seniority fit when matching.
            "domain": domain or "",
            "seniority": seniority or "",
            "ts": time.time(),
        }
        _SESSIONS.move_to_end(session_id)
        _evict_locked()


def set_vectorstore(session_id: str, vs):
    with _LOCK:
        if session_id in _SESSIONS:
            _SESSIONS[session_id]["vectorstore"] = vs
            _SESSIONS.move_to_end(session_id)


def update_profile(session_id: str, skills=None, roles=None):
    """Overwrite the stored skills/roles for a session (user manual override).
    Passing None for either leaves that field unchanged."""
    with _LOCK:
        s = _SESSIONS.get(session_id)
        if not s:
            return
        if skills is not None:
            s["skills"] = skills
        if roles is not None:
            s["roles"] = roles
        _SESSIONS.move_to_end(session_id)


def get_session(session_id):
    if not session_id:
        return None
    with _LOCK:
        s = _SESSIONS.get(session_id)
        if s:
            _SESSIONS.move_to_end(session_id)
        return s


def get_vectorstore(session_id):
    s = get_session(session_id)
    return s.get("vectorstore") if s else None


def get_skills(session_id):
    s = get_session(session_id)
    return s.get("skills", []) if s else []


def get_roles(session_id):
    s = get_session(session_id)
    return s.get("roles", []) if s else []


def get_resume_text(session_id):
    s = get_session(session_id)
    return s.get("resume_text", "") if s else ""


def get_seniority(session_id):
    s = get_session(session_id)
    return s.get("seniority", "") if s else ""


def get_domain(session_id):
    s = get_session(session_id)
    return s.get("domain", "") if s else ""


def is_ready(session_id) -> bool:
    s = get_session(session_id)
    return bool(s and s.get("vectorstore") is not None)
