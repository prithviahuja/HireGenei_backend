# Per-session store.
# Previously this was a single global vectorstore/skills/roles shared across ALL
# users, which meant concurrent uploads overwrote each other. We now key state by
# a session_id returned at resume-upload time. An LRU cap keeps memory bounded on
# small hosts (e.g. Render free tier).

import time
import threading
from collections import OrderedDict

_LOCK = threading.Lock()
_SESSIONS: "OrderedDict[str, dict]" = OrderedDict()
_MAX_SESSIONS = 10  # evict oldest beyond this to bound memory


def _evict_locked():
    while len(_SESSIONS) > _MAX_SESSIONS:
        _SESSIONS.popitem(last=False)


def create_session(session_id: str, skills, roles):
    with _LOCK:
        _SESSIONS[session_id] = {
            "vectorstore": None,
            "skills": skills,
            "roles": roles,
            "ts": time.time(),
        }
        _SESSIONS.move_to_end(session_id)
        _evict_locked()


def set_vectorstore(session_id: str, vs):
    with _LOCK:
        if session_id in _SESSIONS:
            _SESSIONS[session_id]["vectorstore"] = vs
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


def is_ready(session_id) -> bool:
    s = get_session(session_id)
    return bool(s and s.get("vectorstore") is not None)
