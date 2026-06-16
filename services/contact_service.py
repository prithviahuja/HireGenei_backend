"""Find a company's contact email + phone for a cold-outreach email.

Strategy (per product decision):
  1. Look inside the job description / company text first (regex).
  2. If nothing useful is found, fall back to a keyless DuckDuckGo search agent:
     find the company's official site, fetch its homepage + a contact page, and
     scrape public email/phone from there.

No API keys required. Results are best-effort and surfaced with a confidence
level so the user knows whether to trust them before sending.
"""

import re
import logging
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

EMAIL_RE = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
# Phone: optional +, then 8–15 digits possibly separated by spaces/dashes/parens.
PHONE_RE = re.compile(r'(\+?\d[\d\s\-().]{7,}\d)')

# Domains that yield junk emails (trackers, CDNs, the platform itself, etc.).
_BLOCKED_EMAIL_DOMAINS = {
    "linkedin.com", "example.com", "example.org", "sentry.io", "sentry-next.wixpress.com",
    "wixpress.com", "w3.org", "schema.org", "googleapis.com", "gstatic.com", "google.com",
    "googlemail.com", "cloudflare.com", "godaddy.com", "wordpress.com", "wordpress.org",
    "jquery.com", "fontawesome.com", " instagram.com".strip(), "facebook.com", "twitter.com",
    "youtube.com", "domain.com", "yourdomain.com", "email.com", "test.com",
}
_BLOCKED_EMAIL_PREFIXES = ("noreply", "no-reply", "donotreply", "do-not-reply")
# File extensions sometimes captured as fake emails (e.g. icon@2x.png).
_IMG_EXT = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".css", ".js", ".ico")


def _ua_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }


def _clean_emails(text: str, company: str = "") -> list[str]:
    found = {}
    company_token = re.sub(r'[^a-z0-9]', '', (company or "").lower())
    for raw in EMAIL_RE.findall(text or ""):
        email = raw.strip().strip('.').lower()
        if email.endswith(_IMG_EXT):
            continue
        domain = email.split("@")[-1]
        local = email.split("@")[0]
        if domain in _BLOCKED_EMAIL_DOMAINS:
            continue
        if any(local.startswith(p) for p in _BLOCKED_EMAIL_PREFIXES):
            # keep but rank low (still a valid channel)
            score = 0
        else:
            score = 1
        # Prefer emails whose domain matches the company name.
        if company_token and company_token[:5] and company_token[:5] in domain:
            score += 2
        # Prefer role inboxes that are usually monitored.
        if any(local.startswith(p) for p in ("careers", "jobs", "hr", "hello", "contact", "info", "recruit")):
            score += 1
        found[email] = max(found.get(email, -1), score)
    return [e for e, _ in sorted(found.items(), key=lambda kv: kv[1], reverse=True)]


def _clean_phones(text: str) -> list[str]:
    out = []
    seen = set()
    for raw in PHONE_RE.findall(text or ""):
        digits = re.sub(r'\D', '', raw)
        if not (8 <= len(digits) <= 15):
            continue
        if digits in seen:
            continue
        seen.add(digits)
        out.append(raw.strip())
        if len(out) >= 3:
            break
    return out


def _from_text(text: str, company: str) -> dict:
    return {"emails": _clean_emails(text, company), "phones": _clean_phones(text)}


# ---------------- DuckDuckGo keyless web-search fallback ----------------

def _ddg_search(query: str, max_results: int = 6) -> list[dict]:
    try:
        try:
            from ddgs import DDGS  # current package name
        except ImportError:
            from duckduckgo_search import DDGS  # older name
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))
    except Exception as e:
        logger.warning(f"DuckDuckGo search failed: {str(e)}")
        return []


_NON_OFFICIAL_HOSTS = (
    "linkedin.com", "indeed.com", "glassdoor.", "facebook.com", "twitter.com", "x.com",
    "instagram.com", "youtube.com", "crunchbase.com", "wikipedia.org", "ambitionbox.com",
    "naukri.com", "zaubacorp.com", "tracxn.com", "bloomberg.com", "github.com",
)


def _find_official_site(company: str) -> str | None:
    results = _ddg_search(f'{company} official website contact')
    for r in results:
        href = r.get("href") or r.get("url") or ""
        host = urlparse(href).netloc.lower()
        if host and not any(bad in host for bad in _NON_OFFICIAL_HOSTS):
            return f"{urlparse(href).scheme or 'https'}://{host}"
    return None


def _fetch_page(url: str) -> str:
    try:
        resp = requests.get(url, headers=_ua_headers(), timeout=10)
        if resp.status_code == 200 and resp.text:
            soup = BeautifulSoup(resp.text, "html.parser")
            # mailto links are the most reliable email source
            mailtos = " ".join(
                a.get("href", "")[7:] for a in soup.find_all("a", href=True)
                if a.get("href", "").lower().startswith("mailto:")
            )
            return mailtos + "\n" + soup.get_text(separator=" ")
    except Exception as e:
        logger.warning(f"Failed to fetch {url[:60]}: {str(e)}")
    return ""


def _from_web(company: str) -> dict:
    site = _find_official_site(company)
    if not site:
        return {"emails": [], "phones": [], "site": None}

    text = _fetch_page(site)
    # Also try common contact pages.
    for path in ("/contact", "/contact-us", "/about", "/careers"):
        if text and (_clean_emails(text, company) or _clean_phones(text)):
            break
        text += "\n" + _fetch_page(site.rstrip("/") + path)

    return {
        "emails": _clean_emails(text, company),
        "phones": _clean_phones(text),
        "site": site,
    }


def find_contacts(jd_text: str, company: str) -> dict:
    """Returns {emails, phones, source, confidence, site}."""
    # 1) Straight from the job description / company text.
    jd = _from_text(jd_text or "", company)
    if jd["emails"] or jd["phones"]:
        logger.info(f"Contacts found in job description for '{company}'.")
        return {
            "emails": jd["emails"][:3],
            "phones": jd["phones"][:3],
            "source": "job description",
            "confidence": "high" if jd["emails"] else "medium",
            "site": None,
        }

    # 2) Keyless web search fallback.
    if company:
        logger.info(f"No contacts in JD; web-searching for '{company}'.")
        web = _from_web(company)
        if web["emails"] or web["phones"]:
            return {
                "emails": web["emails"][:3],
                "phones": web["phones"][:3],
                "source": "company website (web search)",
                "confidence": "medium" if web["emails"] else "low",
                "site": web.get("site"),
            }

    return {"emails": [], "phones": [], "source": "none", "confidence": "none", "site": None}
