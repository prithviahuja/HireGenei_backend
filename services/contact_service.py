"""Find a company's contact email + phone for a cold-outreach email.

Strategy (per product decision — only EVER surface contacts we actually find;
never guess/fabricate addresses):
  1. Look inside the job description / company text first (regex).
  2. Otherwise, keyless DuckDuckGo search to locate the company's OWN domain
     (matched against the company name so we don't scrape a random data-broker /
     job-board page), then scrape published emails/phones from a few of its
     pages — including Cloudflare-obfuscated and mailto:/tel: links.

No API keys required. Results are best-effort and surfaced with a confidence
level so the user can judge them before sending. If nothing real is found we
return nothing.
"""

import re
import logging
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

EMAIL_RE = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
# Phone: optional +, then digits separated by spaces/dashes/parens/dots.
PHONE_RE = re.compile(r'(\+?\d[\d\s\-().]{7,}\d)')

# Domains that yield junk emails (trackers, CDNs, the platform itself, etc.).
_BLOCKED_EMAIL_DOMAINS = {
    "linkedin.com", "example.com", "example.org", "sentry.io", "sentry-next.wixpress.com",
    "wixpress.com", "w3.org", "schema.org", "googleapis.com", "gstatic.com", "google.com",
    "googlemail.com", "cloudflare.com", "godaddy.com", "wordpress.com", "wordpress.org",
    "jquery.com", "fontawesome.com", "instagram.com", "facebook.com", "twitter.com",
    "youtube.com", "domain.com", "yourdomain.com", "email.com", "test.com", "sentry.wixpress.com",
}
_BLOCKED_EMAIL_PREFIXES = ("noreply", "no-reply", "donotreply", "do-not-reply")
# File extensions sometimes captured as fake emails (e.g. icon@2x.png).
_IMG_EXT = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".css", ".js", ".ico")

# Aggregators / job boards / data brokers — never the company's own site.
_NON_OFFICIAL_HOSTS = (
    "linkedin.com", "indeed.", "glassdoor.", "facebook.com", "twitter.com", "x.com",
    "instagram.com", "youtube.com", "crunchbase.com", "wikipedia.org", "ambitionbox.com",
    "naukri.com", "zaubacorp.com", "tracxn.com", "bloomberg.com", "github.com",
    "zoominfo.com", "rocketreach.co", "apollo.io", "signalhire.com", "lusha.com",
    "leadiq.com", "contactout.com", "sgpgrid.com", "internshala.com", "foundthejob.com",
    "startupterminal.com", "yelp.com", "medium.com", "reddit.com", "wellfound.com",
    "angel.co", "f6s.com", "pitchbook.com", "owler.com", "dnb.com", "google.com",
    "bing.com", "yahoo.com", "play.google.com", "apps.apple.com", "mojeek.com",
    "startpage.com", "yandex.com", "brave.com",
)

# Common words to strip from a company name before matching it to a domain.
_COMPANY_STOPWORDS = {
    "inc", "llc", "ltd", "pvt", "private", "limited", "technologies", "technology",
    "tech", "labs", "lab", "solutions", "solution", "systems", "system", "global",
    "india", "usa", "uk", "the", "and", "co", "corp", "corporation", "company",
    "group", "holdings", "ventures", "services", "service", "software", "consulting",
    "digital", "studio", "studios", "media", "ai", "io", "app", "online",
    "worldwide", "international",
}


def _company_tokens(company: str) -> list[str]:
    base = re.sub(r'[^a-z0-9 ]', ' ', (company or "").lower())
    return [t for t in base.split() if t and t not in _COMPANY_STOPWORDS]


def _ua_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }


def _decode_cfemail(hexstr: str) -> str:
    """Decode a Cloudflare-obfuscated email (data-cfemail / email-protection)."""
    try:
        key = int(hexstr[:2], 16)
        return "".join(chr(int(hexstr[i:i + 2], 16) ^ key) for i in range(2, len(hexstr), 2))
    except Exception:
        return ""


def _clean_emails(text: str, company: str = "") -> list[str]:
    found = {}
    tokens = [t for t in _company_tokens(company) if len(t) >= 3]
    for raw in EMAIL_RE.findall(text or ""):
        email = raw.strip().strip('.').lower()
        if email.endswith(_IMG_EXT):
            continue
        domain = email.split("@")[-1]
        local = email.split("@")[0]
        if domain in _BLOCKED_EMAIL_DOMAINS:
            continue
        score = 0 if any(local.startswith(p) for p in _BLOCKED_EMAIL_PREFIXES) else 1
        # Prefer emails whose domain matches the company name.
        if any(t in domain for t in tokens):
            score += 2
        # Prefer role inboxes that are usually monitored.
        if any(local.startswith(p) for p in ("careers", "jobs", "hr", "hello", "contact", "info", "recruit", "talent", "work")):
            score += 1
        found[email] = max(found.get(email, -1), score)
    return [e for e, _ in sorted(found.items(), key=lambda kv: kv[1], reverse=True)]


def _clean_phones(text: str) -> list[str]:
    out, seen = [], set()
    for raw in PHONE_RE.findall(text or ""):
        cand = raw.strip()
        digits = re.sub(r'\D', '', cand)
        if not (8 <= len(digits) <= 15):
            continue
        # Reject "1 2 3 4 5 ..." style sequences / lists of single spaced digits:
        # a real phone has at least one run of >= 3 consecutive digits.
        if max((len(r) for r in re.findall(r'\d+', cand)), default=0) < 3:
            continue
        if digits in seen:
            continue
        seen.add(digits)
        out.append(cand)
        if len(out) >= 3:
            break
    return out


def _from_text(text: str, company: str) -> dict:
    return {"emails": _clean_emails(text, company), "phones": _clean_phones(text)}


# ---------------- DuckDuckGo keyless web-search fallback ----------------

def _ddg_search(query: str, max_results: int = 10) -> list[dict]:
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


def _domain_core(host: str) -> str:
    host = host.lower()
    return host[4:] if host.startswith("www.") else host


def _fetch_page(url: str) -> str:
    try:
        resp = requests.get(url, headers=_ua_headers(), timeout=(5, 7))
        if resp.status_code == 200 and resp.text:
            soup = BeautifulSoup(resp.text, "html.parser")
            parts = []
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                low = href.lower()
                if low.startswith("mailto:"):
                    parts.append(href[7:].split("?")[0])
                elif low.startswith("tel:"):
                    parts.append(href[4:])
                elif "/cdn-cgi/l/email-protection#" in low:
                    parts.append(_decode_cfemail(href.split("#", 1)[1]))
            # Cloudflare-obfuscated inline emails.
            for el in soup.select("[data-cfemail]"):
                parts.append(_decode_cfemail(el.get("data-cfemail", "")))
            parts.append(soup.get_text(separator=" "))
            return " ".join(parts)
    except Exception as e:
        logger.warning(f"Failed to fetch {url[:60]}: {str(e)}")
    return ""


def _guess_domain(tokens: list[str]) -> str | None:
    """Last resort when search doesn't surface the company's own domain: try the
    obvious domain and confirm the page actually mentions the company."""
    joined = "".join(tokens)
    if not joined:
        return None
    for tld in (".com", ".io", ".co", ".ai"):
        url = f"https://www.{joined}{tld}"
        html = _fetch_page(url)
        if html and any(t in html.lower() for t in tokens):
            logger.info(f"Guessed official domain: {url}")
            return url
    return None


def _find_official_site(company: str) -> str | None:
    """Return the company's OWN site (domain must match the company name), or None
    — we never scrape an unrelated site just because it ranked first."""
    tokens = [t for t in _company_tokens(company) if len(t) >= 3]
    if not tokens:
        return None

    for r in _ddg_search(f"{company} official website", max_results=10):
        href = r.get("href") or r.get("url") or ""
        host = urlparse(href).netloc.lower()
        if not host or any(bad in host for bad in _NON_OFFICIAL_HOSTS):
            continue
        if any(t in _domain_core(host) for t in tokens):
            logger.info(f"Matched official site for '{company}': {host}")
            return f"https://{host}"

    return _guess_domain(tokens)


def _from_web(company: str) -> dict:
    site = _find_official_site(company)
    if not site:
        return {"emails": [], "phones": [], "site": None}

    text = _fetch_page(site)
    # Also try common contact pages (bounded; stop as soon as we have something).
    for path in ("/contact", "/contact-us", "/about", "/about-us", "/careers"):
        if _clean_emails(text, company) or _clean_phones(text):
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

    # 2) Keyless web search on the company's own site.
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
