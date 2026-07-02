import re
import pdfplumber
from rapidfuzz import process
import os
import logging

from utils.embeddings import get_hf_embeddings, get_sentence_model
from utils.vectorstore import set_vectorstore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

logger = logging.getLogger(__name__)

# Skills List
SKILLS_LIST = [
    "python", "cpp", "csharp", "java", "javascript", "typescript", "dart", "go", "rust", "r", "shell", "bash", "sql",
    "scala", "php", "perl", "matlab", "assembly", "swift", "kotlin",
    "html", "css", "react", "angular", "vuejs", "svelte", "nextjs", "tailwind", "bootstrap",
    "nodejs", "expressjs", "django", "flask", "fastapi", "springboot", "dotnet", "laravel",
    "graphql", "grpc", "websockets", "restapi",
    "flutter", "reactnative", "android", "ios",
    "mysql", "postgresql", "sqlite", "mongodb", "redis", "neo4j", "cassandra", "dynamodb",
    "influxdb", "firestore", "supabase",
    "hadoop", "hive", "spark", "kafka", "apachebeam", "airflow", "deltalake", "snowflake",
    "glue", "presto", "flink",
    "git", "github", "gitlab", "bitbucket", "jenkins", "docker", "kubernetes", "helm", "istio",
    "ansible", "terraform", "azuredevops", "argocd", "prometheus", "grafana",
    "aws", "azure", "gcp", "s3", "ec2", "lambda", "cloudfunctions", "firebase", "cloudflare", "cloudformation",
    "numpy", "pandas", "matplotlib", "seaborn", "plotly", "dask", "vaex",
    "powerbi", "tableau", "looker", "superset",
    "scikitlearn", "xgboost", "lightgbm", "catboost", "mlflow", "optuna", "joblib", "ann", "cnn", "rnn", "lstm",
    "tensorflow", "keras", "pytorch", "huggingface", "transformers", "t5", "bert", "gpt", "peft", "lora", "qlora",
    "spacy", "nltk", "gensim", "langchain", "openaiai", "fairseq", "marianmt", "crewai", "haystack",
    "opencv", "yolo", "detectron2", "mediapipe", "paddleocr", "tesseract",
    "bentoml", "torchserve", "sagemaker", "vertexai", "onnx", "tfserving", "gradio", "streamlit","langraph","lanserve",
    "surprise", "lightfm", "implicit", "faiss", "annoy", "milvus",
    "loadbalancing", "messagequeues", "caching", "apigateway", "microservices", "monolith",
    "eventdriven", "pubsub", "modulefederation",
    "stripe", "razorpay", "paypal",
    "zapier", "integromat", "powerapps", "bubble", "ifttt",
    "solidity", "web3js", "ethersjs", "truffle", "hardhat", "ganache", "ipfs",
    "unity", "unrealengine", "godot", "blender", "threejs",
    "nmap", "wireshark", "metasploit", "burpsuite", "owasp", "jwt", "oauth2", "saml", "ssltls",
    "aiagents", "rag", "weaviate", "pinecone", "pyg", "stablebaselines3", "cypher"
]

# Roles Dictionary
JOB_ROLES = {
    "Data Scientist": ["python", "pandas", "numpy", "matplotlib", "seaborn", "plotly", "scikitlearn", "xgboost", "lightgbm", "catboost", "mlflow", "optuna", "statistics", "dask", "vaex", "joblib"],
    "Data Analyst": ["sql", "excel", "powerbi", "tableau", "looker", "superset", "pandas", "numpy", "matplotlib", "seaborn", "statistics"],
    "ML Engineer": ["python", "tensorflow", "keras", "pytorch", "mlflow", "onnx", "huggingface", "transformers", "joblib", "optuna", "torchserve", "tfserving", "gradio", "streamlit", "sagemaker", "vertexai"],
    "Data Engineer": ["spark", "hadoop", "hive", "kafka", "apachebeam", "airflow", "deltalake", "snowflake", "glue", "presto", "flink", "mysql", "postgresql", "mongodb", "neo4j", "cassandra", "dynamodb", "influxdb", "supabase"],
    "Generative AI Engineer": ["transformers", "huggingface", "gpt", "bert", "t5", "lora", "qlora", "peft", "langchain", "openaiai", "crewai", "langraph", "lanserve", "rag", "weaviate", "pinecone", "haystack", "vertexai"],
    "NLP Engineer": ["spacy", "nltk", "gensim", "huggingface", "transformers", "bert", "t5", "marianmt", "fairseq", "gpt", "langchain", "rag", "peft", "haystack", "openaiai"],
    "Computer Vision Engineer": ["opencv", "yolo", "detectron2", "mediapipe", "paddleocr", "tesseract", "cnn"],
    "Web Developer": ["html", "css", "javascript", "react", "vuejs", "angular", "svelte", "nextjs", "tailwind", "bootstrap", "nodejs", "expressjs", "restapi", "graphql", "grpc", "websockets", "flask", "django", "fastapi"],
    "Backend Developer": ["nodejs", "expressjs", "flask", "django", "fastapi", "springboot", "dotnet", "graphql", "grpc", "restapi", "mysql", "postgresql", "mongodb", "redis", "neo4j"],
    "Frontend Developer": ["html", "css", "javascript", "typescript", "react", "angular", "vuejs", "svelte", "nextjs", "tailwind", "bootstrap"],
    "Mobile App Developer": ["flutter", "reactnative", "android", "ios", "dart", "kotlin", "swift"],
    "DevOps Engineer": ["git", "github", "gitlab", "bitbucket", "jenkins", "docker", "kubernetes", "helm", "istio", "ansible", "terraform", "azuredevops", "argocd", "prometheus", "grafana"],
    "Cloud Engineer": ["aws", "azure", "gcp", "s3", "ec2", "lambda", "cloudfunctions", "firebase", "cloudformation", "cloudflare"],
    "MLOps Engineer": ["mlflow", "bentoml", "torchserve", "sagemaker", "vertexai", "onnx", "tfserving", "gradio", "streamlit", "docker", "kubernetes"],
    "AI Researcher": ["pytorch", "tensorflow", "huggingface", "gpt", "bert", "t5", "cnn", "rnn", "lstm", "transformers", "peft", "qlora", "fairseq", "marianmt"],
    "Recommender Systems Engineer": ["surprise", "lightfm", "implicit", "faiss", "annoy", "milvus"],
    "Cybersecurity Engineer": ["nmap", "wireshark", "metasploit", "burpsuite", "owasp", "jwt", "oauth2", "saml", "ssltls"],
    "Game Developer": ["unity", "unrealengine", "godot", "blender", "threejs"],
    "Blockchain Developer": ["solidity", "web3js", "ethersjs", "truffle", "hardhat", "ganache", "ipfs"],
    "Automation Specialist / No-Code Developer": ["zapier", "integromat", "powerapps", "bubble", "ifttt"],
    "System Architect": ["loadbalancing", "messagequeues", "caching", "apigateway", "microservices", "monolith", "eventdriven", "pubsub", "modulefederation"],
    "Payment Integration Engineer": ["stripe", "razorpay", "paypal"]
}

def clean_resume_text(text: str) -> str:
    text = text.replace('\u200b', ' ')
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'●', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def extract_resume_text(pdf_path: str) -> str:
    """Readable full-text extraction that PRESERVES line structure.

    Unlike clean_resume_text (which lowercases and collapses everything onto one
    line for skill matching), this keeps newlines and original casing so the
    email LLM can read projects/experience sections naturally.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            pages = [page.extract_text() for page in pdf.pages if page.extract_text()]
        text = "\n".join(pages)
    except Exception as e:
        logger.error(f"Failed to extract readable resume text: {str(e)}")
        return ""
    text = text.replace('​', ' ').replace('●', '• ')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ---- Applicant contact details (for the email signature) ----
_EMAIL_RE = re.compile(r'[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}')
_PHONE_RE = re.compile(r'(\+?\d[\d\s\-().]{7,}\d)')
_LINKEDIN_RE = re.compile(r'(?:https?://)?(?:www\.)?linkedin\.com/(?:in|pub)/[A-Za-z0-9\-_%]+', re.I)
_GITHUB_RE = re.compile(r'(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9\-_.]+', re.I)


def _normalize_url(url: str) -> str:
    url = url.strip().rstrip('/.,)')
    if not url.lower().startswith('http'):
        url = 'https://' + url
    return url


def extract_applicant_contacts(resume_text: str) -> dict:
    """Pull the applicant's own contact details out of their resume so the
    generated email is signed correctly. Name is left to the LLM (hard to
    regex reliably)."""
    if not resume_text:
        return {}
    contacts: dict = {}

    email_m = _EMAIL_RE.search(resume_text)
    if email_m:
        contacts["email"] = email_m.group(0).strip()

    li_m = _LINKEDIN_RE.search(resume_text)
    if li_m:
        contacts["linkedin"] = _normalize_url(li_m.group(0))

    gh_m = _GITHUB_RE.search(resume_text)
    if gh_m and not gh_m.group(0).rstrip('/').lower().endswith('github.com'):
        contacts["github"] = _normalize_url(gh_m.group(0))

    for cand in _PHONE_RE.findall(resume_text):
        digits = re.sub(r'\D', '', cand)
        if 8 <= len(digits) <= 15:
            contacts["phone"] = cand.strip()
            break

    return contacts


# Precompute heavy ML embeddings on startup so they don't block requests
PRECOMPUTED_SKILL_EMBEDDINGS = None
logger.info("Skill embeddings will be pre-compiled lazily on demand.")


def _cos_sim(a, b):
    import torch
    import torch.nn.functional as F
    # Handle 1D HF API vectors by adding a leading batch dimension.
    if a.dim() == 1:
        a = a.unsqueeze(0)
    # Assuming a is (n, d), b is (m, d)
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.t())


def get_precomputed_skill_embeddings():
    global PRECOMPUTED_SKILL_EMBEDDINGS
    if PRECOMPUTED_SKILL_EMBEDDINGS is None:
        logger.info("Pre-compiling skill embeddings...")
        PRECOMPUTED_SKILL_EMBEDDINGS = get_sentence_model().encode(SKILLS_LIST, convert_to_tensor=True)
        logger.info("Skill embeddings pre-compiled successfully.")
    return PRECOMPUTED_SKILL_EMBEDDINGS


def match_skills_in_text(raw_text: str, use_semantic: bool = True) -> list[str]:
    """Core skill matcher (exact + fuzzy + optional semantic) over already-cleaned
    text. Reused for both resumes and job descriptions. Returns raw skill keys
    from SKILLS_LIST (not prettified)."""
    if not raw_text:
        return []

    exact_matches = set()
    for skill in SKILLS_LIST:
        pattern = re.escape(skill.lower())
        if re.search(r'\b'+pattern+r'\b', raw_text):
            exact_matches.add(skill)

    words = set(re.findall(r"\b[a-zA-Z\#\+\-\.]{2,}\b", raw_text))
    fuzzy_matches = set()
    for word in words:
        match = process.extractOne(word, SKILLS_LIST, score_cutoff=93)
        if match:
            fuzzy_matches.add(match[0])

    semantic_matches = set()
    if use_semantic and words:
        import time
        start_time = time.time()
        # We only encode the dynamic words heavily now, bypassing ~200 redundant vector calcs
        sentence_model = get_sentence_model()
        text_embeddings = sentence_model.encode(list(words), convert_to_tensor=True)
        logger.info(f"Encoded {len(words)} dynamic text words in {time.time() - start_time:.2f}s")

        threshold = 0.75
        precomputed_skill_embeddings = get_precomputed_skill_embeddings()
        for vec in text_embeddings:
            scores = _cos_sim(vec, precomputed_skill_embeddings)[0]
            best_idx = scores.argmax().item()
            if float(scores[best_idx]) >= threshold:
                semantic_matches.add(SKILLS_LIST[best_idx])

    final_skills = list(exact_matches.union(fuzzy_matches).union(semantic_matches))
    logger.info(f"Skill match done: {len(exact_matches)} exact, {len(fuzzy_matches)} fuzzy, {len(semantic_matches)} semantic.")
    return final_skills


def skills_extraction(pdf_path: str) -> list[str]:
    with pdfplumber.open(pdf_path) as pdf:
        raw_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    raw_text = clean_resume_text(raw_text)
    return match_skills_in_text(raw_text, use_semantic=True)

logger.info("Role embeddings will be pre-compiled lazily on demand.")
roles_text_global = [" ".join(value) for value in JOB_ROLES.values()]
roles_list_global = list(JOB_ROLES.keys())
PRECOMPUTED_ROLE_EMBEDDINGS = None


def get_precomputed_role_embeddings():
    global PRECOMPUTED_ROLE_EMBEDDINGS
    if PRECOMPUTED_ROLE_EMBEDDINGS is None:
        logger.info("Pre-compiling role embeddings...")
        PRECOMPUTED_ROLE_EMBEDDINGS = get_sentence_model().encode(roles_text_global, convert_to_tensor=True)
        logger.info("Role embeddings pre-compiled successfully.")
    return PRECOMPUTED_ROLE_EMBEDDINGS


def roles_score(user_skills: list[str]) -> list[str]:
    if not user_skills:
        return []

    user_text = " ".join(user_skills)

    sentence_model = get_sentence_model()
    user_embed = sentence_model.encode(user_text, convert_to_tensor=True)

    precomputed_role_embeddings = get_precomputed_role_embeddings()
    cosine_scores = _cos_sim(user_embed, precomputed_role_embeddings)[0]
    scores = cosine_scores.tolist()

    roles_scores = []
    for i in range(len(roles_list_global)):
        roles_scores.append((roles_list_global[i], scores[i]))

    roles_scores.sort(key=lambda x: x[1], reverse=True)
    return [role for role, score in roles_scores[:5]]

# ---- Display-name map so the UI shows "Scikit-learn" not "scikitlearn" ----
SKILL_DISPLAY = {
    "cpp": "C++", "csharp": "C#", "javascript": "JavaScript", "typescript": "TypeScript",
    "nodejs": "Node.js", "expressjs": "Express.js", "nextjs": "Next.js", "vuejs": "Vue.js",
    "reactnative": "React Native", "springboot": "Spring Boot", "dotnet": ".NET",
    "fastapi": "FastAPI", "restapi": "REST API", "graphql": "GraphQL", "grpc": "gRPC",
    "websockets": "WebSockets", "html": "HTML", "css": "CSS", "sql": "SQL",
    "postgresql": "PostgreSQL", "mysql": "MySQL", "mongodb": "MongoDB", "sqlite": "SQLite",
    "dynamodb": "DynamoDB", "influxdb": "InfluxDB", "scikitlearn": "Scikit-learn",
    "xgboost": "XGBoost", "lightgbm": "LightGBM", "catboost": "CatBoost", "mlflow": "MLflow",
    "tensorflow": "TensorFlow", "pytorch": "PyTorch", "huggingface": "Hugging Face",
    "openaiai": "OpenAI", "langchain": "LangChain", "langraph": "LangGraph", "lanserve": "LangServe",
    "spacy": "spaCy", "nltk": "NLTK", "opencv": "OpenCV", "yolo": "YOLO", "paddleocr": "PaddleOCR",
    "onnx": "ONNX", "tfserving": "TF Serving", "bentoml": "BentoML", "sagemaker": "SageMaker",
    "vertexai": "Vertex AI", "powerbi": "Power BI", "aws": "AWS", "gcp": "GCP", "azure": "Azure",
    "s3": "S3", "ec2": "EC2", "threejs": "Three.js", "web3js": "Web3.js",
    "ethersjs": "Ethers.js", "ipfs": "IPFS", "unrealengine": "Unreal Engine", "jwt": "JWT",
    "oauth2": "OAuth2", "saml": "SAML", "ssltls": "SSL/TLS", "owasp": "OWASP", "rag": "RAG",
    "pyg": "PyG", "ann": "ANN", "cnn": "CNN", "rnn": "RNN", "lstm": "LSTM", "bert": "BERT",
    "gpt": "GPT", "t5": "T5", "peft": "PEFT", "lora": "LoRA", "qlora": "QLoRA", "ios": "iOS",
    "android": "Android", "firebase": "Firebase", "cloudflare": "Cloudflare",
}

def prettify_skill(skill: str) -> str:
    if skill in SKILL_DISPLAY:
        return SKILL_DISPLAY[skill]
    if len(skill) <= 3:
        return skill.upper()
    return skill.capitalize()


def compute_resume_score(skills, roles) -> int:
    """Legacy count-only score. Kept for backward compatibility; new code should use
    score_resume() which reads the actual resume content."""
    skill_points = min(len(skills) * 3, 55)
    role_points = min(len(roles) * 5, 25)
    return max(35, min(20 + skill_points + role_points, 99))


# ---- Content-aware resume scoring -----------------------------------------
# The old score just counted skills/roles, so two very different resumes with the
# same keyword count scored identically. This version reads the resume text and
# grades the things recruiters/ATS actually look at: quantified impact, strong
# action verbs, section + contact completeness, length, and skill/role breadth.
# It's deterministic (fast, free, explainable) and degrades gracefully on thin text.

_ACTION_VERBS = {
    "led", "built", "designed", "developed", "created", "launched", "implemented",
    "managed", "improved", "increased", "reduced", "optimized", "automated",
    "delivered", "achieved", "spearheaded", "architected", "engineered", "drove",
    "grew", "scaled", "streamlined", "deployed", "migrated", "established",
    "initiated", "mentored", "analyzed", "researched", "produced", "negotiated",
    "coordinated", "executed", "boosted", "cut", "saved", "generated",
    "transformed", "accelerated", "resolved", "trained", "supervised", "founded",
    "owned", "shipped", "championed",
}
_WEAK_PHRASES = (
    "responsible for", "duties included", "worked on", "helped with",
    "involved in", "tasked with", "responsibilities included",
)
# A line "quantifies impact" if it ties a number to a result (%, currency, scale
# words, or a 3+ digit figure).
_IMPACT_HINT_RE = re.compile(
    r'(\d+\s?%|[₹$]\s?\d|\busd\b|\binr\b|'
    r'\b\d+\s?(?:k|m|x|cr|lakh|lakhs|crore|users?|customers?|clients?|projects?|'
    r'people|members?|hours?|days?|weeks?|months?|years?)\b|\b\d{3,}\b)',
    re.I,
)

_SECTION_PATTERNS = {
    "experience": r'\b(experience|employment|work history|professional background)\b',
    "education": r'\b(education|bachelor|master|b\.?tech|m\.?tech|b\.?sc|university|college|degree|diploma)\b',
    "skills": r'\b(skills|technologies|technical|tools|competenc)\b',
    "summary/projects": r'\b(projects?|portfolio|summary|objective|profile|about me)\b',
    "certifications": r'\b(certification|certificate|licen[cs]e|accredit)\b',
}


def _clamp(v, lo=0.0, hi=100.0):
    return max(lo, min(v, hi))


def score_resume(resume_text: str, skills, roles) -> dict:
    """Return {score, breakdown, suggestions, strengths}.

    score: 0-100 overall readiness.
    breakdown: per-dimension 0-100 sub-scores (so the number isn't a black box).
    suggestions: concrete, resume-specific improvement tips (weakest areas first).
    strengths: what's already working.
    """
    text = resume_text or ""
    skills = skills or []
    roles = roles or []
    low = text.lower()

    # Substantive lines (skip headers/short fragments).
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 25]
    total_lines = len(lines)

    # 1) Impact & quantification
    quantified = sum(1 for ln in lines if _IMPACT_HINT_RE.search(ln))
    quant_ratio = (quantified / total_lines) if total_lines else 0.0
    impact = _clamp(min(quantified / 6.0, 1.0) * 70 + min(quant_ratio * 2.0, 1.0) * 30)

    # 2) Action verbs vs. weak phrases
    action_hits = 0
    for ln in lines:
        head = re.sub(r'^[\-•‣◦\*·\s]+', '', ln).split()
        if head and head[0].lower().rstrip(",.:;") in _ACTION_VERBS:
            action_hits += 1
    weak_hits = sum(low.count(p) for p in _WEAK_PHRASES)
    action = _clamp(min(action_hits / 6.0, 1.0) * 100 - weak_hits * 8)

    # 3) Length / readability
    words = len(text.split())
    if words < 150:
        length = words / 150 * 55
    elif words < 300:
        length = 55 + (words - 150) / 150 * 35
    elif words <= 900:
        length = 100
    elif words <= 1300:
        length = 90
    else:
        length = 70
    length = _clamp(length)

    # 4) Completeness: sections + contact details
    missing_sections = [name for name, pat in _SECTION_PATTERNS.items() if not re.search(pat, low)]
    sections_found = len(_SECTION_PATTERNS) - len(missing_sections)
    has_email = bool(_EMAIL_RE.search(text))
    has_phone = any(8 <= len(re.sub(r'\D', '', c)) <= 15 for c in _PHONE_RE.findall(text))
    has_linkedin = bool(_LINKEDIN_RE.search(text))
    contact_found = sum([has_email, has_phone, has_linkedin])
    completeness = _clamp((sections_found / len(_SECTION_PATTERNS)) * 60 + (contact_found / 3) * 40)

    # 5) Breadth
    skills_breadth = _clamp(min(len(skills) / 12.0, 1.0) * 100)
    role_fit = _clamp(min(len(roles) / 4.0, 1.0) * 100)

    breakdown = {
        "Impact & metrics": round(impact),
        "Action verbs": round(action),
        "Completeness": round(completeness),
        "Skills breadth": round(skills_breadth),
        "Length": round(length),
        "Role fit": round(role_fit),
    }
    weights = {
        "Impact & metrics": 0.25,
        "Completeness": 0.20,
        "Skills breadth": 0.20,
        "Action verbs": 0.15,
        "Length": 0.10,
        "Role fit": 0.10,
    }
    raw = sum(breakdown[k] * w for k, w in weights.items())
    score = int(_clamp(round(raw), 20, 99))

    # ---- Resume-specific suggestions (weakest dimensions first) ----
    suggestions: list[str] = []
    if impact < 70:
        suggestions.append(
            f"Quantify your impact — only {quantified} line(s) include a number or metric. "
            "Add results like “reduced costs 30%” or “handled 200+ clients/month.”"
        )
    if missing_sections:
        suggestions.append(f"Add the missing section(s): {', '.join(missing_sections[:3])}.")
    if contact_found < 3:
        need = [n for n, ok in (("email", has_email), ("phone", has_phone), ("LinkedIn", has_linkedin)) if not ok]
        if need:
            suggestions.append(f"Add your {', '.join(need)} so recruiters can reach you.")
    if action < 70:
        if weak_hits:
            suggestions.append(
                "Replace weak phrases like “responsible for” with strong action verbs "
                "(Led, Built, Designed, Improved)."
            )
        else:
            suggestions.append("Start each bullet with a strong action verb (Led, Built, Designed) to sound results-driven.")
    if words < 200:
        suggestions.append(f"Your resume looks thin (~{words} words). Expand on your projects, responsibilities, and outcomes.")
    elif words > 1100:
        suggestions.append(f"It’s quite long (~{words} words) — tighten toward 1–2 pages and keep only the most relevant points.")
    if skills_breadth < 60:
        suggestions.append("List more concrete skills/tools relevant to your target roles.")

    suggestions = suggestions[:4]
    if not suggestions:
        suggestions = ["Strong resume — tailor the keywords to each specific job description before applying."]

    # ---- Strengths (what's already working) ----
    strengths: list[str] = []
    if impact >= 70:
        strengths.append("Achievements are backed by concrete numbers and metrics.")
    if action >= 70:
        strengths.append("Bullets lead with strong, results-oriented action verbs.")
    if completeness >= 80:
        strengths.append("All key sections and contact details are present.")
    if skills_breadth >= 70:
        strengths.append(f"Broad skill set — {len(skills)} skills detected.")
    if 300 <= words <= 900:
        strengths.append("Well-balanced length and level of detail.")
    if not strengths and skills:
        strengths = [f"Relevant skill: {s}" for s in skills[:3]]

    return {"score": score, "breakdown": breakdown, "suggestions": suggestions, "strengths": strengths}


def extract_resume_details(pdf_path: str):
    logger.info(f"Extracting skills from {pdf_path}")
    extracted_skills = skills_extraction(pdf_path)

    logger.info(f"Extracting roles from skills")
    matched_roles = roles_score(extracted_skills)

    pretty_skills = [prettify_skill(s) for s in extracted_skills]
    score = compute_resume_score(pretty_skills, matched_roles)
    return pretty_skills, matched_roles, score


def analyze_resume(pdf_path: str) -> dict:
    """Domain-agnostic resume analysis. Tries the LLM extractor first (works for
    ANY field — healthcare, finance, design, etc.), then falls back to the legacy
    tech keyword matcher so behaviour never regresses if the LLM is down.

    Returns a dict: {skills, roles, score, breakdown, suggestions, strengths,
    resume_text, domain, seniority}. The PDF text is extracted once here and
    reused for the vectorstore/email later.
    """
    resume_text = extract_resume_text(pdf_path)

    skills: list[str] = []
    roles: list[str] = []
    domain = ""
    seniority = ""

    # 1) LLM-first (domain-agnostic).
    try:
        from services.llm_extract_service import extract_profile_llm
        profile = extract_profile_llm(resume_text)
    except Exception as e:
        logger.warning(f"LLM extractor errored, will fall back: {str(e)}")
        profile = None

    if profile and profile.get("skills"):
        skills = profile["skills"]
        roles = profile.get("roles") or []
        domain = profile.get("domain", "")
        seniority = profile.get("seniority", "")
        logger.info("Resume analyzed via LLM extractor.")

    # 2) Keyword fallback for skills (tech taxonomy) when the LLM gave us nothing.
    if not skills:
        logger.info("Falling back to keyword skill matcher.")
        kw_skills, kw_roles, _ = extract_resume_details(pdf_path)
        skills = kw_skills
        if not roles:
            roles = kw_roles

    # 3) If we have skills but no roles yet, derive roles from the keyword model.
    if skills and not roles:
        try:
            raw = match_skills_in_text(clean_resume_text(resume_text), use_semantic=False)
            roles = roles_score(raw)
        except Exception as e:
            logger.warning(f"Role derivation fallback failed: {str(e)}")

    scored = score_resume(resume_text, skills, roles)
    return {
        "skills": skills,
        "roles": roles,
        "score": scored["score"],
        "breakdown": scored["breakdown"],
        "suggestions": scored["suggestions"],
        "strengths": scored["strengths"],
        "resume_text": resume_text,
        "domain": domain,
        "seniority": seniority,
    }

def build_vectorstore_bg(pdf_path: str, session_id: str):
    try:
        logger.info(f"Building vectorstore from document in background...")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(splits, get_hf_embeddings())
        set_vectorstore(session_id, vectorstore)
        logger.info("Vectorstore build complete.")
    except Exception as e:
        logger.error(f"Failed to build vectorstore: {str(e)}")
    finally:
        # Clean up temp file
        try:
            os.unlink(pdf_path)
            logger.info(f"Cleaned up temp file: {pdf_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp file {pdf_path}: {str(e)}")
