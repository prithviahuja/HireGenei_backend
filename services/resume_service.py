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


def skills_extraction(pdf_path: str) -> list[str]:
    with pdfplumber.open(pdf_path) as pdf:
        raw_text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    raw_text = clean_resume_text(raw_text)

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

    import time
    start_time = time.time()
    # We only encode the dynamic words heavily now, bypassing ~200 redundant vector calcs
    sentence_model = get_sentence_model()
    text_embeddings = sentence_model.encode(list(words), convert_to_tensor=True)
    encode_time = time.time() - start_time
    logger.info(f"Encoded {len(words)} dynamic text words in {encode_time:.2f}s")

    semantic_matches = set()
    threshold = 0.75
    precomputed_skill_embeddings = get_precomputed_skill_embeddings()
    for i, vec in enumerate(text_embeddings):
        scores = _cos_sim(vec, precomputed_skill_embeddings)[0]
        best_idx = scores.argmax().item()
        if float(scores[best_idx]) >= threshold:
            semantic_matches.add(SKILLS_LIST[best_idx])

    final_skills = list(exact_matches.union(fuzzy_matches).union(semantic_matches))
    logger.info(f"Skill extraction done: {len(exact_matches)} exact, {len(fuzzy_matches)} fuzzy, {len(semantic_matches)} semantic matches.")
    return final_skills

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
    """Simple deterministic 0-100 readiness score."""
    skill_points = min(len(skills) * 3, 55)
    role_points = min(len(roles) * 5, 25)
    return max(35, min(20 + skill_points + role_points, 99))


def extract_resume_details(pdf_path: str):
    logger.info(f"Extracting skills from {pdf_path}")
    extracted_skills = skills_extraction(pdf_path)

    logger.info(f"Extracting roles from skills")
    matched_roles = roles_score(extracted_skills)

    pretty_skills = [prettify_skill(s) for s in extracted_skills]
    score = compute_resume_score(pretty_skills, matched_roles)
    return pretty_skills, matched_roles, score

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
