import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import logging
import requests
from core.config import settings

logging.info("Embeddings module loaded; using HF API.")

_HF_EMBEDDINGS = None


def _normalize_embedding_response(result):
    if isinstance(result, dict):
        if "error" in result:
            raise RuntimeError(f"HF embedding API returned error: {result['error']}")
        if "embedding" in result:
            return result["embedding"]
        if "data" in result and isinstance(result["data"], list):
            return [item.get("embedding", item) for item in result["data"]]

    if isinstance(result, list) and len(result) > 0:
        if all(isinstance(item, dict) for item in result):
            if "embedding" in result[0]:
                return [item["embedding"] for item in result]
            if "data" in result[0] and isinstance(result[0]["data"], list):
                return [item["data"][0].get("embedding", item["data"][0]) for item in result]

    return result


def get_embedding(text):
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text

    response = requests.post(
        "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
        headers={"Authorization": f"Bearer {settings.HF_TOKEN}"},
        json={"inputs": texts},
        timeout=60
    )
    result = response.json()

    if response.status_code != 200:
        raise RuntimeError(f"HF embedding API request failed ({response.status_code}): {result}")

    normalized = _normalize_embedding_response(result)
    if isinstance(text, str):
        if isinstance(normalized, list) and len(normalized) == 1 and isinstance(normalized[0], (list, tuple)):
            return normalized[0]
        return normalized

    return normalized


def _load_embedding_models():
    global _HF_EMBEDDINGS
    if _HF_EMBEDDINGS is None:
        from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings

        logging.info("Initializing HF Inference API embeddings...")
        _HF_EMBEDDINGS = HuggingFaceInferenceAPIEmbeddings(
            api_key=settings.HF_TOKEN,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        logging.info("HF Inference API embeddings initialized successfully.")

    return _HF_EMBEDDINGS


def get_hf_embeddings():
    return _load_embedding_models()


def get_sentence_model():
    # For compatibility, return a mock object that has encode method
    class MockSentenceModel:
        def encode(self, texts, convert_to_tensor=False):
            embeddings = get_embedding(texts)
            if convert_to_tensor:
                import torch
                if isinstance(embeddings, dict):
                    raise RuntimeError("Unexpected embedding response format: dict")
                return torch.tensor(embeddings, dtype=torch.float32)
            return embeddings
    return MockSentenceModel()
