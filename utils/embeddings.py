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

    import time
    for attempt in range(3):
        response = requests.post(
            "https://router.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
            headers={"Authorization": f"Bearer {settings.HF_TOKEN}"},
            json={"inputs": texts}
        )
        if response.status_code == 503:
            wait = response.json().get("estimated_time", 20)
            time.sleep(min(wait, 30))
            continue
        if response.status_code != 200:
            raise RuntimeError(f"HF embedding API request failed ({response.status_code}): {response.text}")
        
        import torch
        embeddings = torch.tensor(response.json())
        if isinstance(text, str):
            return embeddings[0]  # Return single embedding for single input
        return embeddings
    
    raise RuntimeError("HF model failed to wake up after 3 attempts")


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
                return embeddings  # already a tensor
            return embeddings.tolist()  # convert to list if not tensor
    return MockSentenceModel()
