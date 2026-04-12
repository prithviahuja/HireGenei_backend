import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import logging
import requests
from core.config import settings

logging.info("Embeddings module loaded; using HF API.")

_HF_EMBEDDINGS = None


def get_embedding(text):
    if isinstance(text, str):
        texts = [text]
    else:
        texts = text
    response = requests.post(
        "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2",
        headers={"Authorization": f"Bearer {settings.HF_TOKEN}"},
        json={"inputs": texts}
    )
    result = response.json()
    if isinstance(text, str):
        return result[0] if isinstance(result, list) and len(result) > 0 else result
    return result


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
                return torch.tensor(embeddings)
            return embeddings
    return MockSentenceModel()
