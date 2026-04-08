import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import logging

logging.info("Embeddings module loaded; models will initialize lazily on first request.")

_HF_EMBEDDINGS = None
_SENTENCE_MODEL = None


def _load_embedding_models():
    global _HF_EMBEDDINGS, _SENTENCE_MODEL
    if _HF_EMBEDDINGS is None or _SENTENCE_MODEL is None:
        from sentence_transformers import SentenceTransformer
        from langchain_huggingface import HuggingFaceEmbeddings

        logging.info("Loading embedding models into memory...")
        _SENTENCE_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _HF_EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        logging.info("Embedding models initialized successfully.")

    return _HF_EMBEDDINGS, _SENTENCE_MODEL


def get_hf_embeddings():
    return _load_embedding_models()[0]


def get_sentence_model():
    return _load_embedding_models()[1]
