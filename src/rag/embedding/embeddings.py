from __future__ import annotations

import os
from typing import Optional

import torch

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
NORMALIZE_EMBEDDINGS = os.getenv("NORMALIZE_EMBEDDINGS", "true").lower() == "true"
HF_CACHE_DIR = os.getenv("HF_HOME", None)

_EMBEDDINGS = None


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_embedding_model():
    global _EMBEDDINGS

    if _EMBEDDINGS is not None:
        return _EMBEDDINGS

    device = get_device()

    print("=" * 80)
    print(f"[INFO] Loading embedding model: {MODEL_NAME}")
    print(f"[INFO] Embedding device: {device}")
    print(f"[INFO] Embedding batch size: {EMBEDDING_BATCH_SIZE}")
    print(f"[INFO] Normalize embeddings: {NORMALIZE_EMBEDDINGS}")
    print("=" * 80)

    _EMBEDDINGS = HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={
            "device": device,
        },
        encode_kwargs={
            "batch_size": EMBEDDING_BATCH_SIZE,
            "normalize_embeddings": NORMALIZE_EMBEDDINGS,
        },
        cache_folder=HF_CACHE_DIR,
    )

    # gắn thêm metadata để file khác log cho dễ
    setattr(_EMBEDDINGS, "_runtime_device", device)
    setattr(_EMBEDDINGS, "_runtime_model_name", MODEL_NAME)
    setattr(_EMBEDDINGS, "_runtime_batch_size", EMBEDDING_BATCH_SIZE)

    print("[DONE] Embedding model loaded successfully.")
    return _EMBEDDINGS