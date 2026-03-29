from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_SOURCE_DIR = Path(os.getenv("DATA_SOURCE_DIR", ROOT_DIR / "data_source"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", DATA_SOURCE_DIR / "processed"))

DOCUMENTS_PATH = Path(os.getenv("DOCUMENTS_PATH", PROCESSED_DIR / "documents.jsonl"))
TICKETS_PATH = Path(os.getenv("TICKETS_PATH", PROCESSED_DIR / "tickets.jsonl"))

COLLECTION_NAME = os.getenv("COLLECTION_NAME", "internal_support_copilot")

INCLUDE_TICKETS = os.getenv("INCLUDE_TICKETS", "true").lower() == "true"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "8"))
FINAL_TOP_K = int(os.getenv("FINAL_TOP_K", "4"))

# Cross-encoder rerank
USE_CROSS_ENCODER = os.getenv("USE_CROSS_ENCODER", "true").lower() == "true"
CROSS_ENCODER_MODEL_NAME = os.getenv(
    "CROSS_ENCODER_MODEL_NAME",
    "cross-encoder/ms-marco-MiniLM-L6-v2",
)
CROSS_ENCODER_TOP_K = int(os.getenv("CROSS_ENCODER_TOP_K", "5"))
CROSS_ENCODER_BATCH_SIZE = int(os.getenv("CROSS_ENCODER_BATCH_SIZE", "16"))

QDRANT_MODE = os.getenv("QDRANT_MODE", "server")  # server | local
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "internal_support_copilot_qdrant")

QDRANT_VECTOR_NAME = os.getenv("QDRANT_VECTOR_NAME", "dense")
QDRANT_SPARSE_VECTOR_NAME = os.getenv("QDRANT_SPARSE_VECTOR_NAME", "sparse")
QDRANT_PREFER_GRPC = os.getenv("QDRANT_PREFER_GRPC", "true").lower() == "true"
USE_QDRANT_HYBRID = os.getenv("USE_QDRANT_HYBRID", "true").lower() == "true"