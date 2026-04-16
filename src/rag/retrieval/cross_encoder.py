from __future__ import annotations

import logging
from functools import lru_cache

try:
    from sentence_transformers import CrossEncoder
except Exception:
    CrossEncoder = None

from src.core.settings import (
    CROSS_ENCODER_BATCH_SIZE,
    CROSS_ENCODER_MODEL_NAME,
    USE_CROSS_ENCODER,
)
from src.rag.embedding.embeddings import get_device

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_cross_encoder():
    """Lazy load the optional cross-encoder."""
    if not USE_CROSS_ENCODER:
        logger.info(
            "Cross-encoder disabled by settings",
            extra={"event": "cross_encoder.disabled"},
        )
        return None

    if CrossEncoder is None:
        logger.warning(
            "CrossEncoder import is unavailable",
            extra={"event": "cross_encoder.unavailable"},
        )
        return None

    device = get_device()
    logger.info(
        "Loading cross-encoder",
        extra={
            "event": "cross_encoder.load.started",
            "model_name": CROSS_ENCODER_MODEL_NAME,
            "device": device,
            "batch_size": CROSS_ENCODER_BATCH_SIZE,
        },
    )

    model = CrossEncoder(
        CROSS_ENCODER_MODEL_NAME,
        device=device,
    )

    logger.info(
        "Cross-encoder loaded successfully",
        extra={
            "event": "cross_encoder.load.completed",
            "model_name": CROSS_ENCODER_MODEL_NAME,
        },
    )
    return model


def rerank_with_cross_encoder(query: str, docs: list, top_k: int) -> list:
    """Rerank documents with the cross-encoder over the final candidate set."""
    if not docs:
        return []

    model = get_cross_encoder()
    if model is None:
        return docs[:top_k]

    pairs = [[query, doc.page_content] for doc in docs]

    scores = model.predict(
        pairs,
        batch_size=CROSS_ENCODER_BATCH_SIZE,
        show_progress_bar=False,
    )

    scored = []
    for doc, score in zip(docs, scores):
        ce_score = float(score)
        doc.metadata["ce_score"] = ce_score
        scored.append((doc, ce_score))

    scored.sort(
        key=lambda x: (
            x[1],
            x[0].metadata.get("rerank_score", 0.0),
        ),
        reverse=True,
    )

    return [doc for doc, _ in scored[:top_k]]
