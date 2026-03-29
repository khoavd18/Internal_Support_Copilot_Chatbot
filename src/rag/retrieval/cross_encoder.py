from __future__ import annotations

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


@lru_cache(maxsize=1)
def get_cross_encoder():
    """
    Lazy load cross-encoder.
    Nếu thư viện chưa có hoặc user tắt thì trả None.
    """
    if not USE_CROSS_ENCODER:
        print("[INFO] Cross-encoder disabled by settings.")
        return None

    if CrossEncoder is None:
        print("[WARN] sentence_transformers.CrossEncoder chưa import được.")
        print("[WARN] Nếu cần, cài: pip install sentence-transformers")
        return None

    device = get_device()

    print("=" * 80)
    print(f"[INFO] Loading cross-encoder: {CROSS_ENCODER_MODEL_NAME}")
    print(f"[INFO] Cross-encoder device: {device}")
    print(f"[INFO] Cross-encoder batch size: {CROSS_ENCODER_BATCH_SIZE}")
    print("=" * 80)

    model = CrossEncoder(
        CROSS_ENCODER_MODEL_NAME,
        device=device,
    )

    print("[DONE] Cross-encoder loaded successfully.")
    return model


def rerank_with_cross_encoder(query: str, docs: list, top_k: int) -> list:
    """
    Rerank docs bằng cross-encoder trên top candidates.
    Trả lại docs đã được sắp xếp lại theo ce_score giảm dần.
    """
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