from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from src.core.settings import (
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_PREFER_GRPC,
    QDRANT_SPARSE_VECTOR_NAME,
    QDRANT_URL,
    QDRANT_VECTOR_NAME,
)
from src.data.enterprise_support_documents import build_enterprise_support_documents
from src.data.enterprise_support_loader import load_enterprise_support_dataset
from src.rag.embedding.embeddings import get_embedding_model
from src.rag.indexing.qdrant_store import _get_sparse_embeddings

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ENTERPRISE_DATA_DIR = PROJECT_ROOT / "data" / "sample_enterprise_support"

ENTERPRISE_SOURCE = "enterprise_support"
FILTER_FIELDS = ("source_type", "customer_id", "ticket_id", "service_id", "product_id")
RRF_K = 60.0
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")
STOPWORDS = {
    "about",
    "after",
    "also",
    "and",
    "are",
    "can",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "should",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


@dataclass(frozen=True)
class HybridSearchHit:
    document: Document
    rank: int
    score: float
    channel: str


def retrieve_enterprise_hybrid_documents(
    query: str,
    *,
    top_k: int = 5,
    filters: dict[str, Any] | None = None,
    dense_limit: int | None = None,
    sparse_limit: int | None = None,
    data_dir: Path = DEFAULT_ENTERPRISE_DATA_DIR,
) -> dict[str, Any]:
    normalized_filters = normalize_enterprise_filters(filters)
    candidate_limit = max(top_k * 4, 12)
    dense_limit = dense_limit or candidate_limit
    sparse_limit = sparse_limit or candidate_limit

    dense_hits: list[HybridSearchHit] = []
    sparse_hits: list[HybridSearchHit] = []
    dense_error = ""
    sparse_error = ""
    sparse_mode = "qdrant_sparse"

    try:
        dense_hits = _search_qdrant(
            query=query,
            mode=RetrievalMode.DENSE,
            limit=dense_limit,
            filters=normalized_filters,
            channel="dense",
        )
    except Exception as exc:
        dense_error = _safe_error(exc)

    try:
        sparse_hits = _search_qdrant(
            query=query,
            mode=RetrievalMode.SPARSE,
            limit=sparse_limit,
            filters=normalized_filters,
            channel="sparse",
        )
    except Exception as exc:
        sparse_error = _safe_error(exc)
        sparse_mode = "local_lexical_fallback"
        sparse_hits = _lexical_search(
            query=query,
            limit=sparse_limit,
            filters=normalized_filters,
            data_dir=data_dir,
        )

    if not sparse_hits and sparse_mode == "qdrant_sparse":
        sparse_mode = "local_lexical_fallback"
        sparse_hits = _lexical_search(
            query=query,
            limit=sparse_limit,
            filters=normalized_filters,
            data_dir=data_dir,
        )

    fused_documents, debug = fuse_enterprise_search_results(
        dense_hits=dense_hits,
        sparse_or_lexical_hits=sparse_hits,
        top_k=top_k,
    )

    return {
        "documents": fused_documents,
        "debug": debug,
        "stats": {
            "mode": "enterprise_hybrid",
            "top_k": top_k,
            "dense_count": len(dense_hits),
            "sparse_or_lexical_count": len(sparse_hits),
            "fused_count": len(fused_documents),
            "sparse_mode": sparse_mode,
            "filters": normalized_filters,
            "dense_error": dense_error,
            "sparse_error": sparse_error,
            "qdrant_filter_applied": bool(normalized_filters),
        },
    }


def normalize_enterprise_filters(filters: dict[str, Any] | None) -> dict[str, list[str]]:
    normalized: dict[str, list[str]] = {}
    for field in FILTER_FIELDS:
        raw_value = (filters or {}).get(field)
        values = _normalize_filter_values(raw_value)
        if values:
            normalized[field] = values
    return normalized


def build_enterprise_qdrant_filter(filters: dict[str, Any] | None = None) -> models.Filter:
    normalized_filters = normalize_enterprise_filters(filters)
    conditions: list[models.FieldCondition] = [
        models.FieldCondition(
            key="metadata.source",
            match=models.MatchValue(value=ENTERPRISE_SOURCE),
        )
    ]

    for field, values in normalized_filters.items():
        match: models.MatchAny | models.MatchValue
        if len(values) == 1:
            match = models.MatchValue(value=values[0])
        else:
            match = models.MatchAny(any=values)
        conditions.append(models.FieldCondition(key=f"metadata.{field}", match=match))

    return models.Filter(must=conditions)


def fuse_enterprise_search_results(
    *,
    dense_hits: list[HybridSearchHit],
    sparse_or_lexical_hits: list[HybridSearchHit],
    top_k: int,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> tuple[list[Document], list[dict[str, Any]]]:
    candidates: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for hit in dense_hits:
        key = _document_identity(hit.document)
        state = candidates.setdefault(key, {"document": hit.document, "channels": set()})
        if key not in order:
            order.append(key)
        state["dense_rank"] = hit.rank
        state["dense_score"] = hit.score
        state["channels"].add("dense")

    for hit in sparse_or_lexical_hits:
        key = _document_identity(hit.document)
        state = candidates.setdefault(key, {"document": hit.document, "channels": set()})
        if key not in order:
            order.append(key)
        score_key = "lexical_score" if hit.channel == "lexical" else "sparse_score"
        rank_key = "lexical_rank" if hit.channel == "lexical" else "sparse_rank"
        state[rank_key] = hit.rank
        state[score_key] = hit.score
        state["sparse_or_lexical_rank"] = hit.rank
        state["sparse_or_lexical_score"] = hit.score
        state["sparse_or_lexical_channel"] = hit.channel
        state["channels"].add(hit.channel)

    scored: list[tuple[float, int, str, dict[str, Any]]] = []
    for order_index, key in enumerate(order):
        state = candidates[key]
        fused_score = 0.0
        dense_rank = state.get("dense_rank")
        sparse_rank = state.get("sparse_or_lexical_rank")
        if dense_rank:
            fused_score += dense_weight / (RRF_K + float(dense_rank))
        if sparse_rank:
            fused_score += sparse_weight / (RRF_K + float(sparse_rank))
        scored.append((fused_score, -order_index, key, state))

    scored.sort(reverse=True)

    fused_documents: list[Document] = []
    debug: list[dict[str, Any]] = []
    for rank, (fused_score, _, key, state) in enumerate(scored[:top_k], start=1):
        document = _copy_document_with_debug(
            state["document"],
            rank=rank,
            fused_score=fused_score,
            state=state,
        )
        fused_documents.append(document)
        debug.append(
            {
                "rank": rank,
                "id": key,
                "dense_score": state.get("dense_score"),
                "sparse_score": state.get("sparse_score"),
                "lexical_score": state.get("lexical_score"),
                "sparse_lexical_score": state.get("sparse_or_lexical_score"),
                "fused_score": fused_score,
                "channels": sorted(state["channels"]),
                "matched_metadata": _matched_metadata(document.metadata),
            }
        )

    return fused_documents, debug


def metadata_matches_filter(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    normalized_filters = normalize_enterprise_filters(filters)
    for field, allowed_values in normalized_filters.items():
        metadata_value = str(metadata.get(field) or "").strip()
        if metadata_value not in allowed_values:
            return False
    return True


def _search_qdrant(
    *,
    query: str,
    mode: RetrievalMode,
    limit: int,
    filters: dict[str, list[str]],
    channel: str,
) -> list[HybridSearchHit]:
    store = _load_qdrant_store(mode)
    qdrant_filter = build_enterprise_qdrant_filter(filters)
    scored_documents = store.similarity_search_with_score(
        query,
        k=limit,
        filter=qdrant_filter,
    )
    return [
        HybridSearchHit(document=document, rank=rank, score=float(score), channel=channel)
        for rank, (document, score) in enumerate(scored_documents, start=1)
    ]


def _load_qdrant_store(mode: RetrievalMode) -> QdrantVectorStore:
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
        prefer_grpc=QDRANT_PREFER_GRPC,
    )

    if mode == RetrievalMode.DENSE:
        return QdrantVectorStore(
            client=client,
            collection_name=QDRANT_COLLECTION_NAME,
            embedding=get_embedding_model(),
            retrieval_mode=RetrievalMode.DENSE,
            vector_name=QDRANT_VECTOR_NAME,
        )

    return QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        sparse_embedding=_get_sparse_embeddings(),
        retrieval_mode=RetrievalMode.SPARSE,
        sparse_vector_name=QDRANT_SPARSE_VECTOR_NAME,
    )


def _lexical_search(
    *,
    query: str,
    limit: int,
    filters: dict[str, list[str]],
    data_dir: Path,
) -> list[HybridSearchHit]:
    query_tokens = _tokenize(query)
    query_lower = query.lower()
    scored: list[tuple[float, str, Document]] = []

    for document in _load_enterprise_documents(str(data_dir)):
        metadata = document.metadata or {}
        if not metadata_matches_filter(metadata, filters):
            continue

        metadata_values = _metadata_values(metadata)
        haystack = " ".join([document.page_content, " ".join(metadata_values)]).lower()
        document_tokens = _tokenize(haystack)
        overlap = query_tokens.intersection(document_tokens)
        score = float(len(overlap))

        title = str(metadata.get("title") or "").lower()
        if title and any(token in title for token in query_tokens):
            score += 2.0

        for value in metadata_values:
            normalized_value = value.lower()
            if normalized_value and normalized_value in query_lower:
                score += 4.0

        if score > 0:
            scored.append((score, _document_identity(document), document))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [
        HybridSearchHit(document=document, rank=rank, score=score, channel="lexical")
        for rank, (score, _, document) in enumerate(scored[:limit], start=1)
    ]


@lru_cache(maxsize=4)
def _load_enterprise_documents(data_dir: str) -> tuple[Document, ...]:
    dataset = load_enterprise_support_dataset(Path(data_dir))
    documents = build_enterprise_support_documents(dataset)
    langchain_documents = []
    for document in documents:
        metadata = dict(document.get("metadata") or {})
        metadata["doc_id"] = str(document.get("id") or "")
        metadata["source_chunk_id"] = str(document.get("id") or "")
        langchain_documents.append(
            Document(
                page_content=str(document.get("text") or ""),
                metadata=metadata,
            )
        )
    return tuple(langchain_documents)


def _copy_document_with_debug(
    document: Document,
    *,
    rank: int,
    fused_score: float,
    state: dict[str, Any],
) -> Document:
    metadata = dict(document.metadata or {})
    retrieval_debug = {
        "rank": rank,
        "dense_rank": state.get("dense_rank"),
        "sparse_rank": state.get("sparse_rank"),
        "lexical_rank": state.get("lexical_rank"),
        "dense_score": state.get("dense_score"),
        "sparse_score": state.get("sparse_score"),
        "lexical_score": state.get("lexical_score"),
        "sparse_lexical_score": state.get("sparse_or_lexical_score"),
        "fused_score": fused_score,
        "channels": sorted(state["channels"]),
        "matched_metadata": _matched_metadata(metadata),
    }
    metadata.update(
        {
            "dense_score": state.get("dense_score"),
            "sparse_score": state.get("sparse_score"),
            "lexical_score": state.get("lexical_score"),
            "sparse_lexical_score": state.get("sparse_or_lexical_score"),
            "fused_score": fused_score,
            "retrieval_rank": rank,
            "retrieval_channels": sorted(state["channels"]),
            "retrieval_debug": retrieval_debug,
            "matched_metadata": retrieval_debug["matched_metadata"],
        }
    )
    return Document(page_content=document.page_content, metadata=metadata)


def _document_identity(document: Document) -> str:
    metadata = document.metadata or {}
    source_type = str(metadata.get("source_type") or "").strip()
    entity_id = str(metadata.get("entity_id") or "").strip()
    if source_type and entity_id:
        return f"{source_type}:{entity_id}"

    for field in (
        "doc_id",
        "source_chunk_id",
        "ticket_id",
        "customer_id",
        "service_id",
        "product_id",
        "policy_id",
        "risk_event_id",
        "issue_id",
    ):
        value = str(metadata.get(field) or "").strip()
        if value:
            return value

    return document.page_content[:120]


def _metadata_values(metadata: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for field in (
        "entity_id",
        "source_type",
        "title",
        "customer_id",
        "account_id",
        "ticket_id",
        "product_id",
        "service_id",
        "policy_id",
        "risk_event_id",
        "issue_id",
    ):
        value = str(metadata.get(field) or "").strip()
        if value:
            values.append(value)
    return values


def _matched_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "source_type",
        "entity_id",
        "customer_id",
        "ticket_id",
        "service_id",
        "product_id",
        "policy_id",
        "title",
    )
    return {key: metadata[key] for key in keys if metadata.get(key)}


def _normalize_filter_values(raw_value: Any) -> list[str]:
    if raw_value in (None, "", []):
        return []
    if isinstance(raw_value, (list, tuple, set)):
        raw_values = raw_value
    else:
        raw_values = [raw_value]
    return [str(value).strip() for value in raw_values if str(value).strip()]


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall(str(text or "").lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _safe_error(exc: Exception) -> str:
    text = str(exc).strip()
    if len(text) > 240:
        text = text[:237] + "..."
    return text
