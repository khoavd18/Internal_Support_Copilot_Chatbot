from __future__ import annotations

import logging

from qdrant_client import models

from src.core.observability import observe_duration
from src.core.settings import CROSS_ENCODER_TOP_K, USE_CROSS_ENCODER
from src.rag.indexing.store_manager import get_vector_store
from src.rag.ingest.hierarchical_loader import PARENT_PATH, load_parent_map
from src.rag.retrieval.cross_encoder import rerank_with_cross_encoder
from src.rag.retrieval.parent_merge import merge_leaf_results_with_parents
from src.rag.retrieval.query_analyzer import extract_keyword_groups
from src.rag.retrieval.reranker import rerank_documents

logger = logging.getLogger(__name__)


def _run_rerank_stage(
    *,
    stage: str,
    strategy: str,
    docs: list,
    top_k: int,
    operation,
) -> list:
    with observe_duration(
        "rag.rerank",
        metric_name="rag.rerank.duration_ms",
        metric_attributes={
            "stage": stage,
            "strategy": strategy,
        },
        span_attributes={
            "stage": stage,
            "strategy": strategy,
            "docs_in": len(docs),
            "top_k": top_k,
        },
    ) as observation:
        reranked = operation()
        observation.set_attribute("docs_out", len(reranked))
        return reranked


def _load_parent_map_if_available() -> dict[str, dict]:
    if PARENT_PATH.exists():
        return load_parent_map()
    return {}


def _looks_like_leaf(doc) -> bool:
    metadata = doc.metadata or {}
    return (
        str(metadata.get("node_type") or "").lower() == "leaf"
        and bool(metadata.get("parent_id"))
    )


def _origin_key(doc) -> str:
    md = doc.metadata or {}
    return (
        md.get("origin_doc_id")
        or md.get("doc_id")
        or md.get("path")
        or md.get("url")
        or ""
    )


def _node_priority(doc) -> int:
    md = doc.metadata or {}
    node_type = str(md.get("node_type") or "").lower()

    if node_type == "parent_pack":
        return 3
    if node_type == "parent":
        return 2
    if node_type == "leaf":
        return 1
    return 0


def _final_dedupe_by_origin(docs: list, top_k: int) -> list:
    kept = []
    seen = {}

    for doc in docs:
        key = _origin_key(doc)
        if not key:
            kept.append(doc)
            continue

        if key not in seen:
            seen[key] = doc
            continue

        old_doc = seen[key]
        if _node_priority(doc) > _node_priority(old_doc):
            seen[key] = doc

    used_keys = set()
    for doc in docs:
        key = _origin_key(doc)
        if not key:
            if len(kept) < top_k:
                kept.append(doc)
            continue

        if key in used_keys:
            continue

        chosen = seen[key]
        if chosen is doc:
            kept.append(doc)
            used_keys.add(key)

        if len(kept) >= top_k:
            break

    return kept[:top_k]


def _prefilter_candidates_for_auth_query(query: str, docs: list, top_k: int) -> list:
    keyword_groups = extract_keyword_groups(query)
    intent_labels = set(keyword_groups.get("intent_labels", []))

    if "github_authentication" not in intent_labels:
        return docs

    filtered = []
    for doc in docs:
        md = doc.metadata or {}
        source = str(md.get("source") or "").lower()
        title = str(md.get("title") or "").lower()
        path = str(md.get("path") or "").lower()
        content_head = doc.page_content[:1200].lower()

        auth_hit = (
            "passkey" in title
            or "passkey" in path
            or "authentication" in title
            or "authentication" in path
            or "sign in" in title
            or "login" in title
            or "passkey" in content_head
            or "authentication" in content_head
        )

        if source == "github_docs" and auth_hit:
            filtered.append(doc)

    if len(filtered) >= max(top_k, 4):
        return filtered

    return docs


def retrieve_documents(
    query: str,
    top_k: int = 4,
    rebuild: bool = False,
    qdrant_filter: models.Filter | None = None,
):
    """
    Retrieval flow:
    1) Qdrant hybrid similarity search
    2) light intent prefilter when useful
    3) heuristic rerank
    4) optional parent/sibling merge
    5) heuristic rerank after merge
    6) optional cross-encoder rerank
    7) final dedupe by origin document
    """
    with observe_duration(
        "rag.retrieve",
        metric_name="rag.retrieval.duration_ms",
        metric_attributes={
            "top_k": top_k,
            "rebuild": rebuild,
            "has_filter": qdrant_filter is not None,
        },
        span_attributes={
            "top_k": top_k,
            "rebuild": rebuild,
            "has_filter": qdrant_filter is not None,
            "query_length": len(query or ""),
        },
    ) as observation:
        vector_store = get_vector_store(rebuild=rebuild)

        candidate_k = max(top_k * 6, 18)
        rerank_k = max(top_k * 3, 8)
        ce_k = max(CROSS_ENCODER_TOP_K, top_k)
        observation.set_attribute("candidate_k", candidate_k)
        observation.set_attribute("rerank_k", rerank_k)
        observation.set_attribute("cross_encoder_top_k", ce_k)

        logger.info(
            "Retriever started",
            extra={
                "event": "retrieval.started",
                "query_length": len(query or ""),
                "top_k": top_k,
                "candidate_k": candidate_k,
                "rerank_k": rerank_k,
                "ce_k": ce_k,
                "rebuild": rebuild,
                "qdrant_filter_applied": qdrant_filter is not None,
                "cross_encoder_enabled": USE_CROSS_ENCODER,
            },
        )

        candidate_docs = vector_store.similarity_search(
            query,
            k=candidate_k,
            filter=qdrant_filter,
        )

        if not candidate_docs:
            observation.set_attribute("candidate_docs", 0)
            observation.set_attribute("final_docs", 0)
            logger.info(
                "Retriever completed with no candidates",
                extra={"event": "retrieval.completed", "candidate_docs": 0, "final_docs": 0},
            )
            return []

        candidate_docs = _prefilter_candidates_for_auth_query(
            query=query,
            docs=candidate_docs,
            top_k=top_k,
        )
        observation.set_attribute("candidate_docs", len(candidate_docs))

        stage1_docs = _run_rerank_stage(
            stage="stage1",
            strategy="heuristic",
            docs=candidate_docs,
            top_k=rerank_k,
            operation=lambda: rerank_documents(
                query=query,
                docs=candidate_docs,
                top_k=rerank_k,
            ),
        )

        if not stage1_docs:
            stage1_docs = candidate_docs[:rerank_k]
        observation.set_attribute("stage1_docs", len(stage1_docs))

        parent_map = _load_parent_map_if_available()

        if not parent_map or not any(_looks_like_leaf(doc) for doc in stage1_docs):
            observation.set_attribute("used_parent_merge", False)
            if USE_CROSS_ENCODER:
                final_docs = _run_rerank_stage(
                    stage="final",
                    strategy="cross_encoder",
                    docs=stage1_docs,
                    top_k=max(top_k * 2, top_k),
                    operation=lambda: rerank_with_cross_encoder(
                        query=query,
                        docs=stage1_docs,
                        top_k=max(top_k * 2, top_k),
                    ),
                )
                deduped = _final_dedupe_by_origin(final_docs, top_k=top_k)
                observation.set_attribute("used_cross_encoder", True)
                observation.set_attribute("final_docs", len(deduped))
                logger.info(
                    "Retriever completed without hierarchical merge",
                    extra={
                        "event": "retrieval.completed",
                        "candidate_docs": len(candidate_docs),
                        "stage1_docs": len(stage1_docs),
                        "merged_docs": len(stage1_docs),
                        "final_docs": len(deduped),
                        "used_cross_encoder": True,
                        "used_parent_merge": False,
                    },
                )
                return deduped

            final_docs = _run_rerank_stage(
                stage="final",
                strategy="heuristic",
                docs=stage1_docs,
                top_k=top_k,
                operation=lambda: rerank_documents(
                    query=query,
                    docs=stage1_docs,
                    top_k=top_k,
                ),
            )
            deduped = _final_dedupe_by_origin(final_docs, top_k=top_k)
            observation.set_attribute("used_cross_encoder", False)
            observation.set_attribute("final_docs", len(deduped))
            logger.info(
                "Retriever completed without hierarchical merge",
                extra={
                    "event": "retrieval.completed",
                    "candidate_docs": len(candidate_docs),
                    "stage1_docs": len(stage1_docs),
                    "merged_docs": len(stage1_docs),
                    "final_docs": len(deduped),
                    "used_cross_encoder": False,
                    "used_parent_merge": False,
                },
            )
            return deduped

        merged_docs = merge_leaf_results_with_parents(
            retrieved_docs=stage1_docs,
            parent_map=parent_map,
            min_children_to_merge=2,
            min_sibling_parents_to_merge=2,
            max_results=rerank_k,
        )
        observation.set_attribute("used_parent_merge", True)
        observation.set_attribute("merged_docs", len(merged_docs))

        stage3_docs = _run_rerank_stage(
            stage="stage3",
            strategy="heuristic",
            docs=merged_docs,
            top_k=ce_k,
            operation=lambda: rerank_documents(
                query=query,
                docs=merged_docs,
                top_k=ce_k,
            ),
        )

        if not stage3_docs:
            stage3_docs = merged_docs[:ce_k]
        observation.set_attribute("stage3_docs", len(stage3_docs))

        if USE_CROSS_ENCODER:
            final_docs = _run_rerank_stage(
                stage="final",
                strategy="cross_encoder",
                docs=stage3_docs,
                top_k=max(top_k * 2, top_k),
                operation=lambda: rerank_with_cross_encoder(
                    query=query,
                    docs=stage3_docs,
                    top_k=max(top_k * 2, top_k),
                ),
            )
            deduped = _final_dedupe_by_origin(final_docs, top_k=top_k)
            observation.set_attribute("used_cross_encoder", True)
            observation.set_attribute("final_docs", len(deduped))
            logger.info(
                "Retriever completed",
                extra={
                    "event": "retrieval.completed",
                    "candidate_docs": len(candidate_docs),
                    "stage1_docs": len(stage1_docs),
                    "merged_docs": len(merged_docs),
                    "stage3_docs": len(stage3_docs),
                    "final_docs": len(deduped),
                    "used_cross_encoder": True,
                    "used_parent_merge": True,
                },
            )
            return deduped

        deduped = _final_dedupe_by_origin(stage3_docs, top_k=top_k)
        observation.set_attribute("used_cross_encoder", False)
        observation.set_attribute("final_docs", len(deduped))
        logger.info(
            "Retriever completed",
            extra={
                "event": "retrieval.completed",
                "candidate_docs": len(candidate_docs),
                "stage1_docs": len(stage1_docs),
                "merged_docs": len(merged_docs),
                "stage3_docs": len(stage3_docs),
                "final_docs": len(deduped),
                "used_cross_encoder": False,
                "used_parent_merge": True,
            },
        )
        return deduped
