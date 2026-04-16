from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, Optional

from langchain_core.documents import Document
from qdrant_client import models

from src.pipeline import LocalRAGPipeline, build_pipeline, get_default_pipeline
from src.rag.generation.answer_postprocess import build_debug_rows, build_sources
from src.rag.retrieval.retriever import retrieve_documents

logger = logging.getLogger(__name__)


def _doc_preview(doc: Document, index: int) -> Dict[str, Any]:
    metadata = doc.metadata or {}
    return {
        "index": index,
        "title": metadata.get("title") or metadata.get("path") or metadata.get("doc_id") or f"Document {index}",
        "source": metadata.get("source") or "unknown",
        "path": metadata.get("path") or "",
        "url": metadata.get("url") or "",
        "doc_id": metadata.get("doc_id"),
        "source_type": metadata.get("source_type"),
        "rerank_score": metadata.get("rerank_score"),
        "preview": doc.page_content[:220].strip(),
    }


def _normalize_set(values: Iterable[str]) -> set[str]:
    return {str(v).strip().lower() for v in values if str(v).strip()}


def _build_source_filter(
    allowed_sources: set[str],
    allowed_source_types: set[str],
    *,
    allow_source_type_fallback: bool,
) -> models.Filter | None:
    should_conditions: list[models.FieldCondition] = []

    if allowed_sources:
        should_conditions.append(
            models.FieldCondition(
                key="metadata.source",
                match=models.MatchAny(any=sorted(allowed_sources)),
            )
        )

    if allow_source_type_fallback and allowed_source_types:
        should_conditions.append(
            models.FieldCondition(
                key="metadata.source_type",
                match=models.MatchAny(any=sorted(allowed_source_types)),
            )
        )

    if not should_conditions:
        return None

    if len(should_conditions) == 1:
        return models.Filter(must=[should_conditions[0]])

    return models.Filter(should=should_conditions)


def _search_by_source(
    query: str,
    top_k: int,
    allowed_sources: Iterable[str],
    allowed_source_types: Iterable[str],
    rebuild: bool = False,
    tool_name: str = "search_by_source",
    allow_source_type_fallback: bool = False,
) -> Dict[str, Any]:
    qdrant_filter = _build_source_filter(
        allowed_sources=_normalize_set(allowed_sources),
        allowed_source_types=_normalize_set(allowed_source_types),
        allow_source_type_fallback=allow_source_type_fallback,
    )

    logger.info(
        "Source-scoped retrieval started",
        extra={
            "event": "tool.search.started",
            "tool_name": tool_name,
            "query_length": len(query or ""),
            "top_k": top_k,
            "rebuild": rebuild,
            "source_filter_applied": qdrant_filter is not None,
        },
    )
    docs = retrieve_documents(
        query=query,
        top_k=top_k,
        rebuild=rebuild,
        qdrant_filter=qdrant_filter,
    )
    logger.info(
        "Source-scoped retrieval completed",
        extra={
            "event": "tool.search.completed",
            "tool_name": tool_name,
            "retrieved_docs": len(docs),
            "source_filter_applied": qdrant_filter is not None,
        },
    )

    return {
        "documents_raw": docs,
        "documents": [_doc_preview(doc, i) for i, doc in enumerate(docs, start=1)],
        "sources": build_sources(docs),
        "debug": build_debug_rows(docs),
        "stats": {
            "retrieved_docs": len(docs),
            "top_k_requested": top_k,
            "tool": tool_name,
            "query_filter_applied": qdrant_filter is not None,
        },
    }


def search_knowledge_base(query: str, top_k: int = 4, rebuild: bool = False) -> Dict[str, Any]:
    logger.info(
        "Knowledge-base retrieval started",
        extra={
            "event": "tool.search.started",
            "tool_name": "search_knowledge_base",
            "query_length": len(query or ""),
            "top_k": top_k,
            "rebuild": rebuild,
        },
    )
    docs = retrieve_documents(
        query=query,
        top_k=top_k,
        rebuild=rebuild,
    )
    logger.info(
        "Knowledge-base retrieval completed",
        extra={
            "event": "tool.search.completed",
            "tool_name": "search_knowledge_base",
            "retrieved_docs": len(docs),
        },
    )

    return {
        "documents_raw": docs,
        "documents": [_doc_preview(doc, i) for i, doc in enumerate(docs, start=1)],
        "sources": build_sources(docs),
        "debug": build_debug_rows(docs),
        "stats": {
            "retrieved_docs": len(docs),
            "top_k_requested": top_k,
            "tool": "search_knowledge_base",
        },
    }


def search_github_docs(query: str, top_k: int = 4, rebuild: bool = False) -> Dict[str, Any]:
    return _search_by_source(
        query=query,
        top_k=top_k,
        rebuild=rebuild,
        allowed_sources={"github_docs"},
        allowed_source_types={"document"},
        tool_name="search_github_docs",
        allow_source_type_fallback=False,
    )


def search_gitlab_handbook(query: str, top_k: int = 4, rebuild: bool = False) -> Dict[str, Any]:
    return _search_by_source(
        query=query,
        top_k=top_k,
        rebuild=rebuild,
        allowed_sources={"gitlab_handbook"},
        allowed_source_types={"document"},
        tool_name="search_gitlab_handbook",
        allow_source_type_fallback=False,
    )


def search_github_issues(query: str, top_k: int = 4, rebuild: bool = False) -> Dict[str, Any]:
    return _search_by_source(
        query=query,
        top_k=top_k,
        rebuild=rebuild,
        allowed_sources={"github_issues"},
        allowed_source_types={"ticket", "issue", "github_issue", "github_issues"},
        tool_name="search_github_issues",
        allow_source_type_fallback=True,
    )


def answer_from_knowledge_base(
    question: str,
    top_k: int = 4,
    debug: bool = False,
    pipeline: Optional[LocalRAGPipeline] = None,
) -> Dict[str, Any]:
    logger.info(
        "Answer-from-knowledge-base tool started",
        extra={
            "event": "tool.answer.started",
            "tool_name": "answer_from_knowledge_base",
            "question_length": len(question or ""),
            "top_k": top_k,
            "debug_requested": debug,
        },
    )
    if pipeline is None:
        if top_k == 4:
            pipeline = get_default_pipeline()
        else:
            pipeline = build_pipeline(top_k=top_k, rebuild=False)

    result = pipeline.ask(question, debug=debug)
    result.setdefault("stats", {})
    result["stats"]["tool"] = "answer_from_knowledge_base"
    logger.info(
        "Answer-from-knowledge-base tool completed",
        extra={
            "event": "tool.answer.completed",
            "tool_name": "answer_from_knowledge_base",
            "retrieved_docs": result.get("stats", {}).get("retrieved_docs", 0),
            "used_fallback": result.get("stats", {}).get("used_fallback", False),
        },
    )
    return result
