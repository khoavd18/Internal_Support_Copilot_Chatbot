from __future__ import annotations

from typing import Any, Dict, Optional

from langchain_core.documents import Document

from src.pipeline import LocalRAGPipeline, build_pipeline, get_default_pipeline
from src.rag.generation.answer_postprocess import build_debug_rows, build_sources
from src.rag.retrieval.retriever import retrieve_documents


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


def search_knowledge_base(query: str, top_k: int = 4, rebuild: bool = False) -> Dict[str, Any]:
    docs = retrieve_documents(
        query=query,
        top_k=top_k,
        rebuild=rebuild,
    )

    return {
        "documents": [_doc_preview(doc, i) for i, doc in enumerate(docs, start=1)],
        "sources": build_sources(docs),
        "debug": build_debug_rows(docs),
        "stats": {
            "retrieved_docs": len(docs),
            "top_k_requested": top_k,
            "tool": "search_knowledge_base",
        },
    }


def answer_from_knowledge_base(
    question: str,
    top_k: int = 4,
    debug: bool = False,
    pipeline: Optional[LocalRAGPipeline] = None,
) -> Dict[str, Any]:
    if pipeline is None:
        if top_k == 4:
            pipeline = get_default_pipeline()
        else:
            pipeline = build_pipeline(top_k=top_k, rebuild=False)

    result = pipeline.ask(question, debug=debug)
    result.setdefault("stats", {})
    result["stats"]["tool"] = "answer_from_knowledge_base"
    return result