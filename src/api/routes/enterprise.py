from __future__ import annotations

import logging
from collections import Counter
from time import perf_counter
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from src.api.schemas.enterprise import GraphRAGAskRequest, GraphRAGAskResponse
from src.core.observability import increment_counter, record_histogram
from src.rag.enterprise_agentic_retrieval import retrieve_enterprise_context_agentically
from src.rag.graphrag import (
    build_grounded_enterprise_answer,
    format_context_for_answer,
    retrieve_enterprise_context,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _enterprise_retrieval_filters(payload: GraphRAGAskRequest) -> dict[str, str]:
    filters = {}
    for field in ("source_type", "customer_id", "ticket_id", "service_id", "product_id"):
        value = str(getattr(payload, field, "") or "").strip()
        if value:
            filters[field] = value
    return filters


def _enterprise_latency_breakdown(
    context: dict[str, Any],
    grounded_answer: dict[str, Any],
    total_latency_ms: float,
) -> dict[str, float]:
    breakdown: dict[str, float] = {}
    breakdown.update(context.get("stats", {}).get("latency_ms") or {})
    breakdown.update(grounded_answer.get("latency_ms") or {})
    breakdown["total_latency_ms"] = total_latency_ms
    return breakdown


def _enterprise_debug_payload(
    context: dict[str, Any],
    grounded_answer: dict[str, Any],
    latency_breakdown: dict[str, float],
) -> dict[str, Any]:
    merged_context = list(context.get("merged_context") or [])
    source_type_counts = Counter(
        str(item.get("source_type") or (item.get("metadata") or {}).get("source_type") or "unknown")
        for item in merged_context
    )
    return {
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "evidence_count": len(merged_context),
        "top_evidence_ids": [str(item.get("id") or "") for item in merged_context[:5]],
        "latency_breakdown": latency_breakdown,
        "hybrid_debug": list(context.get("stats", {}).get("hybrid_debug") or [])[:5],
        "evidence_sufficiency": grounded_answer.get("evidence_sufficiency", {}),
        "agentic_trace": context.get("stats", {}).get("agentic_trace", {}),
    }


@router.post("/enterprise/ask", response_model=GraphRAGAskResponse, tags=["enterprise"])
def enterprise_ask(payload: GraphRAGAskRequest, request: Request) -> GraphRAGAskResponse:
    total_start = perf_counter()
    increment_counter("enterprise_ask_requests_total")
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    try:
        request_id = str(getattr(request.state, "request_id", "") or "")
        filters = _enterprise_retrieval_filters(payload)
        if payload.use_agentic_retrieval:
            context = retrieve_enterprise_context_agentically(
                question,
                top_k=payload.top_k,
                graph_depth=payload.graph_depth,
                base_filters=filters or None,
            )
        else:
            context = retrieve_enterprise_context(
                question,
                top_k=payload.top_k,
                graph_depth=payload.graph_depth,
                filters=filters or None,
            )
        grounded_answer = build_grounded_enterprise_answer(question, context)
        total_latency_ms = round((perf_counter() - total_start) * 1000, 2)
        latency_breakdown = _enterprise_latency_breakdown(
            context,
            grounded_answer,
            total_latency_ms,
        )
        confidence = grounded_answer["confidence"]
        sufficiency_level = grounded_answer["evidence_sufficiency"]["level"]
        if confidence == "low":
            increment_counter("enterprise_low_confidence_total")
        record_histogram(
            "enterprise_ask_latency_ms",
            total_latency_ms,
            attributes={
                "confidence": confidence,
                "evidence_sufficiency_level": sufficiency_level,
            },
        )
        logger.info(
            "Enterprise ask completed",
            extra={
                "event": "enterprise.ask.completed",
                "query_length": len(question),
                "top_k": payload.top_k,
                "graph_depth": payload.graph_depth,
                "confidence": confidence,
                "evidence_sufficiency_level": sufficiency_level,
                "evidence_count": len(context.get("merged_context") or []),
                "duration_ms": total_latency_ms,
                "debug_requested": payload.debug,
                "agentic_retrieval": payload.use_agentic_retrieval,
            },
        )
        metadata = {
            "formatted_context": format_context_for_answer(context),
            "query": question,
            "request_id": request_id,
            "mode": grounded_answer["mode"],
            "retrieval_mode": "agentic" if payload.use_agentic_retrieval else "direct",
            "missing_information": grounded_answer["missing_information"],
            "evidence_sufficiency": grounded_answer["evidence_sufficiency"],
            "filters": filters,
        }
        if payload.use_agentic_retrieval:
            metadata["agentic_trace"] = context.get("stats", {}).get("agentic_trace", {})
        if payload.debug:
            metadata["debug"] = _enterprise_debug_payload(
                context,
                grounded_answer,
                latency_breakdown,
            )
        return GraphRAGAskResponse(
            answer=grounded_answer["answer"],
            confidence=confidence,
            vector_evidence=context.get("vector_evidence", []),
            graph_evidence=context.get("graph_evidence", []),
            merged_context=context.get("merged_context", []),
            citations=grounded_answer["citations"],
            metadata=metadata,
            stats={
                **context.get("stats", {}),
                "confidence": confidence,
                "missing_information_count": len(grounded_answer["missing_information"]),
                "evidence_sufficiency_score": grounded_answer["evidence_sufficiency"]["score"],
                "evidence_sufficiency_level": sufficiency_level,
                "missing_source_types": grounded_answer["evidence_sufficiency"][
                    "missing_source_types"
                ],
                "latency_ms": latency_breakdown,
            },
        )
    except Exception as exc:
        total_latency_ms = round((perf_counter() - total_start) * 1000, 2)
        increment_counter("enterprise_ask_errors_total")
        record_histogram(
            "enterprise_ask_latency_ms",
            total_latency_ms,
            attributes={"status": "error"},
        )
        logger.exception(
            "Enterprise GraphRAG endpoint failed",
            extra={
                "event": "enterprise.graphrag.failed",
                "query_length": len(question),
                "duration_ms": total_latency_ms,
            },
        )
        raise HTTPException(status_code=500, detail="Enterprise GraphRAG request failed.") from exc
