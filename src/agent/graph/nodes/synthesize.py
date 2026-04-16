from __future__ import annotations

import logging
from typing import Any, Dict, List

from langchain_core.documents import Document

from src.agent.guardrails import evaluate_answer_guardrails
from src.pipeline import build_pipeline, get_default_pipeline

logger = logging.getLogger(__name__)


def _get_pipeline(top_k: int):
    if top_k == 4:
        return get_default_pipeline()
    return build_pipeline(top_k=top_k, rebuild=False)


def _score_value(value: Any) -> float:
    try:
        if value is None or value == "":
            return float("-inf")
        return float(value)
    except (TypeError, ValueError):
        return float("-inf")


def _document_key(doc: Document) -> tuple[str, str, str, str, str]:
    metadata = doc.metadata or {}
    return (
        str(metadata.get("doc_id") or ""),
        str(metadata.get("path") or ""),
        str(metadata.get("url") or ""),
        str(metadata.get("title") or ""),
        str(metadata.get("source") or ""),
    )


def _sort_documents_by_score(documents: List[Document]) -> List[Document]:
    return sorted(
        documents,
        key=lambda doc: _score_value((doc.metadata or {}).get("rerank_score")),
        reverse=True,
    )


def _sort_sources_by_score(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        sources,
        key=lambda item: _score_value((item or {}).get("rerank_score")),
        reverse=True,
    )


def _dedupe_and_reindex_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()

    for item in _sort_sources_by_score(sources):
        normalized = dict(item or {})
        key = (
            normalized.get("doc_id") or "",
            normalized.get("path") or "",
            normalized.get("url") or "",
            normalized.get("title") or "",
            normalized.get("source") or "",
        )
        if key in seen:
            continue

        seen.add(key)
        normalized["index"] = len(deduped) + 1
        deduped.append(normalized)

    return deduped


def _dedupe_documents(documents: List[Document]) -> List[Document]:
    deduped: List[Document] = []
    seen = set()

    for doc in _sort_documents_by_score(documents):
        key = _document_key(doc)
        if key in seen:
            continue

        seen.add(key)
        deduped.append(doc)

    return deduped


def merge_results_node(state: Dict[str, Any]) -> Dict[str, Any]:
    merged_documents = []
    merged_sources = []
    merged_debug = []

    for key in ["github_docs_result", "gitlab_result", "issues_result"]:
        result = state.get(key, {}) or {}
        merged_documents.extend(result.get("documents_raw", []) or [])
        merged_sources.extend(result.get("sources", []) or [])
        merged_debug.extend(result.get("debug", []) or [])

    merged = {
        "merged_documents": _dedupe_documents(merged_documents),
        "merged_sources": _dedupe_and_reindex_sources(merged_sources),
        "merged_debug": merged_debug,
    }
    logger.info(
        "Supervisor merged subgraph results",
        extra={
            "event": "supervisor.merge.completed",
            "merged_documents": len(merged["merged_documents"]),
            "merged_sources": len(merged["merged_sources"]),
            "merged_debug_rows": len(merged["merged_debug"]),
        },
    )
    return merged


def _build_no_docs_answer() -> str:
    return (
        "I ran the relevant agents, but there is still not enough trustworthy evidence to answer confidently.\n\n"
        "Please ask a more specific question with the exact platform or task involved."
    )


def _build_clarify_answer(reason: str) -> str:
    return (
        "The current question is not specific enough to synthesize a confident answer.\n\n"
        f"Reason: {reason}\n\n"
        "Please restate it with more precise platform or task details."
    )


def _build_guardrail_clarify_answer(reason: str) -> str:
    return (
        "I gathered evidence from multiple agents, but the signal is still too weak to answer confidently.\n\n"
        f"Fallback reason: {reason}\n\n"
        "Please include the exact task, platform, or error so I can choose stronger evidence."
    )


def _format_search_only_answer(sources: List[Dict[str, Any]], route_reason: str) -> str:
    if not sources:
        return (
            "Search mode completed, but no sufficiently relevant sources were found.\n\n"
            "Please restate the request with more specific context."
        )

    lines = [
        "Search mode is active, so I am returning the most relevant sources only.",
        f"Route reason: {route_reason}",
        "",
    ]

    for item in sources:
        title = item.get("title", "Unknown")
        source = item.get("source", "unknown")
        path = item.get("path", "")
        url = item.get("url", "")
        rerank_score = item.get("rerank_score")

        lines.append(f"{item.get('index', '?')}. {title} ({source})")
        if path:
            lines.append(f"   - path: {path}")
        if url:
            lines.append(f"   - url: {url}")
        if rerank_score is not None:
            lines.append(f"   - rerank_score: {rerank_score}")

    lines.extend(
        [
            "",
            "Switch to mode='answer' if you want a synthesized final answer.",
        ]
    )
    return "\n".join(lines)


def _build_tool_calls(state: Dict[str, Any], selected_agents: List[str]) -> List[Dict[str, Any]]:
    tool_calls: List[Dict[str, Any]] = []

    for state_key, tool_name, selected_name in [
        ("github_docs_result", "github_docs_agent", "github_docs"),
        ("gitlab_result", "gitlab_agent", "gitlab"),
        ("issues_result", "issues_agent", "issues"),
    ]:
        if selected_name not in selected_agents:
            continue

        result = state.get(state_key, {}) or {}
        tool_calls.append(
            {
                "tool_name": tool_name,
                "status": "ok",
                "note": f"documents_count={len(result.get('documents_raw', []) or [])}",
            }
        )

    return tool_calls


def synthesize_answer_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = (state.get("effective_question") or state.get("question") or "").strip()
    top_k = int(state.get("top_k", 4))
    debug = bool(state.get("debug", False))

    merged_documents: List[Any] = state.get("merged_documents", []) or []
    merged_sources: List[Dict[str, Any]] = state.get("merged_sources", []) or []
    merged_debug: List[Dict[str, Any]] = state.get("merged_debug", []) or []
    selected_agents = state.get("selected_agents", []) or []
    route_reason = state.get("route_reason", "")
    response_route = state.get("response_route", "answer_from_kb")
    tool_calls = _build_tool_calls(state, selected_agents)
    logger.info(
        "Supervisor synthesize node started",
        extra={
            "event": "supervisor.synthesize.started",
            "selected_agents": selected_agents,
            "response_route": response_route,
            "merged_documents": len(merged_documents),
            "merged_sources": len(merged_sources),
        },
    )

    if not question:
        return {
            "answer": "Question is empty.",
            "sources": merged_sources,
            "stats": {
                "retrieved_docs": 0,
                "used_fallback": True,
                "stage": "empty_question",
                "selected_agents": selected_agents,
                "route_reason": route_reason,
                "response_route": "clarify",
                "backend_mode": "multi_agent",
            },
            "agent": {
                "route": "clarify",
                "reason": "Question is empty.",
                "tool_calls": tool_calls,
            },
            "debug": merged_debug if debug else [],
        }

    if response_route == "clarify":
        return {
            "answer": _build_clarify_answer(route_reason or "Question is not specific enough."),
            "sources": merged_sources,
            "stats": {
                "retrieved_docs": len(merged_sources),
                "used_fallback": True,
                "stage": "clarify",
                "selected_agents": selected_agents,
                "route_reason": route_reason,
                "response_route": "clarify",
                "backend_mode": "multi_agent",
            },
            "agent": {
                "route": "clarify",
                "reason": route_reason or "Question is not specific enough.",
                "tool_calls": tool_calls,
            },
            "debug": merged_debug if debug else [],
        }

    if response_route == "retrieve_only":
        return {
            "answer": _format_search_only_answer(merged_sources, route_reason),
            "sources": merged_sources,
            "stats": {
                "retrieved_docs": len(merged_sources),
                "used_fallback": False,
                "stage": "search_only",
                "selected_agents": selected_agents,
                "route_reason": route_reason,
                "response_route": "retrieve_only",
                "backend_mode": "multi_agent",
                "merged_documents_count": len(merged_documents),
                "merged_sources_count": len(merged_sources),
            },
            "agent": {
                "route": "retrieve_only",
                "reason": route_reason or "Search mode returns sources only.",
                "tool_calls": tool_calls,
            },
            "debug": merged_debug if debug else [],
        }

    if not merged_documents:
        return {
            "answer": _build_no_docs_answer(),
            "sources": merged_sources,
            "stats": {
                "retrieved_docs": 0,
                "used_fallback": True,
                "stage": "no_merged_documents",
                "selected_agents": selected_agents,
                "route_reason": route_reason,
                "response_route": "clarify",
                "backend_mode": "multi_agent",
            },
            "agent": {
                "route": "clarify",
                "reason": "Sub-agents did not return enough evidence to synthesize an answer.",
                "tool_calls": tool_calls,
            },
            "debug": merged_debug if debug else [],
        }

    guardrail = evaluate_answer_guardrails(
        {
            "sources": merged_sources,
            "stats": {
                "retrieved_docs": len(merged_documents),
            },
        },
        requested_top_k=top_k,
    )

    if guardrail.action == "fallback_to_clarify":
        return {
            "answer": _build_guardrail_clarify_answer(guardrail.reason),
            "sources": merged_sources,
            "stats": {
                "retrieved_docs": len(merged_documents),
                "used_fallback": True,
                "stage": "guardrail_clarify",
                "selected_agents": selected_agents,
                "route_reason": route_reason,
                "response_route": "clarify",
                "backend_mode": "multi_agent",
                "guardrail_action": guardrail.action,
                "guardrail_reason": guardrail.reason,
                **guardrail.metrics,
            },
            "agent": {
                "route": "clarify",
                "reason": f"Guardrail blocked synthesize: {guardrail.reason}",
                "tool_calls": tool_calls,
            },
            "debug": merged_debug if debug else [],
        }

    if guardrail.action == "fallback_to_search":
        return {
            "answer": _format_search_only_answer(merged_sources, guardrail.reason),
            "sources": merged_sources,
            "stats": {
                "retrieved_docs": len(merged_documents),
                "used_fallback": True,
                "stage": "guardrail_search_only",
                "selected_agents": selected_agents,
                "route_reason": route_reason,
                "response_route": "retrieve_only",
                "backend_mode": "multi_agent",
                "guardrail_action": guardrail.action,
                "guardrail_reason": guardrail.reason,
                "merged_documents_count": len(merged_documents),
                "merged_sources_count": len(merged_sources),
                **guardrail.metrics,
            },
            "agent": {
                "route": "retrieve_only",
                "reason": f"Guardrail switched to search-only: {guardrail.reason}",
                "tool_calls": tool_calls,
            },
            "debug": merged_debug if debug else [],
        }

    pipeline = _get_pipeline(top_k)
    result = pipeline.answer_from_documents(
        question=question,
        documents=merged_documents,
        debug=debug,
    )

    final_stats = {
        **dict(result.get("stats", {})),
        "selected_agents": selected_agents,
        "route_reason": route_reason,
        "response_route": "answer_from_kb",
        "backend_mode": "multi_agent",
        "merged_documents_count": len(merged_documents),
        "merged_sources_count": len(merged_sources),
        "stage": "synthesized",
        "guardrail_action": guardrail.action,
        "guardrail_reason": guardrail.reason,
        **guardrail.metrics,
    }

    logger.info(
        "Supervisor synthesize node completed",
        extra={
            "event": "supervisor.synthesize.completed",
            "selected_agents": selected_agents,
            "retrieved_docs": final_stats.get("retrieved_docs", 0),
            "used_fallback": final_stats.get("used_fallback", False),
            "guardrail_action": guardrail.action,
        },
    )
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []) or merged_sources,
        "stats": final_stats,
        "debug": result.get("debug", []) if debug else [],
        "agent": {
            "route": "answer_from_kb",
            "reason": route_reason or "Supervisor synthesized results from the selected sub-agents.",
            "tool_calls": tool_calls,
        },
    }
