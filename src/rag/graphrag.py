from __future__ import annotations

import logging
import re
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_core.documents import Document
from src.data.enterprise_support_loader import load_enterprise_support_dataset
from src.kg.builder import build_graph_from_enterprise_support_dataset
from src.kg.retriever import retrieve_graph_context
from src.kg.schema import GraphContext, KGNode
from src.kg.store import get_default_graph
from src.rag.evidence_sufficiency import score_evidence_sufficiency
from src.rag.retrieval.enterprise_hybrid import retrieve_enterprise_hybrid_documents

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENTERPRISE_DATA_DIR = PROJECT_ROOT / "data" / "sample_enterprise_support"
logger = logging.getLogger(__name__)

SOURCE_TYPE_TO_NODE_TYPE = {
    "account": "Account",
    "customer": "Customer",
    "github_issue": "GitHubIssue",
    "knowledge_base": "Policy",
    "policy": "Policy",
    "product": "Product",
    "risk_event": "RiskEvent",
    "service": "Service",
    "ticket": "Ticket",
    "ticket_message": "TicketMessage",
}

MAX_ANSWER_EVIDENCE_ITEMS = 4
MAX_SNIPPET_CHARS = 320
ENTITY_ID_PATTERN = re.compile(
    r"\b(?:acct|cust|gh|msg|pol|prod|res|risk|svc|tkt)_[a-z0-9_]+\b",
    re.IGNORECASE,
)
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
SENSITIVE_DETAIL_PHRASES = {
    "credit card": "credit card details",
    "home address": "home address",
    "mobile number": "mobile number",
    "password value": "password value",
    "personal address": "personal address",
    "phone number": "phone number",
    "private phone": "private phone number",
    "social security": "Social Security number",
    "ssn": "Social Security number",
}


def retrieve_enterprise_context(
    query: str,
    top_k: int = 5,
    graph_depth: int = 2,
    filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_start = perf_counter()
    latency_ms: dict[str, float] = {}
    vector_results: list[Any] = []
    vector_error = ""
    hybrid_result: dict[str, Any] = {"debug": [], "stats": {}}

    vector_start = perf_counter()
    try:
        hybrid_result = retrieve_enterprise_hybrid_documents(
            query=query,
            top_k=top_k,
            filters=filters,
        )
        vector_results = list(hybrid_result.get("documents") or [])
    except Exception as exc:
        vector_error = _safe_error(exc)
    latency_ms["vector_retrieval_ms"] = _elapsed_ms(vector_start)
    logger.info(
        "Enterprise vector retrieval completed",
        extra={
            "event": "enterprise.rag.vector_retrieval.completed",
            "query_length": len(query or ""),
            "top_k": top_k,
            "result_count": len(vector_results),
            "duration_ms": latency_ms["vector_retrieval_ms"],
            "has_filters": bool(filters),
            "status": "error" if vector_error else "ok",
        },
    )

    hybrid_stats = dict(hybrid_result.get("stats") or {})
    vector_error = vector_error or " | ".join(
        error
        for error in [
            str(hybrid_stats.get("dense_error") or ""),
            str(hybrid_stats.get("sparse_error") or ""),
        ]
        if error
    )
    graph_results = _retrieve_graph_context(
        query=query,
        depth=graph_depth,
        limit=top_k,
        latency_ms=latency_ms,
    )
    fusion_start = perf_counter()
    merged_context = merge_vector_and_graph_context(vector_results, graph_results)
    latency_ms["fusion_ms"] = _elapsed_ms(fusion_start)
    logger.info(
        "Enterprise GraphRAG fusion completed",
        extra={
            "event": "enterprise.rag.fusion.completed",
            "vector_count": len(vector_results),
            "graph_node_count": len(graph_results.context_nodes),
            "merged_count": len(merged_context),
            "duration_ms": latency_ms["fusion_ms"],
        },
    )
    formatted_context = format_context_for_answer(merged_context)
    latency_ms["context_retrieval_total_ms"] = _elapsed_ms(total_start)

    return {
        "query": query,
        "vector_evidence": _normalize_vector_results(vector_results),
        "graph_evidence": _normalize_graph_results(graph_results),
        "merged_context": merged_context,
        "formatted_context": formatted_context,
        "citations": _build_citations(merged_context, query=query),
        "stats": {
            "top_k": top_k,
            "graph_depth": graph_depth,
            "vector_count": len(vector_results),
            "graph_node_count": len(graph_results.context_nodes),
            "graph_edge_count": len(graph_results.context_edges),
            "merged_count": len(merged_context),
            "vector_error": vector_error,
            "hybrid_retrieval": hybrid_stats,
            "hybrid_debug": list(hybrid_result.get("debug") or []),
            "latency_ms": latency_ms,
        },
    }


def build_grounded_enterprise_answer(query: str, context: dict[str, Any]) -> dict[str, Any]:
    total_start = perf_counter()
    latency_ms: dict[str, float] = {}
    merged_context = list(context.get("merged_context") or [])
    sufficiency_start = perf_counter()
    evidence_sufficiency = score_evidence_sufficiency(query, merged_context)
    latency_ms["evidence_sufficiency_scoring_ms"] = _elapsed_ms(sufficiency_start)
    logger.info(
        "Enterprise evidence sufficiency scoring completed",
        extra={
            "event": "enterprise.rag.evidence_sufficiency.completed",
            "query_length": len(query or ""),
            "evidence_count": len(merged_context),
            "score": evidence_sufficiency["score"],
            "level": evidence_sufficiency["level"],
            "duration_ms": latency_ms["evidence_sufficiency_scoring_ms"],
        },
    )

    generation_start = perf_counter()
    citations = _build_citations(merged_context, query=query)
    used_citations = _mark_answer_citations(citations)
    missing_information = _dedupe_strings(
        [
            *_detect_missing_information(query, used_citations),
            *_missing_information_from_sufficiency(evidence_sufficiency),
        ]
    )
    confidence = _cap_confidence_by_sufficiency(
        _estimate_confidence(
            query=query,
            citations=used_citations,
            missing_information=missing_information,
        ),
        evidence_sufficiency["level"],
    )
    latency_ms["answer_generation_ms"] = _elapsed_ms(generation_start)
    latency_ms["answer_total_ms"] = _elapsed_ms(total_start)
    logger.info(
        "Enterprise grounded answer generation completed",
        extra={
            "event": "enterprise.rag.answer_generation.completed",
            "query_length": len(query or ""),
            "citation_count": len(citations),
            "used_citation_count": len(used_citations),
            "confidence": confidence,
            "duration_ms": latency_ms["answer_generation_ms"],
        },
    )

    return {
        "answer": _compose_grounded_answer(
            citations=used_citations,
            missing_information=missing_information,
        ),
        "citations": citations,
        "confidence": confidence,
        "missing_information": missing_information,
        "evidence_sufficiency": evidence_sufficiency,
        "latency_ms": latency_ms,
        "mode": "deterministic_grounded_generation",
    }


def merge_vector_and_graph_context(vector_results, graph_results) -> list[dict[str, Any]]:
    vector_evidence = _normalize_vector_results(vector_results)
    graph_evidence = _normalize_graph_results(graph_results)

    merged_by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for item in vector_evidence:
        key = _evidence_key(item)
        merged_by_key[key] = dict(item)
        order.append(key)

    for item in graph_evidence:
        key = _evidence_key(item)
        if key not in merged_by_key:
            merged_by_key[key] = dict(item)
            order.append(key)
            continue

        merged_by_key[key] = _merge_evidence_items(merged_by_key[key], item)

    return [merged_by_key[key] for key in order]


def format_context_for_answer(context) -> str:
    if isinstance(context, dict):
        evidence_items = context.get("merged_context", [])
    else:
        evidence_items = context or []

    lines = ["Enterprise GraphRAG Context"]
    if not evidence_items:
        lines.append("")
        lines.append("No enterprise vector or graph evidence was found.")
        return "\n".join(lines)

    for index, item in enumerate(evidence_items, start=1):
        metadata = item.get("metadata") or {}
        title = item.get("title") or metadata.get("title") or item.get("id") or "Untitled"
        text = str(item.get("text") or "").strip()
        if len(text) > 700:
            text = text[:697] + "..."

        lines.extend(
            [
                "",
                f"[{index}] {title}",
                f"source: {item.get('context_source', 'unknown')}",
                f"source_type: {item.get('source_type') or metadata.get('source_type') or ''}",
                f"id: {item.get('id') or ''}",
                f"metadata: {metadata}",
                f"text: {text}",
            ]
        )

    return "\n".join(lines).strip()


def _retrieve_graph_context(
    query: str,
    depth: int,
    limit: int,
    latency_ms: dict[str, float] | None = None,
) -> GraphContext:
    load_start = perf_counter()
    cache_hit = True
    try:
        graph = get_default_graph()
    except RuntimeError:
        cache_hit = False
        dataset = load_enterprise_support_dataset(DEFAULT_ENTERPRISE_DATA_DIR)
        graph = build_graph_from_enterprise_support_dataset(dataset)
    dataset_context_load_ms = _elapsed_ms(load_start)
    if latency_ms is not None:
        latency_ms["dataset_context_load_ms"] = dataset_context_load_ms
    logger.info(
        "Enterprise dataset/context load completed",
        extra={
            "event": "enterprise.rag.context_load.completed",
            "cache_hit": cache_hit,
            "duration_ms": dataset_context_load_ms,
        },
    )

    graph_start = perf_counter()
    graph_context = retrieve_graph_context(query=query, depth=depth, graph=graph, limit=limit)
    graph_retrieval_ms = _elapsed_ms(graph_start)
    if latency_ms is not None:
        latency_ms["graph_retrieval_ms"] = graph_retrieval_ms
    logger.info(
        "Enterprise graph retrieval completed",
        extra={
            "event": "enterprise.rag.graph_retrieval.completed",
            "query_length": len(query or ""),
            "graph_depth": depth,
            "node_count": len(graph_context.context_nodes),
            "edge_count": len(graph_context.context_edges),
            "duration_ms": graph_retrieval_ms,
        },
    )
    return graph_context


def _normalize_vector_results(vector_results) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for index, result in enumerate(vector_results or [], start=1):
        if isinstance(result, Document):
            text = result.page_content
            metadata = dict(result.metadata or {})
        elif isinstance(result, dict):
            text = str(result.get("text") or result.get("page_content") or "")
            metadata = dict(result.get("metadata") or {})
        else:
            text = str(result)
            metadata = {}

        evidence_id = _vector_evidence_id(metadata, index)
        evidence.append(
            {
                "id": evidence_id,
                "text": text,
                "metadata": metadata,
                "context_source": "vector",
                "source_type": str(metadata.get("source_type") or ""),
                "title": str(metadata.get("title") or evidence_id),
            }
        )
    return evidence


def _normalize_graph_results(graph_results) -> list[dict[str, Any]]:
    if isinstance(graph_results, GraphContext):
        nodes = graph_results.context_nodes
    elif isinstance(graph_results, dict):
        nodes = graph_results.get("context_nodes") or graph_results.get("matched_nodes") or []
    else:
        nodes = graph_results or []

    evidence: list[dict[str, Any]] = []
    for node in nodes:
        if isinstance(node, KGNode):
            metadata = {
                **node.properties,
                "kg_node_id": node.id,
                "kg_node_type": node.type,
                "entity_id": _entity_id_from_node_id(node.id),
                "title": node.label,
                "source_type": _node_type_to_source_type(node.type),
            }
            evidence.append(
                {
                    "id": node.id,
                    "text": node.text,
                    "metadata": metadata,
                    "context_source": "graph",
                    "source_type": _node_type_to_source_type(node.type),
                    "title": node.label,
                }
            )
            continue

        if isinstance(node, dict):
            metadata = dict(node.get("metadata") or node.get("properties") or {})
            evidence.append(
                {
                    "id": str(node.get("id") or metadata.get("kg_node_id") or ""),
                    "text": str(node.get("text") or ""),
                    "metadata": metadata,
                    "context_source": "graph",
                    "source_type": str(
                        node.get("source_type") or metadata.get("source_type") or ""
                    ),
                    "title": str(node.get("title") or metadata.get("title") or ""),
                }
            )
    return evidence


def _merge_evidence_items(
    vector_item: dict[str, Any],
    graph_item: dict[str, Any],
) -> dict[str, Any]:
    vector_text = str(vector_item.get("text") or "").strip()
    graph_text = str(graph_item.get("text") or "").strip()
    if vector_text and graph_text and vector_text != graph_text:
        text = f"Vector evidence:\n{vector_text}\n\nGraph evidence:\n{graph_text}"
    else:
        text = vector_text or graph_text

    vector_metadata = dict(vector_item.get("metadata") or {})
    graph_metadata = dict(graph_item.get("metadata") or {})
    metadata = {
        **graph_metadata,
        **vector_metadata,
        "vector_metadata": vector_metadata,
        "graph_metadata": graph_metadata,
    }

    return {
        "id": graph_item.get("id") or vector_item.get("id"),
        "text": text,
        "metadata": metadata,
        "context_source": "both",
        "source_type": vector_item.get("source_type") or graph_item.get("source_type") or "",
        "title": vector_item.get("title") or graph_item.get("title") or "",
    }


def _vector_evidence_id(metadata: dict[str, Any], index: int) -> str:
    canonical = _canonical_node_id_from_metadata(metadata)
    if canonical:
        return canonical
    return str(
        metadata.get("doc_id")
        or metadata.get("source_chunk_id")
        or metadata.get("path")
        or metadata.get("title")
        or f"vector:{index}"
    )


def _evidence_key(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    canonical = _canonical_node_id_from_metadata(metadata)
    if canonical:
        return canonical
    item_id = str(item.get("id") or "")
    if ":" in item_id:
        return item_id
    return item_id or str(item.get("title") or "")


def _canonical_node_id_from_metadata(metadata: dict[str, Any]) -> str:
    kg_node_id = str(metadata.get("kg_node_id") or "").strip()
    if kg_node_id:
        return kg_node_id

    source_type = str(metadata.get("source_type") or "").strip().lower()
    entity_id = str(metadata.get("entity_id") or "").strip()

    if not entity_id:
        for field in (
            "ticket_id",
            "customer_id",
            "account_id",
            "product_id",
            "service_id",
            "policy_id",
            "risk_event_id",
            "issue_id",
            "message_id",
        ):
            value = str(metadata.get(field) or "").strip()
            if value:
                entity_id = value
                source_type = _source_type_from_id_field(field, source_type)
                break

    node_type = SOURCE_TYPE_TO_NODE_TYPE.get(source_type)
    if node_type and entity_id:
        return f"{node_type}:{entity_id}"
    return ""


def _source_type_from_id_field(field: str, fallback: str) -> str:
    mapping = {
        "account_id": "account",
        "customer_id": "customer",
        "issue_id": "github_issue",
        "message_id": "ticket_message",
        "policy_id": "knowledge_base",
        "product_id": "product",
        "risk_event_id": "risk_event",
        "service_id": "service",
        "ticket_id": "ticket",
    }
    return mapping.get(field, fallback)


def _node_type_to_source_type(node_type: str) -> str:
    mapping = {
        "Account": "account",
        "Customer": "customer",
        "GitHubIssue": "github_issue",
        "Policy": "knowledge_base",
        "Product": "product",
        "RiskEvent": "risk_event",
        "Service": "service",
        "Team": "team",
        "Ticket": "ticket",
        "TicketMessage": "ticket_message",
    }
    return mapping.get(node_type, node_type.lower())


def _entity_id_from_node_id(node_id: str) -> str:
    if ":" not in node_id:
        return node_id
    return node_id.split(":", 1)[1]


def _build_citations(
    merged_context: list[dict[str, Any]],
    query: str = "",
) -> list[dict[str, Any]]:
    citations = []
    for index, item in enumerate(merged_context, start=1):
        metadata = item.get("metadata") or {}
        source_type = item.get("source_type") or metadata.get("source_type") or ""
        item_id = str(item.get("id") or "")
        citations.append(
            {
                "index": index,
                "id": item_id,
                "entity_id": _entity_id_from_item(item, source_type),
                "title": item.get("title") or metadata.get("title") or "",
                "source_type": source_type,
                "context_source": item.get("context_source"),
                "snippet": _best_snippet(str(item.get("text") or ""), query=query),
                "claim_index": None,
                "used_for_answer": False,
                "metadata": metadata,
            }
        )
    return citations


def _mark_answer_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    used_citations: list[dict[str, Any]] = []
    for citation in citations:
        if len(used_citations) >= MAX_ANSWER_EVIDENCE_ITEMS:
            break
        if not str(citation.get("snippet") or "").strip():
            continue

        citation["used_for_answer"] = True
        citation["claim_index"] = len(used_citations) + 1
        used_citations.append(citation)

    return used_citations


def _compose_grounded_answer(
    *,
    citations: list[dict[str, Any]],
    missing_information: list[str],
) -> str:
    if not citations:
        missing = "; ".join(missing_information) or "relevant retrieved enterprise support evidence"
        return (
            "I do not have enough retrieved enterprise support evidence to answer this question. "
            f"Missing information: {missing}."
        )

    if missing_information:
        lines = [
            "Based only on the retrieved enterprise support evidence, I can answer the "
            "supported parts but some requested details are missing."
        ]
    else:
        lines = ["Based only on the retrieved enterprise support evidence:"]

    for citation in citations:
        source_label = _format_citation_source(citation)
        snippet = str(citation.get("snippet") or "").strip()
        lines.append(f"- [{citation['claim_index']}] {source_label}: {snippet}")

    if missing_information:
        lines.append(f"Missing information: {'; '.join(missing_information)}.")

    return "\n".join(lines)


def _format_citation_source(citation: dict[str, Any]) -> str:
    source_type = str(citation.get("source_type") or "source").replace("_", " ")
    entity_id = str(citation.get("entity_id") or citation.get("id") or "").strip()
    title = str(citation.get("title") or "").strip()

    if title and entity_id:
        return f"{source_type} {entity_id} ({title})"
    if entity_id:
        return f"{source_type} {entity_id}"
    if title:
        return f"{source_type} ({title})"
    return source_type


def _detect_missing_information(
    query: str,
    citations: list[dict[str, Any]],
) -> list[str]:
    missing: list[str] = []
    if not citations:
        return ["relevant retrieved enterprise support evidence"]

    query_ids = _extract_entity_ids(query)
    citation_ids = _citation_entity_ids(citations)
    missing_ids = sorted(query_ids - citation_ids)
    if missing_ids:
        missing.append(f"retrieved evidence for entity ID(s): {', '.join(missing_ids)}")

    query_lower = query.lower()
    sensitive_details = [
        label for phrase, label in SENSITIVE_DETAIL_PHRASES.items() if phrase in query_lower
    ]
    if sensitive_details:
        missing.append(
            "retrieved evidence containing requested sensitive detail(s): "
            + ", ".join(sorted(set(sensitive_details)))
        )

    query_tokens = _tokenize(query)
    evidence_tokens = _tokenize(
        " ".join(str(citation.get("snippet") or "") for citation in citations)
    )
    if query_tokens and not query_tokens.intersection(evidence_tokens.union(citation_ids)):
        missing.append("specific evidence matching the requested detail")

    return _dedupe_strings(missing)


def _estimate_confidence(
    *,
    query: str,
    citations: list[dict[str, Any]],
    missing_information: list[str],
) -> str:
    if not citations:
        return "low"

    source_types = {
        str(citation.get("source_type") or "").strip()
        for citation in citations
        if str(citation.get("source_type") or "").strip()
    }
    query_ids = _extract_entity_ids(query)
    citation_ids = _citation_entity_ids(citations)
    exact_entity_match = bool(query_ids and query_ids.intersection(citation_ids))
    query_tokens = _tokenize(query)
    evidence_tokens = _tokenize(
        " ".join(str(citation.get("snippet") or "") for citation in citations)
    )
    token_overlap_count = len(query_tokens.intersection(evidence_tokens))

    if missing_information:
        if len(citations) >= 3 and len(source_types) >= 2 and exact_entity_match:
            return "medium"
        return "low"

    if (
        len(citations) >= 4
        and len(source_types) >= 2
        and (exact_entity_match or token_overlap_count >= 3)
    ):
        return "high"

    if len(citations) >= 2 and (
        len(source_types) >= 2 or exact_entity_match or token_overlap_count >= 2
    ):
        return "medium"

    return "low"


def _missing_information_from_sufficiency(evidence_sufficiency: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if evidence_sufficiency.get("level") != "low":
        return missing

    missing_source_types = [
        str(source_type).strip()
        for source_type in evidence_sufficiency.get("missing_source_types", [])
        if str(source_type).strip()
    ]
    if missing_source_types:
        missing.append("critical source type(s): " + ", ".join(sorted(set(missing_source_types))))
    else:
        missing.append("additional corroborating enterprise support evidence")

    return missing


def _cap_confidence_by_sufficiency(confidence: str, sufficiency_level: str) -> str:
    order = {"low": 0, "medium": 1, "high": 2}
    confidence_value = order.get(confidence, 0)
    sufficiency_value = order.get(sufficiency_level, 0)
    capped_value = min(confidence_value, sufficiency_value)
    for label, value in order.items():
        if value == capped_value:
            return label
    return "low"


def _entity_id_from_item(item: dict[str, Any], source_type: str) -> str:
    metadata = item.get("metadata") or {}
    direct_entity_id = str(metadata.get("entity_id") or "").strip()
    if direct_entity_id:
        return direct_entity_id

    preferred_fields = {
        "account": "account_id",
        "customer": "customer_id",
        "github_issue": "issue_id",
        "knowledge_base": "policy_id",
        "policy": "policy_id",
        "product": "product_id",
        "risk_event": "risk_event_id",
        "service": "service_id",
        "ticket": "ticket_id",
        "ticket_message": "message_id",
    }
    preferred_field = preferred_fields.get(str(source_type or "").lower())
    if preferred_field:
        value = str(metadata.get(preferred_field) or "").strip()
        if value:
            return value

    for field in (
        "ticket_id",
        "customer_id",
        "account_id",
        "product_id",
        "service_id",
        "policy_id",
        "risk_event_id",
        "issue_id",
        "message_id",
    ):
        value = str(metadata.get(field) or "").strip()
        if value:
            return value

    item_id = str(item.get("id") or "").strip()
    if ":" in item_id:
        return item_id.split(":", 1)[1]
    return item_id


def _best_snippet(text: str, query: str = "") -> str:
    candidates = _snippet_candidates(text)
    if not candidates:
        return ""

    query_tokens = _tokenize(query)
    query_ids = _extract_entity_ids(query)
    if not query_tokens and not query_ids:
        return _truncate_snippet(candidates[0])

    risk_query = _is_risk_or_escalation_query(query)

    def _score(candidate: str) -> tuple[int, int, int]:
        candidate_tokens = _tokenize(candidate)
        candidate_ids = _extract_entity_ids(candidate)
        return (
            len(query_tokens.intersection(candidate_tokens)) + _risk_signal_score(candidate)
            if risk_query
            else len(query_tokens.intersection(candidate_tokens)),
            len(query_ids.intersection(candidate_ids)),
            -len(candidate),
        )

    best_candidate = max(candidates, key=_score)
    return _truncate_snippet(best_candidate)


def _snippet_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    for line in str(text or "").splitlines():
        cleaned = line.strip(" -\t")
        if not cleaned:
            continue
        if cleaned.endswith(":") and len(cleaned) <= 40:
            continue
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        candidates.extend(part.strip() for part in parts if part.strip())
    return candidates


def _truncate_snippet(text: str, max_chars: int = MAX_SNIPPET_CHARS) -> str:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def _extract_entity_ids(text: str) -> set[str]:
    return {match.group(0).lower() for match in ENTITY_ID_PATTERN.finditer(str(text or ""))}


def _is_risk_or_escalation_query(query: str) -> bool:
    query_tokens = _tokenize(query)
    return bool(
        query_tokens.intersection(
            {
                "breach",
                "breached",
                "escalate",
                "escalation",
                "risk",
                "risky",
                "sla",
                "severity",
            }
        )
    )


def _risk_signal_score(text: str) -> int:
    text_tokens = _tokenize(text)
    return len(
        text_tokens.intersection(
            {
                "breach",
                "breached",
                "critical",
                "escalate",
                "escalation",
                "risk",
                "sla",
                "severity",
            }
        )
    )


def _citation_entity_ids(citations: list[dict[str, Any]]) -> set[str]:
    ids: set[str] = set()
    for citation in citations:
        for key in ("entity_id", "id"):
            value = str(citation.get(key) or "").strip()
            if not value:
                continue
            if ":" in value:
                value = value.split(":", 1)[1]
            ids.update(_extract_entity_ids(value) or {value.lower()})
    return ids


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall(str(text or "").lower())
        if len(token) > 2 and token not in STOPWORDS
    }


def _dedupe_strings(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        normalized = value.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped


def _safe_error(exc: Exception) -> str:
    text = str(exc).strip()
    if len(text) > 240:
        text = text[:237] + "..."
    return text


def _elapsed_ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)
