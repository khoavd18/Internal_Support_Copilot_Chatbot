from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Literal

from src.rag.evidence_sufficiency import score_evidence_sufficiency
from src.rag.graphrag import retrieve_enterprise_context

EnterpriseIntent = Literal[
    "customer_summary",
    "ticket_triage",
    "policy_lookup",
    "service_owner",
    "risk_explanation",
    "general",
]

MAX_RETRIEVAL_ATTEMPTS = 2
SUFFICIENT_LEVELS = {"medium", "high"}
ENTITY_ID_PATTERN = re.compile(
    r"\b(?:acct|cust|gh|msg|pol|prod|res|risk|svc|tkt)_[a-z0-9_]+\b",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")

INTENT_SOURCE_TYPES: dict[EnterpriseIntent, list[str]] = {
    "customer_summary": ["customer", "account", "ticket", "risk_event"],
    "ticket_triage": ["ticket", "ticket_message", "knowledge_base", "service", "risk_event"],
    "policy_lookup": ["knowledge_base"],
    "service_owner": ["service", "github_issue"],
    "risk_explanation": ["risk_event", "ticket", "customer", "service"],
    "general": [],
}

INTENT_EXPANSIONS: dict[EnterpriseIntent, str] = {
    "customer_summary": "customer account ticket risk health renewal summary",
    "ticket_triage": "ticket priority status severity SLA escalation service policy",
    "policy_lookup": "knowledge base policy runbook SLA access refund security retention",
    "service_owner": "service catalog owner team engineering GitHub issue",
    "risk_explanation": "risk event anomaly escalation incident ticket customer service",
    "general": "customer ticket policy service risk evidence",
}

Retriever = Callable[[str, int, int, dict[str, Any] | None], dict[str, Any]]


def classify_query_intent(query: str) -> EnterpriseIntent:
    tokens = _tokens(query)

    if tokens & {"owner", "owns", "team", "service_catalog"} or "service owner" in _lower(query):
        return "service_owner"
    if tokens & {"risk", "risky", "anomaly", "churn", "breach", "breached", "incident"}:
        return "risk_explanation"
    if tokens & {
        "policy",
        "policies",
        "runbook",
        "sla",
        "refund",
        "access",
        "retention",
        "security",
        "troubleshooting",
    }:
        return "policy_lookup"
    if tokens & {"triage", "priority", "severity", "escalate", "escalation"}:
        return "ticket_triage"
    if tokens & {"summary", "summarize", "customer", "account", "health", "renewal"}:
        return "customer_summary"
    return "general"


def choose_retrieval_filters(
    *,
    intent: EnterpriseIntent,
    query: str,
    base_filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    filters = _normalize_filters(base_filters)
    if "source_type" not in filters:
        source_types = INTENT_SOURCE_TYPES[intent]
        if source_types:
            filters["source_type"] = list(source_types)

    for entity_id in _extract_entity_ids(query):
        field = _filter_field_for_entity_id(entity_id)
        if field and field not in filters:
            filters[field] = entity_id

    return filters


def retrieve_enterprise_context_agentically(
    query: str,
    top_k: int = 5,
    graph_depth: int = 2,
    base_filters: dict[str, Any] | None = None,
    retriever: Retriever | None = None,
) -> dict[str, Any]:
    active_retriever = retriever or retrieve_enterprise_context
    intent = classify_query_intent(query)
    original_filters = choose_retrieval_filters(
        intent=intent,
        query=query,
        base_filters=base_filters,
    )
    protected_filter_keys = set(_normalize_filters(base_filters))

    attempts: list[dict[str, Any]] = []
    best_context: dict[str, Any] | None = None
    best_sufficiency: dict[str, Any] | None = None
    current_query = query
    current_filters = dict(original_filters)
    stop_reason = "max_attempts_reached"

    for attempt_number in range(1, MAX_RETRIEVAL_ATTEMPTS + 1):
        filters_for_attempt = dict(current_filters)
        context = active_retriever(
            current_query,
            top_k,
            graph_depth,
            filters_for_attempt or None,
        )
        sufficiency = score_evidence_sufficiency(query, context.get("merged_context") or [])
        attempts.append(
            {
                "attempt": attempt_number,
                "query": current_query,
                "filters": filters_for_attempt,
                "sufficiency": sufficiency,
            }
        )

        if _is_better_sufficiency(sufficiency, best_sufficiency):
            best_context = context
            best_sufficiency = sufficiency

        if sufficiency["level"] in SUFFICIENT_LEVELS:
            stop_reason = "sufficient_evidence"
            best_context = context
            best_sufficiency = sufficiency
            break

        if attempt_number >= MAX_RETRIEVAL_ATTEMPTS:
            break

        current_query = rewrite_query_once(
            query=query,
            intent=intent,
            sufficiency=sufficiency,
        )
        current_filters = _expand_filters_for_retry(
            filters_for_attempt,
            missing_source_types=sufficiency.get("missing_source_types") or [],
            protected_filter_keys=protected_filter_keys,
        )

    final_context = dict(best_context or {})
    final_context["query"] = query
    stats = dict(final_context.get("stats") or {})
    trace = _build_trace(
        intent=intent,
        attempts=attempts,
        stop_reason=stop_reason,
        best_sufficiency=best_sufficiency,
    )
    stats["agentic_retrieval"] = True
    stats["agentic_trace"] = trace
    final_context["stats"] = stats
    return final_context


def rewrite_query_once(
    *,
    query: str,
    intent: EnterpriseIntent,
    sufficiency: dict[str, Any],
) -> str:
    missing_source_types = ", ".join(sufficiency.get("missing_source_types") or [])
    expansion_parts = [query.strip(), INTENT_EXPANSIONS[intent]]
    if missing_source_types:
        expansion_parts.append(f"include missing source types: {missing_source_types}")
    return " ".join(part for part in expansion_parts if part)


def _build_trace(
    *,
    intent: EnterpriseIntent,
    attempts: list[dict[str, Any]],
    stop_reason: str,
    best_sufficiency: dict[str, Any] | None,
) -> dict[str, Any]:
    sufficiency_before = attempts[0]["sufficiency"] if attempts else {}
    sufficiency_after = best_sufficiency or sufficiency_before
    return {
        "intent": intent,
        "retrieval_attempts": len(attempts),
        "filters_used": [attempt["filters"] for attempt in attempts],
        "sufficiency_before": sufficiency_before,
        "sufficiency_after": sufficiency_after,
        "stop_reason": stop_reason,
        "attempts": attempts,
    }


def _expand_filters_for_retry(
    filters: dict[str, Any],
    *,
    missing_source_types: list[str],
    protected_filter_keys: set[str] | None = None,
) -> dict[str, Any]:
    expanded = dict(filters)
    protected = protected_filter_keys or set()
    if missing_source_types:
        for field in ("customer_id", "ticket_id", "service_id", "product_id"):
            if field not in protected:
                expanded.pop(field, None)

        source_types = set(_list_filter_values(expanded.get("source_type")))
        source_types.update(str(source_type) for source_type in missing_source_types if source_type)
        if source_types:
            expanded["source_type"] = sorted(source_types)
    return expanded


def _is_better_sufficiency(candidate: dict[str, Any], current_best: dict[str, Any] | None) -> bool:
    if current_best is None:
        return True
    return float(candidate.get("score") or 0.0) >= float(current_best.get("score") or 0.0)


def _normalize_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in (filters or {}).items():
        if value is None:
            continue
        if isinstance(value, str):
            stripped = value.strip()
            if stripped:
                normalized[key] = stripped
            continue
        if isinstance(value, (list, tuple, set)):
            values = [str(item).strip() for item in value if str(item).strip()]
            if values:
                normalized[key] = values
            continue
        normalized[key] = value
    return normalized


def _filter_field_for_entity_id(entity_id: str) -> str:
    normalized = entity_id.lower()
    if normalized.startswith("cust_"):
        return "customer_id"
    if normalized.startswith("tkt_"):
        return "ticket_id"
    if normalized.startswith("svc_"):
        return "service_id"
    if normalized.startswith("prod_"):
        return "product_id"
    return ""


def _extract_entity_ids(query: str) -> list[str]:
    seen: set[str] = set()
    entity_ids: list[str] = []
    for match in ENTITY_ID_PATTERN.finditer(query or ""):
        entity_id = match.group(0).lower()
        if entity_id not in seen:
            seen.add(entity_id)
            entity_ids.append(entity_id)
    return entity_ids


def _list_filter_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return [str(value)]


def _tokens(query: str) -> set[str]:
    return set(TOKEN_PATTERN.findall(_lower(query)))


def _lower(query: str) -> str:
    return str(query or "").lower()
