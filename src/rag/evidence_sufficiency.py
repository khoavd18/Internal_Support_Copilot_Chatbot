from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

ENTITY_ID_PATTERN = re.compile(
    r"\b(?:acct|cust|gh|msg|pol|prod|res|risk|svc|tkt)_[a-z0-9_]+\b",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+")

POLICY_TERMS = {
    "access",
    "policy",
    "refund",
    "retention",
    "runbook",
    "sla",
    "troubleshooting",
}
TICKET_TERMS = {
    "case",
    "escalate",
    "escalation",
    "issue",
    "priority",
    "reply",
    "sla",
    "status",
    "ticket",
    "triage",
}
CUSTOMER_TERMS = {
    "account",
    "customer",
    "health",
    "northstar",
    "renewal",
    "summary",
}
SERVICE_TERMS = {
    "api",
    "gateway",
    "integration",
    "owner",
    "service",
    "sync",
    "timeout",
}
RISK_TERMS = {
    "anomaly",
    "breach",
    "breached",
    "churn",
    "risk",
    "risky",
}

METADATA_ID_FIELDS = (
    "entity_id",
    "customer_id",
    "account_id",
    "ticket_id",
    "product_id",
    "service_id",
    "policy_id",
    "risk_event_id",
    "issue_id",
    "message_id",
)


def score_evidence_sufficiency(query, merged_context) -> dict:
    evidence_items = list(merged_context or [])
    source_types = _source_types(evidence_items)
    required_source_types = _infer_required_source_types(str(query or ""))
    missing_source_types = sorted(required_source_types - source_types)
    query_entity_ids = _extract_entity_ids(str(query or ""))
    evidence_entity_ids = _evidence_entity_ids(evidence_items)
    matched_entity_ids = sorted(query_entity_ids & evidence_entity_ids)
    contradictions = _detect_contradictions(evidence_items)
    freshness = _freshness_signal(evidence_items)

    evidence_count_score = min(len(evidence_items) / 4.0, 1.0) * 0.22
    diversity_score = min(len(source_types) / 3.0, 1.0) * 0.18
    entity_match_score = _entity_match_score(query_entity_ids, evidence_entity_ids) * 0.18
    coverage_score = _coverage_score(required_source_types, source_types) * 0.25
    freshness_score = freshness["score"] * 0.07
    contradiction_penalty = 0.15 if contradictions else 0.0

    raw_score = (
        evidence_count_score
        + diversity_score
        + entity_match_score
        + coverage_score
        + freshness_score
        - contradiction_penalty
    )
    score = _clamp(raw_score)
    level = _level(score)

    return {
        "score": round(score, 4),
        "level": level,
        "reasons": _build_reasons(
            evidence_items=evidence_items,
            source_types=source_types,
            required_source_types=required_source_types,
            missing_source_types=missing_source_types,
            query_entity_ids=query_entity_ids,
            matched_entity_ids=matched_entity_ids,
            contradictions=contradictions,
            freshness=freshness,
        ),
        "missing_source_types": missing_source_types,
    }


def _infer_required_source_types(query: str) -> set[str]:
    tokens = _tokenize(query)
    query_ids = _extract_entity_ids(query)
    required: set[str] = set()

    if query_ids.intersection(_ids_with_prefix(query_ids, "tkt_")) or tokens & TICKET_TERMS:
        required.add("ticket")
    if query_ids.intersection(_ids_with_prefix(query_ids, "cust_")) or tokens & CUSTOMER_TERMS:
        required.add("customer")
    if query_ids.intersection(_ids_with_prefix(query_ids, "svc_")) or tokens & SERVICE_TERMS:
        required.add("service")
    if query_ids.intersection(_ids_with_prefix(query_ids, "pol_")) or tokens & POLICY_TERMS:
        required.add("knowledge_base")
    if query_ids.intersection(_ids_with_prefix(query_ids, "risk_")) or tokens & RISK_TERMS:
        required.add("risk_event")

    if "sla" in tokens:
        required.update({"ticket", "knowledge_base"})

    if not required and evidence_question_needs_context(tokens):
        required.update({"ticket", "customer"})

    return required


def evidence_question_needs_context(tokens: set[str]) -> bool:
    return bool(tokens & {"explain", "summarize", "summary", "why"})


def _source_types(evidence_items: list[dict[str, Any]]) -> set[str]:
    source_types: set[str] = set()
    for item in evidence_items:
        metadata = item.get("metadata") or {}
        for value in [
            item.get("source_type"),
            metadata.get("source_type"),
            (metadata.get("vector_metadata") or {}).get("source_type")
            if isinstance(metadata.get("vector_metadata"), dict)
            else "",
            (metadata.get("graph_metadata") or {}).get("source_type")
            if isinstance(metadata.get("graph_metadata"), dict)
            else "",
        ]:
            canonical = _canonical_source_type(value)
            if canonical:
                source_types.add(canonical)
    return source_types


def _canonical_source_type(value: Any) -> str:
    source_type = str(value or "").strip().lower()
    if source_type in {"policy", "kb", "knowledge"}:
        return "knowledge_base"
    return source_type


def _entity_match_score(query_entity_ids: set[str], evidence_entity_ids: set[str]) -> float:
    if not query_entity_ids:
        return 0.35 if evidence_entity_ids else 0.0
    matches = query_entity_ids & evidence_entity_ids
    return len(matches) / len(query_entity_ids)


def _coverage_score(required_source_types: set[str], source_types: set[str]) -> float:
    if not required_source_types:
        return min(len(source_types) / 2.0, 1.0)
    return len(required_source_types & source_types) / len(required_source_types)


def _detect_contradictions(evidence_items: list[dict[str, Any]]) -> list[str]:
    field_values: dict[tuple[str, str], set[str]] = {}
    for item in evidence_items:
        metadata = item.get("metadata") or {}
        entity_id = _item_entity_id(item)
        field_values.setdefault(("status", entity_id), set()).update(
            _values_for_field(item, metadata, "status")
        )
        field_values.setdefault(("priority", entity_id), set()).update(
            _values_for_field(item, metadata, "priority")
        )
        field_values.setdefault(("sla_status", entity_id), set()).update(
            _values_for_field(item, metadata, "sla_status")
        )

    contradictions = []
    for (field, entity_id), values in field_values.items():
        clean_values = {value for value in values if value}
        if len(clean_values) > 1:
            label = field.replace("_", " ")
            entity_label = f" for {entity_id}" if entity_id else ""
            contradictions.append(
                f"Conflicting {label}{entity_label}: {', '.join(sorted(clean_values))}"
            )
    return contradictions


def _values_for_field(item: dict[str, Any], metadata: dict[str, Any], field: str) -> set[str]:
    values: set[str] = set()
    for key in {field, field.replace("_", " ")}:
        metadata_value = str(metadata.get(key) or "").strip().lower()
        if metadata_value:
            values.add(metadata_value)

    label_pattern = field.replace("_", r"[_ ]")
    for match in re.finditer(
        rf"(?im)^\s*{label_pattern}\s*:\s*([^\n]+)",
        str(item.get("text") or ""),
    ):
        value = match.group(1).strip().lower()
        if value:
            values.add(value)

    return values


def _freshness_signal(evidence_items: list[dict[str, Any]]) -> dict[str, Any]:
    datetimes = []
    for item in evidence_items:
        parsed = _parse_item_datetime(item)
        if parsed is not None:
            datetimes.append(parsed)

    if not datetimes:
        return {
            "score": 0.4,
            "reason": "No created_at metadata was available for freshness scoring.",
        }

    newest = max(datetimes)
    age_days = (datetime.now(timezone.utc) - newest).days
    if age_days <= 180:
        return {
            "score": 1.0,
            "reason": f"Newest evidence timestamp is recent: {newest.date().isoformat()}.",
        }
    if age_days <= 365:
        return {
            "score": 0.6,
            "reason": f"Newest evidence timestamp is within one year: {newest.date().isoformat()}.",
        }
    return {
        "score": 0.1,
        "reason": f"Newest evidence timestamp is stale: {newest.date().isoformat()}.",
    }


def _parse_item_datetime(item: dict[str, Any]) -> datetime | None:
    metadata = item.get("metadata") or {}
    for key in ("created_at", "updated_at", "detected_at"):
        value = str(metadata.get(key) or "").strip()
        parsed = _parse_datetime(value)
        if parsed is not None:
            return parsed
    return None


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        try:
            parsed = datetime.strptime(value[:10], "%Y-%m-%d")
        except ValueError:
            return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _build_reasons(
    *,
    evidence_items: list[dict[str, Any]],
    source_types: set[str],
    required_source_types: set[str],
    missing_source_types: list[str],
    query_entity_ids: set[str],
    matched_entity_ids: list[str],
    contradictions: list[str],
    freshness: dict[str, Any],
) -> list[str]:
    reasons = [
        f"Evidence item count: {len(evidence_items)}.",
        f"Source diversity: {len(source_types)} source type(s): {', '.join(sorted(source_types)) or 'none'}.",
    ]

    if query_entity_ids:
        if matched_entity_ids:
            reasons.append(f"Exact entity matches found: {', '.join(matched_entity_ids)}.")
        else:
            reasons.append("No exact entity IDs from the query were found in evidence.")
    else:
        reasons.append("No explicit entity ID was requested in the query.")

    if required_source_types:
        covered = sorted(required_source_types - set(missing_source_types))
        reasons.append(
            f"Covered critical source types: {', '.join(covered) if covered else 'none'}."
        )
    else:
        reasons.append("No critical source types were inferred from the query.")

    if missing_source_types:
        reasons.append(f"Missing critical source types: {', '.join(missing_source_types)}.")

    if contradictions:
        reasons.extend(contradictions)
    else:
        reasons.append("No deterministic contradiction signals were detected.")

    reasons.append(str(freshness["reason"]))
    return reasons


def _evidence_entity_ids(evidence_items: list[dict[str, Any]]) -> set[str]:
    entity_ids: set[str] = set()
    for item in evidence_items:
        metadata = item.get("metadata") or {}
        for field in METADATA_ID_FIELDS:
            value = str(metadata.get(field) or "").strip()
            if value:
                entity_ids.add(value.lower())
        item_id = str(item.get("id") or "").strip()
        if ":" in item_id:
            item_id = item_id.split(":", 1)[1]
        if item_id:
            entity_ids.add(item_id.lower())
        entity_ids.update(_extract_entity_ids(str(item.get("text") or "")))
    return entity_ids


def _item_entity_id(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    for field in METADATA_ID_FIELDS:
        value = str(metadata.get(field) or "").strip()
        if value:
            return value
    item_id = str(item.get("id") or "").strip()
    if ":" in item_id:
        return item_id.split(":", 1)[1]
    return item_id


def _extract_entity_ids(text: str) -> set[str]:
    return {match.group(0).lower() for match in ENTITY_ID_PATTERN.finditer(str(text or ""))}


def _ids_with_prefix(entity_ids: set[str], prefix: str) -> set[str]:
    return {entity_id for entity_id in entity_ids if entity_id.startswith(prefix)}


def _tokenize(text: str) -> set[str]:
    return {token for token in TOKEN_PATTERN.findall(str(text or "").lower()) if len(token) > 2}


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"
