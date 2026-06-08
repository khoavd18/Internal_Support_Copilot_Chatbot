from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from src.data.enterprise_support_loader import load_enterprise_support_dataset

Record = dict[str, Any]
Dataset = dict[str, list[Record]]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENTERPRISE_SUPPORT_DATA_DIR = PROJECT_ROOT / "data" / "sample_enterprise_support"

OPEN_STATUSES = {"open", "pending_customer", "pending_engineering"}
CLOSED_STATUSES = {"resolved", "closed"}
ESCALATION_SLA_STATUSES = {"at_risk", "breached"}


class EnterpriseSupportDataError(ValueError):
    """Raised when requested synthetic enterprise support data is unavailable."""


@lru_cache(maxsize=1)
def _load_cached_dataset(data_dir: str = str(DEFAULT_ENTERPRISE_SUPPORT_DATA_DIR)) -> Dataset:
    return load_enterprise_support_dataset(Path(data_dir))


def get_enterprise_support_dataset(data_dir: Path | None = None) -> Dataset:
    if data_dir is None:
        return _load_cached_dataset()
    return load_enterprise_support_dataset(data_dir)


def _value(record: Record | None, key: str, default: str = "") -> str:
    if not record:
        return default
    value = record.get(key, default)
    if value is None:
        return default
    return str(value).strip()


def _find_by_id(records: list[Record], key: str, value: str, entity_name: str) -> Record:
    normalized = str(value or "").strip()
    for record in records:
        if _value(record, key) == normalized:
            return record
    raise EnterpriseSupportDataError(f"{entity_name} not found: {normalized}")


def _index_by(records: list[Record], key: str) -> dict[str, Record]:
    return {_value(record, key): record for record in records if _value(record, key)}


def _group_by(records: list[Record], key: str) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = {}
    for record in records:
        value = _value(record, key)
        if value:
            grouped.setdefault(value, []).append(record)
    return grouped


def _context_item(
    *,
    source_type: str,
    entity_id: str,
    title: str = "",
    summary: str = "",
    metadata: Record | None = None,
) -> Record:
    return {
        "source_type": source_type,
        "entity_id": entity_id,
        "title": title,
        "summary": summary,
        "metadata": metadata or {},
    }


def _ticket_context(ticket: Record) -> Record:
    return _context_item(
        source_type="ticket",
        entity_id=_value(ticket, "ticket_id"),
        title=_value(ticket, "title"),
        summary="; ".join(
            part
            for part in [
                f"priority={_value(ticket, 'priority')}",
                f"status={_value(ticket, 'status')}",
                f"sla={_value(ticket, 'sla_status')}",
                _value(ticket, "description"),
            ]
            if part
        ),
        metadata={
            "customer_id": _value(ticket, "customer_id"),
            "ticket_id": _value(ticket, "ticket_id"),
            "product_id": _value(ticket, "product_id"),
            "service_id": _value(ticket, "service_id"),
            "created_at": _value(ticket, "created_at"),
        },
    )


def _risk_context(risk: Record) -> Record:
    return _context_item(
        source_type="risk_event",
        entity_id=_value(risk, "risk_event_id"),
        title=_value(risk, "event_type"),
        summary=_value(risk, "summary"),
        metadata={
            "customer_id": _value(risk, "customer_id"),
            "ticket_id": _value(risk, "ticket_id"),
            "product_id": _value(risk, "product_id"),
            "service_id": _value(risk, "service_id"),
            "created_at": _value(risk, "detected_at"),
            "severity": _value(risk, "severity"),
            "risk_score": _value(risk, "risk_score"),
            "status": _value(risk, "status"),
        },
    )


def _policy_context(article: Record) -> Record:
    return _context_item(
        source_type="knowledge_base",
        entity_id=_value(article, "policy_id"),
        title=_value(article, "title"),
        summary=_value(article, "summary"),
        metadata={
            "policy_id": _value(article, "policy_id"),
            "product_id": _value(article, "product_id"),
            "service_id": _value(article, "service_id"),
            "path": _value(article, "path"),
        },
    )


def _ticket_is_active(ticket: Record) -> bool:
    return _value(ticket, "status").lower() in OPEN_STATUSES


def _risk_is_active(risk: Record) -> bool:
    return _value(risk, "status").lower() not in {"closed", "mitigated"}


def _severity_rank(priority: str) -> int:
    priority = priority.lower()
    ranks = {"p1": 4, "p2": 3, "p3": 2, "p4": 1}
    return ranks.get(priority, 0)


def _max_priority(current: str, candidate: str) -> str:
    return candidate if _severity_rank(candidate) > _severity_rank(current) else current


def build_customer_summary(customer_id: str) -> Record:
    dataset = get_enterprise_support_dataset()
    customer = _find_by_id(dataset["customers"], "customer_id", customer_id, "Customer")
    accounts_by_id = _index_by(dataset["accounts"], "account_id")
    account = accounts_by_id.get(_value(customer, "account_id"), {})
    tickets = [
        ticket
        for ticket in dataset["tickets"]
        if _value(ticket, "customer_id") == _value(customer, "customer_id")
    ]
    risks = [
        risk
        for risk in dataset["risk_events"]
        if _value(risk, "customer_id") == _value(customer, "customer_id")
    ]

    active_tickets = [ticket for ticket in tickets if _ticket_is_active(ticket)]
    active_risks = [risk for risk in risks if _risk_is_active(risk)]
    high_risks = [
        risk
        for risk in risks
        if _value(risk, "severity").lower() in {"high", "critical"}
        or int(_value(risk, "risk_score", "0") or 0) >= 70
    ]

    summary = (
        f"{_value(customer, 'full_name')} is a {_value(customer, 'role')} at "
        f"{_value(account, 'account_name', 'an unknown account')} with "
        f"{_value(customer, 'support_tier')} support. The account risk level is "
        f"{_value(account, 'risk_level', 'unknown')} with health score "
        f"{_value(account, 'health_score', 'unknown')}. There are {len(active_tickets)} "
        f"active tickets and {len(active_risks)} active risk events in the synthetic dataset."
    )

    return {
        "customer_id": _value(customer, "customer_id"),
        "customer_name": _value(customer, "full_name"),
        "account_id": _value(customer, "account_id"),
        "account_name": _value(account, "account_name"),
        "summary": summary,
        "tickets": [_ticket_context(ticket) for ticket in tickets],
        "risk_events": [_risk_context(risk) for risk in risks],
        "stats": {
            "ticket_count": len(tickets),
            "active_ticket_count": len(active_tickets),
            "risk_event_count": len(risks),
            "active_risk_event_count": len(active_risks),
            "high_risk_event_count": len(high_risks),
        },
    }


def _triage_ticket_record(ticket: Record) -> tuple[str, str, bool, list[str]]:
    current_priority = _value(ticket, "priority").lower()
    recommended_priority = current_priority or "p4"
    status = _value(ticket, "status").lower()
    category = _value(ticket, "category").lower()
    severity = _value(ticket, "severity").lower()
    sla_status = _value(ticket, "sla_status").lower()
    reasoning: list[str] = []

    if current_priority == "p1" or severity == "sev1":
        recommended_priority = "p1"
        reasoning.append("Ticket is already marked P1 or Sev1.")

    if category in {"incident", "security"} or severity == "sev2":
        recommended_priority = _max_priority(recommended_priority, "p2")
        reasoning.append("Incident, security, or Sev2 tickets should be at least P2.")

    if sla_status in ESCALATION_SLA_STATUSES:
        recommended_priority = _max_priority(recommended_priority, "p2")
        reasoning.append(f"SLA status is {sla_status}, so escalation should be considered.")

    if status == "pending_engineering":
        reasoning.append("Ticket is already waiting on engineering.")

    escalation_required = (
        sla_status in ESCALATION_SLA_STATUSES
        or recommended_priority == "p1"
        or (recommended_priority == "p2" and status in {"open", "pending_engineering"})
    )
    recommended_status = status
    if escalation_required and status not in CLOSED_STATUSES:
        recommended_status = "pending_engineering"

    if not reasoning:
        reasoning.append("No high-risk signal was found; keep the current ticket handling.")

    return recommended_priority, recommended_status, escalation_required, reasoning


def triage_ticket(ticket_id: str) -> Record:
    dataset = get_enterprise_support_dataset()
    ticket = _find_by_id(dataset["tickets"], "ticket_id", ticket_id, "Ticket")
    products_by_id = _index_by(dataset["products"], "product_id")
    services_by_id = _index_by(dataset["service_catalog"], "service_id")
    product = products_by_id.get(_value(ticket, "product_id"), {})
    service = services_by_id.get(_value(ticket, "service_id"), {})
    recommended_priority, recommended_status, escalation_required, reasoning = (
        _triage_ticket_record(ticket)
    )

    return {
        "ticket_id": _value(ticket, "ticket_id"),
        "current_priority": _value(ticket, "priority"),
        "recommended_priority": recommended_priority,
        "current_status": _value(ticket, "status"),
        "recommended_status": recommended_status,
        "escalation_required": escalation_required,
        "reasoning": reasoning,
        "context": [
            _ticket_context(ticket),
            _context_item(
                source_type="product",
                entity_id=_value(product, "product_id"),
                title=_value(product, "product_name"),
                metadata={"product_id": _value(product, "product_id")},
            ),
            _context_item(
                source_type="service",
                entity_id=_value(service, "service_id"),
                title=_value(service, "service_name"),
                summary=_value(service, "description"),
                metadata={
                    "service_id": _value(service, "service_id"),
                    "product_id": _value(service, "product_id"),
                    "owner_team": _value(service, "owner_team"),
                    "support_escalation_team": _value(service, "support_escalation_team"),
                },
            ),
        ],
    }


def _select_policy_ids(ticket: Record) -> list[str]:
    haystack = " ".join(
        [
            _value(ticket, "title"),
            _value(ticket, "description"),
            _value(ticket, "category"),
            _value(ticket, "priority"),
            _value(ticket, "severity"),
            _value(ticket, "sla_status"),
            _value(ticket, "tags"),
            _value(ticket, "product_id"),
            _value(ticket, "service_id"),
        ]
    ).lower()
    policy_ids: list[str] = []

    rules = [
        ("pol_sla", ["sla", "p1", "p2", "breach", "incident", "outage"]),
        ("pol_api_timeout", ["api", "timeout", "latency", "gateway", "webhook"]),
        ("pol_login_troubleshooting", ["login", "sso", "scim", "session"]),
        ("pol_security_escalation", ["security", "mfa", "api_key", "lost_device"]),
        ("pol_refund", ["billing", "refund", "credit", "tax", "invoice"]),
        ("pol_data_retention", ["export", "retention", "data"]),
        ("pol_access", ["access", "permission", "admin", "invite", "oauth"]),
        ("pol_incident_response", ["incident", "outage", "sev1", "duplicate_orders"]),
    ]
    for policy_id, keywords in rules:
        if any(keyword in haystack for keyword in keywords):
            policy_ids.append(policy_id)

    if not policy_ids:
        policy_ids.append("pol_enterprise_support")

    return list(dict.fromkeys(policy_ids))


def _selected_policies(ticket: Record, dataset: Dataset) -> list[Record]:
    policies_by_id = _index_by(dataset["knowledge_base_docs"], "policy_id")
    return [
        policies_by_id[policy_id]
        for policy_id in _select_policy_ids(ticket)
        if policy_id in policies_by_id
    ]


def suggest_ticket_reply(ticket_id: str) -> Record:
    dataset = get_enterprise_support_dataset()
    ticket = _find_by_id(dataset["tickets"], "ticket_id", ticket_id, "Ticket")
    customer = _index_by(dataset["customers"], "customer_id").get(_value(ticket, "customer_id"), {})
    product = _index_by(dataset["products"], "product_id").get(_value(ticket, "product_id"), {})
    service = _index_by(dataset["service_catalog"], "service_id").get(
        _value(ticket, "service_id"), {}
    )
    messages = [
        message
        for message in dataset["ticket_messages"]
        if _value(message, "ticket_id") == _value(ticket, "ticket_id")
        and _value(message, "visibility") == "public"
    ]
    resolutions = [
        resolution
        for resolution in dataset["ticket_resolutions"]
        if _value(resolution, "ticket_id") == _value(ticket, "ticket_id")
    ]
    policies = _selected_policies(ticket, dataset)
    triage = triage_ticket(ticket_id)

    greeting_name = _value(customer, "full_name").split(" ", 1)[0] or "there"
    latest_customer_message = next(
        (message for message in reversed(messages) if _value(message, "author_type") == "customer"),
        {},
    )
    resolution = resolutions[0] if resolutions else {}

    draft_parts = [
        f"Hi {greeting_name},",
        (
            f"Thanks for contacting us about {_value(ticket, 'title')}. "
            f"We see this affects {_value(product, 'product_name', 'the product')} "
            f"and {_value(service, 'service_name', 'the related service')}."
        ),
    ]

    if latest_customer_message:
        draft_parts.append(
            f"Based on your latest update, we are tracking: {_value(latest_customer_message, 'body')}."
        )

    if resolution:
        draft_parts.append(_value(resolution, "customer_visible_response"))
    elif triage["escalation_required"]:
        draft_parts.append(
            "This has been flagged for escalation based on priority, status, and SLA signals."
        )
    else:
        draft_parts.append(
            "We are continuing the investigation and will follow up with the next concrete finding."
        )

    if policies:
        policy_titles = ", ".join(_value(policy, "title") for policy in policies)
        draft_parts.append(f"We are using the relevant support guidance: {policy_titles}.")

    draft_parts.append("We will keep the response customer-safe and avoid exposing internal notes.")

    return {
        "ticket_id": _value(ticket, "ticket_id"),
        "draft_reply": "\n\n".join(part for part in draft_parts if part),
        "used_policy_ids": [_value(policy, "policy_id") for policy in policies],
        "evidence": [
            _ticket_context(ticket),
            *[_policy_context(policy) for policy in policies],
        ],
    }


def check_ticket_sla(ticket_id: str) -> Record:
    dataset = get_enterprise_support_dataset()
    ticket = _find_by_id(dataset["tickets"], "ticket_id", ticket_id, "Ticket")
    sla_policy = _index_by(dataset["knowledge_base_docs"], "policy_id").get("pol_sla", {})
    priority = _value(ticket, "priority").lower()
    status = _value(ticket, "status").lower()
    sla_status = _value(ticket, "sla_status").lower()

    reasoning: list[str] = []
    if sla_status == "breached":
        reasoning.append("Ticket SLA status is breached.")
    elif sla_status == "at_risk":
        reasoning.append("Ticket SLA status is at_risk.")
    else:
        reasoning.append(f"Ticket SLA status is {sla_status or 'unknown'}.")

    if priority == "p1" and status not in CLOSED_STATUSES:
        reasoning.append("Open P1 tickets require active escalation under the SLA policy.")
    if priority == "p2" and status == "pending_engineering":
        reasoning.append("P2 tickets waiting on engineering should be monitored for escalation.")

    escalation_required = sla_status in ESCALATION_SLA_STATUSES or (
        priority == "p1" and status not in CLOSED_STATUSES
    )

    if escalation_required:
        recommendation = "Escalate to the owning support and engineering teams."
    else:
        recommendation = "No immediate SLA escalation is required based on current ticket fields."

    return {
        "ticket_id": _value(ticket, "ticket_id"),
        "priority": _value(ticket, "priority"),
        "status": _value(ticket, "status"),
        "sla_status": _value(ticket, "sla_status"),
        "first_response_due_at": _value(ticket, "first_response_due_at"),
        "resolution_due_at": _value(ticket, "resolution_due_at"),
        "escalation_required": escalation_required,
        "recommendation": recommendation,
        "reasoning": reasoning,
        "policy": _policy_context(sla_policy) if sla_policy else None,
    }
