from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

from src.ml.schemas import FEATURE_NAMES

Record = dict[str, Any]
Dataset = dict[str, list[Record]]

LOGIN_TERMS = ("login", "sso", "scim", "session", "mfa", "auth")
API_TIMEOUT_TERMS = ("api", "timeout", "latency", "gateway", "webhook", "504")
REFUND_TERMS = ("refund", "billing", "credit", "invoice", "tax", "sla_credit")
NEGATIVE_RISK_TYPES = {
    "churn_risk",
    "incident",
    "security",
    "sentiment_drop",
    "sla_breach",
    "usage_anomaly",
}
NEGATIVE_SEVERITIES = {"high", "critical"}
ESCALATION_SLA_STATUSES = {"at_risk", "breached"}


def build_customer_risk_features(dataset: dict) -> list[dict]:
    typed_dataset: Dataset = dataset
    anchor = _dataset_anchor_datetime(typed_dataset)
    rows: list[dict] = []

    for customer in typed_dataset.get("customers", []):
        customer_id = _value(customer, "customer_id")
        tickets = [
            ticket
            for ticket in typed_dataset.get("tickets", [])
            if _value(ticket, "customer_id") == customer_id
        ]
        risk_events = [
            event
            for event in typed_dataset.get("risk_events", [])
            if _value(event, "customer_id") == customer_id
        ]

        recent_7d_tickets = [
            ticket for ticket in tickets if _within_days(ticket, "created_at", anchor, 7)
        ]
        recent_30d_tickets = [
            ticket for ticket in tickets if _within_days(ticket, "created_at", anchor, 30)
        ]
        recent_7d_risks = [
            event for event in risk_events if _within_days(event, "detected_at", anchor, 7)
        ]
        recent_30d_risks = [
            event for event in risk_events if _within_days(event, "detected_at", anchor, 30)
        ]

        features = {
            "ticket_count_7d": float(len(recent_7d_tickets)),
            "critical_ticket_count_30d": float(
                sum(1 for ticket in recent_30d_tickets if _is_critical_ticket(ticket))
            ),
            "escalation_count_30d": float(
                sum(1 for ticket in recent_30d_tickets if _needs_escalation(ticket))
                + sum(1 for event in recent_30d_risks if _is_escalation_risk(event))
            ),
            "failed_login_count_7d": float(
                sum(
                    1
                    for record in [*recent_7d_tickets, *recent_7d_risks]
                    if _has_terms(record, LOGIN_TERMS)
                )
            ),
            "api_timeout_count_7d": float(
                sum(
                    1
                    for record in [*recent_7d_tickets, *recent_7d_risks]
                    if _has_terms(record, API_TIMEOUT_TERMS)
                )
            ),
            "refund_request_count_30d": float(
                sum(
                    1
                    for record in [*recent_30d_tickets, *recent_30d_risks]
                    if _has_terms(record, REFUND_TERMS)
                )
            ),
            "negative_signal_count_30d": float(
                sum(1 for ticket in recent_30d_tickets if _is_negative_ticket(ticket))
                + sum(1 for event in recent_30d_risks if _is_negative_risk(event))
            ),
        }

        rows.append(
            {
                "customer_id": customer_id,
                "features": {name: features[name] for name in FEATURE_NAMES},
                "related_ticket_ids": [
                    _value(ticket, "ticket_id") for ticket in recent_30d_tickets
                ],
                "related_risk_event_ids": [
                    _value(event, "risk_event_id") for event in recent_30d_risks
                ],
                "anchor_date": anchor.isoformat().replace("+00:00", "Z"),
            }
        )

    return rows


def _dataset_anchor_datetime(dataset: Dataset) -> datetime:
    timestamps: list[datetime] = []
    for ticket in dataset.get("tickets", []):
        for field in ("created_at", "updated_at"):
            parsed = _parse_datetime(_value(ticket, field))
            if parsed:
                timestamps.append(parsed)
    for event in dataset.get("risk_events", []):
        parsed = _parse_datetime(_value(event, "detected_at"))
        if parsed:
            timestamps.append(parsed)

    if not timestamps:
        return datetime.now(timezone.utc)
    return max(timestamps)


def _within_days(record: Record, field: str, anchor: datetime, days: int) -> bool:
    parsed = _parse_datetime(_value(record, field))
    if parsed is None:
        return False
    delta = anchor - parsed
    return timedelta(0) <= delta <= timedelta(days=days)


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _is_critical_ticket(ticket: Record) -> bool:
    return (
        _value(ticket, "priority").lower() == "p1"
        or _value(ticket, "severity").lower() == "sev1"
        or _value(ticket, "category").lower() == "incident"
    )


def _needs_escalation(ticket: Record) -> bool:
    return (
        _value(ticket, "sla_status").lower() in ESCALATION_SLA_STATUSES
        or _value(ticket, "status").lower() == "pending_engineering"
        or _value(ticket, "priority").lower() == "p1"
    )


def _is_escalation_risk(event: Record) -> bool:
    return _value(event, "severity").lower() in NEGATIVE_SEVERITIES or _value(
        event, "event_type"
    ).lower() in {"incident", "security", "sla_breach"}


def _is_negative_ticket(ticket: Record) -> bool:
    text = _record_text(ticket)
    return (
        _needs_escalation(ticket)
        or "frustrated" in text
        or "urgent" in text
        or _value(ticket, "priority").lower() in {"p1", "p2"}
    )


def _is_negative_risk(event: Record) -> bool:
    return (
        _value(event, "event_type").lower() in NEGATIVE_RISK_TYPES
        or _value(event, "severity").lower() in NEGATIVE_SEVERITIES
        or _safe_int(_value(event, "risk_score")) >= 70
    )


def _has_terms(record: Record, terms: tuple[str, ...]) -> bool:
    text = _record_text(record)
    return any(term in text for term in terms)


def _record_text(record: Record) -> str:
    values: list[str] = []
    for value in record.values():
        if isinstance(value, (dict, list)):
            continue
        values.append(str(value))
    return re.sub(r"\s+", " ", " ".join(values).lower())


def _value(record: Record | None, key: str, default: str = "") -> str:
    if not record:
        return default
    value = record.get(key, default)
    if value is None:
        return default
    return str(value).strip()


def _safe_int(value: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
