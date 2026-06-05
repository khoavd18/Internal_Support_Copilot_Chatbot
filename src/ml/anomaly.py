from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from src.ml.risk_features import build_customer_risk_features
from src.ml.schemas import FEATURE_NAMES

FEATURE_WEIGHTS = {
    "ticket_count_7d": 8.0,
    "critical_ticket_count_30d": 20.0,
    "escalation_count_30d": 20.0,
    "failed_login_count_7d": 12.0,
    "api_timeout_count_7d": 12.0,
    "refund_request_count_30d": 8.0,
    "negative_signal_count_30d": 20.0,
}

FEATURE_REASON_LABELS = {
    "ticket_count_7d": "recent support ticket volume",
    "critical_ticket_count_30d": "critical or incident ticket activity",
    "escalation_count_30d": "SLA or engineering escalation activity",
    "failed_login_count_7d": "recent login, SSO, MFA, or auth failures",
    "api_timeout_count_7d": "recent API timeout or latency signals",
    "refund_request_count_30d": "billing, refund, credit, or invoice signals",
    "negative_signal_count_30d": "negative risk, incident, sentiment, or SLA signals",
}


class RiskScoringError(ValueError):
    """Raised when a customer cannot be scored from the synthetic dataset."""


@dataclass(frozen=True)
class HeuristicAnomalyModel:
    feature_names: tuple[str, ...]
    baselines: dict[str, float]
    maxima: dict[str, float]
    model_type: str = "heuristic_anomaly_baseline"

    def score_features(self, features: dict[str, float]) -> float:
        weighted_score = 0.0
        anomaly_bonus = 0.0

        for feature_name in self.feature_names:
            value = float(features.get(feature_name, 0.0))
            maximum = max(self.maxima.get(feature_name, 0.0), 1.0)
            weighted_score += (value / maximum) * FEATURE_WEIGHTS[feature_name]

            baseline = self.baselines.get(feature_name, 0.0)
            if value > baseline:
                spread = max(maximum - baseline, 1.0)
                anomaly_bonus += ((value - baseline) / spread) * 3.0

        return max(0.0, min(100.0, weighted_score + anomaly_bonus))


def train_anomaly_model(features: Iterable[dict] | None) -> HeuristicAnomalyModel:
    rows = list(features or [])
    baselines: dict[str, float] = {}
    maxima: dict[str, float] = {}

    for feature_name in FEATURE_NAMES:
        values = [float(row.get("features", {}).get(feature_name, 0.0)) for row in rows]
        baselines[feature_name] = sum(values) / len(values) if values else 0.0
        maxima[feature_name] = max(values) if values else 0.0

    return HeuristicAnomalyModel(
        feature_names=FEATURE_NAMES,
        baselines=baselines,
        maxima=maxima,
    )


def score_customer_risk(customer_id: str, dataset: dict) -> dict[str, Any]:
    feature_rows = build_customer_risk_features(dataset)
    row = _find_feature_row(customer_id, feature_rows)
    model = train_anomaly_model(feature_rows)
    score = round(model.score_features(row["features"]), 2)

    return {
        "customer_id": row["customer_id"],
        "risk_score": score,
        "risk_level": _risk_level(score),
        "features": row["features"],
        "related_ticket_ids": row["related_ticket_ids"],
        "related_risk_event_ids": row["related_risk_event_ids"],
        "model_metadata": {
            "model_type": model.model_type,
            "feature_names": list(model.feature_names),
            "anchor_date": row.get("anchor_date", ""),
            "dependency": "scikit-learn not configured; using deterministic heuristic baseline",
        },
    }


def explain_risk_score(customer_id: str, dataset: dict) -> dict[str, Any]:
    score_result = score_customer_risk(customer_id, dataset)
    top_reasons = _top_reasons(score_result["features"])
    related_events = _related_events(
        dataset,
        customer_id=str(customer_id),
        related_ticket_ids=set(score_result.get("related_ticket_ids", [])),
        related_risk_event_ids=set(score_result.get("related_risk_event_ids", [])),
    )

    return {
        "customer_id": score_result["customer_id"],
        "risk_score": score_result["risk_score"],
        "risk_level": score_result["risk_level"],
        "top_reasons": top_reasons,
        "related_events": related_events,
        "features": score_result["features"],
        "model_metadata": score_result["model_metadata"],
    }


def _find_feature_row(customer_id: str, feature_rows: list[dict]) -> dict:
    normalized = str(customer_id or "").strip()
    for row in feature_rows:
        if str(row.get("customer_id") or "").strip() == normalized:
            return row
    raise RiskScoringError(f"Customer not found: {normalized}")


def _top_reasons(features: dict[str, float], limit: int = 4) -> list[str]:
    ranked = sorted(
        (
            (float(features.get(feature_name, 0.0)), FEATURE_WEIGHTS[feature_name], feature_name)
            for feature_name in FEATURE_NAMES
        ),
        key=lambda item: item[0] * item[1],
        reverse=True,
    )
    reasons = []
    for value, weight, feature_name in ranked:
        value = float(features.get(feature_name, 0.0))
        if value <= 0:
            continue
        reasons.append(
            f"{FEATURE_REASON_LABELS[feature_name]}: {value:g} signal(s), weight {weight:.0f}"
        )
        if len(reasons) >= limit:
            break

    if not reasons:
        reasons.append("No recent risk feature signals were found for this customer.")
    return reasons


def _related_events(
    dataset: dict,
    *,
    customer_id: str,
    related_ticket_ids: set[str],
    related_risk_event_ids: set[str],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []

    for ticket in dataset.get("tickets", []):
        ticket_id = _value(ticket, "ticket_id")
        if ticket_id not in related_ticket_ids:
            continue
        events.append(
            {
                "source_type": "ticket",
                "entity_id": ticket_id,
                "title": _value(ticket, "title"),
                "summary": _value(ticket, "description"),
                "created_at": _value(ticket, "created_at"),
                "metadata": {
                    "customer_id": customer_id,
                    "ticket_id": ticket_id,
                    "priority": _value(ticket, "priority"),
                    "severity": _value(ticket, "severity"),
                    "status": _value(ticket, "status"),
                    "sla_status": _value(ticket, "sla_status"),
                    "product_id": _value(ticket, "product_id"),
                    "service_id": _value(ticket, "service_id"),
                },
            }
        )

    for event in dataset.get("risk_events", []):
        risk_event_id = _value(event, "risk_event_id")
        if risk_event_id not in related_risk_event_ids:
            continue
        events.append(
            {
                "source_type": "risk_event",
                "entity_id": risk_event_id,
                "title": _value(event, "event_type"),
                "summary": _value(event, "summary"),
                "created_at": _value(event, "detected_at"),
                "metadata": {
                    "customer_id": customer_id,
                    "ticket_id": _value(event, "ticket_id"),
                    "risk_score": _value(event, "risk_score"),
                    "severity": _value(event, "severity"),
                    "status": _value(event, "status"),
                    "product_id": _value(event, "product_id"),
                    "service_id": _value(event, "service_id"),
                },
            }
        )

    events.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    return events[:12]


def _risk_level(score: float) -> str:
    if score >= 80:
        return "critical"
    if score >= 60:
        return "high"
    if score >= 35:
        return "medium"
    return "low"


def _value(record: dict | None, key: str, default: str = "") -> str:
    if not record:
        return default
    value = record.get(key, default)
    if value is None:
        return default
    return str(value).strip()
