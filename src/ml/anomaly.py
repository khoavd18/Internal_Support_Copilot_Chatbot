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


class OptionalMLDependencyError(RiskScoringError):
    """Raised when optional ML scoring is requested without its dependency."""


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


def build_customer_feature_table(dataset: dict) -> list[dict[str, Any]]:
    """Return one flat numeric feature row per customer for ML baselines."""

    rows: list[dict[str, Any]] = []
    for row in build_customer_risk_features(dataset):
        flattened: dict[str, Any] = {
            "customer_id": row["customer_id"],
            "anchor_date": row.get("anchor_date", ""),
            "related_ticket_ids": list(row.get("related_ticket_ids", [])),
            "related_risk_event_ids": list(row.get("related_risk_event_ids", [])),
        }
        flattened.update(
            {
                feature_name: float(row.get("features", {}).get(feature_name, 0.0))
                for feature_name in FEATURE_NAMES
            }
        )
        rows.append(flattened)

    rows.sort(key=lambda item: str(item.get("customer_id", "")))
    return rows


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


def train_isolation_forest(features: Iterable[dict]):
    """Train an optional IsolationForest model from flat customer feature rows."""

    isolation_forest_cls = _load_isolation_forest_class()
    if isolation_forest_cls is None:
        raise OptionalMLDependencyError(
            "scikit-learn is not installed; install it to enable IsolationForest risk scoring."
        )

    rows = list(features or [])
    if len(rows) < 2:
        raise RiskScoringError(
            "At least two customer feature rows are required to train IsolationForest."
        )

    _, matrix = _feature_matrix(rows)
    model = isolation_forest_cls(
        n_estimators=100,
        contamination="auto",
        random_state=42,
    )
    model.fit(matrix)
    model.feature_names_ = list(FEATURE_NAMES)
    return model


def score_with_isolation_forest(
    customer_id: str, model, features: Iterable[dict]
) -> dict[str, Any]:
    """Score one customer with a trained IsolationForest model on a 0-100 risk scale."""

    rows = list(features or [])
    row = _find_feature_row(customer_id, rows)
    _, matrix = _feature_matrix(rows)
    vector = [_feature_value(row, feature_name) for feature_name in FEATURE_NAMES]

    cohort_scores = [float(score) for score in model.decision_function(matrix)]
    customer_score = float(model.decision_function([vector])[0])
    risk_score = _normalize_isolation_score(customer_score, cohort_scores)

    return {
        "customer_id": row["customer_id"],
        "risk_score": round(risk_score, 2),
        "risk_level": _risk_level(risk_score),
        "features": {
            feature_name: _feature_value(row, feature_name) for feature_name in FEATURE_NAMES
        },
        "related_ticket_ids": list(row.get("related_ticket_ids", [])),
        "related_risk_event_ids": list(row.get("related_risk_event_ids", [])),
        "model_metadata": {
            "model_type": "isolation_forest",
            "feature_names": list(FEATURE_NAMES),
            "anchor_date": row.get("anchor_date", ""),
            "dependency": "scikit-learn",
            "fallback_used": False,
        },
    }


def explain_anomaly_with_feature_deviation(
    customer_id: str,
    features: Iterable[dict],
    limit: int = 4,
) -> list[str]:
    """Explain anomaly score by comparing customer features to cohort averages."""

    rows = list(features or [])
    row = _find_feature_row(customer_id, rows)
    baselines = _feature_baselines(rows)
    deviations: list[tuple[float, float, str, float]] = []

    for feature_name in FEATURE_NAMES:
        value = _feature_value(row, feature_name)
        baseline = baselines.get(feature_name, 0.0)
        deviation = value - baseline
        relative = deviation / max(baseline, 1.0)
        if value > 0 and deviation > 0:
            deviations.append((relative, deviation, feature_name, value))

    deviations.sort(reverse=True)
    reasons = [
        (
            f"{FEATURE_REASON_LABELS[feature_name]} is above cohort baseline: "
            f"{value:g} signal(s) vs {baselines.get(feature_name, 0.0):.2f} average"
        )
        for _, _, feature_name, value in deviations[:limit]
    ]

    if not reasons:
        reasons.append("No feature materially exceeds the synthetic cohort baseline.")
    return reasons


def score_customer_risk(customer_id: str, dataset: dict, mode: str = "heuristic") -> dict[str, Any]:
    feature_rows = build_customer_risk_features(dataset)
    row = _find_feature_row(customer_id, feature_rows)

    if mode == "ml":
        return _score_customer_risk_with_optional_ml(customer_id, dataset)
    if mode != "heuristic":
        raise RiskScoringError(f"Unsupported risk scoring mode: {mode}")

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
            "fallback_used": False,
        },
    }


def explain_risk_score(customer_id: str, dataset: dict, mode: str = "heuristic") -> dict[str, Any]:
    score_result = score_customer_risk(customer_id, dataset, mode=mode)
    if score_result["model_metadata"].get("model_type") == "isolation_forest":
        top_reasons = explain_anomaly_with_feature_deviation(
            score_result["customer_id"],
            build_customer_feature_table(dataset),
        )
    else:
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


def _score_customer_risk_with_optional_ml(customer_id: str, dataset: dict) -> dict[str, Any]:
    feature_table = build_customer_feature_table(dataset)
    _find_feature_row(customer_id, feature_table)

    try:
        model = train_isolation_forest(feature_table)
        return score_with_isolation_forest(customer_id, model, feature_table)
    except RiskScoringError as exc:
        fallback = score_customer_risk(customer_id, dataset, mode="heuristic")
        fallback["model_metadata"] = {
            **fallback["model_metadata"],
            "requested_mode": "ml",
            "fallback_used": True,
            "fallback_reason": str(exc),
        }
        return fallback


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


def _feature_matrix(rows: list[dict]) -> tuple[list[str], list[list[float]]]:
    return (
        list(FEATURE_NAMES),
        [[_feature_value(row, feature_name) for feature_name in FEATURE_NAMES] for row in rows],
    )


def _feature_value(row: dict, feature_name: str) -> float:
    if "features" in row and isinstance(row["features"], dict):
        return float(row["features"].get(feature_name, 0.0))
    return float(row.get(feature_name, 0.0))


def _feature_baselines(rows: list[dict]) -> dict[str, float]:
    baselines: dict[str, float] = {}
    for feature_name in FEATURE_NAMES:
        values = [_feature_value(row, feature_name) for row in rows]
        baselines[feature_name] = sum(values) / len(values) if values else 0.0
    return baselines


def _normalize_isolation_score(customer_score: float, cohort_scores: list[float]) -> float:
    """Convert IsolationForest decision scores to risk: lower decision score means higher risk."""

    if not cohort_scores:
        return 50.0

    low = min(cohort_scores)
    high = max(cohort_scores)
    if high <= low:
        return 50.0

    normalized_normality = (customer_score - low) / (high - low)
    return max(0.0, min(100.0, (1.0 - normalized_normality) * 100.0))


def _load_isolation_forest_class():
    try:
        from sklearn.ensemble import IsolationForest
    except ImportError:
        return None
    return IsolationForest


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
