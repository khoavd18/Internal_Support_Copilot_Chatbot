from __future__ import annotations

from pathlib import Path

import pytest
from src.data.enterprise_support_loader import load_enterprise_support_dataset
from src.ml.anomaly import RiskScoringError, explain_risk_score, score_customer_risk
from src.ml.risk_features import build_customer_risk_features
from src.ml.schemas import FEATURE_NAMES

DATA_DIR = Path("data/sample_enterprise_support")


def _load_dataset() -> dict:
    return load_enterprise_support_dataset(DATA_DIR)


def test_build_customer_risk_features_extracts_expected_signals() -> None:
    rows = build_customer_risk_features(_load_dataset())

    assert len(rows) == 10
    assert all(set(row["features"]) == set(FEATURE_NAMES) for row in rows)
    assert all(row["anchor_date"] for row in rows)

    cust_009 = next(row for row in rows if row["customer_id"] == "cust_009")
    assert cust_009["features"]["critical_ticket_count_30d"] >= 1
    assert cust_009["features"]["escalation_count_30d"] >= 1
    assert cust_009["features"]["negative_signal_count_30d"] >= 1
    assert "tkt_029" in cust_009["related_ticket_ids"]
    assert "risk_029" in cust_009["related_risk_event_ids"]


def test_score_customer_risk_returns_valid_range_and_metadata() -> None:
    result = score_customer_risk("cust_009", _load_dataset())

    assert result["customer_id"] == "cust_009"
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_level"] in {"low", "medium", "high", "critical"}
    assert set(result["features"]) == set(FEATURE_NAMES)
    assert result["model_metadata"]["model_type"] == "heuristic_anomaly_baseline"


def test_explain_risk_score_includes_reasons_and_related_events() -> None:
    explanation = explain_risk_score("cust_009", _load_dataset())

    assert explanation["customer_id"] == "cust_009"
    assert explanation["top_reasons"]
    assert any("escalation" in reason.lower() for reason in explanation["top_reasons"])
    assert explanation["related_events"]
    assert {event["source_type"] for event in explanation["related_events"]} >= {
        "ticket",
        "risk_event",
    }


def test_score_customer_risk_rejects_unknown_customer() -> None:
    with pytest.raises(RiskScoringError, match="Customer not found"):
        score_customer_risk("cust_missing", _load_dataset())
