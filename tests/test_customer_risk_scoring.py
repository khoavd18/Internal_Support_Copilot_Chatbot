from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
import src.ml.anomaly as anomaly
from scripts.train_risk_model import run_training
from src.data.enterprise_support_loader import load_enterprise_support_dataset
from src.ml.anomaly import (
    RiskScoringError,
    build_customer_feature_table,
    explain_anomaly_with_feature_deviation,
    explain_risk_score,
    score_customer_risk,
    score_with_isolation_forest,
    train_isolation_forest,
)
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


def test_build_customer_feature_table_returns_flat_model_rows() -> None:
    rows = build_customer_feature_table(_load_dataset())

    assert len(rows) == 10
    assert all("customer_id" in row for row in rows)
    assert all(set(FEATURE_NAMES).issubset(row) for row in rows)
    assert all(isinstance(row[name], float) for row in rows for name in FEATURE_NAMES)


def test_score_customer_risk_returns_valid_range_and_metadata() -> None:
    result = score_customer_risk("cust_009", _load_dataset())

    assert result["customer_id"] == "cust_009"
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_level"] in {"low", "medium", "high", "critical"}
    assert set(result["features"]) == set(FEATURE_NAMES)
    assert result["model_metadata"]["model_type"] == "heuristic_anomaly_baseline"


def test_isolation_forest_trains_and_scores_when_dependency_is_available() -> None:
    pytest.importorskip("sklearn.ensemble")
    feature_table = build_customer_feature_table(_load_dataset())

    model = train_isolation_forest(feature_table)
    result = score_with_isolation_forest("cust_009", model, feature_table)
    reasons = explain_anomaly_with_feature_deviation("cust_009", feature_table)

    assert result["customer_id"] == "cust_009"
    assert 0 <= result["risk_score"] <= 100
    assert result["risk_level"] in {"low", "medium", "high", "critical"}
    assert result["model_metadata"]["model_type"] == "isolation_forest"
    assert reasons


def test_ml_mode_falls_back_to_heuristic_when_sklearn_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(anomaly, "_load_isolation_forest_class", lambda: None)

    result = score_customer_risk("cust_009", _load_dataset(), mode="ml")

    assert result["model_metadata"]["model_type"] == "heuristic_anomaly_baseline"
    assert result["model_metadata"]["requested_mode"] == "ml"
    assert result["model_metadata"]["fallback_used"] is True
    assert "scikit-learn is not installed" in result["model_metadata"]["fallback_reason"]


def test_train_risk_model_dry_run_does_not_write_artifact(tmp_path: Path) -> None:
    stdout = StringIO()
    stderr = StringIO()
    output_path = tmp_path / "risk_model.pkl"

    exit_code = run_training(
        data_dir=DATA_DIR,
        output_path=output_path,
        dry_run=True,
        stdout=stdout,
        stderr=stderr,
    )

    assert exit_code == 0
    assert not output_path.exists()
    assert "Customer rows: 10" in stdout.getvalue()
    assert "Dry run: no model trained or written." in stdout.getvalue()
    assert stderr.getvalue() == ""


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
