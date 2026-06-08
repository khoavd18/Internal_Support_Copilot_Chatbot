from __future__ import annotations

import src.ml.anomaly as anomaly
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_enterprise_related_routes_are_registered() -> None:
    registered = {
        (route.path, method) for route in app.routes for method in getattr(route, "methods", set())
    }

    assert ("/crm/customer-summary", "POST") in registered
    assert ("/support/ticket-triage", "POST") in registered
    assert ("/support/suggest-reply", "POST") in registered
    assert ("/support/sla-check", "POST") in registered
    assert ("/risk/customer-score", "POST") in registered
    assert ("/enterprise/ask", "POST") in registered


def test_customer_summary_endpoint_returns_customer_account_ticket_and_risk_context() -> None:
    response = client.post("/crm/customer-summary", json={"customer_id": "cust_001"})

    assert response.status_code == 200
    body = response.json()
    assert body["customer_id"] == "cust_001"
    assert body["customer_name"] == "Avery Chen"
    assert body["account_id"] == "acct_001"
    assert body["account_name"] == "Northstar Analytics"
    assert "active tickets" in body["summary"]
    assert body["stats"]["ticket_count"] == 3
    assert body["stats"]["risk_event_count"] >= 3
    assert any(item["entity_id"] == "tkt_001" for item in body["tickets"])
    assert any(item["source_type"] == "risk_event" for item in body["risk_events"])


def test_ticket_triage_endpoint_returns_simple_rule_based_recommendation() -> None:
    response = client.post("/support/ticket-triage", json={"ticket_id": "tkt_001"})

    assert response.status_code == 200
    body = response.json()
    assert body["ticket_id"] == "tkt_001"
    assert body["current_priority"] == "p1"
    assert body["recommended_priority"] == "p1"
    assert body["recommended_status"] == "pending_engineering"
    assert body["escalation_required"] is True
    assert any("SLA status is breached" in reason for reason in body["reasoning"])
    assert any(item["source_type"] == "service" for item in body["context"])


def test_suggest_reply_endpoint_returns_grounded_non_llm_draft() -> None:
    response = client.post("/support/suggest-reply", json={"ticket_id": "tkt_001"})

    assert response.status_code == 200
    body = response.json()
    assert body["ticket_id"] == "tkt_001"
    assert "Hi Avery" in body["draft_reply"]
    assert "API timeout during batch sync" in body["draft_reply"]
    assert "customer-safe" in body["draft_reply"]
    assert "pol_sla" in body["used_policy_ids"]
    assert "pol_api_timeout" in body["used_policy_ids"]
    assert any(item["source_type"] == "knowledge_base" for item in body["evidence"])


def test_sla_check_endpoint_uses_ticket_sla_fields_and_policy_context() -> None:
    response = client.post("/support/sla-check", json={"ticket_id": "tkt_026"})

    assert response.status_code == 200
    body = response.json()
    assert body["ticket_id"] == "tkt_026"
    assert body["priority"] == "p2"
    assert body["sla_status"] == "at_risk"
    assert body["escalation_required"] is True
    assert "Escalate" in body["recommendation"]
    assert body["policy"]["entity_id"] == "pol_sla"


def test_customer_risk_score_endpoint_returns_explanation_and_events() -> None:
    response = client.post("/risk/customer-score", json={"customer_id": "cust_009"})

    assert response.status_code == 200
    body = response.json()
    assert body["customer_id"] == "cust_009"
    assert 0 <= body["risk_score"] <= 100
    assert body["risk_level"] in {"low", "medium", "high", "critical"}
    assert body["top_reasons"]
    assert any("escalation" in reason.lower() for reason in body["top_reasons"])
    assert any(item["source_type"] == "risk_event" for item in body["related_events"])
    assert body["features"]["negative_signal_count_30d"] >= 1


def test_customer_risk_score_endpoint_accepts_ml_mode_with_fallback(monkeypatch) -> None:
    monkeypatch.setattr(anomaly, "_load_isolation_forest_class", lambda: None)

    response = client.post("/risk/customer-score", json={"customer_id": "cust_009", "mode": "ml"})

    assert response.status_code == 200
    body = response.json()
    assert body["customer_id"] == "cust_009"
    assert body["model_metadata"]["requested_mode"] == "ml"
    assert body["model_metadata"]["fallback_used"] is True
    assert body["model_metadata"]["model_type"] == "heuristic_anomaly_baseline"


def test_enterprise_support_endpoints_return_404_for_unknown_ids() -> None:
    response = client.post("/support/ticket-triage", json={"ticket_id": "tkt_missing"})

    assert response.status_code == 404
    assert "Ticket not found" in response.json()["detail"]


def test_customer_risk_score_endpoint_returns_404_for_unknown_customer() -> None:
    response = client.post("/risk/customer-score", json={"customer_id": "cust_missing"})

    assert response.status_code == 404
    assert "Customer not found" in response.json()["detail"]


def test_enterprise_ask_endpoint_returns_graphrag_evidence(monkeypatch) -> None:
    def _fake_retrieve_enterprise_context(
        query: str,
        top_k: int,
        graph_depth: int,
        filters=None,
    ):
        assert query == "Why is the API timeout risky?"
        assert top_k == 2
        assert graph_depth == 1
        assert filters is None
        return {
            "vector_evidence": [
                {
                    "id": "Ticket:tkt_001",
                    "text": "Vector ticket evidence",
                    "metadata": {"ticket_id": "tkt_001", "source_type": "ticket"},
                    "context_source": "vector",
                    "source_type": "ticket",
                    "title": "API timeout during batch sync",
                }
            ],
            "graph_evidence": [
                {
                    "id": "RiskEvent:risk_001",
                    "text": "Graph risk evidence",
                    "metadata": {"risk_event_id": "risk_001", "source_type": "risk_event"},
                    "context_source": "graph",
                    "source_type": "risk_event",
                    "title": "P1 API timeout breached resolution target",
                }
            ],
            "merged_context": [
                {
                    "id": "Ticket:tkt_001",
                    "text": (
                        "Support Ticket\n"
                        "Ticket ID: tkt_001\n"
                        "Title: API timeout during batch sync\n"
                        "SLA status: breached"
                    ),
                    "metadata": {
                        "ticket_id": "tkt_001",
                        "entity_id": "tkt_001",
                        "source_type": "ticket",
                    },
                    "context_source": "both",
                    "source_type": "ticket",
                    "title": "API timeout during batch sync",
                },
                {
                    "id": "RiskEvent:risk_001",
                    "text": (
                        "Risk Event\n"
                        "Risk event ID: risk_001\n"
                        "Summary: P1 API timeout breached resolution target"
                    ),
                    "metadata": {
                        "risk_event_id": "risk_001",
                        "entity_id": "risk_001",
                        "source_type": "risk_event",
                    },
                    "context_source": "graph",
                    "source_type": "risk_event",
                    "title": "P1 API timeout breached resolution target",
                },
            ],
            "citations": [
                {
                    "index": 1,
                    "id": "Ticket:tkt_001",
                    "title": "API timeout during batch sync",
                    "source_type": "ticket",
                    "context_source": "both",
                    "metadata": {"ticket_id": "tkt_001"},
                }
            ],
            "stats": {
                "top_k": top_k,
                "graph_depth": graph_depth,
                "merged_count": 1,
                "vector_error": "",
            },
        }

    monkeypatch.setattr(
        "src.api.routes.enterprise.retrieve_enterprise_context",
        _fake_retrieve_enterprise_context,
    )

    response = client.post(
        "/enterprise/ask",
        json={
            "question": "Why is the API timeout risky?",
            "top_k": 2,
            "graph_depth": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "Based only on the retrieved enterprise support evidence" in body["answer"]
    assert "SLA status: breached" in body["answer"]
    assert body["confidence"] == "medium"
    assert body["vector_evidence"][0]["id"] == "Ticket:tkt_001"
    assert body["graph_evidence"][0]["id"] == "RiskEvent:risk_001"
    assert body["merged_context"][0]["context_source"] == "both"
    assert body["citations"][0]["id"] == "Ticket:tkt_001"
    assert body["citations"][0]["entity_id"] == "tkt_001"
    assert body["citations"][0]["source_type"] == "ticket"
    assert body["citations"][0]["title"] == "API timeout during batch sync"
    assert body["citations"][0]["snippet"]
    assert body["citations"][0]["used_for_answer"] is True
    assert body["metadata"]["mode"] == "deterministic_grounded_generation"
    assert body["metadata"]["evidence_sufficiency"]["level"] == "medium"
    assert body["stats"]["evidence_sufficiency_level"] == "medium"
    assert body["stats"]["missing_source_types"] == ["service"]


def test_enterprise_ask_endpoint_reports_missing_information_without_hallucinating(
    monkeypatch,
) -> None:
    def _fake_retrieve_enterprise_context(
        query: str,
        top_k: int,
        graph_depth: int,
        filters=None,
    ):
        assert query == "What is cust_001's private phone number?"
        assert filters is None
        return {
            "vector_evidence": [],
            "graph_evidence": [
                {
                    "id": "Customer:cust_001",
                    "text": "Customer Profile\nCustomer ID: cust_001\nName: Avery Chen",
                    "metadata": {
                        "customer_id": "cust_001",
                        "entity_id": "cust_001",
                        "source_type": "customer",
                    },
                    "context_source": "graph",
                    "source_type": "customer",
                    "title": "Avery Chen",
                }
            ],
            "merged_context": [
                {
                    "id": "Customer:cust_001",
                    "text": (
                        "Customer Profile\n"
                        "Customer ID: cust_001\n"
                        "Name: Avery Chen\n"
                        "Preferred contact channel: Slack"
                    ),
                    "metadata": {
                        "customer_id": "cust_001",
                        "entity_id": "cust_001",
                        "source_type": "customer",
                    },
                    "context_source": "graph",
                    "source_type": "customer",
                    "title": "Avery Chen",
                }
            ],
            "citations": [],
            "stats": {
                "top_k": top_k,
                "graph_depth": graph_depth,
                "merged_count": 1,
                "vector_error": "",
            },
        }

    monkeypatch.setattr(
        "src.api.routes.enterprise.retrieve_enterprise_context",
        _fake_retrieve_enterprise_context,
    )

    response = client.post(
        "/enterprise/ask",
        json={
            "question": "What is cust_001's private phone number?",
            "top_k": 2,
            "graph_depth": 1,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["confidence"] == "low"
    assert "Missing information" in body["answer"]
    assert "private phone number" in body["answer"]
    assert "555" not in body["answer"]
    assert body["citations"][0]["entity_id"] == "cust_001"
    assert body["metadata"]["missing_information"]
    assert body["stats"]["missing_information_count"] == 1
    assert body["metadata"]["evidence_sufficiency"]["level"] == "medium"


def test_enterprise_ask_endpoint_passes_metadata_filters(monkeypatch) -> None:
    captured = {}

    def _fake_retrieve_enterprise_context(
        query: str,
        top_k: int,
        graph_depth: int,
        filters=None,
    ):
        captured["filters"] = filters
        return {
            "vector_evidence": [],
            "graph_evidence": [],
            "merged_context": [],
            "citations": [],
            "stats": {
                "top_k": top_k,
                "graph_depth": graph_depth,
                "merged_count": 0,
                "vector_error": "",
            },
        }

    monkeypatch.setattr(
        "src.api.routes.enterprise.retrieve_enterprise_context",
        _fake_retrieve_enterprise_context,
    )

    response = client.post(
        "/enterprise/ask",
        json={
            "question": "Show only tkt_001 API timeout evidence",
            "top_k": 2,
            "graph_depth": 1,
            "source_type": "ticket",
            "customer_id": "cust_001",
            "ticket_id": "tkt_001",
        },
    )

    assert response.status_code == 200
    assert captured["filters"] == {
        "source_type": "ticket",
        "customer_id": "cust_001",
        "ticket_id": "tkt_001",
    }
    body = response.json()
    assert body["metadata"]["filters"] == captured["filters"]
    assert body["metadata"]["evidence_sufficiency"]["level"] == "low"
    assert body["stats"]["missing_source_types"] == ["service", "ticket"]


def test_enterprise_ask_endpoint_uses_agentic_retrieval_when_requested(monkeypatch) -> None:
    captured = {}

    def _fake_agentic_retrieval(
        query: str,
        top_k: int,
        graph_depth: int,
        base_filters=None,
    ):
        captured["query"] = query
        captured["top_k"] = top_k
        captured["graph_depth"] = graph_depth
        captured["base_filters"] = base_filters
        trace = {
            "intent": "risk_explanation",
            "retrieval_attempts": 2,
            "filters_used": [
                {"source_type": ["risk_event", "ticket", "customer", "service"]},
                {
                    "source_type": [
                        "knowledge_base",
                        "risk_event",
                        "service",
                        "ticket",
                    ]
                },
            ],
            "sufficiency_before": {"score": 0.1, "level": "low", "missing_source_types": []},
            "sufficiency_after": {
                "score": 0.82,
                "level": "high",
                "missing_source_types": [],
            },
            "stop_reason": "sufficient_evidence",
        }
        return {
            "query": query,
            "vector_evidence": [],
            "graph_evidence": [],
            "merged_context": [
                {
                    "id": "Ticket:tkt_001",
                    "text": (
                        "Support Ticket\n"
                        "Ticket ID: tkt_001\n"
                        "Title: API timeout during batch sync\n"
                        "SLA status: breached"
                    ),
                    "metadata": {
                        "entity_id": "tkt_001",
                        "ticket_id": "tkt_001",
                        "source_type": "ticket",
                    },
                    "context_source": "both",
                    "source_type": "ticket",
                    "title": "API timeout during batch sync",
                },
                {
                    "id": "Service:svc_api_gateway",
                    "text": "Service ID: svc_api_gateway\nOwner team: Reliability Engineering",
                    "metadata": {
                        "entity_id": "svc_api_gateway",
                        "service_id": "svc_api_gateway",
                        "source_type": "service",
                    },
                    "context_source": "graph",
                    "source_type": "service",
                    "title": "API Gateway",
                },
            ],
            "citations": [],
            "stats": {"merged_count": 2, "agentic_retrieval": True, "agentic_trace": trace},
        }

    monkeypatch.setattr(
        "src.api.routes.enterprise.retrieve_enterprise_context_agentically",
        _fake_agentic_retrieval,
    )

    response = client.post(
        "/enterprise/ask",
        json={
            "question": "Why is tkt_001 risky?",
            "top_k": 2,
            "graph_depth": 1,
            "use_agentic_retrieval": True,
            "debug": True,
        },
    )

    assert response.status_code == 200
    assert captured == {
        "query": "Why is tkt_001 risky?",
        "top_k": 2,
        "graph_depth": 1,
        "base_filters": None,
    }
    body = response.json()
    assert body["metadata"]["retrieval_mode"] == "agentic"
    assert body["metadata"]["agentic_trace"]["retrieval_attempts"] == 2
    assert body["stats"]["agentic_trace"]["intent"] == "risk_explanation"
    assert body["metadata"]["debug"]["agentic_trace"]["stop_reason"] == "sufficient_evidence"
