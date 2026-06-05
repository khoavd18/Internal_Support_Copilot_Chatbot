from __future__ import annotations

from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


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


def test_enterprise_support_endpoints_return_404_for_unknown_ids() -> None:
    response = client.post("/support/ticket-triage", json={"ticket_id": "tkt_missing"})

    assert response.status_code == 404
    assert "Ticket not found" in response.json()["detail"]


def test_enterprise_ask_endpoint_returns_graphrag_evidence(monkeypatch) -> None:
    def _fake_retrieve_enterprise_context(query: str, top_k: int, graph_depth: int):
        assert query == "Why is the API timeout risky?"
        assert top_k == 2
        assert graph_depth == 1
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
                    "text": "Merged ticket evidence",
                    "metadata": {"ticket_id": "tkt_001", "source_type": "ticket"},
                    "context_source": "both",
                    "source_type": "ticket",
                    "title": "API timeout during batch sync",
                }
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
        "src.api.main.retrieve_enterprise_context", _fake_retrieve_enterprise_context
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
    assert "GraphRAG evidence was retrieved" in body["answer"]
    assert body["vector_evidence"][0]["id"] == "Ticket:tkt_001"
    assert body["graph_evidence"][0]["id"] == "RiskEvent:risk_001"
    assert body["merged_context"][0]["context_source"] == "both"
    assert body["citations"][0]["id"] == "Ticket:tkt_001"
    assert body["metadata"]["mode"] == "graphrag_placeholder"
