from __future__ import annotations

from src.rag.enterprise_agentic_retrieval import (
    choose_retrieval_filters,
    classify_query_intent,
    retrieve_enterprise_context_agentically,
)


def _empty_context(query: str) -> dict:
    return {
        "query": query,
        "vector_evidence": [],
        "graph_evidence": [],
        "merged_context": [],
        "citations": [],
        "stats": {"merged_count": 0},
    }


def _sufficient_context(query: str) -> dict:
    return {
        "query": query,
        "vector_evidence": [],
        "graph_evidence": [],
        "merged_context": [
            {
                "id": "Ticket:tkt_001",
                "text": "Ticket ID: tkt_001\nSLA status: breached\nPriority: p1",
                "metadata": {
                    "entity_id": "tkt_001",
                    "ticket_id": "tkt_001",
                    "source_type": "ticket",
                    "created_at": "2026-05-01T13:20:00Z",
                },
                "context_source": "both",
                "source_type": "ticket",
                "title": "API timeout during batch sync",
            },
            {
                "id": "Policy:pol_sla",
                "text": "SLA policy requires escalation for breached p1 incidents.",
                "metadata": {
                    "entity_id": "pol_sla",
                    "policy_id": "pol_sla",
                    "source_type": "knowledge_base",
                },
                "context_source": "graph",
                "source_type": "knowledge_base",
                "title": "SLA policy",
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
        "stats": {"merged_count": 3},
    }


def test_classify_query_intent_routes_common_enterprise_questions() -> None:
    assert classify_query_intent("Summarize customer cust_001 health") == "customer_summary"
    assert classify_query_intent("Triage tkt_001 priority and escalation") == "ticket_triage"
    assert classify_query_intent("What refund policy applies?") == "policy_lookup"
    assert classify_query_intent("Who owns svc_api_gateway?") == "service_owner"
    assert classify_query_intent("Why is this API timeout risky?") == "risk_explanation"
    assert classify_query_intent("Find related context") == "general"


def test_choose_retrieval_filters_uses_intent_and_existing_filters() -> None:
    filters = choose_retrieval_filters(
        intent="risk_explanation",
        query="Explain risk for cust_001",
        base_filters={"customer_id": "cust_001"},
    )

    assert filters["customer_id"] == "cust_001"
    assert filters["source_type"] == ["risk_event", "ticket", "customer", "service"]


def test_low_sufficiency_triggers_one_retry() -> None:
    calls = []

    def _fake_retriever(query: str, top_k: int, graph_depth: int, filters=None):
        calls.append({"query": query, "filters": filters, "top_k": top_k, "depth": graph_depth})
        if len(calls) == 1:
            return _empty_context(query)
        return _sufficient_context(query)

    result = retrieve_enterprise_context_agentically(
        "Should tkt_001 escalate under the SLA for the API gateway service?",
        top_k=3,
        graph_depth=2,
        retriever=_fake_retriever,
    )

    trace = result["stats"]["agentic_trace"]
    assert len(calls) == 2
    assert trace["intent"] == "policy_lookup"
    assert trace["retrieval_attempts"] == 2
    assert trace["sufficiency_before"]["level"] == "low"
    assert trace["sufficiency_after"]["level"] in {"medium", "high"}
    assert trace["stop_reason"] == "sufficient_evidence"
    assert "knowledge base policy runbook" in calls[1]["query"]
    assert calls[0]["filters"]["ticket_id"] == "tkt_001"
    assert "ticket_id" not in calls[1]["filters"]


def test_max_attempts_are_respected_when_evidence_stays_low() -> None:
    calls = []

    def _fake_retriever(query: str, top_k: int, graph_depth: int, filters=None):
        calls.append(query)
        return _empty_context(query)

    result = retrieve_enterprise_context_agentically(
        "Who approved the undocumented private renewal exception?",
        retriever=_fake_retriever,
    )

    trace = result["stats"]["agentic_trace"]
    assert len(calls) == 2
    assert trace["retrieval_attempts"] == 2
    assert trace["stop_reason"] == "max_attempts_reached"
    assert trace["sufficiency_after"]["level"] == "low"
