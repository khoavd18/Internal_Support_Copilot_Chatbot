from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document
from src.data.enterprise_support_loader import load_enterprise_support_dataset
from src.kg.builder import build_graph_from_enterprise_support_dataset
from src.rag import graphrag

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_enterprise_support"


def _build_real_enterprise_graph():
    dataset = load_enterprise_support_dataset(DATA_DIR)
    return build_graph_from_enterprise_support_dataset(dataset)


def test_retrieve_enterprise_context_merges_vector_and_graph_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_real_enterprise_graph()

    def _fake_retrieve_enterprise_hybrid_documents(*, query: str, top_k: int, filters=None):
        assert "api timeout" in query
        assert top_k == 5
        assert filters is None
        return {
            "documents": [
                Document(
                    page_content="Vector hit for ticket tkt_001 API timeout and Northstar escalation.",
                    metadata={
                        "source": "enterprise_support",
                        "source_type": "ticket",
                        "entity_id": "tkt_001",
                        "ticket_id": "tkt_001",
                        "title": "API timeout during batch sync",
                        "dense_score": 0.91,
                        "lexical_score": 7.0,
                        "fused_score": 0.032,
                    },
                )
            ],
            "debug": [
                {
                    "id": "ticket:tkt_001",
                    "dense_score": 0.91,
                    "lexical_score": 7.0,
                    "fused_score": 0.032,
                }
            ],
            "stats": {
                "mode": "enterprise_hybrid",
                "dense_count": 1,
                "sparse_or_lexical_count": 1,
                "dense_error": "",
                "sparse_error": "",
            },
        }

    monkeypatch.setattr(
        graphrag,
        "retrieve_enterprise_hybrid_documents",
        _fake_retrieve_enterprise_hybrid_documents,
    )

    result = graphrag.retrieve_enterprise_context("api timeout northstar escalation")

    assert result["stats"]["vector_count"] == 1
    assert result["stats"]["graph_node_count"] > 0
    assert result["stats"]["vector_error"] == ""
    assert result["vector_evidence"][0]["context_source"] == "vector"
    assert any(item["context_source"] == "graph" for item in result["graph_evidence"])
    assert result["citations"][0]["entity_id"] == "tkt_001"
    assert result["citations"][0]["snippet"]
    assert result["stats"]["hybrid_retrieval"]["mode"] == "enterprise_hybrid"
    assert result["stats"]["hybrid_debug"][0]["fused_score"] == 0.032

    ticket_item = next(item for item in result["merged_context"] if item["id"] == "Ticket:tkt_001")
    assert ticket_item["context_source"] == "both"
    assert ticket_item["metadata"]["ticket_id"] == "tkt_001"
    assert ticket_item["metadata"]["vector_metadata"]["entity_id"] == "tkt_001"
    assert ticket_item["metadata"]["graph_metadata"]["kg_node_id"] == "Ticket:tkt_001"


def test_retrieve_enterprise_context_continues_when_vector_retrieval_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _build_real_enterprise_graph()

    def _raise_vector_error(*args, **kwargs):
        raise RuntimeError("qdrant unavailable")

    monkeypatch.setattr(graphrag, "retrieve_enterprise_hybrid_documents", _raise_vector_error)

    result = graphrag.retrieve_enterprise_context("refund policy duplicate seats", top_k=3)

    assert result["vector_evidence"] == []
    assert result["graph_evidence"]
    assert result["merged_context"]
    assert "qdrant unavailable" in result["stats"]["vector_error"]


def test_format_context_for_answer_includes_sources_and_metadata() -> None:
    context = [
        {
            "id": "Ticket:tkt_001",
            "text": "Ticket text",
            "metadata": {"ticket_id": "tkt_001", "source_type": "ticket"},
            "context_source": "both",
            "source_type": "ticket",
            "title": "API timeout during batch sync",
        }
    ]

    formatted = graphrag.format_context_for_answer(context)

    assert "Enterprise GraphRAG Context" in formatted
    assert "[1] API timeout during batch sync" in formatted
    assert "source: both" in formatted
    assert "Ticket:tkt_001" in formatted


def test_build_grounded_enterprise_answer_uses_evidence_and_citations() -> None:
    context = {
        "merged_context": [
            {
                "id": "Ticket:tkt_001",
                "text": (
                    "Support Ticket\n"
                    "Ticket ID: tkt_001\n"
                    "Title: API timeout during batch sync\n"
                    "SLA status: breached\n"
                    "Customer: Avery Chen"
                ),
                "metadata": {
                    "ticket_id": "tkt_001",
                    "source_type": "ticket",
                    "entity_id": "tkt_001",
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
                    "Summary: P1 API timeout breached resolution target\n"
                    "Recommended action: Escalate to the Reliability team."
                ),
                "metadata": {
                    "risk_event_id": "risk_001",
                    "source_type": "risk_event",
                    "entity_id": "risk_001",
                },
                "context_source": "graph",
                "source_type": "risk_event",
                "title": "P1 API timeout breached resolution target",
            },
        ]
    }

    result = graphrag.build_grounded_enterprise_answer(
        "Why is tkt_001 API timeout risky?",
        context,
    )

    assert result["mode"] == "deterministic_grounded_generation"
    assert "Based only on the retrieved enterprise support evidence" in result["answer"]
    assert "SLA status: breached" in result["answer"]
    assert "[1]" in result["answer"]
    assert result["citations"][0]["entity_id"] == "tkt_001"
    assert result["citations"][0]["source_type"] == "ticket"
    assert result["citations"][0]["used_for_answer"] is True
    assert result["citations"][0]["claim_index"] == 1
    assert result["citations"][0]["snippet"]
    assert result["confidence"] == "medium"
    assert result["evidence_sufficiency"]["level"] == "medium"
    assert result["evidence_sufficiency"]["missing_source_types"] == ["service"]


def test_build_grounded_enterprise_answer_handles_missing_information() -> None:
    context = {
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
                    "source_type": "customer",
                    "entity_id": "cust_001",
                },
                "context_source": "graph",
                "source_type": "customer",
                "title": "Avery Chen",
            }
        ]
    }

    result = graphrag.build_grounded_enterprise_answer(
        "What is cust_001's private phone number?",
        context,
    )

    assert result["confidence"] == "low"
    assert "Missing information" in result["answer"]
    assert "private phone number" in result["answer"]
    assert "555" not in result["answer"]
    assert result["citations"][0]["entity_id"] == "cust_001"
    assert result["evidence_sufficiency"]["level"] == "medium"


def test_build_grounded_enterprise_answer_is_low_confidence_without_evidence() -> None:
    result = graphrag.build_grounded_enterprise_answer(
        "Who approved the renewal discount?",
        {"merged_context": []},
    )

    assert result["confidence"] == "low"
    assert result["citations"] == []
    assert "do not have enough retrieved enterprise support evidence" in result["answer"]
    assert result["evidence_sufficiency"]["level"] == "low"
