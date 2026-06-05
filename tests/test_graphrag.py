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

    def _fake_retrieve_documents(*, query: str, top_k: int, rebuild: bool):
        assert "api timeout" in query
        assert top_k == 5
        assert rebuild is False
        return [
            Document(
                page_content="Vector hit for ticket tkt_001 API timeout and Northstar escalation.",
                metadata={
                    "source": "enterprise_support",
                    "source_type": "ticket",
                    "entity_id": "tkt_001",
                    "ticket_id": "tkt_001",
                    "title": "API timeout during batch sync",
                },
            )
        ]

    monkeypatch.setattr(graphrag, "retrieve_documents", _fake_retrieve_documents)

    result = graphrag.retrieve_enterprise_context("api timeout northstar escalation")

    assert result["stats"]["vector_count"] == 1
    assert result["stats"]["graph_node_count"] > 0
    assert result["stats"]["vector_error"] == ""
    assert result["vector_evidence"][0]["context_source"] == "vector"
    assert any(item["context_source"] == "graph" for item in result["graph_evidence"])

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

    monkeypatch.setattr(graphrag, "retrieve_documents", _raise_vector_error)

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
