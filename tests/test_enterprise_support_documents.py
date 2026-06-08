from __future__ import annotations

from pathlib import Path

from src.data.enterprise_support_documents import (
    build_customer_documents,
    build_enterprise_support_documents,
    build_github_issue_documents,
    build_knowledge_base_documents,
    build_risk_event_documents,
    build_service_documents,
    build_ticket_documents,
)
from src.data.enterprise_support_loader import load_enterprise_support_dataset

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_enterprise_support"


def _dataset() -> dict:
    return load_enterprise_support_dataset(DATA_DIR)


def test_build_enterprise_support_documents_generates_unique_documents() -> None:
    documents = build_enterprise_support_documents(_dataset())

    assert len(documents) == 128
    assert len({document["id"] for document in documents}) == len(documents)
    for document in documents:
        assert set(document) == {"id", "text", "metadata"}
        assert document["id"].startswith("enterprise_support::")
        assert document["text"].strip()
        assert document["metadata"]["source"] == "enterprise_support"
        assert document["metadata"]["source_type"]
        assert document["metadata"]["entity_id"]


def test_builder_counts_match_dataset_entities() -> None:
    dataset = _dataset()

    assert len(build_customer_documents(dataset)) == 10
    assert len(build_ticket_documents(dataset)) == 30
    assert len(build_knowledge_base_documents(dataset)) == 10
    assert len(build_service_documents(dataset)) == 8
    assert len(build_github_issue_documents(dataset)) == 30
    assert len(build_risk_event_documents(dataset)) == 40


def test_knowledge_base_markdown_becomes_rag_documents() -> None:
    documents = build_knowledge_base_documents(_dataset())
    sla_document = next(
        document
        for document in documents
        if document["id"] == "enterprise_support::knowledge_base::pol_sla"
    )

    assert "Knowledge Base Article" in sla_document["text"]
    assert "Enterprise SLA Policy" in sla_document["text"]
    assert "Enterprise Targets" in sla_document["text"]
    assert "p1" in sla_document["text"]
    assert sla_document["metadata"]["source_type"] == "knowledge_base"
    assert sla_document["metadata"]["entity_id"] == "pol_sla"
    assert sla_document["metadata"]["policy_id"] == "pol_sla"
    assert sla_document["metadata"]["path"] == "knowledge_base/sla_policy.md"


def test_ticket_documents_include_customer_and_product_context() -> None:
    documents = build_ticket_documents(_dataset())
    ticket_document = next(
        document
        for document in documents
        if document["id"] == "enterprise_support::ticket::tkt_001"
    )

    assert "Support Ticket" in ticket_document["text"]
    assert "API timeout during batch sync" in ticket_document["text"]
    assert "Avery Chen" in ticket_document["text"]
    assert "Northstar Analytics" in ticket_document["text"]
    assert "Developer API" in ticket_document["text"]
    assert "API Gateway" in ticket_document["text"]
    assert "Gateway timeout under batch sync load" in ticket_document["text"]
    assert "Risk events" in ticket_document["text"]
    assert ticket_document["metadata"]["source_type"] == "ticket"
    assert ticket_document["metadata"]["entity_id"] == "tkt_001"
    assert ticket_document["metadata"]["ticket_id"] == "tkt_001"
    assert ticket_document["metadata"]["customer_id"] == "cust_001"
    assert ticket_document["metadata"]["product_id"] == "prod_api"
    assert ticket_document["metadata"]["service_id"] == "svc_api_gateway"
    assert ticket_document["metadata"]["created_at"] == "2026-05-01T09:15:00Z"


def test_required_metadata_exists_for_relationship_documents() -> None:
    dataset = _dataset()
    service_document = next(
        document
        for document in build_service_documents(dataset)
        if document["id"] == "enterprise_support::service::svc_api_gateway"
    )
    issue_document = next(
        document
        for document in build_github_issue_documents(dataset)
        if document["id"] == "enterprise_support::github_issue::gh_001"
    )
    risk_document = next(
        document
        for document in build_risk_event_documents(dataset)
        if document["id"] == "enterprise_support::risk_event::risk_001"
    )

    assert service_document["metadata"]["source_type"] == "service"
    assert service_document["metadata"]["service_id"] == "svc_api_gateway"
    assert service_document["metadata"]["product_id"] == "prod_api"
    assert service_document["metadata"]["policy_id"] == "pol_api_timeout"

    assert issue_document["metadata"]["source_type"] == "github_issue"
    assert issue_document["metadata"]["entity_id"] == "gh_001"
    assert issue_document["metadata"]["product_id"] == "prod_api"
    assert issue_document["metadata"]["service_id"] == "svc_api_gateway"
    assert issue_document["metadata"]["created_at"] == "2026-05-01T09:45:00Z"
    assert issue_document["metadata"]["linked_ticket_ids"] == ["tkt_001", "tkt_019"]

    assert risk_document["metadata"]["source_type"] == "risk_event"
    assert risk_document["metadata"]["entity_id"] == "risk_001"
    assert risk_document["metadata"]["customer_id"] == "cust_001"
    assert risk_document["metadata"]["ticket_id"] == "tkt_001"
    assert risk_document["metadata"]["product_id"] == "prod_api"
    assert risk_document["metadata"]["service_id"] == "svc_api_gateway"
    assert risk_document["metadata"]["created_at"] == "2026-05-01T13:20:00Z"
