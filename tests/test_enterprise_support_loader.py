from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.data.enterprise_support_loader import (
    load_enterprise_support_dataset,
    load_github_issues,
    load_knowledge_base_docs,
    load_ticket_messages,
    load_tickets,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_enterprise_support"


def test_load_enterprise_support_dataset_counts() -> None:
    dataset = load_enterprise_support_dataset(DATA_DIR)

    assert set(dataset) == {
        "customers",
        "accounts",
        "products",
        "tickets",
        "ticket_messages",
        "ticket_resolutions",
        "service_catalog",
        "github_issues",
        "risk_events",
        "knowledge_base_docs",
    }
    assert len(dataset["customers"]) == 10
    assert len(dataset["accounts"]) == 10
    assert len(dataset["products"]) == 5
    assert len(dataset["tickets"]) == 30
    assert len(dataset["ticket_messages"]) == 80
    assert len(dataset["ticket_resolutions"]) == 20
    assert len(dataset["service_catalog"]) == 8
    assert len(dataset["github_issues"]) == 30
    assert len(dataset["risk_events"]) == 40
    assert len(dataset["knowledge_base_docs"]) == 10


def test_load_tickets_adds_normalized_metadata() -> None:
    tickets = load_tickets(DATA_DIR)
    ticket = next(row for row in tickets if row["ticket_id"] == "tkt_001")

    assert ticket["metadata"] == {
        "source": "enterprise_support",
        "source_type": "ticket",
        "entity_id": "tkt_001",
        "title": "API timeout during batch sync",
        "customer_id": "cust_001",
        "ticket_id": "tkt_001",
        "product_id": "prod_api",
        "service_id": "svc_api_gateway",
        "path": "support/tickets.csv",
    }


def test_load_ticket_messages_infers_customer_id_for_customer_messages() -> None:
    messages = load_ticket_messages(DATA_DIR)
    message = next(row for row in messages if row["message_id"] == "msg_001")

    assert message["metadata"]["source_type"] == "ticket_message"
    assert message["metadata"]["entity_id"] == "msg_001"
    assert message["metadata"]["ticket_id"] == "tkt_001"
    assert message["metadata"]["customer_id"] == "cust_001"


def test_load_github_issues_preserves_json_types_and_metadata() -> None:
    issues = load_github_issues(DATA_DIR)
    issue = next(row for row in issues if row["issue_id"] == "gh_001")

    assert issue["number"] == 101
    assert issue["linked_ticket_ids"] == ["tkt_001", "tkt_019"]
    assert issue["metadata"]["source_type"] == "github_issue"
    assert issue["metadata"]["entity_id"] == "gh_001"
    assert issue["metadata"]["title"] == "Gateway timeout under batch sync load"
    assert issue["metadata"]["product_id"] == "prod_api"
    assert issue["metadata"]["service_id"] == "svc_api_gateway"


def test_load_knowledge_base_docs_parses_front_matter_and_content() -> None:
    docs = load_knowledge_base_docs(DATA_DIR)
    sla_doc = next(row for row in docs if row["policy_id"] == "pol_sla")

    assert sla_doc["title"] == "Enterprise SLA Policy"
    assert sla_doc["tags"] == ["sla", "escalation", "enterprise"]
    assert sla_doc["content"].startswith("# Enterprise SLA Policy")
    assert not sla_doc["content"].startswith("---")
    assert sla_doc["metadata"]["source_type"] == "knowledge_base"
    assert sla_doc["metadata"]["entity_id"] == "pol_sla"
    assert sla_doc["metadata"]["title"] == "Enterprise SLA Policy"
    assert sla_doc["metadata"]["path"] == "knowledge_base/sla_policy.md"


def test_load_github_issues_reports_invalid_jsonl(tmp_path: Path) -> None:
    issues_dir = tmp_path / "engineering"
    issues_dir.mkdir()
    (issues_dir / "github_issues.jsonl").write_text(
        json.dumps({"issue_id": "gh_test"}) + "\n{not valid json}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_github_issues(tmp_path)
