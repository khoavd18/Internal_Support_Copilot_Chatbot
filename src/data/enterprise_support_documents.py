from __future__ import annotations

from collections import defaultdict
from typing import Any

Record = dict[str, Any]
Dataset = dict[str, list[Record]]
RagDocument = dict[str, Any]

ENTERPRISE_SUPPORT_SOURCE = "enterprise_support"


def _value(record: Record | None, key: str, default: str = "") -> str:
    if not record:
        return default
    value = record.get(key, default)
    if value is None:
        return default
    return str(value).strip()


def _list_value(record: Record | None, key: str) -> list[str]:
    if not record:
        return []
    value = record.get(key)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in value.split("|") if item.strip()]
    return []


def _index_by(records: list[Record], key: str) -> dict[str, Record]:
    return {_value(record, key): record for record in records if _value(record, key)}


def _group_by(records: list[Record], key: str) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        value = _value(record, key)
        if value:
            grouped[value].append(record)
    return dict(grouped)


def _group_issues_by_ticket_id(issues: list[Record]) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for issue in issues:
        for ticket_id in _list_value(issue, "linked_ticket_ids"):
            grouped[ticket_id].append(issue)
    return dict(grouped)


def _line(label: str, value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        clean_items = [str(item).strip() for item in value if str(item).strip()]
        value = ", ".join(clean_items)
    else:
        value = str(value).strip()
    if not value:
        return ""
    return f"{label}: {value}"


def _join_lines(lines: list[str]) -> str:
    return "\n".join(line for line in lines if line).strip()


def _base_metadata(
    *,
    source_type: str,
    entity_id: str,
    title: str = "",
    customer_id: str = "",
    ticket_id: str = "",
    product_id: str = "",
    service_id: str = "",
    policy_id: str = "",
    created_at: str = "",
    account_id: str = "",
    path: str = "",
    extra: Record | None = None,
) -> Record:
    metadata: Record = {
        "source": ENTERPRISE_SUPPORT_SOURCE,
        "source_type": source_type,
        "entity_id": entity_id,
    }
    optional_fields = {
        "title": title,
        "customer_id": customer_id,
        "ticket_id": ticket_id,
        "product_id": product_id,
        "service_id": service_id,
        "policy_id": policy_id,
        "created_at": created_at,
        "account_id": account_id,
        "path": path,
    }
    for key, value in optional_fields.items():
        if value:
            metadata[key] = value

    if extra:
        for key, value in extra.items():
            if value not in ("", None, []):
                metadata[key] = value

    return metadata


def _document(document_id: str, text: str, metadata: Record) -> RagDocument:
    return {
        "id": document_id,
        "text": text,
        "metadata": metadata,
    }


def _ticket_summary(ticket: Record) -> str:
    return " - ".join(
        item
        for item in [
            _value(ticket, "ticket_id"),
            _value(ticket, "title"),
            _value(ticket, "status"),
            _value(ticket, "priority"),
        ]
        if item
    )


def _risk_summary(risk_event: Record) -> str:
    return " - ".join(
        item
        for item in [
            _value(risk_event, "risk_event_id"),
            _value(risk_event, "event_type"),
            _value(risk_event, "severity"),
            _value(risk_event, "summary"),
        ]
        if item
    )


def _message_summary(message: Record) -> str:
    parts = [
        _value(message, "created_at"),
        _value(message, "author_type"),
        _value(message, "visibility"),
        _value(message, "message_type"),
    ]
    prefix = " | ".join(part for part in parts if part)
    body = _value(message, "body")
    return f"{prefix}: {body}" if prefix else body


def _issue_summary(issue: Record) -> str:
    return " - ".join(
        item
        for item in [
            _value(issue, "issue_id"),
            _value(issue, "title"),
            _value(issue, "state"),
            _value(issue, "severity"),
        ]
        if item
    )


def build_customer_documents(dataset: dict) -> list[dict]:
    accounts_by_id = _index_by(dataset.get("accounts", []), "account_id")
    tickets_by_customer_id = _group_by(dataset.get("tickets", []), "customer_id")
    risks_by_customer_id = _group_by(dataset.get("risk_events", []), "customer_id")

    documents: list[RagDocument] = []
    for customer in dataset.get("customers", []):
        customer_id = _value(customer, "customer_id")
        account = accounts_by_id.get(_value(customer, "account_id"))
        customer_tickets = tickets_by_customer_id.get(customer_id, [])
        customer_risks = risks_by_customer_id.get(customer_id, [])

        text = _join_lines(
            [
                "Customer Profile",
                _line("Customer ID", customer_id),
                _line("Name", _value(customer, "full_name")),
                _line("Email", _value(customer, "email")),
                _line("Role", _value(customer, "role")),
                _line("Region", _value(customer, "region")),
                _line("Support tier", _value(customer, "support_tier")),
                _line("Preferred contact channel", _value(customer, "preferred_contact_channel")),
                _line("Account ID", _value(customer, "account_id")),
                _line("Account", _value(account, "account_name")),
                _line("Account segment", _value(account, "segment")),
                _line("Account risk level", _value(account, "risk_level")),
                _line("Account health score", _value(account, "health_score")),
                _line("Related tickets", [_ticket_summary(ticket) for ticket in customer_tickets]),
                _line("Risk events", [_risk_summary(risk) for risk in customer_risks]),
            ]
        )

        documents.append(
            _document(
                f"enterprise_support::customer::{customer_id}",
                text,
                _base_metadata(
                    source_type="customer",
                    entity_id=customer_id,
                    title=_value(customer, "full_name"),
                    customer_id=customer_id,
                    account_id=_value(customer, "account_id"),
                    created_at=_value(customer, "created_at"),
                ),
            )
        )

    return documents


def build_ticket_documents(dataset: dict) -> list[dict]:
    customers_by_id = _index_by(dataset.get("customers", []), "customer_id")
    accounts_by_id = _index_by(dataset.get("accounts", []), "account_id")
    products_by_id = _index_by(dataset.get("products", []), "product_id")
    services_by_id = _index_by(dataset.get("service_catalog", []), "service_id")
    messages_by_ticket_id = _group_by(dataset.get("ticket_messages", []), "ticket_id")
    resolutions_by_ticket_id = _group_by(dataset.get("ticket_resolutions", []), "ticket_id")
    risks_by_ticket_id = _group_by(dataset.get("risk_events", []), "ticket_id")
    issues_by_ticket_id = _group_issues_by_ticket_id(dataset.get("github_issues", []))

    documents: list[RagDocument] = []
    for ticket in dataset.get("tickets", []):
        ticket_id = _value(ticket, "ticket_id")
        customer = customers_by_id.get(_value(ticket, "customer_id"))
        account = accounts_by_id.get(_value(ticket, "account_id"))
        product = products_by_id.get(_value(ticket, "product_id"))
        service = services_by_id.get(_value(ticket, "service_id"))
        messages = messages_by_ticket_id.get(ticket_id, [])
        resolutions = resolutions_by_ticket_id.get(ticket_id, [])
        risks = risks_by_ticket_id.get(ticket_id, [])
        issues = issues_by_ticket_id.get(ticket_id, [])

        text = _join_lines(
            [
                "Support Ticket",
                _line("Ticket ID", ticket_id),
                _line("Title", _value(ticket, "title")),
                _line("Description", _value(ticket, "description")),
                _line("Category", _value(ticket, "category")),
                _line("Priority", _value(ticket, "priority")),
                _line("Severity", _value(ticket, "severity")),
                _line("Status", _value(ticket, "status")),
                _line("SLA status", _value(ticket, "sla_status")),
                _line("Created at", _value(ticket, "created_at")),
                _line("Updated at", _value(ticket, "updated_at")),
                _line("First response due at", _value(ticket, "first_response_due_at")),
                _line("Resolution due at", _value(ticket, "resolution_due_at")),
                _line("Assigned team", _value(ticket, "assigned_team")),
                _line("Assignee", _value(ticket, "assignee")),
                _line("Tags", _value(ticket, "tags")),
                _line("Customer", _value(customer, "full_name")),
                _line("Customer role", _value(customer, "role")),
                _line("Customer support tier", _value(customer, "support_tier")),
                _line("Account", _value(account, "account_name")),
                _line("Account risk level", _value(account, "risk_level")),
                _line("Product", _value(product, "product_name")),
                _line("Product family", _value(product, "product_family")),
                _line("Service", _value(service, "service_name")),
                _line("Service owner team", _value(service, "owner_team")),
                _line("Support escalation team", _value(service, "support_escalation_team")),
                _line("Messages", [_message_summary(message) for message in messages]),
                _line(
                    "Resolutions",
                    [
                        " - ".join(
                            item
                            for item in [
                                _value(resolution, "resolution_id"),
                                _value(resolution, "resolution_type"),
                                _value(resolution, "summary"),
                                _value(resolution, "customer_visible_response"),
                            ]
                            if item
                        )
                        for resolution in resolutions
                    ],
                ),
                _line("Related engineering issues", [_issue_summary(issue) for issue in issues]),
                _line("Risk events", [_risk_summary(risk) for risk in risks]),
            ]
        )

        documents.append(
            _document(
                f"enterprise_support::ticket::{ticket_id}",
                text,
                _base_metadata(
                    source_type="ticket",
                    entity_id=ticket_id,
                    title=_value(ticket, "title"),
                    customer_id=_value(ticket, "customer_id"),
                    ticket_id=ticket_id,
                    product_id=_value(ticket, "product_id"),
                    service_id=_value(ticket, "service_id"),
                    created_at=_value(ticket, "created_at"),
                    account_id=_value(ticket, "account_id"),
                ),
            )
        )

    return documents


def build_knowledge_base_documents(dataset: dict) -> list[dict]:
    documents: list[RagDocument] = []
    for article in dataset.get("knowledge_base_docs", []):
        policy_id = _value(article, "policy_id")
        text = _join_lines(
            [
                "Knowledge Base Article",
                _line("Policy ID", policy_id),
                _line("Title", _value(article, "title")),
                _line("Policy type", _value(article, "policy_type")),
                _line("Owner team", _value(article, "owner_team")),
                _line("Product ID", _value(article, "product_id")),
                _line("Service ID", _value(article, "service_id")),
                _line("Effective date", _value(article, "effective_date")),
                _line("Review date", _value(article, "review_date")),
                _line("Tags", _list_value(article, "tags")),
                _line("Summary", _value(article, "summary")),
                _line("Content", _value(article, "content") or _value(article, "text")),
            ]
        )

        documents.append(
            _document(
                f"enterprise_support::knowledge_base::{policy_id}",
                text,
                _base_metadata(
                    source_type="knowledge_base",
                    entity_id=policy_id,
                    title=_value(article, "title"),
                    product_id=_value(article, "product_id"),
                    service_id=_value(article, "service_id"),
                    policy_id=policy_id,
                    path=_value(article, "path"),
                ),
            )
        )

    return documents


def build_service_documents(dataset: dict) -> list[dict]:
    products_by_id = _index_by(dataset.get("products", []), "product_id")
    tickets_by_service_id = _group_by(dataset.get("tickets", []), "service_id")
    issues_by_service_id = _group_by(dataset.get("github_issues", []), "service_id")
    risks_by_service_id = _group_by(dataset.get("risk_events", []), "service_id")

    documents: list[RagDocument] = []
    for service in dataset.get("service_catalog", []):
        service_id = _value(service, "service_id")
        product = products_by_id.get(_value(service, "product_id"))
        tickets = tickets_by_service_id.get(service_id, [])
        issues = issues_by_service_id.get(service_id, [])
        risks = risks_by_service_id.get(service_id, [])

        text = _join_lines(
            [
                "Service Catalog Entry",
                _line("Service ID", service_id),
                _line("Service", _value(service, "service_name")),
                _line("Description", _value(service, "description")),
                _line("Product", _value(product, "product_name")),
                _line("Product ID", _value(service, "product_id")),
                _line("Owner team", _value(service, "owner_team")),
                _line("Support escalation team", _value(service, "support_escalation_team")),
                _line("Tier", _value(service, "tier")),
                _line("Slack channel", _value(service, "slack_channel")),
                _line("Runbook policy ID", _value(service, "runbook_policy_id")),
                _line("On-call rotation", _value(service, "on_call_rotation")),
                _line("Repository", _value(service, "repo")),
                _line("Status", _value(service, "status")),
                _line("Related tickets", [_ticket_summary(ticket) for ticket in tickets]),
                _line("Related engineering issues", [_issue_summary(issue) for issue in issues]),
                _line("Risk events", [_risk_summary(risk) for risk in risks]),
            ]
        )

        documents.append(
            _document(
                f"enterprise_support::service::{service_id}",
                text,
                _base_metadata(
                    source_type="service",
                    entity_id=service_id,
                    title=_value(service, "service_name"),
                    product_id=_value(service, "product_id"),
                    service_id=service_id,
                    policy_id=_value(service, "runbook_policy_id"),
                ),
            )
        )

    return documents


def build_github_issue_documents(dataset: dict) -> list[dict]:
    products_by_id = _index_by(dataset.get("products", []), "product_id")
    services_by_id = _index_by(dataset.get("service_catalog", []), "service_id")
    tickets_by_id = _index_by(dataset.get("tickets", []), "ticket_id")

    documents: list[RagDocument] = []
    for issue in dataset.get("github_issues", []):
        issue_id = _value(issue, "issue_id")
        linked_ticket_ids = _list_value(issue, "linked_ticket_ids")
        linked_tickets = [
            tickets_by_id[ticket_id]
            for ticket_id in linked_ticket_ids
            if ticket_id in tickets_by_id
        ]
        product = products_by_id.get(_value(issue, "product_id"))
        service = services_by_id.get(_value(issue, "service_id"))

        text = _join_lines(
            [
                "Engineering GitHub Issue",
                _line("Issue ID", issue_id),
                _line("Repository", _value(issue, "repo")),
                _line("Number", _value(issue, "number")),
                _line("Title", _value(issue, "title")),
                _line("State", _value(issue, "state")),
                _line("Severity", _value(issue, "severity")),
                _line("Labels", _list_value(issue, "labels")),
                _line("Product", _value(product, "product_name")),
                _line("Service", _value(service, "service_name")),
                _line("Assignee team", _value(issue, "assignee_team")),
                _line("Created at", _value(issue, "created_at")),
                _line("Updated at", _value(issue, "updated_at")),
                _line("Closed at", _value(issue, "closed_at")),
                _line("Body", _value(issue, "body")),
                _line("Linked tickets", [_ticket_summary(ticket) for ticket in linked_tickets]),
                _line("Resolution summary", _value(issue, "resolution_summary")),
            ]
        )

        extra: Record = {}
        if linked_ticket_ids:
            extra["linked_ticket_ids"] = linked_ticket_ids

        documents.append(
            _document(
                f"enterprise_support::github_issue::{issue_id}",
                text,
                _base_metadata(
                    source_type="github_issue",
                    entity_id=issue_id,
                    title=_value(issue, "title"),
                    ticket_id=linked_ticket_ids[0] if len(linked_ticket_ids) == 1 else "",
                    product_id=_value(issue, "product_id"),
                    service_id=_value(issue, "service_id"),
                    created_at=_value(issue, "created_at"),
                    path=_value(issue, "repo"),
                    extra=extra,
                ),
            )
        )

    return documents


def build_risk_event_documents(dataset: dict) -> list[dict]:
    customers_by_id = _index_by(dataset.get("customers", []), "customer_id")
    accounts_by_id = _index_by(dataset.get("accounts", []), "account_id")
    products_by_id = _index_by(dataset.get("products", []), "product_id")
    services_by_id = _index_by(dataset.get("service_catalog", []), "service_id")
    tickets_by_id = _index_by(dataset.get("tickets", []), "ticket_id")

    documents: list[RagDocument] = []
    for risk in dataset.get("risk_events", []):
        risk_event_id = _value(risk, "risk_event_id")
        customer = customers_by_id.get(_value(risk, "customer_id"))
        account = accounts_by_id.get(_value(risk, "account_id"))
        product = products_by_id.get(_value(risk, "product_id"))
        service = services_by_id.get(_value(risk, "service_id"))
        ticket = tickets_by_id.get(_value(risk, "ticket_id"))

        text = _join_lines(
            [
                "Risk Event",
                _line("Risk event ID", risk_event_id),
                _line("Event type", _value(risk, "event_type")),
                _line("Severity", _value(risk, "severity")),
                _line("Risk score", _value(risk, "risk_score")),
                _line("Detected at", _value(risk, "detected_at")),
                _line("Status", _value(risk, "status")),
                _line("Source", _value(risk, "source")),
                _line("Summary", _value(risk, "summary")),
                _line("Recommended action", _value(risk, "recommended_action")),
                _line("Evidence references", _value(risk, "evidence_refs")),
                _line("Customer", _value(customer, "full_name")),
                _line("Account", _value(account, "account_name")),
                _line("Ticket", _ticket_summary(ticket) if ticket else ""),
                _line("Product", _value(product, "product_name")),
                _line("Service", _value(service, "service_name")),
            ]
        )

        documents.append(
            _document(
                f"enterprise_support::risk_event::{risk_event_id}",
                text,
                _base_metadata(
                    source_type="risk_event",
                    entity_id=risk_event_id,
                    title=_value(risk, "summary"),
                    customer_id=_value(risk, "customer_id"),
                    ticket_id=_value(risk, "ticket_id"),
                    product_id=_value(risk, "product_id"),
                    service_id=_value(risk, "service_id"),
                    created_at=_value(risk, "detected_at"),
                    account_id=_value(risk, "account_id"),
                ),
            )
        )

    return documents


def _ensure_unique_document_ids(documents: list[RagDocument]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for document in documents:
        document_id = _value(document, "id")
        if document_id in seen:
            duplicates.add(document_id)
        seen.add(document_id)

    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate enterprise support document IDs: {duplicate_list}")


def build_enterprise_support_documents(dataset: dict) -> list[dict]:
    documents: list[RagDocument] = [
        *build_customer_documents(dataset),
        *build_ticket_documents(dataset),
        *build_knowledge_base_documents(dataset),
        *build_service_documents(dataset),
        *build_github_issue_documents(dataset),
        *build_risk_event_documents(dataset),
    ]
    _ensure_unique_document_ids(documents)
    return documents
