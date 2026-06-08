from __future__ import annotations

import re
from typing import Any

from src.kg.schema import EdgeType, NodeType
from src.kg.store import InMemoryKnowledgeGraph, set_default_graph

Record = dict[str, Any]
Dataset = dict[str, list[Record]]


def _value(record: Record | None, key: str, default: str = "") -> str:
    if not record:
        return default
    value = record.get(key, default)
    if value is None:
        return default
    return str(value).strip()


def _node_id(node_type: NodeType, entity_id: str) -> str:
    return f"{node_type}:{entity_id}"


def _team_id(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name or "").strip())
    normalized = normalized.strip("_")
    return _node_id("Team", normalized)


def _list_value(record: Record, key: str) -> list[str]:
    value = record.get(key)
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in value.split("|") if item.strip()]
    return []


def _line(label: str, value: Any) -> str:
    if value in ("", None, []):
        return ""
    if isinstance(value, list):
        value = ", ".join(str(item).strip() for item in value if str(item).strip())
    return f"{label}: {value}"


def _join_lines(lines: list[str]) -> str:
    return "\n".join(line for line in lines if line).strip()


def _add_team(graph: InMemoryKnowledgeGraph, team_name: str) -> str:
    if not str(team_name or "").strip():
        return ""
    node_id = _team_id(team_name)
    graph.add_node(
        node_id,
        "Team",
        str(team_name).strip(),
        properties={"team_name": str(team_name).strip()},
        text=f"Team {team_name}",
    )
    return node_id


def _add_edge_if_possible(
    graph: InMemoryKnowledgeGraph,
    source_id: str,
    target_id: str,
    edge_type: EdgeType,
    *,
    properties: Record | None = None,
) -> None:
    if source_id and target_id and source_id in graph.nodes and target_id in graph.nodes:
        graph.add_edge(source_id, target_id, edge_type, properties=properties)


def _add_owned_by_team_edge(
    graph: InMemoryKnowledgeGraph,
    source_id: str,
    team_name: str,
    *,
    role: str,
) -> None:
    team_id = _add_team(graph, team_name)
    _add_edge_if_possible(
        graph,
        source_id,
        team_id,
        "OWNED_BY_TEAM",
        properties={"role": role},
    )


def _build_account_nodes(graph: InMemoryKnowledgeGraph, dataset: Dataset) -> None:
    for account in dataset.get("accounts", []):
        account_id = _value(account, "account_id")
        graph.add_node(
            _node_id("Account", account_id),
            "Account",
            _value(account, "account_name"),
            properties=dict(account),
            text=_join_lines(
                [
                    _line("Account", _value(account, "account_name")),
                    _line("Industry", _value(account, "industry")),
                    _line("Segment", _value(account, "segment")),
                    _line("Region", _value(account, "region")),
                    _line("Risk level", _value(account, "risk_level")),
                    _line("Health score", _value(account, "health_score")),
                ]
            ),
        )
        _add_owned_by_team_edge(
            graph,
            _node_id("Account", account_id),
            _value(account, "account_owner"),
            role="account_owner",
        )


def _build_customer_nodes(graph: InMemoryKnowledgeGraph, dataset: Dataset) -> None:
    for customer in dataset.get("customers", []):
        customer_id = _value(customer, "customer_id")
        customer_node_id = _node_id("Customer", customer_id)
        graph.add_node(
            customer_node_id,
            "Customer",
            _value(customer, "full_name"),
            properties=dict(customer),
            text=_join_lines(
                [
                    _line("Customer", _value(customer, "full_name")),
                    _line("Role", _value(customer, "role")),
                    _line("Region", _value(customer, "region")),
                    _line("Support tier", _value(customer, "support_tier")),
                    _line("Preferred contact", _value(customer, "preferred_contact_channel")),
                ]
            ),
        )
        _add_edge_if_possible(
            graph,
            customer_node_id,
            _node_id("Account", _value(customer, "account_id")),
            "HAS_ACCOUNT",
        )


def _build_product_nodes(graph: InMemoryKnowledgeGraph, dataset: Dataset) -> None:
    for product in dataset.get("products", []):
        product_id = _value(product, "product_id")
        product_node_id = _node_id("Product", product_id)
        graph.add_node(
            product_node_id,
            "Product",
            _value(product, "product_name"),
            properties=dict(product),
            text=_join_lines(
                [
                    _line("Product", _value(product, "product_name")),
                    _line("Family", _value(product, "product_family")),
                    _line("Plan", _value(product, "plan_name")),
                    _line("Lifecycle stage", _value(product, "lifecycle_stage")),
                ]
            ),
        )
        _add_owned_by_team_edge(
            graph,
            product_node_id,
            _value(product, "support_owner_team"),
            role="support_owner_team",
        )
        _add_owned_by_team_edge(
            graph,
            product_node_id,
            _value(product, "engineering_owner_team"),
            role="engineering_owner_team",
        )


def _build_policy_nodes(graph: InMemoryKnowledgeGraph, dataset: Dataset) -> None:
    for policy in dataset.get("knowledge_base_docs", []):
        policy_id = _value(policy, "policy_id")
        policy_node_id = _node_id("Policy", policy_id)
        graph.add_node(
            policy_node_id,
            "Policy",
            _value(policy, "title"),
            properties=dict(policy),
            text=_join_lines(
                [
                    _line("Policy", _value(policy, "title")),
                    _line("Type", _value(policy, "policy_type")),
                    _line("Summary", _value(policy, "summary")),
                    _line("Content", _value(policy, "content")[:1000]),
                ]
            ),
        )
        _add_owned_by_team_edge(
            graph,
            policy_node_id,
            _value(policy, "owner_team"),
            role="policy_owner",
        )
        _add_edge_if_possible(
            graph,
            policy_node_id,
            _node_id("Product", _value(policy, "product_id")),
            "MENTIONS_PRODUCT",
        )
        _add_edge_if_possible(
            graph,
            policy_node_id,
            _node_id("Service", _value(policy, "service_id")),
            "AFFECTS_SERVICE",
        )


def _build_service_nodes(graph: InMemoryKnowledgeGraph, dataset: Dataset) -> None:
    for service in dataset.get("service_catalog", []):
        service_id = _value(service, "service_id")
        service_node_id = _node_id("Service", service_id)
        graph.add_node(
            service_node_id,
            "Service",
            _value(service, "service_name"),
            properties=dict(service),
            text=_join_lines(
                [
                    _line("Service", _value(service, "service_name")),
                    _line("Description", _value(service, "description")),
                    _line("Tier", _value(service, "tier")),
                    _line("Repository", _value(service, "repo")),
                    _line("Status", _value(service, "status")),
                ]
            ),
        )
        _add_edge_if_possible(
            graph,
            _node_id("Product", _value(service, "product_id")),
            service_node_id,
            "AFFECTS_SERVICE",
        )
        _add_owned_by_team_edge(
            graph,
            service_node_id,
            _value(service, "owner_team"),
            role="engineering_owner",
        )
        _add_owned_by_team_edge(
            graph,
            service_node_id,
            _value(service, "support_escalation_team"),
            role="support_escalation_team",
        )
        _add_edge_if_possible(
            graph,
            service_node_id,
            _node_id("Policy", _value(service, "runbook_policy_id")),
            "REFERENCES_POLICY",
        )


def _build_ticket_nodes(graph: InMemoryKnowledgeGraph, dataset: Dataset) -> None:
    for ticket in dataset.get("tickets", []):
        ticket_id = _value(ticket, "ticket_id")
        ticket_node_id = _node_id("Ticket", ticket_id)
        graph.add_node(
            ticket_node_id,
            "Ticket",
            _value(ticket, "title"),
            properties=dict(ticket),
            text=_join_lines(
                [
                    _line("Ticket", _value(ticket, "title")),
                    _line("Description", _value(ticket, "description")),
                    _line("Category", _value(ticket, "category")),
                    _line("Priority", _value(ticket, "priority")),
                    _line("Severity", _value(ticket, "severity")),
                    _line("Status", _value(ticket, "status")),
                    _line("SLA status", _value(ticket, "sla_status")),
                    _line("Tags", _value(ticket, "tags")),
                ]
            ),
        )
        _add_edge_if_possible(
            graph,
            _node_id("Customer", _value(ticket, "customer_id")),
            ticket_node_id,
            "CREATED_TICKET",
        )
        _add_edge_if_possible(
            graph,
            ticket_node_id,
            _node_id("Product", _value(ticket, "product_id")),
            "MENTIONS_PRODUCT",
        )
        _add_edge_if_possible(
            graph,
            ticket_node_id,
            _node_id("Service", _value(ticket, "service_id")),
            "AFFECTS_SERVICE",
        )
        _add_owned_by_team_edge(
            graph,
            ticket_node_id,
            _value(ticket, "assigned_team"),
            role="assigned_team",
        )


def _build_ticket_message_nodes(graph: InMemoryKnowledgeGraph, dataset: Dataset) -> None:
    for message in dataset.get("ticket_messages", []):
        message_id = _value(message, "message_id")
        body = _value(message, "body")
        message_node_id = _node_id("TicketMessage", message_id)
        graph.add_node(
            message_node_id,
            "TicketMessage",
            f"{_value(message, 'author_type')} {_value(message, 'message_type')}".strip(),
            properties=dict(message),
            text=_join_lines(
                [
                    _line("Message", body),
                    _line("Author type", _value(message, "author_type")),
                    _line("Visibility", _value(message, "visibility")),
                    _line("Sentiment", _value(message, "sentiment")),
                ]
            ),
        )
        _add_edge_if_possible(
            graph,
            _node_id("Ticket", _value(message, "ticket_id")),
            message_node_id,
            "HAS_MESSAGE",
        )


def _build_github_issue_nodes(graph: InMemoryKnowledgeGraph, dataset: Dataset) -> None:
    for issue in dataset.get("github_issues", []):
        issue_id = _value(issue, "issue_id")
        issue_node_id = _node_id("GitHubIssue", issue_id)
        graph.add_node(
            issue_node_id,
            "GitHubIssue",
            _value(issue, "title"),
            properties=dict(issue),
            text=_join_lines(
                [
                    _line("GitHub issue", _value(issue, "title")),
                    _line("Repository", _value(issue, "repo")),
                    _line("State", _value(issue, "state")),
                    _line("Severity", _value(issue, "severity")),
                    _line("Labels", _list_value(issue, "labels")),
                    _line("Body", _value(issue, "body")),
                    _line("Resolution", _value(issue, "resolution_summary")),
                ]
            ),
        )
        _add_edge_if_possible(
            graph,
            issue_node_id,
            _node_id("Product", _value(issue, "product_id")),
            "MENTIONS_PRODUCT",
        )
        _add_edge_if_possible(
            graph,
            issue_node_id,
            _node_id("Service", _value(issue, "service_id")),
            "AFFECTS_SERVICE",
        )
        _add_owned_by_team_edge(
            graph,
            issue_node_id,
            _value(issue, "assignee_team"),
            role="assignee_team",
        )
        for ticket_id in _list_value(issue, "linked_ticket_ids"):
            _add_edge_if_possible(
                graph,
                _node_id("Ticket", ticket_id),
                issue_node_id,
                "RELATED_TO_ISSUE",
            )


def _build_risk_event_nodes(graph: InMemoryKnowledgeGraph, dataset: Dataset) -> None:
    for risk in dataset.get("risk_events", []):
        risk_event_id = _value(risk, "risk_event_id")
        risk_node_id = _node_id("RiskEvent", risk_event_id)
        graph.add_node(
            risk_node_id,
            "RiskEvent",
            _value(risk, "summary") or _value(risk, "event_type"),
            properties=dict(risk),
            text=_join_lines(
                [
                    _line("Risk event", _value(risk, "summary")),
                    _line("Event type", _value(risk, "event_type")),
                    _line("Severity", _value(risk, "severity")),
                    _line("Risk score", _value(risk, "risk_score")),
                    _line("Recommended action", _value(risk, "recommended_action")),
                    _line("Evidence", _value(risk, "evidence_refs")),
                ]
            ),
        )
        for source_type, source_key in [
            ("Customer", "customer_id"),
            ("Account", "account_id"),
            ("Ticket", "ticket_id"),
            ("Product", "product_id"),
            ("Service", "service_id"),
        ]:
            _add_edge_if_possible(
                graph,
                _node_id(source_type, _value(risk, source_key)),
                risk_node_id,
                "HAS_RISK_EVENT",
            )
        _add_edge_if_possible(
            graph,
            risk_node_id,
            _node_id("Product", _value(risk, "product_id")),
            "MENTIONS_PRODUCT",
        )
        _add_edge_if_possible(
            graph,
            risk_node_id,
            _node_id("Service", _value(risk, "service_id")),
            "AFFECTS_SERVICE",
        )


def _add_resolution_edges(graph: InMemoryKnowledgeGraph, dataset: Dataset) -> None:
    for resolution in dataset.get("ticket_resolutions", []):
        ticket_node_id = _node_id("Ticket", _value(resolution, "ticket_id"))
        policy_node_id = _node_id("Policy", _value(resolution, "linked_policy_id"))
        issue_node_id = _node_id("GitHubIssue", _value(resolution, "linked_issue_id"))

        edge_properties = {
            "resolution_id": _value(resolution, "resolution_id"),
            "resolution_type": _value(resolution, "resolution_type"),
            "summary": _value(resolution, "summary"),
            "resolved_at": _value(resolution, "resolved_at"),
        }
        _add_edge_if_possible(
            graph,
            ticket_node_id,
            policy_node_id,
            "HAS_RESOLUTION",
            properties=edge_properties,
        )
        _add_edge_if_possible(
            graph,
            ticket_node_id,
            policy_node_id,
            "REFERENCES_POLICY",
            properties={"resolution_id": _value(resolution, "resolution_id")},
        )
        _add_edge_if_possible(
            graph,
            ticket_node_id,
            issue_node_id,
            "RELATED_TO_ISSUE",
            properties={"resolution_id": _value(resolution, "resolution_id")},
        )


def _ticket_policy_ids(ticket: Record) -> list[str]:
    haystack = " ".join(
        [
            _value(ticket, "title"),
            _value(ticket, "description"),
            _value(ticket, "category"),
            _value(ticket, "priority"),
            _value(ticket, "severity"),
            _value(ticket, "sla_status"),
            _value(ticket, "tags"),
            _value(ticket, "product_id"),
            _value(ticket, "service_id"),
        ]
    ).lower()
    rules = [
        ("pol_sla", ["sla", "p1", "p2", "breach", "breached", "incident", "outage"]),
        ("pol_api_timeout", ["api", "timeout", "latency", "gateway", "webhook"]),
        ("pol_login_troubleshooting", ["login", "sso", "scim", "session"]),
        ("pol_security_escalation", ["security", "mfa", "api_key", "lost_device"]),
        ("pol_refund", ["billing", "refund", "credit", "tax", "invoice"]),
        ("pol_data_retention", ["export", "retention", "data"]),
        ("pol_access", ["access", "permission", "admin", "invite", "oauth"]),
        ("pol_incident_response", ["incident", "outage", "sev1", "duplicate_orders"]),
    ]
    policy_ids = [
        policy_id
        for policy_id, keywords in rules
        if any(keyword in haystack for keyword in keywords)
    ]
    return list(dict.fromkeys(policy_ids))


def _add_ticket_policy_edges(graph: InMemoryKnowledgeGraph, dataset: Dataset) -> None:
    for ticket in dataset.get("tickets", []):
        ticket_node_id = _node_id("Ticket", _value(ticket, "ticket_id"))
        for policy_id in _ticket_policy_ids(ticket):
            _add_edge_if_possible(
                graph,
                ticket_node_id,
                _node_id("Policy", policy_id),
                "REFERENCES_POLICY",
                properties={"reason": "rule_based_ticket_context"},
            )


def build_graph_from_enterprise_support_dataset(dataset: dict) -> InMemoryKnowledgeGraph:
    graph = InMemoryKnowledgeGraph()
    typed_dataset: Dataset = dataset

    _build_account_nodes(graph, typed_dataset)
    _build_customer_nodes(graph, typed_dataset)
    _build_product_nodes(graph, typed_dataset)
    _build_service_nodes(graph, typed_dataset)
    _build_policy_nodes(graph, typed_dataset)
    _build_ticket_nodes(graph, typed_dataset)
    _build_ticket_message_nodes(graph, typed_dataset)
    _build_github_issue_nodes(graph, typed_dataset)
    _build_risk_event_nodes(graph, typed_dataset)
    _add_ticket_policy_edges(graph, typed_dataset)
    _add_resolution_edges(graph, typed_dataset)

    set_default_graph(graph)
    return graph
