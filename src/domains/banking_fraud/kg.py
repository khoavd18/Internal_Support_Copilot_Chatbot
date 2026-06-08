from __future__ import annotations

from typing import Any

from src.domains.banking_fraud.schemas import Dataset, Record


def build_banking_fraud_graph(dataset: Dataset) -> dict[str, Any]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, str]] = []

    for customer in dataset.get("customers", []):
        _add_node(
            nodes,
            "Customer",
            _value(customer, "customer_id"),
            _value(customer, "full_name"),
            customer,
        )
    for account in dataset.get("accounts", []):
        account_id = _value(account, "account_id")
        _add_node(nodes, "Account", account_id, account_id, account)
        _add_edge(
            edges, "Customer", _value(account, "customer_id"), "Account", account_id, "HAS_ACCOUNT"
        )
    for merchant in dataset.get("merchants", []):
        _add_node(
            nodes,
            "Merchant",
            _value(merchant, "merchant_id"),
            _value(merchant, "merchant_name"),
            merchant,
        )
    for transaction in dataset.get("transactions", []):
        transaction_id = _value(transaction, "transaction_id")
        _add_node(nodes, "Transaction", transaction_id, transaction_id, transaction)
        _add_edge(
            edges,
            "Account",
            _value(transaction, "account_id"),
            "Transaction",
            transaction_id,
            "MADE_TRANSACTION",
        )
        _add_edge(
            edges,
            "Customer",
            _value(transaction, "customer_id"),
            "Transaction",
            transaction_id,
            "INITIATED_TRANSACTION",
        )
        _add_edge(
            edges,
            "Transaction",
            transaction_id,
            "Merchant",
            _value(transaction, "merchant_id"),
            "PAID_MERCHANT",
        )
    for alert in dataset.get("fraud_alerts", []):
        alert_id = _value(alert, "alert_id")
        _add_node(nodes, "FraudAlert", alert_id, _value(alert, "summary"), alert)
        _add_edge(
            edges,
            "Transaction",
            _value(alert, "transaction_id"),
            "FraudAlert",
            alert_id,
            "TRIGGERED_ALERT",
        )
        _add_edge(
            edges, "Customer", _value(alert, "customer_id"), "FraudAlert", alert_id, "HAS_ALERT"
        )
    for case in dataset.get("aml_cases", []):
        case_id = _value(case, "case_id")
        _add_node(nodes, "AMLCase", case_id, _value(case, "summary"), case)
        _add_edge(
            edges, "Customer", _value(case, "customer_id"), "AMLCase", case_id, "HAS_AML_CASE"
        )
        for alert_id in _split_pipe(_value(case, "linked_alert_ids")):
            _add_edge(edges, "FraudAlert", alert_id, "AMLCase", case_id, "LINKED_TO_CASE")
    for policy in dataset.get("policies", []):
        _add_node(nodes, "Policy", _value(policy, "policy_id"), _value(policy, "title"), policy)

    return {"domain": "banking_fraud", "nodes": nodes, "edges": edges}


def _add_node(
    nodes: dict[str, dict[str, Any]],
    node_type: str,
    entity_id: str,
    label: str,
    properties: Record,
) -> None:
    if not entity_id:
        return
    node_id = f"{node_type}:{entity_id}"
    nodes[node_id] = {
        "id": node_id,
        "type": node_type,
        "label": label or entity_id,
        "properties": dict(properties),
    }


def _add_edge(
    edges: list[dict[str, str]],
    source_type: str,
    source_id: str,
    target_type: str,
    target_id: str,
    edge_type: str,
) -> None:
    if not source_id or not target_id:
        return
    edges.append(
        {
            "source": f"{source_type}:{source_id}",
            "target": f"{target_type}:{target_id}",
            "type": edge_type,
        }
    )


def _split_pipe(value: str) -> list[str]:
    return [part.strip() for part in value.split("|") if part.strip()]


def _value(record: Record | None, key: str, default: str = "") -> str:
    if not record:
        return default
    value = record.get(key, default)
    if value is None:
        return default
    return str(value).strip()
