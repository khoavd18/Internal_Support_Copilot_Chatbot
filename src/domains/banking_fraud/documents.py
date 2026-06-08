from __future__ import annotations

from collections import defaultdict
from typing import Any

from src.domains.banking_fraud.schemas import Dataset, Record


def build_banking_fraud_documents(dataset: Dataset) -> list[dict[str, Any]]:
    accounts_by_customer = _group_by(dataset.get("accounts", []), "customer_id")
    transactions_by_customer = _group_by(dataset.get("transactions", []), "customer_id")
    alerts_by_customer = _group_by(dataset.get("fraud_alerts", []), "customer_id")
    cases_by_customer = _group_by(dataset.get("aml_cases", []), "customer_id")
    merchants_by_id = _index_by(dataset.get("merchants", []), "merchant_id")
    transactions_by_id = _index_by(dataset.get("transactions", []), "transaction_id")
    alerts_by_id = _index_by(dataset.get("fraud_alerts", []), "alert_id")

    documents: list[dict[str, Any]] = []
    for customer in dataset.get("customers", []):
        customer_id = _value(customer, "customer_id")
        documents.append(
            _document(
                "customer",
                customer_id,
                [
                    "Banking Customer",
                    _line("Customer ID", customer_id),
                    _line("Name", _value(customer, "full_name")),
                    _line("Segment", _value(customer, "segment")),
                    _line("Region", _value(customer, "region")),
                    _line("Risk tier", _value(customer, "risk_tier")),
                    _line("KYC status", _value(customer, "kyc_status")),
                    _line(
                        "Accounts",
                        [
                            _summary(account, "account_id", "account_type", "status")
                            for account in accounts_by_customer[customer_id]
                        ],
                    ),
                    _line(
                        "Recent transactions",
                        [
                            _transaction_summary(txn, merchants_by_id)
                            for txn in transactions_by_customer[customer_id]
                        ],
                    ),
                    _line(
                        "Fraud alerts",
                        [
                            _summary(alert, "alert_id", "alert_type", "severity")
                            for alert in alerts_by_customer[customer_id]
                        ],
                    ),
                    _line(
                        "AML cases",
                        [
                            _summary(case, "case_id", "case_type", "status")
                            for case in cases_by_customer[customer_id]
                        ],
                    ),
                ],
                title=_value(customer, "full_name"),
                metadata={
                    "customer_id": customer_id,
                    "created_at": _value(customer, "created_at"),
                },
            )
        )

    for account in dataset.get("accounts", []):
        account_id = _value(account, "account_id")
        documents.append(
            _document(
                "account",
                account_id,
                [
                    "Banking Account",
                    _line("Account ID", account_id),
                    _line("Customer ID", _value(account, "customer_id")),
                    _line("Type", _value(account, "account_type")),
                    _line("Status", _value(account, "status")),
                    _line("Balance USD", _value(account, "balance_usd")),
                    _line("Online banking enabled", _value(account, "online_banking_enabled")),
                ],
                title=account_id,
                metadata={
                    "account_id": account_id,
                    "customer_id": _value(account, "customer_id"),
                    "created_at": _value(account, "opened_at"),
                },
            )
        )

    for transaction in dataset.get("transactions", []):
        transaction_id = _value(transaction, "transaction_id")
        merchant = merchants_by_id.get(_value(transaction, "merchant_id"), {})
        documents.append(
            _document(
                "transaction",
                transaction_id,
                [
                    "Banking Transaction",
                    _line("Transaction ID", transaction_id),
                    _line("Customer ID", _value(transaction, "customer_id")),
                    _line("Account ID", _value(transaction, "account_id")),
                    _line("Merchant", _value(merchant, "merchant_name")),
                    _line("Merchant risk", _value(merchant, "risk_level")),
                    _line("Amount USD", _value(transaction, "amount_usd")),
                    _line("Type", _value(transaction, "transaction_type")),
                    _line("Channel", _value(transaction, "channel")),
                    _line("Country", _value(transaction, "country")),
                    _line("IP country", _value(transaction, "ip_country")),
                    _line("Status", _value(transaction, "status")),
                    _line("Created at", _value(transaction, "created_at")),
                ],
                title=f"{transaction_id} {_value(transaction, 'transaction_type')}",
                metadata={
                    "transaction_id": transaction_id,
                    "customer_id": _value(transaction, "customer_id"),
                    "account_id": _value(transaction, "account_id"),
                    "merchant_id": _value(transaction, "merchant_id"),
                    "created_at": _value(transaction, "created_at"),
                },
            )
        )

    for merchant in dataset.get("merchants", []):
        merchant_id = _value(merchant, "merchant_id")
        documents.append(
            _document(
                "merchant",
                merchant_id,
                [
                    "Banking Merchant",
                    _line("Merchant ID", merchant_id),
                    _line("Name", _value(merchant, "merchant_name")),
                    _line("Category", _value(merchant, "category")),
                    _line("Country", _value(merchant, "country")),
                    _line("Risk level", _value(merchant, "risk_level")),
                ],
                title=_value(merchant, "merchant_name"),
                metadata={"merchant_id": merchant_id},
            )
        )

    for alert in dataset.get("fraud_alerts", []):
        alert_id = _value(alert, "alert_id")
        transaction = transactions_by_id.get(_value(alert, "transaction_id"), {})
        documents.append(
            _document(
                "fraud_alert",
                alert_id,
                [
                    "Fraud Alert",
                    _line("Alert ID", alert_id),
                    _line("Alert type", _value(alert, "alert_type")),
                    _line("Severity", _value(alert, "severity")),
                    _line("Status", _value(alert, "status")),
                    _line("Rule", _value(alert, "rule_name")),
                    _line("Summary", _value(alert, "summary")),
                    _line("Recommended action", _value(alert, "recommended_action")),
                    _line("Transaction", _transaction_summary(transaction, merchants_by_id)),
                ],
                title=_value(alert, "summary"),
                metadata={
                    "alert_id": alert_id,
                    "transaction_id": _value(alert, "transaction_id"),
                    "customer_id": _value(alert, "customer_id"),
                    "account_id": _value(alert, "account_id"),
                    "created_at": _value(alert, "created_at"),
                },
            )
        )

    for case in dataset.get("aml_cases", []):
        case_id = _value(case, "case_id")
        linked_alert_ids = _split_pipe(_value(case, "linked_alert_ids"))
        documents.append(
            _document(
                "aml_case",
                case_id,
                [
                    "AML Case",
                    _line("Case ID", case_id),
                    _line("Customer ID", _value(case, "customer_id")),
                    _line("Account ID", _value(case, "account_id")),
                    _line("Case type", _value(case, "case_type")),
                    _line("Priority", _value(case, "priority")),
                    _line("Status", _value(case, "status")),
                    _line("Summary", _value(case, "summary")),
                    _line("Linked alerts", linked_alert_ids),
                    _line(
                        "Alert summaries",
                        [
                            _value(alerts_by_id.get(alert_id), "summary")
                            for alert_id in linked_alert_ids
                        ],
                    ),
                    _line("Recommended action", _value(case, "recommended_action")),
                ],
                title=_value(case, "summary"),
                metadata={
                    "case_id": case_id,
                    "customer_id": _value(case, "customer_id"),
                    "account_id": _value(case, "account_id"),
                    "alert_id": linked_alert_ids,
                    "created_at": _value(case, "opened_at"),
                },
            )
        )

    for policy in dataset.get("policies", []):
        policy_id = _value(policy, "policy_id")
        documents.append(
            _document(
                "policy",
                policy_id,
                [
                    "Banking Fraud Policy",
                    _line("Policy ID", policy_id),
                    _line("Title", _value(policy, "title")),
                    _line("Path", _value(policy, "path")),
                    _value(policy, "content"),
                ],
                title=_value(policy, "title"),
                metadata={"policy_id": policy_id, "path": _value(policy, "path")},
            )
        )

    return documents


def _document(
    source_type: str,
    entity_id: str,
    lines: list[str],
    *,
    title: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    clean_lines = [line for line in lines if line]
    return {
        "id": f"banking_fraud::{source_type}::{entity_id}",
        "text": "\n".join(clean_lines),
        "metadata": {
            "source": "banking_fraud",
            "source_type": source_type,
            "entity_id": entity_id,
            "title": title,
            **metadata,
        },
    }


def _group_by(records: list[Record], field: str) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[_value(record, field)].append(record)
    return grouped


def _index_by(records: list[Record], field: str) -> dict[str, Record]:
    return {_value(record, field): record for record in records if _value(record, field)}


def _line(label: str, value: Any) -> str:
    if isinstance(value, list):
        clean = [str(item).strip() for item in value if str(item).strip()]
        if not clean:
            return ""
        return f"{label}: {', '.join(clean)}"
    text = str(value or "").strip()
    return f"{label}: {text}" if text else ""


def _summary(record: Record, id_field: str, label_field: str, status_field: str) -> str:
    if not record:
        return ""
    return " - ".join(
        part
        for part in [
            _value(record, id_field),
            _value(record, label_field),
            _value(record, status_field),
        ]
        if part
    )


def _transaction_summary(transaction: Record, merchants_by_id: dict[str, Record]) -> str:
    if not transaction:
        return ""
    merchant = merchants_by_id.get(_value(transaction, "merchant_id"), {})
    return (
        f"{_value(transaction, 'transaction_id')} - {_value(transaction, 'amount_usd')} "
        f"{_value(transaction, 'currency')} - {_value(transaction, 'transaction_type')} - "
        f"{_value(merchant, 'merchant_name')} - {_value(transaction, 'status')}"
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
