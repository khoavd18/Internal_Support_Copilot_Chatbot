from __future__ import annotations

from typing import Any

Record = dict[str, Any]
Dataset = dict[str, list[Record]]

DATASET_KEYS = (
    "customers",
    "accounts",
    "transactions",
    "merchants",
    "fraud_alerts",
    "aml_cases",
    "policies",
)

REQUIRED_COLUMNS: dict[str, tuple[str, ...]] = {
    "customers": (
        "customer_id",
        "full_name",
        "email",
        "segment",
        "region",
        "risk_tier",
        "kyc_status",
        "created_at",
    ),
    "accounts": (
        "account_id",
        "customer_id",
        "account_type",
        "status",
        "opened_at",
        "balance_usd",
        "online_banking_enabled",
    ),
    "transactions": (
        "transaction_id",
        "account_id",
        "customer_id",
        "merchant_id",
        "amount_usd",
        "currency",
        "transaction_type",
        "channel",
        "country",
        "created_at",
        "status",
        "device_id",
        "ip_country",
    ),
    "merchants": ("merchant_id", "merchant_name", "category", "country", "risk_level"),
    "fraud_alerts": (
        "alert_id",
        "transaction_id",
        "customer_id",
        "account_id",
        "alert_type",
        "severity",
        "status",
        "created_at",
        "rule_name",
        "summary",
        "recommended_action",
    ),
    "aml_cases": (
        "case_id",
        "customer_id",
        "account_id",
        "case_type",
        "status",
        "opened_at",
        "priority",
        "summary",
        "linked_alert_ids",
        "assigned_team",
        "recommended_action",
    ),
    "policies": ("policy_id", "title", "path", "content"),
}

UNIQUE_ID_FIELDS = {
    "customers": "customer_id",
    "accounts": "account_id",
    "transactions": "transaction_id",
    "merchants": "merchant_id",
    "fraud_alerts": "alert_id",
    "aml_cases": "case_id",
    "policies": "policy_id",
}
