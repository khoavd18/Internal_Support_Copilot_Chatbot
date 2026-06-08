from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.domains.banking_fraud.documents import build_banking_fraud_documents
from src.domains.banking_fraud.kg import build_banking_fraud_graph
from src.domains.banking_fraud.loader import load_banking_fraud_dataset
from src.domains.banking_fraud.prompts import get_prompt_templates
from src.domains.banking_fraud.schemas import (
    DATASET_KEYS,
    REQUIRED_COLUMNS,
    UNIQUE_ID_FIELDS,
    Dataset,
    Record,
)
from src.domains.base import DomainAdapter, DomainValidationResult

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "sample_banking_fraud"
DEFAULT_COLLECTION_NAME = "banking_fraud_copilot_qdrant"
DEFAULT_EVAL_QUERIES = PROJECT_ROOT / "eval" / "banking_fraud_queries.jsonl"
RECALL_K = 5

REQUIRED_FILES = (
    "customers.csv",
    "accounts.csv",
    "transactions.csv",
    "merchants.csv",
    "fraud_alerts.csv",
    "aml_cases.csv",
    "policies/aml_policy.md",
    "policies/card_fraud_policy.md",
    "policies/account_takeover_policy.md",
)

FOREIGN_KEYS = (
    ("accounts", "customer_id", "customers", "customer_id"),
    ("transactions", "account_id", "accounts", "account_id"),
    ("transactions", "customer_id", "customers", "customer_id"),
    ("transactions", "merchant_id", "merchants", "merchant_id"),
    ("fraud_alerts", "transaction_id", "transactions", "transaction_id"),
    ("fraud_alerts", "customer_id", "customers", "customer_id"),
    ("fraud_alerts", "account_id", "accounts", "account_id"),
    ("aml_cases", "customer_id", "customers", "customer_id"),
    ("aml_cases", "account_id", "accounts", "account_id"),
)

ENTITY_ID_FIELDS = (
    "entity_id",
    "customer_id",
    "account_id",
    "transaction_id",
    "merchant_id",
    "alert_id",
    "case_id",
    "policy_id",
)


class BankingFraudAdapter(DomainAdapter):
    name = "banking_fraud"
    default_data_dir = DEFAULT_DATA_DIR
    default_collection_name = DEFAULT_COLLECTION_NAME

    def load_dataset(self, data_dir: Path = DEFAULT_DATA_DIR) -> Dataset:
        return load_banking_fraud_dataset(Path(data_dir))

    def validate_dataset(self, dataset: dict[str, Any]) -> DomainValidationResult:
        typed_dataset: Dataset = dataset
        errors: list[str] = []
        counts: dict[str, int] = {}

        for key in DATASET_KEYS:
            records = typed_dataset.get(key)
            if not isinstance(records, list):
                errors.append(f"{key}: expected list")
                counts[key] = 0
                continue
            counts[key] = len(records)
            if not records:
                errors.append(f"{key}: no records loaded")
                continue
            _validate_columns(key, records, errors)

        _validate_unique_ids(typed_dataset, errors)
        _validate_foreign_keys(typed_dataset, errors)
        _validate_linked_alerts(typed_dataset, errors)
        _validate_policy_ids(typed_dataset, errors)

        return DomainValidationResult(
            domain=self.name,
            is_valid=not errors,
            errors=errors,
            counts=counts,
        )

    def validate_data_dir(self, data_dir: Path = DEFAULT_DATA_DIR) -> DomainValidationResult:
        data_dir = Path(data_dir).resolve()
        missing_files = [
            f"{rel_path}: missing required file"
            for rel_path in REQUIRED_FILES
            if not (data_dir / rel_path).is_file()
        ]
        if missing_files:
            return DomainValidationResult(
                domain=self.name,
                data_dir=data_dir,
                is_valid=False,
                errors=missing_files,
                counts={},
            )

        try:
            dataset = self.load_dataset(data_dir)
        except (OSError, ValueError) as exc:
            return DomainValidationResult(
                domain=self.name,
                data_dir=data_dir,
                is_valid=False,
                errors=[str(exc)],
                counts={},
            )

        result = self.validate_dataset(dataset)
        result.data_dir = data_dir
        return result

    def build_documents(self, dataset: dict[str, Any]) -> list[dict[str, Any]]:
        return build_banking_fraud_documents(dataset)

    def build_graph(self, dataset: dict[str, Any]) -> Any:
        return build_banking_fraud_graph(dataset)

    def get_eval_queries(self) -> Path:
        return DEFAULT_EVAL_QUERIES

    def get_prompt_templates(self) -> dict[str, str]:
        return get_prompt_templates()

    def run_evaluation(
        self,
        *,
        data_dir: Path = DEFAULT_DATA_DIR,
        dry_run: bool = False,
        limit: int | None = None,
    ) -> dict[str, Any]:
        dataset = self.load_dataset(Path(data_dir))
        documents = self.build_documents(dataset)
        cases = _load_eval_cases(self.get_eval_queries())
        if limit is not None:
            cases = cases[:limit]

        records = [_evaluate_case(case, documents) for case in cases]
        summary = _summary(
            records,
            data_dir=Path(data_dir),
            mode="dry_run_local" if dry_run else "local_adapter",
        )
        return {"records": records, "summary": summary, "paths": {}}


def _validate_columns(key: str, records: list[Record], errors: list[str]) -> None:
    required = set(REQUIRED_COLUMNS[key])
    for index, record in enumerate(records, start=1):
        missing = sorted(required - set(record))
        if missing:
            errors.append(f"{key}:{index}: missing columns: {', '.join(missing)}")


def _validate_unique_ids(dataset: Dataset, errors: list[str]) -> None:
    for key, field in UNIQUE_ID_FIELDS.items():
        seen: set[str] = set()
        for index, record in enumerate(dataset.get(key, []), start=1):
            value = _value(record, field)
            if not value:
                errors.append(f"{key}:{index}: missing {field}")
                continue
            if value in seen:
                errors.append(f"{key}:{index}: duplicate {field} '{value}'")
            seen.add(value)


def _validate_foreign_keys(dataset: Dataset, errors: list[str]) -> None:
    for source_key, source_field, target_key, target_field in FOREIGN_KEYS:
        target_values = {
            _value(record, target_field)
            for record in dataset.get(target_key, [])
            if _value(record, target_field)
        }
        for index, record in enumerate(dataset.get(source_key, []), start=1):
            value = _value(record, source_field)
            if value not in target_values:
                errors.append(
                    f"{source_key}:{index}: invalid foreign key "
                    f"{source_field}='{value}' -> {target_key}.{target_field}"
                )


def _validate_linked_alerts(dataset: Dataset, errors: list[str]) -> None:
    alert_ids = {
        _value(record, "alert_id")
        for record in dataset.get("fraud_alerts", [])
        if _value(record, "alert_id")
    }
    for index, record in enumerate(dataset.get("aml_cases", []), start=1):
        for alert_id in _split_pipe(_value(record, "linked_alert_ids")):
            if alert_id not in alert_ids:
                errors.append(f"aml_cases:{index}: invalid linked_alert_ids value '{alert_id}'")


def _validate_policy_ids(dataset: Dataset, errors: list[str]) -> None:
    expected = {
        "bf_pol_aml",
        "bf_pol_card_fraud",
        "bf_pol_account_takeover",
    }
    actual = {
        _value(record, "policy_id")
        for record in dataset.get("policies", [])
        if _value(record, "policy_id")
    }
    missing = sorted(expected - actual)
    if missing:
        errors.append(f"policies: missing policy IDs: {', '.join(missing)}")
    for policy in dataset.get("policies", []):
        if not _value(policy, "content"):
            errors.append(f"policies:{_value(policy, 'policy_id')}: empty policy content")


def _load_eval_cases(path: Path) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            missing = {
                "query_id",
                "query",
                "expected_source_types",
                "expected_entity_ids",
                "expected_answer_points",
            } - set(payload)
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing fields: {', '.join(sorted(missing))}"
                )
            cases.append(payload)
    return cases


def _evaluate_case(case: dict[str, Any], documents: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = _local_retrieve(str(case["query"]), documents, limit=RECALL_K)
    retrieved_entity_ids = _retrieved_entity_ids(evidence)
    retrieved_source_types = _retrieved_source_types(evidence)
    expected_entity_ids = {_normalize(value) for value in case["expected_entity_ids"]}
    expected_source_types = {_normalize(value) for value in case["expected_source_types"]}
    entity_hits = sorted(expected_entity_ids & retrieved_entity_ids)
    source_hits = sorted(expected_source_types & retrieved_source_types)
    recall = len(entity_hits) / len(expected_entity_ids) if expected_entity_ids else None

    return {
        "query_id": str(case["query_id"]),
        "query": str(case["query"]),
        "category": str(case.get("category") or ""),
        "expected_source_types": list(case["expected_source_types"]),
        "expected_entity_ids": list(case["expected_entity_ids"]),
        "expected_answer_points": list(case["expected_answer_points"]),
        "retrieved_entity_ids_at_5": sorted(retrieved_entity_ids),
        "retrieved_source_types_at_5": sorted(retrieved_source_types),
        "metrics": {
            "recall_at_5": _round(recall),
            "entity_hits": entity_hits,
            "source_type_hit": bool(source_hits) if expected_source_types else None,
            "source_type_all_hit": expected_source_types.issubset(retrieved_source_types)
            if expected_source_types
            else None,
            "source_type_hits": source_hits,
            "groundedness": bool(evidence and (entity_hits or source_hits)),
            "missing_info_handled": None,
        },
        "evidence": [_compact_evidence(item) for item in evidence],
        "retrieval_stats": {
            "mode": "banking_fraud_local_lexical",
            "top_k": RECALL_K,
            "merged_count": len(evidence),
            "vector_error": "",
        },
    }


def _local_retrieve(
    query: str, documents: list[dict[str, Any]], *, limit: int
) -> list[dict[str, Any]]:
    query_tokens = set(_tokenize(query))
    query_lower = query.lower()
    scored: list[tuple[float, dict[str, Any]]] = []
    for document in documents:
        metadata = dict(document.get("metadata") or {})
        haystack = " ".join(
            [
                str(document.get("id") or ""),
                str(document.get("text") or ""),
                " ".join(str(value) for value in metadata.values()),
            ]
        ).lower()
        score = float(len(query_tokens & set(_tokenize(haystack))))
        for value in _metadata_values(metadata):
            if value and value.lower() in query_lower:
                score += 4.0
        if score > 0:
            scored.append((score, document))

    scored.sort(
        key=lambda item: (
            item[0],
            str(item[1].get("metadata", {}).get("source_type") or ""),
            str(item[1].get("id") or ""),
        ),
        reverse=True,
    )
    return [
        {
            "id": str(document.get("id") or ""),
            "text": str(document.get("text") or ""),
            "metadata": dict(document.get("metadata") or {}),
            "context_source": "vector",
            "source_type": str(document.get("metadata", {}).get("source_type") or ""),
            "title": str(document.get("metadata", {}).get("title") or document.get("id") or ""),
            "score": score,
        }
        for score, document in scored[:limit]
    ]


def _summary(records: list[dict[str, Any]], *, data_dir: Path, mode: str) -> dict[str, Any]:
    recall_values = [
        record["metrics"]["recall_at_5"]
        for record in records
        if record["metrics"]["recall_at_5"] is not None
    ]
    source_hits = [
        record["metrics"]["source_type_hit"]
        for record in records
        if record["metrics"]["source_type_hit"] is not None
    ]
    source_all_hits = [
        record["metrics"]["source_type_all_hit"]
        for record in records
        if record["metrics"]["source_type_all_hit"] is not None
    ]
    grounded = [
        record["metrics"]["groundedness"]
        for record in records
        if record["metrics"]["groundedness"] is not None
    ]
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "queries_path": str(DEFAULT_EVAL_QUERIES),
        "data_dir": str(data_dir),
        "mode": mode,
        "top_k": RECALL_K,
        "recall_k": RECALL_K,
        "graph_depth": 0,
        "query_count": len(records),
        "metrics": {
            "recall_at_5": _round(_average(recall_values)),
            "source_type_hit_rate": _round(_rate(source_hits)),
            "source_type_all_hit_rate": _round(_rate(source_all_hits)),
            "groundedness_rate": _round(_rate(grounded)),
            "missing_info_handling_rate": None,
            "missing_info_cases": 0,
        },
        "vector_error_count": 0,
        "vector_errors_sample": [],
        "weak_cases": [
            {
                "query_id": record["query_id"],
                "category": record["category"],
                "recall_at_5": record["metrics"]["recall_at_5"],
                "source_type_hit": record["metrics"]["source_type_hit"],
                "groundedness": record["metrics"]["groundedness"],
            }
            for record in records
            if (record["metrics"]["recall_at_5"] or 0.0) < 0.4
        ][:10],
    }


def _retrieved_entity_ids(evidence: list[dict[str, Any]]) -> set[str]:
    entity_ids: set[str] = set()
    for item in evidence:
        metadata = item.get("metadata") or {}
        entity_ids.update(_split_entity_identifier(item.get("id")))
        for field in ENTITY_ID_FIELDS:
            value = metadata.get(field)
            if isinstance(value, list):
                for item_value in value:
                    entity_ids.update(_split_entity_identifier(item_value))
            else:
                entity_ids.update(_split_entity_identifier(value))
    return {_normalize(value) for value in entity_ids if _normalize(value)}


def _retrieved_source_types(evidence: list[dict[str, Any]]) -> set[str]:
    return {
        _normalize(item.get("source_type") or item.get("metadata", {}).get("source_type"))
        for item in evidence
        if _normalize(item.get("source_type") or item.get("metadata", {}).get("source_type"))
    }


def _compact_evidence(item: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(item.get("metadata") or {})
    text = str(item.get("text") or "")
    if len(text) > 500:
        text = text[:497] + "..."
    return {
        "id": item.get("id"),
        "title": item.get("title"),
        "source_type": item.get("source_type"),
        "metadata": {
            key: metadata.get(key)
            for key in ENTITY_ID_FIELDS
            if metadata.get(key) not in ("", None, [])
        },
        "text": text,
    }


def _metadata_values(metadata: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for field in ENTITY_ID_FIELDS:
        value = metadata.get(field)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value:
            values.append(str(value))
    title = metadata.get("title")
    if title:
        values.append(str(title))
    return values


def _split_entity_identifier(value: Any) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    aliases = {text}
    if "::" in text:
        aliases.add(text.rsplit("::", 1)[-1])
    if ":" in text:
        aliases.add(text.rsplit(":", 1)[-1])
    return aliases


def _split_pipe(value: str) -> list[str]:
    return [part.strip() for part in value.split("|") if part.strip()]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _value(record: Record | None, key: str, default: str = "") -> str:
    if not record:
        return default
    value = record.get(key, default)
    if value is None:
        return default
    return str(value).strip()


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def _average(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _rate(values: list[bool | None]) -> float | None:
    clean = [bool(value) for value in values if value is not None]
    return sum(1 for value in clean if value) / len(clean) if clean else None


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None
