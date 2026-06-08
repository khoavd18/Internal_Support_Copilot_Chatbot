from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

Record = dict[str, Any]
Dataset = dict[str, list[Record]]

ENTERPRISE_SUPPORT_SOURCE = "enterprise_support"

CUSTOMERS_PATH = Path("crm/customers.csv")
ACCOUNTS_PATH = Path("crm/accounts.csv")
PRODUCTS_PATH = Path("crm/products.csv")
TICKETS_PATH = Path("support/tickets.csv")
TICKET_MESSAGES_PATH = Path("support/ticket_messages.csv")
TICKET_RESOLUTIONS_PATH = Path("support/ticket_resolutions.csv")
SERVICE_CATALOG_PATH = Path("engineering/service_catalog.csv")
GITHUB_ISSUES_PATH = Path("engineering/github_issues.jsonl")
RISK_EVENTS_PATH = Path("risk/risk_events.csv")
KNOWLEDGE_BASE_DIR = Path("knowledge_base")

METADATA_RELATIONSHIP_FIELDS = ("customer_id", "ticket_id", "product_id", "service_id")


def _first_non_empty(record: Record, fields: tuple[str, ...]) -> Any:
    for field in fields:
        value = record.get(field)
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return ""


def _dataset_path(data_dir: Path, relative_path: Path) -> Path:
    return data_dir / relative_path


def _read_csv_records(data_dir: Path, relative_path: Path) -> list[Record]:
    path = _dataset_path(data_dir, relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing enterprise support CSV file: {path}")

    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _read_jsonl_records(data_dir: Path, relative_path: Path) -> list[Record]:
    path = _dataset_path(data_dir, relative_path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing enterprise support JSONL file: {path}")

    records: list[Record] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc.msg}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object in {path} at line {line_number}")

            records.append(record)

    return records


def _build_metadata(
    record: Record,
    *,
    source_type: str,
    entity_id_fields: tuple[str, ...],
    title_fields: tuple[str, ...] = (),
    path: str = "",
) -> Record:
    metadata: Record = {
        "source": ENTERPRISE_SUPPORT_SOURCE,
        "source_type": source_type,
        "entity_id": _first_non_empty(record, entity_id_fields),
    }

    title = _first_non_empty(record, title_fields)
    if title:
        metadata["title"] = title

    for field in METADATA_RELATIONSHIP_FIELDS:
        value = _first_non_empty(record, (field,))
        if value:
            metadata[field] = value

    if path:
        metadata["path"] = path

    return metadata


def _with_metadata(
    record: Record,
    *,
    source_type: str,
    entity_id_fields: tuple[str, ...],
    title_fields: tuple[str, ...] = (),
    path: str = "",
) -> Record:
    loaded = dict(record)
    loaded["metadata"] = _build_metadata(
        loaded,
        source_type=source_type,
        entity_id_fields=entity_id_fields,
        title_fields=title_fields,
        path=path,
    )
    return loaded


def _load_csv_entity(
    data_dir: Path,
    relative_path: Path,
    *,
    source_type: str,
    entity_id_fields: tuple[str, ...],
    title_fields: tuple[str, ...] = (),
) -> list[Record]:
    records = _read_csv_records(data_dir, relative_path)
    return [
        _with_metadata(
            record,
            source_type=source_type,
            entity_id_fields=entity_id_fields,
            title_fields=title_fields,
            path=relative_path.as_posix(),
        )
        for record in records
    ]


def _parse_scalar_front_matter_value(value: str) -> Any:
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        items = value[1:-1].split(",")
        return [item.strip().strip("\"'") for item in items if item.strip()]
    return value.strip("\"'")


def _parse_front_matter(text: str) -> tuple[Record, str]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    if not lines or lines[0].strip() != "---":
        return {}, normalized.strip()

    end_index = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end_index = index
            break

    if end_index is None:
        return {}, normalized.strip()

    metadata: Record = {}
    for line in lines[1:end_index]:
        if not line.strip() or line.lstrip().startswith("#") or ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        if key:
            metadata[key] = _parse_scalar_front_matter_value(value)

    content = "\n".join(lines[end_index + 1 :]).strip()
    return metadata, content


def _extract_markdown_title(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""


def load_customers(data_dir: Path) -> list[dict]:
    return _load_csv_entity(
        data_dir,
        CUSTOMERS_PATH,
        source_type="customer",
        entity_id_fields=("customer_id",),
        title_fields=("full_name", "email"),
    )


def load_accounts(data_dir: Path) -> list[dict]:
    return _load_csv_entity(
        data_dir,
        ACCOUNTS_PATH,
        source_type="account",
        entity_id_fields=("account_id",),
        title_fields=("account_name",),
    )


def load_products(data_dir: Path) -> list[dict]:
    return _load_csv_entity(
        data_dir,
        PRODUCTS_PATH,
        source_type="product",
        entity_id_fields=("product_id",),
        title_fields=("product_name",),
    )


def load_tickets(data_dir: Path) -> list[dict]:
    return _load_csv_entity(
        data_dir,
        TICKETS_PATH,
        source_type="ticket",
        entity_id_fields=("ticket_id",),
        title_fields=("title",),
    )


def load_ticket_messages(data_dir: Path) -> list[dict]:
    messages = _load_csv_entity(
        data_dir,
        TICKET_MESSAGES_PATH,
        source_type="ticket_message",
        entity_id_fields=("message_id",),
        title_fields=("message_type",),
    )

    for message in messages:
        if message.get("author_type") == "customer" and message.get("author_id"):
            message["metadata"]["customer_id"] = message["author_id"]

    return messages


def load_ticket_resolutions(data_dir: Path) -> list[dict]:
    return _load_csv_entity(
        data_dir,
        TICKET_RESOLUTIONS_PATH,
        source_type="ticket_resolution",
        entity_id_fields=("resolution_id",),
        title_fields=("summary", "resolution_type"),
    )


def load_service_catalog(data_dir: Path) -> list[dict]:
    return _load_csv_entity(
        data_dir,
        SERVICE_CATALOG_PATH,
        source_type="service",
        entity_id_fields=("service_id",),
        title_fields=("service_name",),
    )


def load_github_issues(data_dir: Path) -> list[dict]:
    records = _read_jsonl_records(data_dir, GITHUB_ISSUES_PATH)
    return [
        _with_metadata(
            record,
            source_type="github_issue",
            entity_id_fields=("issue_id",),
            title_fields=("title",),
            path=GITHUB_ISSUES_PATH.as_posix(),
        )
        for record in records
    ]


def load_risk_events(data_dir: Path) -> list[dict]:
    return _load_csv_entity(
        data_dir,
        RISK_EVENTS_PATH,
        source_type="risk_event",
        entity_id_fields=("risk_event_id",),
        title_fields=("summary", "event_type"),
    )


def load_knowledge_base_docs(data_dir: Path) -> list[dict]:
    knowledge_base_dir = _dataset_path(data_dir, KNOWLEDGE_BASE_DIR)
    if not knowledge_base_dir.is_dir():
        raise FileNotFoundError(
            f"Missing enterprise support knowledge base directory: {knowledge_base_dir}"
        )

    docs: list[Record] = []
    for path in sorted(knowledge_base_dir.glob("*.md")):
        relative_path = path.relative_to(data_dir).as_posix()
        front_matter, content = _parse_front_matter(path.read_text(encoding="utf-8"))

        title = str(front_matter.get("title") or _extract_markdown_title(content) or path.stem)
        policy_id = str(front_matter.get("policy_id") or path.stem)

        record: Record = {
            **front_matter,
            "policy_id": policy_id,
            "title": title,
            "path": relative_path,
            "content": content,
            "text": content,
        }
        record["metadata"] = _build_metadata(
            record,
            source_type="knowledge_base",
            entity_id_fields=("policy_id",),
            title_fields=("title",),
            path=relative_path,
        )
        docs.append(record)

    return docs


def load_enterprise_support_dataset(data_dir: Path) -> dict:
    return {
        "customers": load_customers(data_dir),
        "accounts": load_accounts(data_dir),
        "products": load_products(data_dir),
        "tickets": load_tickets(data_dir),
        "ticket_messages": load_ticket_messages(data_dir),
        "ticket_resolutions": load_ticket_resolutions(data_dir),
        "service_catalog": load_service_catalog(data_dir),
        "github_issues": load_github_issues(data_dir),
        "risk_events": load_risk_events(data_dir),
        "knowledge_base_docs": load_knowledge_base_docs(data_dir),
    }
