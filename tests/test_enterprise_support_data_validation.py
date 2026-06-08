from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

from scripts.validate_enterprise_support_data import (
    DEFAULT_DATASET_ROOT,
    format_report,
    validate_dataset,
)


def _copy_sample_dataset(tmp_path: Path) -> Path:
    dataset_root = tmp_path / "sample_enterprise_support"
    shutil.copytree(DEFAULT_DATASET_ROOT, dataset_root)
    return dataset_root


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def test_sample_enterprise_support_dataset_is_valid() -> None:
    report = validate_dataset(DEFAULT_DATASET_ROOT)

    assert report.is_valid, format_report(report)
    assert report.counts["crm/customers.csv"] == 10
    assert report.counts["support/tickets.csv"] == 30
    assert report.counts["support/ticket_messages.csv"] == 80
    assert report.counts["support/ticket_resolutions.csv"] == 20
    assert report.counts["engineering/github_issues.jsonl"] == 30
    assert report.counts["risk/risk_events.csv"] == 40


def test_validator_reports_missing_file_missing_column_and_empty_markdown(
    tmp_path: Path,
) -> None:
    dataset_root = _copy_sample_dataset(tmp_path)
    (dataset_root / "crm/accounts.csv").unlink()

    products_path = dataset_root / "crm/products.csv"
    fieldnames, rows = _read_csv(products_path)
    fieldnames.remove("product_name")
    _write_csv(
        products_path,
        fieldnames,
        [{key: value for key, value in row.items() if key in fieldnames} for row in rows],
    )

    (dataset_root / "knowledge_base/sla_policy.md").write_text("", encoding="utf-8")

    report = validate_dataset(dataset_root)

    assert not report.is_valid
    assert "crm/accounts.csv: missing required file" in report.errors
    assert "crm/products.csv: missing required columns: product_name" in report.errors
    assert "knowledge_base/sla_policy.md: markdown file is empty" in report.errors


def test_validator_reports_duplicate_ids_and_invalid_csv_foreign_keys(
    tmp_path: Path,
) -> None:
    dataset_root = _copy_sample_dataset(tmp_path)

    customers_path = dataset_root / "crm/customers.csv"
    customer_fields, customer_rows = _read_csv(customers_path)
    customer_rows[1]["customer_id"] = customer_rows[0]["customer_id"]
    _write_csv(customers_path, customer_fields, customer_rows)

    tickets_path = dataset_root / "support/tickets.csv"
    ticket_fields, ticket_rows = _read_csv(tickets_path)
    ticket_rows[0]["customer_id"] = "cust_missing"
    ticket_rows[1]["product_id"] = "prod_missing"
    _write_csv(tickets_path, ticket_fields, ticket_rows)

    messages_path = dataset_root / "support/ticket_messages.csv"
    message_fields, message_rows = _read_csv(messages_path)
    message_rows[0]["ticket_id"] = "tkt_missing"
    _write_csv(messages_path, message_fields, message_rows)

    resolutions_path = dataset_root / "support/ticket_resolutions.csv"
    resolution_fields, resolution_rows = _read_csv(resolutions_path)
    resolution_rows[0]["ticket_id"] = "tkt_missing"
    _write_csv(resolutions_path, resolution_fields, resolution_rows)

    risks_path = dataset_root / "risk/risk_events.csv"
    risk_fields, risk_rows = _read_csv(risks_path)
    risk_rows[0]["customer_id"] = "cust_missing"
    _write_csv(risks_path, risk_fields, risk_rows)

    report = validate_dataset(dataset_root)

    assert not report.is_valid
    assert any("duplicate customer_id 'cust_001'" in error for error in report.errors)
    assert any(
        "support/tickets.csv:2: invalid foreign key customer_id='cust_missing'" in error
        for error in report.errors
    )
    assert any(
        "support/tickets.csv:3: invalid foreign key product_id='prod_missing'" in error
        for error in report.errors
    )
    assert any(
        "support/ticket_messages.csv:2: invalid foreign key ticket_id='tkt_missing'" in error
        for error in report.errors
    )
    assert any(
        "support/ticket_resolutions.csv:2: invalid foreign key ticket_id='tkt_missing'" in error
        for error in report.errors
    )
    assert any(
        "risk/risk_events.csv:2: invalid foreign key customer_id='cust_missing'" in error
        for error in report.errors
    )


def test_validator_reports_invalid_jsonl_and_issue_service_foreign_key(
    tmp_path: Path,
) -> None:
    dataset_root = _copy_sample_dataset(tmp_path)
    issues_path = dataset_root / "engineering/github_issues.jsonl"

    issue_rows = [
        json.loads(line)
        for line in issues_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    issue_rows[0]["service_id"] = "svc_missing"

    with issues_path.open("w", encoding="utf-8", newline="\n") as file:
        for issue in issue_rows:
            file.write(json.dumps(issue, ensure_ascii=True) + "\n")
        file.write("{not valid json}\n")

    report = validate_dataset(dataset_root)

    assert not report.is_valid
    assert any(
        "engineering/github_issues.jsonl:31: invalid JSON" in error for error in report.errors
    )
    assert any(
        "engineering/github_issues.jsonl:1: invalid foreign key service_id='svc_missing'" in error
        for error in report.errors
    )
