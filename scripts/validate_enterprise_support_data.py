from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "sample_enterprise_support"

REQUIRED_FILES = (
    "crm/customers.csv",
    "crm/accounts.csv",
    "crm/products.csv",
    "support/tickets.csv",
    "support/ticket_messages.csv",
    "support/ticket_resolutions.csv",
    "knowledge_base/sla_policy.md",
    "knowledge_base/access_policy.md",
    "knowledge_base/refund_policy.md",
    "knowledge_base/api_timeout_runbook.md",
    "knowledge_base/login_troubleshooting.md",
    "knowledge_base/security_escalation_policy.md",
    "knowledge_base/incident_response_policy.md",
    "knowledge_base/enterprise_support_policy.md",
    "knowledge_base/customer_risk_policy.md",
    "knowledge_base/data_retention_policy.md",
    "engineering/service_catalog.csv",
    "engineering/github_issues.jsonl",
    "risk/risk_events.csv",
)

REQUIRED_CSV_COLUMNS = {
    "crm/customers.csv": (
        "customer_id",
        "account_id",
        "full_name",
        "email",
        "role",
        "region",
        "timezone",
        "created_at",
        "status",
        "support_tier",
        "preferred_contact_channel",
    ),
    "crm/accounts.csv": (
        "account_id",
        "account_name",
        "industry",
        "segment",
        "region",
        "account_owner",
        "contract_start_date",
        "contract_end_date",
        "arr_usd",
        "health_score",
        "risk_level",
        "created_at",
    ),
    "crm/products.csv": (
        "product_id",
        "product_name",
        "product_family",
        "service_id",
        "plan_name",
        "lifecycle_stage",
        "support_owner_team",
        "engineering_owner_team",
        "docs_url",
    ),
    "support/tickets.csv": (
        "ticket_id",
        "account_id",
        "customer_id",
        "product_id",
        "service_id",
        "title",
        "description",
        "category",
        "priority",
        "severity",
        "status",
        "channel",
        "created_at",
        "updated_at",
        "first_response_due_at",
        "resolution_due_at",
        "sla_status",
        "assigned_team",
        "assignee",
        "tags",
    ),
    "support/ticket_messages.csv": (
        "message_id",
        "ticket_id",
        "author_type",
        "author_id",
        "created_at",
        "visibility",
        "message_type",
        "body",
        "sentiment",
        "contains_action_item",
    ),
    "support/ticket_resolutions.csv": (
        "resolution_id",
        "ticket_id",
        "resolved_at",
        "resolution_type",
        "summary",
        "root_cause",
        "linked_policy_id",
        "linked_issue_id",
        "customer_visible_response",
        "preventive_action",
    ),
    "engineering/service_catalog.csv": (
        "service_id",
        "service_name",
        "description",
        "product_id",
        "owner_team",
        "support_escalation_team",
        "tier",
        "slack_channel",
        "runbook_policy_id",
        "on_call_rotation",
        "repo",
        "status",
    ),
    "risk/risk_events.csv": (
        "risk_event_id",
        "account_id",
        "customer_id",
        "ticket_id",
        "product_id",
        "service_id",
        "event_type",
        "severity",
        "risk_score",
        "detected_at",
        "source",
        "summary",
        "evidence_refs",
        "recommended_action",
        "status",
    ),
}

REQUIRED_GITHUB_ISSUE_FIELDS = (
    "issue_id",
    "repo",
    "number",
    "title",
    "body",
    "state",
    "service_id",
    "product_id",
    "created_at",
    "updated_at",
)

UNIQUE_ID_COLUMNS = {
    "crm/customers.csv": "customer_id",
    "crm/accounts.csv": "account_id",
    "crm/products.csv": "product_id",
    "support/tickets.csv": "ticket_id",
    "support/ticket_messages.csv": "message_id",
    "support/ticket_resolutions.csv": "resolution_id",
    "engineering/service_catalog.csv": "service_id",
    "risk/risk_events.csv": "risk_event_id",
    "engineering/github_issues.jsonl": "issue_id",
}

FOREIGN_KEY_RULES = (
    ("support/tickets.csv", "customer_id", "crm/customers.csv", "customer_id"),
    ("support/tickets.csv", "product_id", "crm/products.csv", "product_id"),
    ("support/ticket_messages.csv", "ticket_id", "support/tickets.csv", "ticket_id"),
    (
        "support/ticket_resolutions.csv",
        "ticket_id",
        "support/tickets.csv",
        "ticket_id",
    ),
    ("risk/risk_events.csv", "customer_id", "crm/customers.csv", "customer_id"),
    (
        "engineering/github_issues.jsonl",
        "service_id",
        "engineering/service_catalog.csv",
        "service_id",
    ),
)


@dataclass
class ValidationReport:
    root: Path
    errors: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors


def _as_posix(path: Path) -> str:
    return path.as_posix()


def _required_file_errors(root: Path, report: ValidationReport) -> None:
    for rel_path in REQUIRED_FILES:
        if not (root / rel_path).is_file():
            report.errors.append(f"{rel_path}: missing required file")


def _read_csv_file(root: Path, rel_path: str, report: ValidationReport) -> list[dict[str, str]]:
    path = root / rel_path
    if not path.is_file():
        report.counts[rel_path] = 0
        return []

    try:
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None:
                report.errors.append(f"{rel_path}: missing CSV header")
                report.counts[rel_path] = 0
                return []

            actual_columns = set(reader.fieldnames)
            required_columns = set(REQUIRED_CSV_COLUMNS[rel_path])
            missing_columns = sorted(required_columns - actual_columns)
            if missing_columns:
                report.errors.append(
                    f"{rel_path}: missing required columns: {', '.join(missing_columns)}"
                )

            rows: list[dict[str, str]] = []
            for line_number, row in enumerate(reader, start=2):
                if None in row:
                    report.errors.append(
                        f"{rel_path}:{line_number}: row has extra unparsed columns"
                    )
                    row.pop(None, None)

                row["_line_number"] = str(line_number)
                rows.append(row)

    except UnicodeDecodeError as exc:
        report.errors.append(f"{rel_path}: could not decode as UTF-8: {exc}")
        report.counts[rel_path] = 0
        return []
    except csv.Error as exc:
        report.errors.append(f"{rel_path}: invalid CSV: {exc}")
        report.counts[rel_path] = 0
        return []

    report.counts[rel_path] = len(rows)
    return rows


def _read_github_issues_jsonl(
    root: Path,
    report: ValidationReport,
) -> list[dict[str, Any]]:
    rel_path = "engineering/github_issues.jsonl"
    path = root / rel_path
    if not path.is_file():
        report.counts[rel_path] = 0
        return []

    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue

                try:
                    item = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    report.errors.append(f"{rel_path}:{line_number}: invalid JSON: {exc.msg}")
                    continue

                if not isinstance(item, dict):
                    report.errors.append(f"{rel_path}:{line_number}: JSONL row must be an object")
                    continue

                missing_fields = [
                    field for field in REQUIRED_GITHUB_ISSUE_FIELDS if field not in item
                ]
                if missing_fields:
                    report.errors.append(
                        f"{rel_path}:{line_number}: missing required fields: "
                        f"{', '.join(missing_fields)}"
                    )

                item["_line_number"] = line_number
                rows.append(item)
    except UnicodeDecodeError as exc:
        report.errors.append(f"{rel_path}: could not decode as UTF-8: {exc}")
        report.counts[rel_path] = 0
        return []

    report.counts[rel_path] = len(rows)
    return rows


def _validate_unique_ids(
    rel_path: str,
    id_column: str,
    rows: list[dict[str, Any]],
    report: ValidationReport,
) -> None:
    seen: dict[str, int] = {}
    for index, row in enumerate(rows, start=1):
        line_number = int(row.get("_line_number") or index)
        value = str(row.get(id_column) or "").strip()
        if not value:
            report.errors.append(f"{rel_path}:{line_number}: missing {id_column}")
            continue

        if value in seen:
            report.errors.append(
                f"{rel_path}:{line_number}: duplicate {id_column} '{value}' "
                f"(first seen on line {seen[value]})"
            )
            continue

        seen[value] = line_number


def _validate_foreign_key(
    source_rel_path: str,
    source_column: str,
    target_rel_path: str,
    target_column: str,
    records_by_path: dict[str, list[dict[str, Any]]],
    report: ValidationReport,
) -> None:
    source_rows = records_by_path.get(source_rel_path, [])
    target_rows = records_by_path.get(target_rel_path, [])

    target_values = {
        str(row.get(target_column) or "").strip()
        for row in target_rows
        if str(row.get(target_column) or "").strip()
    }

    for index, row in enumerate(source_rows, start=1):
        line_number = int(row.get("_line_number") or index)
        value = str(row.get(source_column) or "").strip()
        if not value:
            report.errors.append(
                f"{source_rel_path}:{line_number}: missing foreign key {source_column}"
            )
            continue

        if value not in target_values:
            report.errors.append(
                f"{source_rel_path}:{line_number}: invalid foreign key "
                f"{source_column}='{value}' -> {target_rel_path}.{target_column}"
            )


def _validate_knowledge_base_files(root: Path, report: ValidationReport) -> None:
    for rel_path in REQUIRED_FILES:
        if not rel_path.startswith("knowledge_base/"):
            continue

        path = root / rel_path
        if not path.is_file():
            report.counts[rel_path] = 0
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            report.errors.append(f"{rel_path}: could not decode as UTF-8: {exc}")
            report.counts[rel_path] = 0
            continue

        if not text.strip():
            report.errors.append(f"{rel_path}: markdown file is empty")

        report.counts[rel_path] = 1


def validate_dataset(root: Path = DEFAULT_DATASET_ROOT) -> ValidationReport:
    root = root.resolve()
    report = ValidationReport(root=root)
    _required_file_errors(root, report)

    records_by_path: dict[str, list[dict[str, Any]]] = {}
    for rel_path in REQUIRED_CSV_COLUMNS:
        records_by_path[rel_path] = _read_csv_file(root, rel_path, report)

    records_by_path["engineering/github_issues.jsonl"] = _read_github_issues_jsonl(
        root,
        report,
    )
    _validate_knowledge_base_files(root, report)

    for rel_path, id_column in UNIQUE_ID_COLUMNS.items():
        _validate_unique_ids(
            rel_path,
            id_column,
            records_by_path.get(rel_path, []),
            report,
        )

    for rule in FOREIGN_KEY_RULES:
        _validate_foreign_key(*rule, records_by_path=records_by_path, report=report)

    return report


def format_report(report: ValidationReport) -> str:
    lines = [f"Validated enterprise support dataset: {_as_posix(report.root)}"]

    if report.counts:
        lines.append("")
        lines.append("Counts:")
        for rel_path in sorted(report.counts):
            lines.append(f"  {rel_path}: {report.counts[rel_path]}")

    if report.is_valid:
        lines.append("")
        lines.append("Result: OK")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"Result: FAILED ({len(report.errors)} error(s))")
    for error in report.errors:
        lines.append(f"  - {error}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the synthetic enterprise support sample dataset."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root. Defaults to data/sample_enterprise_support.",
    )
    parsed = parser.parse_args()

    report = validate_dataset(parsed.root)
    output = format_report(report)
    stream = sys.stdout if report.is_valid else sys.stderr
    print(output, file=stream)
    return 0 if report.is_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
