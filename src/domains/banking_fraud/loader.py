from __future__ import annotations

import csv
from pathlib import Path

from src.domains.banking_fraud.schemas import Dataset, Record

CUSTOMERS_PATH = Path("customers.csv")
ACCOUNTS_PATH = Path("accounts.csv")
TRANSACTIONS_PATH = Path("transactions.csv")
MERCHANTS_PATH = Path("merchants.csv")
FRAUD_ALERTS_PATH = Path("fraud_alerts.csv")
AML_CASES_PATH = Path("aml_cases.csv")
POLICIES_DIR = Path("policies")

POLICY_IDS = {
    "aml_policy.md": "bf_pol_aml",
    "card_fraud_policy.md": "bf_pol_card_fraud",
    "account_takeover_policy.md": "bf_pol_account_takeover",
}


def load_banking_fraud_dataset(data_dir: Path) -> Dataset:
    data_dir = Path(data_dir)
    return {
        "customers": _read_csv_records(data_dir / CUSTOMERS_PATH),
        "accounts": _read_csv_records(data_dir / ACCOUNTS_PATH),
        "transactions": _read_csv_records(data_dir / TRANSACTIONS_PATH),
        "merchants": _read_csv_records(data_dir / MERCHANTS_PATH),
        "fraud_alerts": _read_csv_records(data_dir / FRAUD_ALERTS_PATH),
        "aml_cases": _read_csv_records(data_dir / AML_CASES_PATH),
        "policies": _read_policy_docs(data_dir / POLICIES_DIR),
    }


def _read_csv_records(path: Path) -> list[Record]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing banking fraud CSV file: {path}")
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def _read_policy_docs(path: Path) -> list[Record]:
    if not path.is_dir():
        raise FileNotFoundError(f"Missing banking fraud policies directory: {path}")

    policies: list[Record] = []
    for file_path in sorted(path.glob("*.md")):
        content = file_path.read_text(encoding="utf-8")
        policy_id = POLICY_IDS.get(file_path.name, file_path.stem)
        policies.append(
            {
                "policy_id": policy_id,
                "title": _title_from_markdown(content) or file_path.stem.replace("_", " ").title(),
                "path": str(Path("policies") / file_path.name),
                "content": content,
            }
        )
    return policies


def _title_from_markdown(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return ""
