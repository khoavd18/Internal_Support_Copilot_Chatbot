from __future__ import annotations

from pathlib import Path
from typing import Any

from src.domains.base import DomainAdapter, DomainValidationResult
from src.domains.enterprise_support.documents import build_enterprise_support_documents
from src.domains.enterprise_support.kg import build_graph_from_enterprise_support_dataset
from src.domains.enterprise_support.loader import load_enterprise_support_dataset
from src.domains.enterprise_support.prompts import get_prompt_templates

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "sample_enterprise_support"
DEFAULT_COLLECTION_NAME = "enterprise_support_copilot_qdrant"
DEFAULT_EVAL_QUERIES = PROJECT_ROOT / "eval" / "enterprise_support_queries.jsonl"

REQUIRED_DATASET_KEYS = (
    "customers",
    "accounts",
    "products",
    "tickets",
    "ticket_messages",
    "ticket_resolutions",
    "knowledge_base_docs",
    "service_catalog",
    "github_issues",
    "risk_events",
)


class EnterpriseSupportAdapter(DomainAdapter):
    name = "enterprise_support"
    default_data_dir = DEFAULT_DATA_DIR
    default_collection_name = DEFAULT_COLLECTION_NAME

    def load_dataset(self, data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, Any]:
        return load_enterprise_support_dataset(Path(data_dir))

    def validate_dataset(self, dataset: dict[str, Any]) -> DomainValidationResult:
        errors: list[str] = []
        counts: dict[str, int] = {}

        for key in REQUIRED_DATASET_KEYS:
            value = dataset.get(key)
            if not isinstance(value, list):
                errors.append(f"{key}: expected list")
                counts[key] = 0
                continue
            counts[key] = len(value)
            if not value:
                errors.append(f"{key}: no records loaded")

        return DomainValidationResult(
            domain=self.name,
            is_valid=not errors,
            errors=errors,
            counts=counts,
        )

    def validate_data_dir(self, data_dir: Path = DEFAULT_DATA_DIR) -> DomainValidationResult:
        from scripts.validate_enterprise_support_data import validate_dataset

        report = validate_dataset(Path(data_dir))
        return DomainValidationResult(
            domain=self.name,
            data_dir=report.root,
            is_valid=report.is_valid,
            errors=list(report.errors),
            counts=dict(report.counts),
        )

    def build_documents(self, dataset: dict[str, Any]) -> list[dict[str, Any]]:
        return build_enterprise_support_documents(dataset)

    def build_graph(self, dataset: dict[str, Any]) -> Any:
        return build_graph_from_enterprise_support_dataset(dataset)

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
        from eval.evaluate_enterprise_support import run_evaluation

        return run_evaluation(
            queries_path=self.get_eval_queries(),
            data_dir=Path(data_dir),
            dry_run=dry_run,
            limit=limit,
        )
