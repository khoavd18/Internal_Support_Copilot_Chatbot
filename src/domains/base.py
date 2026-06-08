from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DomainValidationResult:
    domain: str
    data_dir: Path | None = None
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)


class DomainAdapter(ABC):
    """Interface for plugging a domain into the shared copilot infrastructure."""

    name: str
    default_data_dir: Path
    default_collection_name: str

    @abstractmethod
    def load_dataset(self, data_dir: Path) -> dict[str, Any]:
        """Load domain data into a structured in-memory dataset."""

    @abstractmethod
    def validate_dataset(self, dataset: dict[str, Any]) -> DomainValidationResult:
        """Validate a loaded dataset."""

    @abstractmethod
    def build_documents(self, dataset: dict[str, Any]) -> list[dict[str, Any]]:
        """Convert a loaded dataset into RAG-ready documents."""

    @abstractmethod
    def build_graph(self, dataset: dict[str, Any]) -> Any:
        """Build a domain-specific Knowledge Graph representation."""

    @abstractmethod
    def get_eval_queries(self) -> Path:
        """Return the default evaluation query file for the domain."""

    @abstractmethod
    def get_prompt_templates(self) -> dict[str, str]:
        """Return domain prompt templates used by generation or future agents."""


def format_validation_result(result: DomainValidationResult) -> str:
    data_dir = result.data_dir.as_posix() if result.data_dir else "loaded dataset"
    lines = [f"Validated domain '{result.domain}' dataset: {data_dir}"]

    if result.counts:
        lines.append("")
        lines.append("Counts:")
        for key in sorted(result.counts):
            lines.append(f"  {key}: {result.counts[key]}")

    if result.is_valid:
        lines.append("")
        lines.append("Result: OK")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"Result: FAILED ({len(result.errors)} error(s))")
    for error in result.errors:
        lines.append(f"  - {error}")
    return "\n".join(lines)
