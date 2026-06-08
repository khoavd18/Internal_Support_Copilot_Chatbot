from __future__ import annotations

from io import StringIO
from pathlib import Path

from eval import evaluate_domain
from scripts import ingest_domain, validate_domain_data
from src.domains.registry import get_domain_adapter

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_enterprise_support"


def test_registry_returns_enterprise_support_adapter() -> None:
    adapter = get_domain_adapter("enterprise_support")

    assert adapter.name == "enterprise_support"
    assert adapter.default_data_dir.name == "sample_enterprise_support"
    assert adapter.default_collection_name == "enterprise_support_copilot_qdrant"


def test_generic_validation_works_for_enterprise_support() -> None:
    adapter = get_domain_adapter("enterprise_support")

    result = validate_domain_data.validate_domain_data(adapter, DATA_DIR)

    assert result.is_valid
    assert result.domain == "enterprise_support"
    assert result.counts["crm/customers.csv"] == 10
    assert result.counts["support/tickets.csv"] == 30


def test_generic_ingestion_dry_run_works_for_enterprise_support() -> None:
    stream = StringIO()
    args = ingest_domain.build_parser().parse_args(
        [
            "--domain",
            "enterprise_support",
            "--data-dir",
            str(DATA_DIR),
            "--collection-name",
            "test_domain_collection",
            "--dry-run",
            "--limit",
            "2",
        ]
    )

    exit_code = ingest_domain.run(args, stream=stream)

    output = stream.getvalue()
    assert exit_code == 0
    assert "Dry run: no documents written to Qdrant." in output
    assert "Domain: enterprise_support" in output
    assert "Target collection: test_domain_collection" in output
    assert "Total documents: 2" in output
    assert "Sample #1" in output


def test_generic_evaluation_dry_run_works_for_enterprise_support(tmp_path: Path) -> None:
    output_path = tmp_path / "domain_eval.json"
    args = evaluate_domain.build_parser().parse_args(
        [
            "--domain",
            "enterprise_support",
            "--dry-run",
            "--limit",
            "2",
            "--json-output",
            str(output_path),
        ]
    )

    exit_code = evaluate_domain.run_cli(args)

    assert exit_code == 0
    assert output_path.is_file()
    payload = output_path.read_text(encoding="utf-8")
    assert '"domain": "enterprise_support"' in payload
    assert '"query_count": 2' in payload
