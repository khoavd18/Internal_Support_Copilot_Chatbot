from __future__ import annotations

from io import StringIO
from pathlib import Path

from eval import evaluate_domain
from scripts import ingest_domain, validate_domain_data
from src.domains.registry import get_domain_adapter

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_banking_fraud"


def test_registry_returns_banking_fraud_adapter() -> None:
    adapter = get_domain_adapter("banking_fraud")

    assert adapter.name == "banking_fraud"
    assert adapter.default_data_dir.name == "sample_banking_fraud"
    assert adapter.default_collection_name == "banking_fraud_copilot_qdrant"


def test_banking_fraud_validation_passes_for_synthetic_dataset() -> None:
    adapter = get_domain_adapter("banking_fraud")

    result = validate_domain_data.validate_domain_data(adapter, DATA_DIR)

    assert result.is_valid, result.errors
    assert result.counts["customers"] == 6
    assert result.counts["accounts"] == 6
    assert result.counts["transactions"] == 15
    assert result.counts["fraud_alerts"] == 7
    assert result.counts["aml_cases"] == 4
    assert result.counts["policies"] == 3


def test_banking_fraud_document_building_and_graph() -> None:
    adapter = get_domain_adapter("banking_fraud")
    dataset = adapter.load_dataset(DATA_DIR)

    documents = adapter.build_documents(dataset)
    graph = adapter.build_graph(dataset)

    assert len(documents) == 49
    assert len({document["id"] for document in documents}) == len(documents)
    assert all(document["metadata"]["source"] == "banking_fraud" for document in documents)
    assert any(document["metadata"]["source_type"] == "aml_case" for document in documents)
    assert "Customer:bf_cust_001" in graph["nodes"]
    assert "FraudAlert:bf_alert_001" in graph["nodes"]
    assert any(edge["type"] == "TRIGGERED_ALERT" for edge in graph["edges"])


def test_banking_fraud_generic_ingestion_dry_run() -> None:
    stream = StringIO()
    args = ingest_domain.build_parser().parse_args(
        [
            "--domain",
            "banking_fraud",
            "--data-dir",
            str(DATA_DIR),
            "--collection-name",
            "test_banking_fraud_collection",
            "--dry-run",
            "--limit",
            "4",
        ]
    )

    exit_code = ingest_domain.run(args, stream=stream)

    output = stream.getvalue()
    assert exit_code == 0
    assert "Domain: banking_fraud" in output
    assert "Target collection: test_banking_fraud_collection" in output
    assert "Total documents: 4" in output
    assert "  customer: 4" in output


def test_banking_fraud_generic_evaluation_dry_run(tmp_path: Path) -> None:
    output_path = tmp_path / "banking_eval.json"
    args = evaluate_domain.build_parser().parse_args(
        [
            "--domain",
            "banking_fraud",
            "--dry-run",
            "--limit",
            "3",
            "--json-output",
            str(output_path),
        ]
    )

    exit_code = evaluate_domain.run_cli(args)

    assert exit_code == 0
    assert output_path.is_file()
    payload = output_path.read_text(encoding="utf-8")
    assert '"domain": "banking_fraud"' in payload
    assert '"query_count": 3' in payload
