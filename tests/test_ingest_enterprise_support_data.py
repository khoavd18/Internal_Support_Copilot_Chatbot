from __future__ import annotations

import argparse
from io import StringIO
from pathlib import Path

import pytest
from scripts import ingest_enterprise_support_data as ingest_script

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "sample_enterprise_support"


def _args(**overrides) -> argparse.Namespace:
    defaults = {
        "data_dir": DATA_DIR,
        "collection_name": "test_enterprise_collection",
        "dry_run": True,
        "limit": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_dry_run_prints_counts_samples_and_skips_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("dry-run should not ingest into Qdrant")

    monkeypatch.setattr(ingest_script, "ingest_documents_into_qdrant", _fail_if_called)
    stream = StringIO()

    result = ingest_script.run(_args(), stream=stream)

    output = stream.getvalue()
    assert result == 0
    assert "Dry run: no documents written to Qdrant." in output
    assert "Target collection: test_enterprise_collection" in output
    assert "Total documents: 128" in output
    assert "  customer: 10" in output
    assert "  ticket: 30" in output
    assert "  knowledge_base: 10" in output
    assert "Sample #1" in output
    assert "metadata:" in output


def test_dry_run_limit_applies_before_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail_if_called(*args, **kwargs):
        raise AssertionError("dry-run should not ingest into Qdrant")

    monkeypatch.setattr(ingest_script, "ingest_documents_into_qdrant", _fail_if_called)
    stream = StringIO()

    result = ingest_script.run(_args(limit=2), stream=stream)

    output = stream.getvalue()
    assert result == 0
    assert "Total documents: 2" in output
    assert "  customer: 2" in output
    assert "Sample #1" in output
    assert "Sample #2" in output
    assert "Sample #3" not in output


def test_main_returns_nonzero_for_invalid_dataset(tmp_path: Path) -> None:
    result = ingest_script.main(["--data-dir", str(tmp_path), "--dry-run"])

    assert result == 1
