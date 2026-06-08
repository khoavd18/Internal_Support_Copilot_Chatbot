from __future__ import annotations

import json

from eval import evaluate_enterprise_support as enterprise_eval


def _summary(*, recall_at_5: float, source_type_hit_rate: float) -> dict:
    return {
        "mode": "dry_run_local",
        "query_count": 20,
        "vector_error_count": 0,
        "metrics": {
            "recall_at_5": recall_at_5,
            "source_type_hit_rate": source_type_hit_rate,
            "source_type_all_hit_rate": 0.25,
            "groundedness_rate": 1.0,
            "missing_info_handling_rate": 1.0,
        },
    }


def test_evaluate_quality_gate_passes_when_metrics_meet_thresholds() -> None:
    gate = enterprise_eval.evaluate_quality_gate(
        _summary(recall_at_5=0.6788, source_type_hit_rate=1.0),
        min_recall_at_5=0.65,
        min_source_hit_rate=0.90,
    )

    assert gate["enabled"] is True
    assert gate["passed"] is True
    assert all(check["passed"] for check in gate["checks"])


def test_evaluate_quality_gate_fails_when_recall_is_below_threshold() -> None:
    gate = enterprise_eval.evaluate_quality_gate(
        _summary(recall_at_5=0.60, source_type_hit_rate=1.0),
        min_recall_at_5=0.65,
        min_source_hit_rate=0.90,
    )

    assert gate["enabled"] is True
    assert gate["passed"] is False
    recall_check = next(check for check in gate["checks"] if check["metric"] == "recall_at_5")
    assert recall_check == {
        "metric": "recall_at_5",
        "actual": 0.60,
        "minimum": 0.65,
        "passed": False,
    }


def test_run_cli_returns_nonzero_for_failed_gate_and_writes_json(tmp_path, monkeypatch) -> None:
    output_path = tmp_path / "enterprise_eval.json"

    monkeypatch.setattr(
        enterprise_eval,
        "run_evaluation",
        lambda **kwargs: {
            "records": [],
            "summary": _summary(recall_at_5=0.60, source_type_hit_rate=1.0),
            "paths": {},
        },
    )

    args = enterprise_eval.parse_args(
        [
            "--dry-run",
            "--fail-under",
            "--json-output",
            str(output_path),
        ]
    )
    exit_code = enterprise_eval.run_cli(args)

    assert exit_code == 1
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["quality_gate"]["passed"] is False
    assert payload["quality_gate"]["checks"][0]["minimum"] == 0.65
    assert payload["quality_gate"]["checks"][1]["minimum"] == 0.90


def test_run_cli_returns_zero_for_passing_custom_thresholds(monkeypatch) -> None:
    monkeypatch.setattr(
        enterprise_eval,
        "run_evaluation",
        lambda **kwargs: {
            "records": [],
            "summary": _summary(recall_at_5=0.70, source_type_hit_rate=0.95),
            "paths": {},
        },
    )

    args = enterprise_eval.parse_args(
        [
            "--dry-run",
            "--min-recall-at-5",
            "0.70",
            "--min-source-hit-rate",
            "0.95",
            "--json-output",
            "-",
        ]
    )

    assert enterprise_eval.run_cli(args) == 0
