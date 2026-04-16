from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from scripts import evaluate_retrieval as eval_script


def test_load_benchmark_cases_supports_jsonl_labels(tmp_path):
    benchmark_path = tmp_path / "benchmark.jsonl"
    benchmark_path.write_text(
        json.dumps(
            {
                "id": "passkey-sign-in",
                "query": "How to sign in with a passkey?",
                "expected_sources": ["github_docs"],
                "expected_documents": ["signing-in-with-a-passkey"],
                "expected_categories": ["authentication"],
                "expected_answer_intent": "github_authentication",
                "notes": "sample",
                "answer_quality": {
                    "reference_answer": "Explain how to sign in with a passkey.",
                    "must_include": ["passkey", "sign in"],
                    "must_not_include": ["ssh key"],
                    "completeness_points": ["passkey", "github"],
                    "minimum_source_count": 1,
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    cases = eval_script.load_benchmark_cases(benchmark_path)

    assert len(cases) == 1
    assert cases[0].case_id == "passkey-sign-in"
    assert cases[0].expected_sources == ["github_docs"]
    assert cases[0].expected_documents == ["signing-in-with-a-passkey"]
    assert cases[0].expected_categories == ["authentication"]
    assert cases[0].expected_answer_intents == ["github_authentication"]
    assert cases[0].notes == "sample"
    assert cases[0].answer_quality.reference_answer == "Explain how to sign in with a passkey."
    assert cases[0].answer_quality.must_include == ["passkey", "sign in"]
    assert cases[0].answer_quality.must_not_include == ["ssh key"]
    assert cases[0].answer_quality.completeness_points == ["passkey", "github"]
    assert cases[0].answer_quality.minimum_source_count == 1


def test_load_benchmark_cases_supports_legacy_text(tmp_path):
    benchmark_path = tmp_path / "queries.txt"
    benchmark_path.write_text(
        "# comment\nHow to sign in with a passkey?\n\nsupport handbook\n",
        encoding="utf-8",
    )

    cases = eval_script.load_benchmark_cases(benchmark_path)

    assert len(cases) == 2
    assert cases[0].query == "How to sign in with a passkey?"
    assert cases[0].has_retrieval_labels is False
    assert cases[0].case_id.startswith("legacy-001-")


def test_run_benchmark_produces_metrics_and_outputs(tmp_path, monkeypatch):
    benchmark_path = tmp_path / "benchmark.jsonl"
    benchmark_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "passkey-sign-in",
                        "query": "How to sign in with a passkey?",
                        "expected_sources": ["github_docs"],
                        "expected_documents": ["signing-in-with-a-passkey"],
                        "expected_categories": ["authentication"],
                        "expected_answer_intent": "github_authentication",
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "engineering-policy",
                        "query": "engineering handbook policy",
                        "expected_sources": ["gitlab_handbook"],
                        "expected_documents": ["engineering handbook"],
                        "expected_categories": ["policy"],
                        "expected_answer_intent": "gitlab_handbook",
                    },
                    ensure_ascii=False,
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    runs_dir = tmp_path / "runs"
    reports_dir = tmp_path / "reports"
    runs_dir.mkdir()
    reports_dir.mkdir()
    monkeypatch.setattr(eval_script, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(eval_script, "REPORTS_DIR", reports_dir)

    def _fake_retrieval_fn(*, query: str, top_k: int, rebuild: bool):
        assert top_k == 3
        assert rebuild is False
        if "passkey" in query.lower():
            return [
                Document(
                    page_content="Passkey sign-in guide",
                    metadata={
                        "source": "github_docs",
                        "source_type": "document",
                        "title": "Signing in with a passkey",
                        "doc_id": "github_docs::authentication/signing-in-with-a-passkey.md",
                        "path": "authentication/signing-in-with-a-passkey.md",
                        "url": "https://docs.github.com/en/authentication/signing-in-with-a-passkey",
                    },
                ),
                Document(
                    page_content="Support handbook overview",
                    metadata={
                        "source": "gitlab_handbook",
                        "source_type": "document",
                        "title": "Support Handbook",
                        "doc_id": "gitlab_handbook::support-handbook",
                        "path": "support/handbook/index.html",
                    },
                ),
            ]

        return [
            Document(
                page_content="Authentication help",
                metadata={
                    "source": "github_docs",
                    "source_type": "document",
                    "title": "Authentication help",
                    "doc_id": "github_docs::authentication/auth-help.md",
                    "path": "authentication/auth-help.md",
                },
            ),
            Document(
                page_content="Engineering handbook policy",
                metadata={
                    "source": "gitlab_handbook",
                    "source_type": "document",
                    "title": "Engineering Handbook Policy",
                    "doc_id": "gitlab_handbook::engineering-handbook-policy",
                    "path": "engineering/handbook/policy.html",
                },
            ),
        ]

    result = eval_script.run_benchmark(
        benchmark_path=benchmark_path,
        top_k=3,
        rebuild=False,
        retrieval_fn=_fake_retrieval_fn,
    )

    summary = result["summary"]
    assert summary["query_count"] == 2
    assert summary["metrics"]["overall"]["queries_evaluated"] == 2
    assert summary["metrics"]["overall"]["hit_at_k"] == 1.0
    assert summary["metrics"]["overall"]["mrr"] == 0.75
    assert summary["metrics"]["source"]["hit_at_k"] == 1.0
    assert summary["metrics"]["intent"]["accuracy"] == 1.0
    assert result["paths"]["run_jsonl"].endswith("_retrieval_benchmark.jsonl")

    run_path = Path(result["paths"]["run_jsonl"])
    summary_json_path = Path(result["paths"]["summary_json"])
    report_path = Path(result["paths"]["report_txt"])

    assert run_path.exists()
    assert summary_json_path.exists()
    assert report_path.exists()
    assert "Overall hit@3" in report_path.read_text(encoding="utf-8")

    run_rows = [
        json.loads(line)
        for line in run_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(run_rows) == 2
    assert run_rows[0]["results"][0]["match"]["any"] is True
    assert run_rows[1]["results"][1]["match"]["document"] is True
