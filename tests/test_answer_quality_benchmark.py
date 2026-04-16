from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from scripts import evaluate_answer_quality as answer_eval


def test_run_answer_quality_benchmark_produces_scores_and_reports(tmp_path, monkeypatch):
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
                        "answer_quality": {
                            "reference_answer": "Explain how to sign in with a passkey on GitHub.",
                            "must_include": ["passkey", "sign in"],
                            "must_not_include": ["ssh key"],
                            "completeness_points": ["passkey", "github"],
                            "minimum_source_count": 1,
                        },
                    },
                    ensure_ascii=False,
                ),
                json.dumps(
                    {
                        "id": "engineering-handbook-policy",
                        "query": "engineering handbook policy",
                        "expected_sources": ["gitlab_handbook"],
                        "expected_documents": ["engineering handbook"],
                        "expected_categories": ["policy"],
                        "answer_quality": {
                            "reference_answer": "Point the user to the engineering handbook policy page.",
                            "must_include": ["handbook", "policy"],
                            "must_not_include": ["passkey"],
                            "completeness_points": ["handbook", "policy"],
                            "minimum_source_count": 1,
                        },
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
    monkeypatch.setattr(answer_eval, "RUNS_DIR", runs_dir)
    monkeypatch.setattr(answer_eval, "REPORTS_DIR", reports_dir)

    def _fake_retrieval_fn(*, query: str, top_k: int, rebuild: bool):
        assert top_k == 3
        assert rebuild is False
        if "passkey" in query.lower():
            return [
                Document(
                    page_content="Use a passkey to sign in to GitHub.",
                    metadata={
                        "source": "github_docs",
                        "source_type": "document",
                        "title": "Signing in with a passkey",
                        "doc_id": "github_docs::authentication/signing-in-with-a-passkey.md",
                        "path": "authentication/signing-in-with-a-passkey.md",
                    },
                )
            ]

        return [
            Document(
                page_content="Authentication help",
                metadata={
                    "source": "github_docs",
                    "source_type": "document",
                    "title": "Authentication help",
                    "doc_id": "github_docs::authentication/help.md",
                    "path": "authentication/help.md",
                },
            )
        ]

    def _fake_answer_fn(*, question: str, documents: list) -> dict:
        if "passkey" in question.lower():
            return {
                "answer": "Use a passkey to sign in to GitHub. See the GitHub passkey guidance.",
                "sources": [
                    {
                        "index": 1,
                        "title": "Signing in with a passkey",
                        "source": "github_docs",
                        "doc_id": "github_docs::authentication/signing-in-with-a-passkey.md",
                    }
                ],
                "stats": {
                    "stage": "ok",
                    "used_fallback": False,
                },
            }

        return {
            "answer": "Use a passkey for this workflow.",
            "sources": [
                {
                    "index": 1,
                    "title": "Authentication help",
                    "source": "github_docs",
                    "doc_id": "github_docs::authentication/help.md",
                }
            ],
            "stats": {
                "stage": "ok",
                "used_fallback": False,
            },
        }

    result = answer_eval.run_answer_quality_benchmark(
        benchmark_path=benchmark_path,
        top_k=3,
        rebuild=False,
        retrieval_fn=_fake_retrieval_fn,
        answer_fn=_fake_answer_fn,
    )

    summary = result["summary"]
    assert summary["query_count"] == 2
    assert summary["metrics"]["overall"]["cases_evaluated"] == 2
    assert summary["metrics"]["overall"]["pass_rate"] == 0.5
    assert summary["metrics"]["correctness"]["pass_rate"] == 0.5
    assert summary["metrics"]["groundedness"]["cases_evaluated"] == 2
    assert summary["metrics"]["citation_relevance"]["pass_rate"] == 0.5
    assert summary["metrics"]["completeness"]["pass_rate"] == 0.5
    assert result["paths"]["run_jsonl"].endswith("_answer_quality_benchmark.jsonl")

    run_path = Path(result["paths"]["run_jsonl"])
    summary_json_path = Path(result["paths"]["summary_json"])
    report_path = Path(result["paths"]["report_txt"])
    latest_report_path = Path(result["paths"]["latest_report_txt"])

    assert run_path.exists()
    assert summary_json_path.exists()
    assert report_path.exists()
    assert latest_report_path.exists()
    assert "Answer Quality Benchmark Summary" in report_path.read_text(encoding="utf-8")

    run_rows = [
        json.loads(line)
        for line in run_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(run_rows) == 2
    assert run_rows[0]["answer"]["metrics"]["overall"]["passed"] is True
    assert run_rows[1]["answer"]["metrics"]["overall"]["passed"] is False
    assert run_rows[1]["answer"]["metrics"]["correctness"]["details"]["must_not_include_hits"] == ["passkey"]
