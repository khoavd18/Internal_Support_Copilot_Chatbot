from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import evaluate_retrieval as retrieval_eval
from src.pipeline import build_pipeline


RUNS_DIR = PROJECT_ROOT / "eval" / "runs"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

TOKEN_PATTERN = re.compile(r"[A-Za-zÀ-ỹ0-9_+-]+", re.UNICODE)
QUALITY_STOPWORDS = {
    "the",
    "and",
    "with",
    "that",
    "this",
    "from",
    "your",
    "into",
    "using",
    "used",
    "have",
    "there",
    "need",
    "then",
    "them",
    "how",
    "what",
    "when",
    "where",
    "which",
    "for",
    "you",
    "are",
    "can",
    "was",
    "were",
    "does",
    "một",
    "những",
    "được",
    "làm",
    "thế",
    "nào",
    "để",
    "cách",
    "với",
    "trong",
    "khi",
    "cho",
    "của",
    "các",
    "là",
}


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = _normalize_text(text)
    normalized_phrase = _normalize_text(phrase)
    if not normalized_phrase:
        return False
    return normalized_phrase in normalized_text


def _matched_phrases(text: str, phrases: Sequence[str]) -> list[str]:
    return [phrase for phrase in phrases if _contains_phrase(text, phrase)]


def _coverage_score(text: str, phrases: Sequence[str]) -> tuple[float | None, list[str]]:
    if not phrases:
        return None, []
    matched = _matched_phrases(text, phrases)
    return len(matched) / len(phrases), matched


def _reference_keyword_recall(answer: str, reference_answer: str) -> tuple[float | None, list[str]]:
    reference_tokens = {
        token
        for token in TOKEN_PATTERN.findall(_normalize_text(reference_answer))
        if len(token) >= 4 and token not in QUALITY_STOPWORDS
    }
    if not reference_tokens:
        return None, []

    answer_tokens = set(TOKEN_PATTERN.findall(_normalize_text(answer)))
    matched = sorted(token for token in reference_tokens if token in answer_tokens)
    return len(matched) / len(reference_tokens), matched


def _dimension_result(
    *,
    evaluated: bool,
    score: float | None,
    passed: bool | None,
    details: dict[str, Any],
) -> dict[str, Any]:
    return {
        "evaluated": evaluated,
        "score": (round(score, 4) if score is not None else None),
        "passed": passed if evaluated else None,
        "details": details,
    }


def _evaluate_correctness(case: retrieval_eval.BenchmarkCase, answer_text: str) -> dict[str, Any]:
    rubric = case.answer_quality
    must_include_score, matched_must_include = _coverage_score(answer_text, rubric.must_include)
    forbidden_hits = _matched_phrases(answer_text, rubric.must_not_include)
    reference_score, reference_hits = _reference_keyword_recall(answer_text, rubric.reference_answer)

    evaluated = bool(
        rubric.must_include
        or rubric.must_not_include
        or rubric.reference_answer
    )
    if not evaluated:
        return _dimension_result(evaluated=False, score=None, passed=None, details={})

    score_parts = [item for item in (must_include_score, reference_score) if item is not None]
    base_score = sum(score_parts) / len(score_parts) if score_parts else 1.0
    score = 0.0 if forbidden_hits else base_score
    passed = (
        not forbidden_hits
        and (must_include_score is None or must_include_score == 1.0)
        and (reference_score is None or reference_score >= 0.5)
    )

    return _dimension_result(
        evaluated=True,
        score=score,
        passed=passed,
        details={
            "must_include_total": len(rubric.must_include),
            "must_include_matched": matched_must_include,
            "must_not_include_hits": forbidden_hits,
            "reference_keyword_hits": reference_hits,
        },
    )


def _evaluate_completeness(case: retrieval_eval.BenchmarkCase, answer_text: str) -> dict[str, Any]:
    rubric = case.answer_quality
    coverage_score, matched_points = _coverage_score(answer_text, rubric.completeness_points)
    if coverage_score is None:
        return _dimension_result(evaluated=False, score=None, passed=None, details={})

    return _dimension_result(
        evaluated=True,
        score=coverage_score,
        passed=coverage_score == 1.0,
        details={
            "points_total": len(rubric.completeness_points),
            "points_matched": matched_points,
        },
    )


def _citation_ranks(rows: Sequence[dict[str, Any]]) -> dict[str, int | None]:
    total_rows = len(rows)
    return {
        "source": retrieval_eval._first_rank(rows, "source", top_k=total_rows),
        "document": retrieval_eval._first_rank(rows, "document", top_k=total_rows),
        "category": retrieval_eval._first_rank(rows, "category", top_k=total_rows),
    }


def _evaluate_citation_relevance(
    case: retrieval_eval.BenchmarkCase,
    citation_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    ranks = _citation_ranks(citation_rows)
    components: list[float] = []
    if case.expected_sources:
        components.append((1.0 / ranks["source"]) if ranks["source"] else 0.0)
    if case.expected_documents:
        components.append((1.0 / ranks["document"]) if ranks["document"] else 0.0)
    if case.expected_categories:
        components.append((1.0 / ranks["category"]) if ranks["category"] else 0.0)

    if not components:
        return _dimension_result(evaluated=False, score=None, passed=None, details={})

    score = sum(components) / len(components)
    return _dimension_result(
        evaluated=True,
        score=score,
        passed=score >= 0.5,
        details={
            "first_relevant_source_rank": ranks["source"],
            "first_relevant_document_rank": ranks["document"],
            "first_relevant_category_rank": ranks["category"],
        },
    )


def _evaluate_groundedness(
    case: retrieval_eval.BenchmarkCase,
    answer_result: dict[str, Any],
    citation_relevance: dict[str, Any],
) -> dict[str, Any]:
    stats = dict(answer_result.get("stats", {}))
    source_count = len(answer_result.get("sources", []) or [])
    required_source_count = max(1, int(case.answer_quality.minimum_source_count or 0))
    stage = str(stats.get("stage") or "").strip()
    used_fallback = bool(stats.get("used_fallback", False))
    stage_ok = stage == "ok" and not used_fallback
    source_count_ok = source_count >= required_source_count

    support_score = citation_relevance.get("score")
    if support_score is None:
        support_score = 1.0 if source_count_ok else 0.0

    score = (
        (1.0 if stage_ok else 0.0)
        + (1.0 if source_count_ok else 0.0)
        + float(support_score)
    ) / 3.0
    passed = stage_ok and source_count_ok and float(support_score) >= 0.5

    return _dimension_result(
        evaluated=True,
        score=score,
        passed=passed,
        details={
            "stage": stage,
            "used_fallback": used_fallback,
            "source_count": source_count,
            "required_source_count": required_source_count,
        },
    )


def _summarize_answer_quality(dimensions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    evaluated = [item for item in dimensions.values() if item["evaluated"]]
    if not evaluated:
        return {
            "evaluated": False,
            "score": None,
            "passed": None,
        }

    score = sum(float(item["score"]) for item in evaluated if item["score"] is not None) / len(evaluated)
    passed = all(bool(item["passed"]) for item in evaluated if item["passed"] is not None)
    return {
        "evaluated": True,
        "score": round(score, 4),
        "passed": passed,
    }


def evaluate_answer_case(
    case: retrieval_eval.BenchmarkCase,
    *,
    top_k: int,
    rebuild: bool = False,
    retrieval_fn: Callable[..., list] = retrieval_eval.retrieve_documents,
    answer_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    documents = retrieval_fn(query=case.query, top_k=top_k, rebuild=rebuild)
    retrieval_record = retrieval_eval.evaluate_case(
        case,
        top_k=top_k,
        rebuild=False,
        retrieval_fn=lambda **kwargs: documents,
    )

    if answer_fn is None:
        pipeline = build_pipeline(top_k=top_k, rebuild=rebuild)

        def _default_answer_fn(*, question: str, documents: list) -> dict[str, Any]:
            return pipeline.answer_from_documents(question=question, documents=documents, debug=False)

        resolved_answer_fn = _default_answer_fn
    else:
        resolved_answer_fn = answer_fn

    answer_result = resolved_answer_fn(question=case.query, documents=documents)
    answer_text = str(answer_result.get("answer") or "").strip()

    citation_rows = [
        retrieval_eval.annotate_result_row(
            retrieval_eval.doc_to_row(rank=index, doc=doc),
            case,
        )
        for index, doc in enumerate(documents, start=1)
    ]

    correctness = _evaluate_correctness(case, answer_text)
    completeness = _evaluate_completeness(case, answer_text)
    citation_relevance = _evaluate_citation_relevance(case, citation_rows)
    groundedness = _evaluate_groundedness(case, answer_result, citation_relevance)

    dimensions = {
        "correctness": correctness,
        "groundedness": groundedness,
        "citation_relevance": citation_relevance,
        "completeness": completeness,
    }
    overall = _summarize_answer_quality(dimensions)

    return {
        "case_id": case.case_id,
        "query": case.query,
        "labels": {
            "expected_sources": case.expected_sources,
            "expected_documents": case.expected_documents,
            "expected_categories": case.expected_categories,
            "expected_answer_intents": case.expected_answer_intents,
            "notes": case.notes,
            "answer_quality": {
                "reference_answer": case.answer_quality.reference_answer,
                "must_include": case.answer_quality.must_include,
                "must_not_include": case.answer_quality.must_not_include,
                "completeness_points": case.answer_quality.completeness_points,
                "minimum_source_count": case.answer_quality.minimum_source_count,
                "notes": case.answer_quality.notes,
            },
        },
        "retrieval": retrieval_record,
        "answer": {
            "text": answer_text,
            "sources": answer_result.get("sources", []),
            "stats": answer_result.get("stats", {}),
            "metrics": {
                "overall": overall,
                **dimensions,
            },
        },
    }


def _aggregate_dimension(records: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    evaluated = [
        record["answer"]["metrics"][key]
        for record in records
        if record["answer"]["metrics"][key]["evaluated"]
    ]
    if not evaluated:
        return {
            "cases_evaluated": 0,
            "passes": 0,
            "pass_rate": None,
            "average_score": None,
        }

    passes = sum(1 for item in evaluated if item["passed"])
    average_score = sum(float(item["score"]) for item in evaluated if item["score"] is not None) / len(evaluated)
    return {
        "cases_evaluated": len(evaluated),
        "passes": passes,
        "pass_rate": round(passes / len(evaluated), 4),
        "average_score": round(average_score, 4),
    }


def build_summary(
    records: Sequence[dict[str, Any]],
    *,
    benchmark_path: Path,
    top_k: int,
    rebuild: bool,
    timestamp: str,
    run_path: Path,
    summary_json_path: Path,
    report_path: Path,
    latest_report_path: Path,
) -> dict[str, Any]:
    stage_counts: Counter[str] = Counter()
    total_sources = 0
    for record in records:
        stage = str(record["answer"]["stats"].get("stage") or "").strip() or "unknown"
        stage_counts[stage] += 1
        total_sources += len(record["answer"].get("sources", []) or [])

    failures = []
    for record in records:
        overall = record["answer"]["metrics"]["overall"]
        if not overall["evaluated"] or overall["passed"]:
            continue
        failed_dimensions = [
            name
            for name, payload in record["answer"]["metrics"].items()
            if name != "overall" and payload["evaluated"] and not payload["passed"]
        ]
        failures.append(
            {
                "case_id": record["case_id"],
                "query": record["query"],
                "overall_score": overall["score"],
                "failed_dimensions": failed_dimensions,
            }
        )

    answer_evaluated = [
        record["answer"]["metrics"]["overall"]
        for record in records
        if record["answer"]["metrics"]["overall"]["evaluated"]
    ]
    overall_passes = sum(1 for item in answer_evaluated if item["passed"])
    overall_score = (
        round(
            sum(float(item["score"]) for item in answer_evaluated if item["score"] is not None)
            / len(answer_evaluated),
            4,
        )
        if answer_evaluated
        else None
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "timestamp": timestamp,
        "benchmark_path": str(benchmark_path),
        "paths": {
            "run_jsonl": str(run_path),
            "summary_json": str(summary_json_path),
            "report_txt": str(report_path),
            "latest_report_txt": str(latest_report_path),
        },
        "config": {
            "top_k": top_k,
            "rebuild": rebuild,
        },
        "query_count": len(records),
        "answer_quality_coverage": {
            "cases_with_any_answer_eval": len(answer_evaluated),
            "correctness_cases": _aggregate_dimension(records, "correctness")["cases_evaluated"],
            "groundedness_cases": _aggregate_dimension(records, "groundedness")["cases_evaluated"],
            "citation_relevance_cases": _aggregate_dimension(records, "citation_relevance")["cases_evaluated"],
            "completeness_cases": _aggregate_dimension(records, "completeness")["cases_evaluated"],
        },
        "metrics": {
            "overall": {
                "cases_evaluated": len(answer_evaluated),
                "passes": overall_passes,
                "pass_rate": round(overall_passes / len(answer_evaluated), 4) if answer_evaluated else None,
                "average_score": overall_score,
            },
            "correctness": _aggregate_dimension(records, "correctness"),
            "groundedness": _aggregate_dimension(records, "groundedness"),
            "citation_relevance": _aggregate_dimension(records, "citation_relevance"),
            "completeness": _aggregate_dimension(records, "completeness"),
        },
        "answer_stats": {
            "average_sources_per_answer": round(total_sources / len(records), 4) if records else 0.0,
            "stage_counts": dict(sorted(stage_counts.items())),
        },
        "failures": failures[:10],
    }


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_run_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def build_human_report(summary: dict[str, Any]) -> str:
    overall = summary["metrics"]["overall"]
    correctness = summary["metrics"]["correctness"]
    groundedness = summary["metrics"]["groundedness"]
    citation_relevance = summary["metrics"]["citation_relevance"]
    completeness = summary["metrics"]["completeness"]

    lines = [
        "Answer Quality Benchmark Summary",
        "=" * 80,
        f"Generated at            : {summary['generated_at']}",
        f"Benchmark file          : {summary['benchmark_path']}",
        f"Top-k                   : {summary['config']['top_k']}",
        f"Rebuild vector store    : {summary['config']['rebuild']}",
        f"Run jsonl               : {summary['paths']['run_jsonl']}",
        f"Summary json            : {summary['paths']['summary_json']}",
        "",
        "Overall",
        "-" * 80,
        f"Cases evaluated         : {overall['cases_evaluated']}",
        f"Overall pass rate       : {_format_metric(overall['pass_rate'])} ({overall['passes']}/{overall['cases_evaluated']})",
        f"Overall average score   : {_format_metric(overall['average_score'])}",
        "",
        "Dimensions",
        "-" * 80,
        f"Correctness pass rate   : {_format_metric(correctness['pass_rate'])} ({correctness['passes']}/{correctness['cases_evaluated']})",
        f"Correctness avg score   : {_format_metric(correctness['average_score'])}",
        f"Groundedness pass rate  : {_format_metric(groundedness['pass_rate'])} ({groundedness['passes']}/{groundedness['cases_evaluated']})",
        f"Groundedness avg score  : {_format_metric(groundedness['average_score'])}",
        f"Citation relevance rate : {_format_metric(citation_relevance['pass_rate'])} ({citation_relevance['passes']}/{citation_relevance['cases_evaluated']})",
        f"Citation avg score      : {_format_metric(citation_relevance['average_score'])}",
        f"Completeness pass rate  : {_format_metric(completeness['pass_rate'])} ({completeness['passes']}/{completeness['cases_evaluated']})",
        f"Completeness avg score  : {_format_metric(completeness['average_score'])}",
        "",
        "Answer Stats",
        "-" * 80,
        f"Average sources / answer: {_format_metric(summary['answer_stats']['average_sources_per_answer'])}",
        f"Stage counts            : {json.dumps(summary['answer_stats']['stage_counts'], ensure_ascii=False)}",
    ]

    if summary["failures"]:
        lines.extend(["", "Top Failures", "-" * 80])
        for failure in summary["failures"]:
            lines.append(
                f"- {failure['case_id']}: score={failure['overall_score']} | failed={', '.join(failure['failed_dimensions'])} | {failure['query']}"
            )

    return "\n".join(lines) + "\n"


def run_answer_quality_benchmark(
    *,
    benchmark_path: Path,
    top_k: int,
    rebuild: bool = False,
    retrieval_fn: Callable[..., list] = retrieval_eval.retrieve_documents,
    answer_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cases = retrieval_eval.load_benchmark_cases(benchmark_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = RUNS_DIR / f"{timestamp}_answer_quality_benchmark.jsonl"
    summary_json_path = RUNS_DIR / f"{timestamp}_answer_quality_summary.json"
    report_path = REPORTS_DIR / f"{timestamp}_answer_quality_summary.txt"
    latest_report_path = REPORTS_DIR / "latest_answer_quality_summary.txt"

    shared_answer_fn = answer_fn
    if shared_answer_fn is None:
        pipeline = build_pipeline(top_k=top_k, rebuild=rebuild)

        def _shared_answer_fn(*, question: str, documents: list) -> dict[str, Any]:
            return pipeline.answer_from_documents(question=question, documents=documents, debug=False)

        shared_answer_fn = _shared_answer_fn

    records = [
        evaluate_answer_case(
            case,
            top_k=top_k,
            rebuild=rebuild,
            retrieval_fn=retrieval_fn,
            answer_fn=shared_answer_fn,
        )
        for case in cases
    ]

    summary = build_summary(
        records,
        benchmark_path=benchmark_path,
        top_k=top_k,
        rebuild=rebuild,
        timestamp=timestamp,
        run_path=run_path,
        summary_json_path=summary_json_path,
        report_path=report_path,
        latest_report_path=latest_report_path,
    )
    report_text = build_human_report(summary)

    _write_run_jsonl(run_path, records)
    _write_json(summary_json_path, summary)
    report_path.write_text(report_text, encoding="utf-8")
    latest_report_path.write_text(report_text, encoding="utf-8")

    return {
        "records": records,
        "summary": summary,
        "paths": summary["paths"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run answer-quality evaluation on top of the retrieval benchmark."
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=retrieval_eval.resolve_default_queries_path(),
        help="Benchmark file (.jsonl/.json or legacy .txt).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved results to provide as answer context.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the vector store before running the benchmark.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_answer_quality_benchmark(
        benchmark_path=args.queries_path,
        top_k=args.top_k,
        rebuild=args.rebuild,
    )
    summary = result["summary"]
    print("=" * 80)
    print("ANSWER QUALITY BENCHMARK")
    print(f"Benchmark file        : {summary['benchmark_path']}")
    print(f"Queries               : {summary['query_count']}")
    print(f"Cases evaluated       : {summary['metrics']['overall']['cases_evaluated']}")
    print(
        f"Overall pass rate     : {_format_metric(summary['metrics']['overall']['pass_rate'])}"
    )
    print(
        f"Overall average score : {_format_metric(summary['metrics']['overall']['average_score'])}"
    )
    print(f"[DONE] Saved run      -> {summary['paths']['run_jsonl']}")
    print(f"[DONE] Saved summary  -> {summary['paths']['summary_json']}")
    print(f"[DONE] Saved report   -> {summary['paths']['report_txt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
