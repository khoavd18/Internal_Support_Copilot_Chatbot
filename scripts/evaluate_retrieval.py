from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.retrieval.query_analyzer import extract_keyword_groups
from src.rag.retrieval.retriever import retrieve_documents


DEFAULT_BENCHMARK_PATH = PROJECT_ROOT / "eval" / "queries" / "retrieval_benchmark.jsonl"
LEGACY_QUERIES_PATH = PROJECT_ROOT / "eval" / "queries" / "evaluation_queries.txt"
RUNS_DIR = PROJECT_ROOT / "eval" / "runs"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DOCUMENT_WEIGHT = 3
CATEGORY_WEIGHT = 2
SOURCE_WEIGHT = 1


@dataclass
class AnswerQualityRubric:
    reference_answer: str = ""
    must_include: list[str] = field(default_factory=list)
    must_not_include: list[str] = field(default_factory=list)
    completeness_points: list[str] = field(default_factory=list)
    minimum_source_count: int = 0
    notes: str = ""

    @property
    def enabled(self) -> bool:
        return bool(
            self.reference_answer
            or self.must_include
            or self.must_not_include
            or self.completeness_points
            or self.minimum_source_count > 0
        )


@dataclass
class BenchmarkCase:
    case_id: str
    query: str
    expected_sources: list[str] = field(default_factory=list)
    expected_documents: list[str] = field(default_factory=list)
    expected_categories: list[str] = field(default_factory=list)
    expected_answer_intents: list[str] = field(default_factory=list)
    notes: str = ""
    answer_quality: AnswerQualityRubric = field(default_factory=AnswerQualityRubric)

    @property
    def has_retrieval_labels(self) -> bool:
        return bool(
            self.expected_sources
            or self.expected_documents
            or self.expected_categories
        )


def _normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return re.sub(r"\s+", " ", text)


def _slugify(value: str, *, fallback: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", _normalize_text(value)).strip("-")
    return normalized[:64] or fallback


def _ensure_string_list(value: Any) -> list[str]:
    if value in (None, "", []):
        return []

    if isinstance(value, (list, tuple, set)):
        raw_items = list(value)
    else:
        raw_items = [value]

    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        cleaned = str(item or "").strip()
        if not cleaned:
            continue
        marker = _normalize_text(cleaned)
        if marker in seen:
            continue
        seen.add(marker)
        normalized.append(cleaned)
    return normalized


def _load_json_cases(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(json.loads(line))
        return rows

    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        items = payload.get("cases", [])
        if isinstance(items, list):
            return [item for item in items if isinstance(item, dict)]
        raise ValueError(f"Unsupported benchmark JSON structure in {path}.")
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    raise ValueError(f"Unsupported benchmark JSON structure in {path}.")


def _legacy_text_case(index: int, query: str) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=f"legacy-{index:03d}-{_slugify(query, fallback='query')}",
        query=query,
    )


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_benchmark_cases(path: Path) -> list[BenchmarkCase]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find benchmark file: {path}")

    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        cases: list[BenchmarkCase] = []
        for index, row in enumerate(_load_json_cases(path), start=1):
            query = str(row.get("query") or "").strip()
            if not query:
                raise ValueError(f"Benchmark row #{index} is missing `query`.")

            case_id = str(row.get("id") or row.get("case_id") or "").strip()
            answer_quality = dict(row.get("answer_quality") or {})
            cases.append(
                BenchmarkCase(
                    case_id=case_id or f"case-{index:03d}-{_slugify(query, fallback='query')}",
                    query=query,
                    expected_sources=_ensure_string_list(
                        row.get("expected_sources") or row.get("expected_source")
                    ),
                    expected_documents=_ensure_string_list(
                        row.get("expected_documents") or row.get("expected_document")
                    ),
                    expected_categories=_ensure_string_list(
                        row.get("expected_categories") or row.get("expected_category")
                    ),
                    expected_answer_intents=_ensure_string_list(
                        row.get("expected_answer_intents")
                        or row.get("expected_answer_intent")
                    ),
                    notes=str(row.get("notes") or "").strip(),
                    answer_quality=AnswerQualityRubric(
                        reference_answer=str(
                            answer_quality.get("reference_answer")
                            or row.get("reference_answer")
                            or ""
                        ).strip(),
                        must_include=_ensure_string_list(
                            answer_quality.get("must_include")
                            or row.get("answer_must_include")
                        ),
                        must_not_include=_ensure_string_list(
                            answer_quality.get("must_not_include")
                            or row.get("answer_must_not_include")
                        ),
                        completeness_points=_ensure_string_list(
                            answer_quality.get("completeness_points")
                            or row.get("answer_completeness_points")
                        ),
                        minimum_source_count=max(
                            0,
                            _coerce_int(
                                answer_quality.get("minimum_source_count")
                                if "minimum_source_count" in answer_quality
                                else row.get("answer_minimum_source_count"),
                                default=0,
                            ),
                        ),
                        notes=str(answer_quality.get("notes") or "").strip(),
                    ),
                )
            )
        return cases

    queries: list[BenchmarkCase] = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        cleaned = line.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        queries.append(_legacy_text_case(len(queries) + 1, cleaned))
    return queries


def resolve_default_queries_path() -> Path:
    if DEFAULT_BENCHMARK_PATH.exists():
        return DEFAULT_BENCHMARK_PATH
    return LEGACY_QUERIES_PATH


def doc_to_row(rank: int, doc) -> dict[str, Any]:
    metadata = doc.metadata or {}
    return {
        "rank": rank,
        "node_type": metadata.get("node_type"),
        "merge_strategy": metadata.get("merge_strategy"),
        "source": metadata.get("source"),
        "source_type": metadata.get("source_type"),
        "title": metadata.get("title"),
        "doc_id": metadata.get("doc_id"),
        "origin_doc_id": metadata.get("origin_doc_id"),
        "parent_id": metadata.get("parent_id"),
        "path": metadata.get("path"),
        "url": metadata.get("url"),
        "category": metadata.get("category"),
        "rerank_score": metadata.get("rerank_score"),
        "preview": doc.page_content[:220],
    }


def _match_source(expected_sources: Sequence[str], row: dict[str, Any]) -> bool:
    if not expected_sources:
        return False

    candidates = {
        _normalize_text(row.get("source")),
        _normalize_text(row.get("source_type")),
    }
    candidates.discard("")
    return any(_normalize_text(item) in candidates for item in expected_sources)


def _match_substring_labels(expected_values: Sequence[str], candidates: Iterable[Any]) -> bool:
    normalized_candidates = [_normalize_text(item) for item in candidates if _normalize_text(item)]
    if not expected_values or not normalized_candidates:
        return False

    for expected in expected_values:
        needle = _normalize_text(expected)
        if not needle:
            continue
        if any(needle in candidate or candidate in needle for candidate in normalized_candidates):
            return True
    return False


def _match_document(expected_documents: Sequence[str], row: dict[str, Any]) -> bool:
    return _match_substring_labels(
        expected_documents,
        [
            row.get("doc_id"),
            row.get("origin_doc_id"),
            row.get("title"),
            row.get("path"),
            row.get("url"),
        ],
    )


def _match_category(expected_categories: Sequence[str], row: dict[str, Any]) -> bool:
    return _match_substring_labels(
        expected_categories,
        [
            row.get("category"),
            row.get("source_type"),
            row.get("title"),
            row.get("path"),
        ],
    )


def annotate_result_row(row: dict[str, Any], case: BenchmarkCase) -> dict[str, Any]:
    source_match = _match_source(case.expected_sources, row)
    document_match = _match_document(case.expected_documents, row)
    category_match = _match_category(case.expected_categories, row)
    relevance = (
        (DOCUMENT_WEIGHT if document_match else 0)
        + (CATEGORY_WEIGHT if category_match else 0)
        + (SOURCE_WEIGHT if source_match else 0)
    )

    annotated = dict(row)
    annotated["match"] = {
        "source": source_match,
        "document": document_match,
        "category": category_match,
        "any": relevance > 0,
    }
    annotated["relevance"] = relevance
    return annotated


def _metric_from_first_match_rank(rank: int | None, *, evaluated: bool) -> dict[str, Any]:
    return {
        "evaluated": evaluated,
        "hit": bool(rank),
        "first_relevant_rank": rank,
        "reciprocal_rank": (1.0 / rank) if rank else 0.0,
    }


def _dcg(gains: Sequence[int]) -> float:
    total = 0.0
    for index, gain in enumerate(gains, start=1):
        if gain <= 0:
            continue
        total += (2**gain - 1) / math.log2(index + 1)
    return total


def _compute_ndcg(rows: Sequence[dict[str, Any]], *, top_k: int, evaluated: bool) -> float | None:
    if not evaluated:
        return None

    gains = [int(row.get("relevance") or 0) for row in rows[:top_k]]
    if not any(gains):
        return 0.0

    actual = _dcg(gains)
    ideal = _dcg(sorted(gains, reverse=True))
    if ideal <= 0:
        return 0.0
    return actual / ideal


def _first_rank(rows: Sequence[dict[str, Any]], key: str, *, top_k: int) -> int | None:
    for row in rows[:top_k]:
        if key == "any":
            if bool(row.get("relevance")):
                return int(row["rank"])
            continue
        if bool((row.get("match") or {}).get(key)):
            return int(row["rank"])
    return None


def evaluate_case(
    case: BenchmarkCase,
    *,
    top_k: int,
    rebuild: bool = False,
    retrieval_fn: Callable[..., list] = retrieve_documents,
) -> dict[str, Any]:
    documents = retrieval_fn(query=case.query, top_k=top_k, rebuild=rebuild)
    result_rows = [
        annotate_result_row(doc_to_row(rank=index, doc=doc), case)
        for index, doc in enumerate(documents, start=1)
    ]

    predicted_intents = extract_keyword_groups(case.query).get("intent_labels", [])
    expected_intents = [_normalize_text(item) for item in case.expected_answer_intents]
    predicted_intent_set = {_normalize_text(item) for item in predicted_intents}
    intent_evaluated = bool(expected_intents)
    intent_hit = set(expected_intents).issubset(predicted_intent_set) if intent_evaluated else False

    metrics = {
        "overall": {
            **_metric_from_first_match_rank(
                _first_rank(result_rows, "any", top_k=top_k),
                evaluated=case.has_retrieval_labels,
            ),
            "ndcg_at_k": _compute_ndcg(
                result_rows,
                top_k=top_k,
                evaluated=case.has_retrieval_labels,
            ),
        },
        "source": _metric_from_first_match_rank(
            _first_rank(result_rows, "source", top_k=top_k),
            evaluated=bool(case.expected_sources),
        ),
        "document": _metric_from_first_match_rank(
            _first_rank(result_rows, "document", top_k=top_k),
            evaluated=bool(case.expected_documents),
        ),
        "category": _metric_from_first_match_rank(
            _first_rank(result_rows, "category", top_k=top_k),
            evaluated=bool(case.expected_categories),
        ),
        "intent": {
            "evaluated": intent_evaluated,
            "hit": intent_hit,
            "expected": case.expected_answer_intents,
            "predicted": predicted_intents,
        },
    }

    return {
        "case_id": case.case_id,
        "query": case.query,
        "labels": {
            "expected_sources": case.expected_sources,
            "expected_documents": case.expected_documents,
            "expected_categories": case.expected_categories,
            "expected_answer_intents": case.expected_answer_intents,
            "notes": case.notes,
        },
        "result_count": len(result_rows),
        "results": result_rows,
        "metrics": metrics,
    }


def _round_metric(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 4)


def _aggregate_rank_metrics(records: Sequence[dict[str, Any]], key: str) -> dict[str, Any]:
    evaluated = [record["metrics"][key] for record in records if record["metrics"][key]["evaluated"]]
    if not evaluated:
        return {
            "queries_evaluated": 0,
            "hits": 0,
            "hit_at_k": None,
            "mrr": None,
            "ndcg_at_k": None,
        }

    hits = sum(1 for item in evaluated if item["hit"])
    mrr = sum(float(item["reciprocal_rank"]) for item in evaluated) / len(evaluated)
    ndcg_values = [item.get("ndcg_at_k") for item in evaluated if item.get("ndcg_at_k") is not None]
    ndcg = (sum(float(item) for item in ndcg_values) / len(ndcg_values)) if ndcg_values else None

    return {
        "queries_evaluated": len(evaluated),
        "hits": hits,
        "hit_at_k": _round_metric(hits / len(evaluated)),
        "mrr": _round_metric(mrr),
        "ndcg_at_k": _round_metric(ndcg),
    }


def _aggregate_intent_metrics(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    evaluated = [record["metrics"]["intent"] for record in records if record["metrics"]["intent"]["evaluated"]]
    if not evaluated:
        return {
            "queries_evaluated": 0,
            "hits": 0,
            "accuracy": None,
        }

    hits = sum(1 for item in evaluated if item["hit"])
    return {
        "queries_evaluated": len(evaluated),
        "hits": hits,
        "accuracy": _round_metric(hits / len(evaluated)),
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
    archived_report_path: Path,
) -> dict[str, Any]:
    node_type_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for record in records:
        for row in record["results"]:
            node_type = str(row.get("node_type") or "").strip()
            source = str(row.get("source") or "").strip()
            if node_type:
                node_type_counts[node_type] += 1
            if source:
                source_counts[source] += 1

    labeled_misses = []
    for record in records:
        overall = record["metrics"]["overall"]
        if not overall["evaluated"] or overall["hit"]:
            continue
        top_result = (record["results"] or [{}])[0]
        labeled_misses.append(
            {
                "case_id": record["case_id"],
                "query": record["query"],
                "top_result_title": top_result.get("title") or "",
                "top_result_source": top_result.get("source") or "",
            }
        )

    average_results = (
        sum(int(record["result_count"]) for record in records) / len(records)
        if records
        else 0.0
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "timestamp": timestamp,
        "benchmark_path": str(benchmark_path),
        "paths": {
            "run_jsonl": str(run_path),
            "summary_json": str(summary_json_path),
            "report_txt": str(report_path),
            "archived_report_txt": str(archived_report_path),
        },
        "config": {
            "top_k": top_k,
            "rebuild": rebuild,
        },
        "query_count": len(records),
        "label_coverage": {
            "retrieval_labeled_queries": sum(
                1 for record in records if record["metrics"]["overall"]["evaluated"]
            ),
            "source_labeled_queries": sum(
                1 for record in records if record["metrics"]["source"]["evaluated"]
            ),
            "document_labeled_queries": sum(
                1 for record in records if record["metrics"]["document"]["evaluated"]
            ),
            "category_labeled_queries": sum(
                1 for record in records if record["metrics"]["category"]["evaluated"]
            ),
            "intent_labeled_queries": sum(
                1 for record in records if record["metrics"]["intent"]["evaluated"]
            ),
        },
        "metrics": {
            "overall": _aggregate_rank_metrics(records, "overall"),
            "source": _aggregate_rank_metrics(records, "source"),
            "document": _aggregate_rank_metrics(records, "document"),
            "category": _aggregate_rank_metrics(records, "category"),
            "intent": _aggregate_intent_metrics(records),
        },
        "result_stats": {
            "total_results": sum(int(record["result_count"]) for record in records),
            "average_results_per_query": _round_metric(average_results),
            "node_type_counts": dict(sorted(node_type_counts.items())),
            "source_counts": dict(sorted(source_counts.items())),
        },
        "misses": labeled_misses[:10],
    }


def _format_metric(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def build_human_report(summary: dict[str, Any]) -> str:
    top_k = summary["config"]["top_k"]
    overall = summary["metrics"]["overall"]
    source = summary["metrics"]["source"]
    document = summary["metrics"]["document"]
    category = summary["metrics"]["category"]
    intent = summary["metrics"]["intent"]
    coverage = summary["label_coverage"]
    stats = summary["result_stats"]

    lines = [
        "Retrieval Benchmark Summary",
        "=" * 80,
        f"Generated at          : {summary['generated_at']}",
        f"Benchmark file        : {summary['benchmark_path']}",
        f"Top-k                 : {top_k}",
        f"Rebuild vector store  : {summary['config']['rebuild']}",
        f"Run jsonl             : {summary['paths']['run_jsonl']}",
        f"Summary json          : {summary['paths']['summary_json']}",
        "",
        "Coverage",
        "-" * 80,
        f"Total queries         : {summary['query_count']}",
        f"Retrieval-labeled     : {coverage['retrieval_labeled_queries']}",
        f"Source-labeled        : {coverage['source_labeled_queries']}",
        f"Document-labeled      : {coverage['document_labeled_queries']}",
        f"Category-labeled      : {coverage['category_labeled_queries']}",
        f"Intent-labeled        : {coverage['intent_labeled_queries']}",
        "",
        "Metrics",
        "-" * 80,
        f"Overall hit@{top_k}    : {_format_metric(overall['hit_at_k'])} ({overall['hits']}/{overall['queries_evaluated']})",
        f"Overall MRR          : {_format_metric(overall['mrr'])}",
        f"Overall nDCG@{top_k}   : {_format_metric(overall['ndcg_at_k'])}",
        f"Source hit@{top_k}     : {_format_metric(source['hit_at_k'])} ({source['hits']}/{source['queries_evaluated']})",
        f"Source MRR           : {_format_metric(source['mrr'])}",
        f"Document hit@{top_k}   : {_format_metric(document['hit_at_k'])} ({document['hits']}/{document['queries_evaluated']})",
        f"Document MRR         : {_format_metric(document['mrr'])}",
        f"Category hit@{top_k}   : {_format_metric(category['hit_at_k'])} ({category['hits']}/{category['queries_evaluated']})",
        f"Category MRR         : {_format_metric(category['mrr'])}",
        f"Intent accuracy      : {_format_metric(intent['accuracy'])} ({intent['hits']}/{intent['queries_evaluated']})",
        "",
        "Result Stats",
        "-" * 80,
        f"Total retrieved docs : {stats['total_results']}",
        f"Avg docs / query     : {_format_metric(stats['average_results_per_query'])}",
        f"Node types           : {json.dumps(stats['node_type_counts'], ensure_ascii=False)}",
        f"Sources              : {json.dumps(stats['source_counts'], ensure_ascii=False)}",
    ]

    if summary["misses"]:
        lines.extend(
            [
                "",
                "Top Misses",
                "-" * 80,
            ]
        )
        for miss in summary["misses"]:
            lines.append(
                f"- {miss['case_id']}: {miss['query']} | top_result={miss['top_result_source']} :: {miss['top_result_title']}"
            )

    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _write_run_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_benchmark(
    *,
    benchmark_path: Path,
    top_k: int,
    rebuild: bool = False,
    retrieval_fn: Callable[..., list] = retrieve_documents,
) -> dict[str, Any]:
    cases = load_benchmark_cases(benchmark_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = RUNS_DIR / f"{timestamp}_retrieval_benchmark.jsonl"
    summary_json_path = RUNS_DIR / f"{timestamp}_retrieval_summary.json"
    archived_report_path = RUNS_DIR / f"{timestamp}_retrieval_summary.txt"
    report_path = REPORTS_DIR / "latest_retrieval_summary.txt"

    records = [
        evaluate_case(
            case,
            top_k=top_k,
            rebuild=rebuild,
            retrieval_fn=retrieval_fn,
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
        archived_report_path=archived_report_path,
    )
    report_text = build_human_report(summary)

    _write_run_jsonl(run_path, records)
    _write_json(summary_json_path, summary)
    archived_report_path.write_text(report_text, encoding="utf-8")
    report_path.write_text(report_text, encoding="utf-8")

    return {
        "records": records,
        "summary": summary,
        "paths": summary["paths"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the retrieval benchmark with optional labels and ranking metrics."
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=resolve_default_queries_path(),
        help="Benchmark file (.jsonl/.json or legacy .txt).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of retrieved results to evaluate per query.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the vector store before running the benchmark.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_benchmark(
        benchmark_path=args.queries_path,
        top_k=args.top_k,
        rebuild=args.rebuild,
    )
    summary = result["summary"]
    top_k = summary["config"]["top_k"]

    print("=" * 80)
    print("RETRIEVAL BENCHMARK")
    print(f"Benchmark file      : {summary['benchmark_path']}")
    print(f"Queries             : {summary['query_count']}")
    print(f"Retrieval-labeled   : {summary['label_coverage']['retrieval_labeled_queries']}")
    print(
        f"Overall hit@{top_k:<2}       : {_format_metric(summary['metrics']['overall']['hit_at_k'])}"
    )
    print(f"Overall MRR         : {_format_metric(summary['metrics']['overall']['mrr'])}")
    print(
        f"Overall nDCG@{top_k:<2}      : {_format_metric(summary['metrics']['overall']['ndcg_at_k'])}"
    )
    print(f"Intent accuracy     : {_format_metric(summary['metrics']['intent']['accuracy'])}")
    print(f"[DONE] Saved run    -> {summary['paths']['run_jsonl']}")
    print(f"[DONE] Saved summary -> {summary['paths']['summary_json']}")
    print(f"[DONE] Saved report  -> {summary['paths']['report_txt']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
