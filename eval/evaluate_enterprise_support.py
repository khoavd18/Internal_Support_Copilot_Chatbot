from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.enterprise_support_documents import build_enterprise_support_documents  # noqa: E402
from src.data.enterprise_support_loader import load_enterprise_support_dataset  # noqa: E402
from src.kg.builder import build_graph_from_enterprise_support_dataset  # noqa: E402
from src.kg.retriever import retrieve_graph_context  # noqa: E402
from src.kg.schema import GraphContext, KGNode  # noqa: E402

DEFAULT_QUERIES_PATH = PROJECT_ROOT / "eval" / "enterprise_support_queries.jsonl"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "sample_enterprise_support"
RUNS_DIR = PROJECT_ROOT / "eval" / "runs"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"
RECALL_K = 5

NODE_TYPE_TO_SOURCE_TYPE = {
    "Account": "account",
    "Customer": "customer",
    "GitHubIssue": "github_issue",
    "Policy": "knowledge_base",
    "Product": "product",
    "RiskEvent": "risk_event",
    "Service": "service",
    "Team": "team",
    "Ticket": "ticket",
    "TicketMessage": "ticket_message",
}

ENTITY_ID_FIELDS = (
    "entity_id",
    "customer_id",
    "account_id",
    "ticket_id",
    "product_id",
    "service_id",
    "policy_id",
    "risk_event_id",
    "issue_id",
    "message_id",
    "kg_node_id",
)


@dataclass(frozen=True)
class EnterpriseEvalCase:
    query_id: str
    query: str
    expected_source_types: list[str]
    expected_entity_ids: list[str]
    expected_answer_points: list[str]
    category: str = ""
    missing_info: bool = False


def load_cases(path: Path) -> list[EnterpriseEvalCase]:
    cases: list[EnterpriseEvalCase] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            missing = _required_fields(payload) - set(payload)
            if missing:
                missing_list = ", ".join(sorted(missing))
                raise ValueError(f"{path}:{line_number} missing fields: {missing_list}")

            cases.append(
                EnterpriseEvalCase(
                    query_id=str(payload["query_id"]),
                    query=str(payload["query"]),
                    expected_source_types=_string_list(payload["expected_source_types"]),
                    expected_entity_ids=_string_list(payload["expected_entity_ids"]),
                    expected_answer_points=_string_list(payload["expected_answer_points"]),
                    category=str(payload.get("category") or ""),
                    missing_info=bool(payload.get("missing_info", False)),
                )
            )
    return cases


def run_evaluation(
    *,
    queries_path: Path = DEFAULT_QUERIES_PATH,
    data_dir: Path = DEFAULT_DATA_DIR,
    top_k: int = RECALL_K,
    graph_depth: int = 2,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    cases = load_cases(queries_path)
    if limit is not None:
        cases = cases[:limit]

    local_context = _build_local_context(data_dir) if dry_run else None
    records = [
        evaluate_case(
            case,
            top_k=top_k,
            graph_depth=graph_depth,
            dry_run=dry_run,
            local_context=local_context,
        )
        for case in cases
    ]
    summary = _build_summary(
        records,
        queries_path=queries_path,
        data_dir=data_dir,
        top_k=top_k,
        graph_depth=graph_depth,
        dry_run=dry_run,
    )

    paths: dict[str, str] = {}
    if not dry_run:
        paths = _write_outputs(records, summary)

    return {
        "records": records,
        "summary": summary,
        "paths": paths,
    }


def evaluate_case(
    case: EnterpriseEvalCase,
    *,
    top_k: int,
    graph_depth: int,
    dry_run: bool,
    local_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = (
        _retrieve_local_context(case.query, top_k=top_k, graph_depth=graph_depth, **local_context)
        if dry_run and local_context
        else _retrieve_graphrag_context(case.query, top_k=top_k, graph_depth=graph_depth)
    )
    evidence = list(context.get("merged_context") or [])
    top_evidence = evidence[:RECALL_K]
    retrieved_entity_ids = _retrieved_entity_ids(top_evidence)
    retrieved_source_types = _retrieved_source_types(top_evidence)
    expected_entity_ids = {_normalize(value) for value in case.expected_entity_ids}
    expected_source_types = {_normalize(value) for value in case.expected_source_types}

    entity_hits = sorted(expected_entity_ids & retrieved_entity_ids)
    source_hits = sorted(expected_source_types & retrieved_source_types)
    recall = len(entity_hits) / len(expected_entity_ids) if expected_entity_ids else None
    source_type_hit = bool(source_hits) if expected_source_types else None
    source_type_all_hit = (
        expected_source_types.issubset(retrieved_source_types) if expected_source_types else None
    )
    groundedness = _groundedness_proxy(
        top_evidence,
        entity_hits=entity_hits,
        source_type_hit=source_type_hit,
        has_expected_entities=bool(expected_entity_ids),
    )
    missing_info_handled = _missing_info_proxy(case, top_evidence, entity_hits, source_hits)

    return {
        "query_id": case.query_id,
        "category": case.category,
        "query": case.query,
        "expected_source_types": case.expected_source_types,
        "expected_entity_ids": case.expected_entity_ids,
        "expected_answer_points": case.expected_answer_points,
        "retrieved_entity_ids_at_5": sorted(retrieved_entity_ids),
        "retrieved_source_types_at_5": sorted(retrieved_source_types),
        "metrics": {
            "recall_at_5": _round(recall),
            "entity_hits": entity_hits,
            "source_type_hit": source_type_hit,
            "source_type_all_hit": source_type_all_hit,
            "source_type_hits": source_hits,
            "groundedness": groundedness,
            "missing_info_handled": missing_info_handled,
        },
        "evidence": [_compact_evidence(item) for item in top_evidence],
        "retrieval_stats": context.get("stats", {}),
    }


def _retrieve_graphrag_context(query: str, *, top_k: int, graph_depth: int) -> dict[str, Any]:
    from src.rag.graphrag import retrieve_enterprise_context

    return retrieve_enterprise_context(query, top_k=top_k, graph_depth=graph_depth)


def _build_local_context(data_dir: Path) -> dict[str, Any]:
    dataset = load_enterprise_support_dataset(data_dir)
    return {
        "documents": build_enterprise_support_documents(dataset),
        "graph": build_graph_from_enterprise_support_dataset(dataset),
    }


def _retrieve_local_context(
    query: str,
    *,
    top_k: int,
    graph_depth: int,
    documents: list[dict],
    graph: Any,
) -> dict[str, Any]:
    vector_evidence = _local_lexical_retrieve(query, documents, limit=top_k)
    graph_context = retrieve_graph_context(query=query, depth=graph_depth, graph=graph, limit=top_k)
    graph_evidence = _normalize_graph_context(graph_context)
    merged_context = _merge_evidence(vector_evidence, graph_evidence)[:top_k]
    return {
        "query": query,
        "vector_evidence": vector_evidence,
        "graph_evidence": graph_evidence,
        "merged_context": merged_context,
        "citations": [_citation(index, item) for index, item in enumerate(merged_context, start=1)],
        "stats": {
            "mode": "dry_run_local_lexical_plus_kg",
            "top_k": top_k,
            "graph_depth": graph_depth,
            "vector_count": len(vector_evidence),
            "graph_node_count": len(graph_context.context_nodes),
            "graph_edge_count": len(graph_context.context_edges),
            "merged_count": len(merged_context),
            "vector_error": "",
        },
    }


def _local_lexical_retrieve(
    query: str, documents: list[dict], *, limit: int
) -> list[dict[str, Any]]:
    query_tokens = set(_tokenize(query))
    scored: list[tuple[float, dict]] = []
    for document in documents:
        metadata = dict(document.get("metadata") or {})
        text = str(document.get("text") or "")
        haystack = " ".join(
            [
                text,
                str(document.get("id") or ""),
                " ".join(str(value) for value in metadata.values()),
            ]
        ).lower()
        doc_tokens = set(_tokenize(haystack))
        score = float(len(query_tokens & doc_tokens))

        for value in _metadata_values(metadata):
            normalized_value = value.lower()
            if normalized_value and normalized_value in query.lower():
                score += 4.0

        if score > 0:
            scored.append((score, document))

    scored.sort(
        key=lambda item: (
            item[0],
            str(item[1].get("metadata", {}).get("source_type") or ""),
            str(item[1].get("id") or ""),
        ),
        reverse=True,
    )
    return [
        {
            "id": str(document.get("id") or ""),
            "text": str(document.get("text") or ""),
            "metadata": dict(document.get("metadata") or {}),
            "context_source": "vector",
            "source_type": str(document.get("metadata", {}).get("source_type") or ""),
            "title": str(document.get("metadata", {}).get("title") or document.get("id") or ""),
            "score": score,
        }
        for score, document in scored[:limit]
    ]


def _normalize_graph_context(graph_context: GraphContext) -> list[dict[str, Any]]:
    nodes: list[KGNode] = []
    seen: set[str] = set()
    for node in [*graph_context.matched_nodes, *graph_context.context_nodes]:
        if node.id in seen:
            continue
        seen.add(node.id)
        nodes.append(node)

    evidence = []
    for node in nodes:
        source_type = NODE_TYPE_TO_SOURCE_TYPE.get(node.type, str(node.type).lower())
        entity_id = _entity_id_from_node_id(node.id)
        metadata = {
            **node.properties,
            "kg_node_id": node.id,
            "kg_node_type": node.type,
            "entity_id": entity_id,
            "source_type": source_type,
            "title": node.label,
        }
        evidence.append(
            {
                "id": node.id,
                "text": node.text,
                "metadata": metadata,
                "context_source": "graph",
                "source_type": source_type,
                "title": node.label,
            }
        )
    return evidence


def _merge_evidence(
    vector_evidence: list[dict[str, Any]],
    graph_evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for item in [*vector_evidence, *graph_evidence]:
        key = _evidence_key(item)
        if key not in merged:
            merged[key] = dict(item)
            order.append(key)
            continue
        existing = merged[key]
        existing["context_source"] = "both"
        if str(item.get("text") or "") not in str(existing.get("text") or ""):
            existing["text"] = f"{existing.get('text', '')}\n\n{item.get('text', '')}".strip()
        existing["metadata"] = {
            **dict(item.get("metadata") or {}),
            **dict(existing.get("metadata") or {}),
        }
    return [merged[key] for key in order]


def _build_summary(
    records: list[dict[str, Any]],
    *,
    queries_path: Path,
    data_dir: Path,
    top_k: int,
    graph_depth: int,
    dry_run: bool,
) -> dict[str, Any]:
    recall_values = [
        record["metrics"]["recall_at_5"]
        for record in records
        if record["metrics"]["recall_at_5"] is not None
    ]
    source_hits = [
        record["metrics"]["source_type_hit"]
        for record in records
        if record["metrics"]["source_type_hit"] is not None
    ]
    source_all_hits = [
        record["metrics"]["source_type_all_hit"]
        for record in records
        if record["metrics"]["source_type_all_hit"] is not None
    ]
    grounded = [
        record["metrics"]["groundedness"]
        for record in records
        if record["metrics"]["groundedness"] is not None
    ]
    missing_info = [
        record["metrics"]["missing_info_handled"]
        for record in records
        if record["metrics"]["missing_info_handled"] is not None
    ]
    vector_errors = [
        record["retrieval_stats"].get("vector_error")
        for record in records
        if record["retrieval_stats"].get("vector_error")
    ]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "queries_path": str(queries_path),
        "data_dir": str(data_dir),
        "mode": "dry_run_local" if dry_run else "graphrag",
        "top_k": top_k,
        "recall_k": RECALL_K,
        "graph_depth": graph_depth,
        "query_count": len(records),
        "metrics": {
            "recall_at_5": _round(_average(recall_values)),
            "source_type_hit_rate": _round(_rate(source_hits)),
            "source_type_all_hit_rate": _round(_rate(source_all_hits)),
            "groundedness_rate": _round(_rate(grounded)),
            "missing_info_handling_rate": _round(_rate(missing_info)) if missing_info else None,
            "missing_info_cases": len(missing_info),
        },
        "vector_error_count": len(vector_errors),
        "vector_errors_sample": vector_errors[:3],
        "weak_cases": [
            {
                "query_id": record["query_id"],
                "category": record["category"],
                "recall_at_5": record["metrics"]["recall_at_5"],
                "source_type_hit": record["metrics"]["source_type_hit"],
                "groundedness": record["metrics"]["groundedness"],
            }
            for record in records
            if (record["metrics"]["recall_at_5"] or 0.0) < 0.4
            or record["metrics"]["source_type_hit"] is False
            or record["metrics"]["groundedness"] is False
        ][:10],
    }


def _write_outputs(records: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, str]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_path = RUNS_DIR / f"{timestamp}_enterprise_support_eval.jsonl"
    summary_path = RUNS_DIR / f"{timestamp}_enterprise_support_summary.json"
    report_path = REPORTS_DIR / "latest_enterprise_support_summary.md"

    with run_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(_format_summary(summary), encoding="utf-8")
    return {
        "run_jsonl": str(run_path),
        "summary_json": str(summary_path),
        "latest_report": str(report_path),
    }


def _format_summary(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    lines = [
        "# Enterprise Support Evaluation Summary",
        "",
        f"- Mode: `{summary['mode']}`",
        f"- Queries: `{summary['query_count']}`",
        f"- Recall@5: `{_format_metric(metrics['recall_at_5'])}`",
        f"- Source type hit rate: `{_format_metric(metrics['source_type_hit_rate'])}`",
        f"- Source type all-hit rate: `{_format_metric(metrics['source_type_all_hit_rate'])}`",
        f"- Groundedness proxy rate: `{_format_metric(metrics['groundedness_rate'])}`",
        f"- Missing-info handling proxy: `{_format_metric(metrics['missing_info_handling_rate'])}`",
        f"- Vector errors: `{summary['vector_error_count']}`",
        "",
        "Weak cases:",
    ]
    weak_cases = summary.get("weak_cases") or []
    if not weak_cases:
        lines.append("- None")
    for case in weak_cases:
        lines.append(
            "- {query_id}: recall={recall_at_5}, source_hit={source_type_hit}, grounded={groundedness}".format(
                **case
            )
        )
    return "\n".join(lines).strip() + "\n"


def _groundedness_proxy(
    evidence: list[dict[str, Any]],
    *,
    entity_hits: list[str],
    source_type_hit: bool | None,
    has_expected_entities: bool,
) -> bool:
    has_cited_metadata = any(
        _retrieved_entity_ids([item]) and _retrieved_source_types([item]) for item in evidence
    )
    if not has_cited_metadata:
        return False
    if has_expected_entities:
        return bool(entity_hits)
    if source_type_hit is not None:
        return source_type_hit
    return bool(evidence)


def _missing_info_proxy(
    case: EnterpriseEvalCase,
    evidence: list[dict[str, Any]],
    entity_hits: list[str],
    source_hits: list[str],
) -> bool | None:
    if not case.missing_info:
        return None

    evidence_text_parts: list[str] = []
    for item in evidence:
        evidence_text_parts.extend(
            [
                str(item.get("text") or ""),
                str(item.get("title") or ""),
                json.dumps(item.get("metadata") or {}, ensure_ascii=False),
            ]
        )
    text = " ".join(evidence_text_parts).lower()
    query = case.query.lower()

    unsupported_terms: list[str] = []
    if "phone" in query:
        unsupported_terms.extend(["private phone", "phone number", "mobile number"])
    if "discount" in query or "approval chain" in query:
        unsupported_terms.extend(["discount approval", "approval chain", "discount percentage"])

    false_support = any(term in text for term in unsupported_terms)
    has_relevant_context = bool(entity_hits or source_hits or not evidence)
    return has_relevant_context and not false_support


def _retrieved_entity_ids(evidence: list[dict[str, Any]]) -> set[str]:
    entity_ids: set[str] = set()
    for item in evidence:
        entity_ids.update(_entity_aliases(item))
    return {_normalize(value) for value in entity_ids if _normalize(value)}


def _retrieved_source_types(evidence: list[dict[str, Any]]) -> set[str]:
    source_types: set[str] = set()
    for item in evidence:
        metadata = item.get("metadata") or {}
        for value in (item.get("source_type"), metadata.get("source_type")):
            normalized = _normalize(value)
            if normalized:
                source_types.add(normalized)
    return source_types


def _entity_aliases(item: dict[str, Any]) -> set[str]:
    aliases: set[str] = set()
    metadata = item.get("metadata") or {}

    for value in (item.get("id"), metadata.get("id")):
        aliases.update(_split_entity_identifier(value))

    for field in ENTITY_ID_FIELDS:
        value = metadata.get(field)
        if isinstance(value, list):
            for item_value in value:
                aliases.update(_split_entity_identifier(item_value))
        else:
            aliases.update(_split_entity_identifier(value))

    linked_ticket_ids = metadata.get("linked_ticket_ids")
    if isinstance(linked_ticket_ids, list):
        aliases.update(str(value) for value in linked_ticket_ids)
    elif isinstance(linked_ticket_ids, str):
        aliases.update(part.strip() for part in linked_ticket_ids.split("|") if part.strip())

    return aliases


def _split_entity_identifier(value: Any) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    aliases = {text}
    if "::" in text:
        aliases.add(text.rsplit("::", 1)[-1])
    if ":" in text:
        aliases.add(text.rsplit(":", 1)[-1])
    return aliases


def _evidence_key(item: dict[str, Any]) -> str:
    source_type = _normalize(item.get("source_type") or item.get("metadata", {}).get("source_type"))
    aliases = sorted(_entity_aliases(item))
    if source_type and aliases:
        return f"{source_type}:{aliases[0]}"
    return str(item.get("id") or item.get("title") or "")


def _compact_evidence(item: dict[str, Any]) -> dict[str, Any]:
    metadata = dict(item.get("metadata") or {})
    text = str(item.get("text") or "").strip()
    if len(text) > 500:
        text = text[:497] + "..."
    return {
        "id": item.get("id"),
        "title": item.get("title") or metadata.get("title") or "",
        "source_type": item.get("source_type") or metadata.get("source_type") or "",
        "context_source": item.get("context_source") or "",
        "metadata": {
            key: metadata.get(key)
            for key in (
                "entity_id",
                "customer_id",
                "account_id",
                "ticket_id",
                "product_id",
                "service_id",
                "policy_id",
                "risk_event_id",
                "issue_id",
                "kg_node_id",
            )
            if metadata.get(key) not in ("", None, [])
        },
        "text": text,
    }


def _citation(index: int, item: dict[str, Any]) -> dict[str, Any]:
    metadata = item.get("metadata") or {}
    return {
        "index": index,
        "id": item.get("id"),
        "title": item.get("title") or metadata.get("title") or "",
        "source_type": item.get("source_type") or metadata.get("source_type") or "",
        "context_source": item.get("context_source") or "",
        "metadata": metadata,
    }


def _required_fields(payload: dict[str, Any]) -> set[str]:
    return {
        "query_id",
        "query",
        "expected_source_types",
        "expected_entity_ids",
        "expected_answer_points",
    }


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _metadata_values(metadata: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for key in ENTITY_ID_FIELDS:
        value = metadata.get(key)
        if isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value:
            values.append(str(value))
    title = metadata.get("title")
    if title:
        values.append(str(title))
    return values


def _entity_id_from_node_id(node_id: str) -> str:
    return node_id.split(":", 1)[1] if ":" in node_id else node_id


def _normalize(value: Any) -> str:
    return str(value or "").strip().lower()


def _average(values: list[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _rate(values: list[bool | None]) -> float | None:
    clean = [bool(value) for value in values if value is not None]
    return sum(1 for value in clean if value) / len(clean) if clean else None


def _round(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate enterprise support retrieval and GraphRAG evidence coverage."
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help="Path to enterprise support evaluation JSONL.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to the synthetic enterprise support dataset.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=RECALL_K,
        help="Number of evidence items to retrieve before Recall@5 scoring.",
    )
    parser.add_argument(
        "--graph-depth",
        type=int,
        default=2,
        help="Knowledge Graph traversal depth.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use local lexical retrieval plus in-memory KG and skip Qdrant/output files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N cases.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_evaluation(
        queries_path=args.queries_path,
        data_dir=args.data_dir,
        top_k=args.top_k,
        graph_depth=args.graph_depth,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    summary = result["summary"]
    metrics = summary["metrics"]
    print("Enterprise Support Evaluation")
    print(f"Mode                 : {summary['mode']}")
    print(f"Queries              : {summary['query_count']}")
    print(f"Recall@5             : {_format_metric(metrics['recall_at_5'])}")
    print(f"Source type hit rate : {_format_metric(metrics['source_type_hit_rate'])}")
    print(f"Source type all-hit  : {_format_metric(metrics['source_type_all_hit_rate'])}")
    print(f"Groundedness proxy   : {_format_metric(metrics['groundedness_rate'])}")
    print(f"Missing-info proxy   : {_format_metric(metrics['missing_info_handling_rate'])}")
    print(f"Vector errors        : {summary['vector_error_count']}")
    if result["paths"]:
        print(f"Run output           : {result['paths']['run_jsonl']}")
        print(f"Summary output       : {result['paths']['summary_json']}")


if __name__ == "__main__":
    main()
