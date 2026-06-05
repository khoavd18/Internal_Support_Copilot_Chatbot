from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from src.data.enterprise_support_loader import load_enterprise_support_dataset
from src.kg.builder import build_graph_from_enterprise_support_dataset
from src.kg.retriever import retrieve_graph_context
from src.kg.schema import GraphContext, KGNode
from src.kg.store import get_default_graph
from src.rag.retrieval.retriever import retrieve_documents

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENTERPRISE_DATA_DIR = PROJECT_ROOT / "data" / "sample_enterprise_support"

SOURCE_TYPE_TO_NODE_TYPE = {
    "account": "Account",
    "customer": "Customer",
    "github_issue": "GitHubIssue",
    "knowledge_base": "Policy",
    "policy": "Policy",
    "product": "Product",
    "risk_event": "RiskEvent",
    "service": "Service",
    "ticket": "Ticket",
    "ticket_message": "TicketMessage",
}


def retrieve_enterprise_context(
    query: str,
    top_k: int = 5,
    graph_depth: int = 2,
) -> dict[str, Any]:
    vector_results: list[Any] = []
    vector_error = ""

    try:
        vector_results = retrieve_documents(query=query, top_k=top_k, rebuild=False)
    except Exception as exc:
        vector_error = _safe_error(exc)

    graph_results = _retrieve_graph_context(query=query, depth=graph_depth, limit=top_k)
    merged_context = merge_vector_and_graph_context(vector_results, graph_results)
    formatted_context = format_context_for_answer(merged_context)

    return {
        "query": query,
        "vector_evidence": _normalize_vector_results(vector_results),
        "graph_evidence": _normalize_graph_results(graph_results),
        "merged_context": merged_context,
        "formatted_context": formatted_context,
        "citations": _build_citations(merged_context),
        "stats": {
            "top_k": top_k,
            "graph_depth": graph_depth,
            "vector_count": len(vector_results),
            "graph_node_count": len(graph_results.context_nodes),
            "graph_edge_count": len(graph_results.context_edges),
            "merged_count": len(merged_context),
            "vector_error": vector_error,
        },
    }


def merge_vector_and_graph_context(vector_results, graph_results) -> list[dict[str, Any]]:
    vector_evidence = _normalize_vector_results(vector_results)
    graph_evidence = _normalize_graph_results(graph_results)

    merged_by_key: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for item in vector_evidence:
        key = _evidence_key(item)
        merged_by_key[key] = dict(item)
        order.append(key)

    for item in graph_evidence:
        key = _evidence_key(item)
        if key not in merged_by_key:
            merged_by_key[key] = dict(item)
            order.append(key)
            continue

        merged_by_key[key] = _merge_evidence_items(merged_by_key[key], item)

    return [merged_by_key[key] for key in order]


def format_context_for_answer(context) -> str:
    if isinstance(context, dict):
        evidence_items = context.get("merged_context", [])
    else:
        evidence_items = context or []

    lines = ["Enterprise GraphRAG Context"]
    if not evidence_items:
        lines.append("")
        lines.append("No enterprise vector or graph evidence was found.")
        return "\n".join(lines)

    for index, item in enumerate(evidence_items, start=1):
        metadata = item.get("metadata") or {}
        title = item.get("title") or metadata.get("title") or item.get("id") or "Untitled"
        text = str(item.get("text") or "").strip()
        if len(text) > 700:
            text = text[:697] + "..."

        lines.extend(
            [
                "",
                f"[{index}] {title}",
                f"source: {item.get('context_source', 'unknown')}",
                f"source_type: {item.get('source_type') or metadata.get('source_type') or ''}",
                f"id: {item.get('id') or ''}",
                f"metadata: {metadata}",
                f"text: {text}",
            ]
        )

    return "\n".join(lines).strip()


def _retrieve_graph_context(query: str, depth: int, limit: int) -> GraphContext:
    try:
        graph = get_default_graph()
    except RuntimeError:
        dataset = load_enterprise_support_dataset(DEFAULT_ENTERPRISE_DATA_DIR)
        graph = build_graph_from_enterprise_support_dataset(dataset)
    return retrieve_graph_context(query=query, depth=depth, graph=graph, limit=limit)


def _normalize_vector_results(vector_results) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for index, result in enumerate(vector_results or [], start=1):
        if isinstance(result, Document):
            text = result.page_content
            metadata = dict(result.metadata or {})
        elif isinstance(result, dict):
            text = str(result.get("text") or result.get("page_content") or "")
            metadata = dict(result.get("metadata") or {})
        else:
            text = str(result)
            metadata = {}

        evidence_id = _vector_evidence_id(metadata, index)
        evidence.append(
            {
                "id": evidence_id,
                "text": text,
                "metadata": metadata,
                "context_source": "vector",
                "source_type": str(metadata.get("source_type") or ""),
                "title": str(metadata.get("title") or evidence_id),
            }
        )
    return evidence


def _normalize_graph_results(graph_results) -> list[dict[str, Any]]:
    if isinstance(graph_results, GraphContext):
        nodes = graph_results.context_nodes
    elif isinstance(graph_results, dict):
        nodes = graph_results.get("context_nodes") or graph_results.get("matched_nodes") or []
    else:
        nodes = graph_results or []

    evidence: list[dict[str, Any]] = []
    for node in nodes:
        if isinstance(node, KGNode):
            metadata = {
                **node.properties,
                "kg_node_id": node.id,
                "kg_node_type": node.type,
                "entity_id": _entity_id_from_node_id(node.id),
                "title": node.label,
                "source_type": _node_type_to_source_type(node.type),
            }
            evidence.append(
                {
                    "id": node.id,
                    "text": node.text,
                    "metadata": metadata,
                    "context_source": "graph",
                    "source_type": _node_type_to_source_type(node.type),
                    "title": node.label,
                }
            )
            continue

        if isinstance(node, dict):
            metadata = dict(node.get("metadata") or node.get("properties") or {})
            evidence.append(
                {
                    "id": str(node.get("id") or metadata.get("kg_node_id") or ""),
                    "text": str(node.get("text") or ""),
                    "metadata": metadata,
                    "context_source": "graph",
                    "source_type": str(
                        node.get("source_type") or metadata.get("source_type") or ""
                    ),
                    "title": str(node.get("title") or metadata.get("title") or ""),
                }
            )
    return evidence


def _merge_evidence_items(
    vector_item: dict[str, Any],
    graph_item: dict[str, Any],
) -> dict[str, Any]:
    vector_text = str(vector_item.get("text") or "").strip()
    graph_text = str(graph_item.get("text") or "").strip()
    if vector_text and graph_text and vector_text != graph_text:
        text = f"Vector evidence:\n{vector_text}\n\nGraph evidence:\n{graph_text}"
    else:
        text = vector_text or graph_text

    vector_metadata = dict(vector_item.get("metadata") or {})
    graph_metadata = dict(graph_item.get("metadata") or {})
    metadata = {
        **graph_metadata,
        **vector_metadata,
        "vector_metadata": vector_metadata,
        "graph_metadata": graph_metadata,
    }

    return {
        "id": graph_item.get("id") or vector_item.get("id"),
        "text": text,
        "metadata": metadata,
        "context_source": "both",
        "source_type": vector_item.get("source_type") or graph_item.get("source_type") or "",
        "title": vector_item.get("title") or graph_item.get("title") or "",
    }


def _vector_evidence_id(metadata: dict[str, Any], index: int) -> str:
    canonical = _canonical_node_id_from_metadata(metadata)
    if canonical:
        return canonical
    return str(
        metadata.get("doc_id")
        or metadata.get("source_chunk_id")
        or metadata.get("path")
        or metadata.get("title")
        or f"vector:{index}"
    )


def _evidence_key(item: dict[str, Any]) -> str:
    metadata = item.get("metadata") or {}
    canonical = _canonical_node_id_from_metadata(metadata)
    if canonical:
        return canonical
    item_id = str(item.get("id") or "")
    if ":" in item_id:
        return item_id
    return item_id or str(item.get("title") or "")


def _canonical_node_id_from_metadata(metadata: dict[str, Any]) -> str:
    kg_node_id = str(metadata.get("kg_node_id") or "").strip()
    if kg_node_id:
        return kg_node_id

    source_type = str(metadata.get("source_type") or "").strip().lower()
    entity_id = str(metadata.get("entity_id") or "").strip()

    if not entity_id:
        for field in (
            "ticket_id",
            "customer_id",
            "account_id",
            "product_id",
            "service_id",
            "policy_id",
            "risk_event_id",
            "issue_id",
            "message_id",
        ):
            value = str(metadata.get(field) or "").strip()
            if value:
                entity_id = value
                source_type = _source_type_from_id_field(field, source_type)
                break

    node_type = SOURCE_TYPE_TO_NODE_TYPE.get(source_type)
    if node_type and entity_id:
        return f"{node_type}:{entity_id}"
    return ""


def _source_type_from_id_field(field: str, fallback: str) -> str:
    mapping = {
        "account_id": "account",
        "customer_id": "customer",
        "issue_id": "github_issue",
        "message_id": "ticket_message",
        "policy_id": "knowledge_base",
        "product_id": "product",
        "risk_event_id": "risk_event",
        "service_id": "service",
        "ticket_id": "ticket",
    }
    return mapping.get(field, fallback)


def _node_type_to_source_type(node_type: str) -> str:
    mapping = {
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
    return mapping.get(node_type, node_type.lower())


def _entity_id_from_node_id(node_id: str) -> str:
    if ":" not in node_id:
        return node_id
    return node_id.split(":", 1)[1]


def _build_citations(merged_context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations = []
    for index, item in enumerate(merged_context, start=1):
        metadata = item.get("metadata") or {}
        citations.append(
            {
                "index": index,
                "id": item.get("id"),
                "title": item.get("title") or metadata.get("title") or "",
                "source_type": item.get("source_type") or metadata.get("source_type") or "",
                "context_source": item.get("context_source"),
                "metadata": metadata,
            }
        )
    return citations


def _safe_error(exc: Exception) -> str:
    text = str(exc).strip()
    if len(text) > 240:
        text = text[:237] + "..."
    return text
