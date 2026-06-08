from __future__ import annotations

from src.kg.schema import GraphContext, KGEdge, KGNode
from src.kg.store import InMemoryKnowledgeGraph, get_default_graph


def search_nodes(
    query: str,
    *,
    graph: InMemoryKnowledgeGraph | None = None,
    limit: int = 10,
) -> list[KGNode]:
    active_graph = graph or get_default_graph()
    return active_graph.search_nodes(query, limit=limit)


def retrieve_graph_context(
    query: str,
    depth: int = 2,
    *,
    graph: InMemoryKnowledgeGraph | None = None,
    limit: int = 5,
) -> GraphContext:
    active_graph = graph or get_default_graph()
    matched_nodes = active_graph.search_nodes(query, limit=limit)

    context_nodes_by_id: dict[str, KGNode] = {}
    for node in matched_nodes:
        context_nodes_by_id[node.id] = node
        for neighbor in active_graph.get_neighbors(node.id, depth=depth):
            context_nodes_by_id[neighbor.id] = neighbor

    context_node_ids = set(context_nodes_by_id)
    context_edges = active_graph.get_context_edges(context_node_ids)
    context_nodes = sorted(
        context_nodes_by_id.values(),
        key=lambda node: (node.type, node.label, node.id),
    )

    return GraphContext(
        query=query,
        text=_format_context_text(matched_nodes, context_nodes, context_edges),
        matched_nodes=matched_nodes,
        context_nodes=context_nodes,
        context_edges=context_edges,
    )


def _format_context_text(
    matched_nodes: list[KGNode],
    context_nodes: list[KGNode],
    context_edges: list[KGEdge],
) -> str:
    lines = ["Graph Context"]
    lines.append("")
    lines.append("Matched nodes:")
    if not matched_nodes:
        lines.append("- No matching nodes found.")
    for node in matched_nodes:
        lines.append(f"- {node.type} {node.id}: {node.label}")

    lines.append("")
    lines.append("Context nodes:")
    for node in context_nodes:
        summary = node.text.replace("\n", " ")
        if len(summary) > 240:
            summary = summary[:237] + "..."
        lines.append(f"- {node.type} {node.id}: {node.label}. {summary}")

    lines.append("")
    lines.append("Context edges:")
    for edge in context_edges:
        lines.append(f"- {edge.source_id} -[{edge.type}]-> {edge.target_id}")

    return "\n".join(lines).strip()
