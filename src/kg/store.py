from __future__ import annotations

import re
from collections import defaultdict, deque
from typing import Any

from src.kg.schema import EDGE_TYPES, NODE_TYPES, EdgeType, KGEdge, KGNode, NodeType


class InMemoryKnowledgeGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, KGNode] = {}
        self.edges: list[KGEdge] = []
        self._edge_keys: set[tuple[str, str, str]] = set()
        self._outgoing: dict[str, list[KGEdge]] = defaultdict(list)
        self._incoming: dict[str, list[KGEdge]] = defaultdict(list)

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        label: str,
        *,
        properties: dict[str, Any] | None = None,
        text: str = "",
    ) -> KGNode:
        if node_type not in NODE_TYPES:
            raise ValueError(f"Unknown KG node type: {node_type}")

        clean_label = str(label or node_id).strip() or node_id
        node = KGNode(
            id=node_id,
            type=node_type,
            label=clean_label,
            properties=properties or {},
            text=text,
        )
        self.nodes[node_id] = node
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        *,
        properties: dict[str, Any] | None = None,
    ) -> KGEdge:
        if edge_type not in EDGE_TYPES:
            raise ValueError(f"Unknown KG edge type: {edge_type}")
        if source_id not in self.nodes:
            raise KeyError(f"Source node does not exist: {source_id}")
        if target_id not in self.nodes:
            raise KeyError(f"Target node does not exist: {target_id}")

        key = (source_id, target_id, edge_type)
        if key in self._edge_keys:
            for edge in self._outgoing[source_id]:
                if (
                    edge.source_id == source_id
                    and edge.target_id == target_id
                    and edge.type == edge_type
                ):
                    return edge

        edge = KGEdge(
            source_id=source_id,
            target_id=target_id,
            type=edge_type,
            properties=properties or {},
        )
        self.edges.append(edge)
        self._edge_keys.add(key)
        self._outgoing[source_id].append(edge)
        self._incoming[target_id].append(edge)
        return edge

    def get_node(self, node_id: str) -> KGNode | None:
        return self.nodes.get(node_id)

    def edge_exists(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
    ) -> bool:
        return (source_id, target_id, edge_type) in self._edge_keys

    def get_edges(self, node_id: str | None = None) -> list[KGEdge]:
        if node_id is None:
            return list(self.edges)
        return list(self._outgoing.get(node_id, [])) + list(self._incoming.get(node_id, []))

    def get_neighbors(self, node_id: str, depth: int = 1) -> list[KGNode]:
        if depth < 1 or node_id not in self.nodes:
            return []

        visited = {node_id}
        queue: deque[tuple[str, int]] = deque([(node_id, 0)])
        neighbors: list[KGNode] = []

        while queue:
            current_id, current_depth = queue.popleft()
            if current_depth >= depth:
                continue

            for edge in self.get_edges(current_id):
                next_id = edge.target_id if edge.source_id == current_id else edge.source_id
                if next_id in visited:
                    continue
                visited.add(next_id)
                neighbors.append(self.nodes[next_id])
                queue.append((next_id, current_depth + 1))

        return neighbors

    def get_context_edges(self, node_ids: set[str]) -> list[KGEdge]:
        return [
            edge for edge in self.edges if edge.source_id in node_ids and edge.target_id in node_ids
        ]

    def search_nodes(self, query: str, limit: int = 10) -> list[KGNode]:
        tokens = _tokenize(query)
        normalized_query = str(query or "").strip().lower()
        if not tokens and not normalized_query:
            return []

        scored: list[tuple[int, KGNode]] = []
        for node in self.nodes.values():
            haystack = _node_search_text(node)
            score = 0
            if normalized_query and normalized_query in haystack:
                score += 5
            score += sum(1 for token in tokens if token in haystack)
            if normalized_query and normalized_query == node.label.lower():
                score += 4
            if score:
                scored.append((score, node))

        scored.sort(key=lambda item: (-item[0], item[1].type, item[1].label, item[1].id))
        return [node for _, node in scored[:limit]]


def _tokenize(query: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-z0-9_]+", str(query or "").lower()) if len(token) > 1
    ]


def _node_search_text(node: KGNode) -> str:
    property_text = " ".join(str(value) for value in node.properties.values())
    return f"{node.id} {node.type} {node.label} {node.text} {property_text}".lower()


_DEFAULT_GRAPH: InMemoryKnowledgeGraph | None = None


def set_default_graph(graph: InMemoryKnowledgeGraph) -> None:
    global _DEFAULT_GRAPH
    _DEFAULT_GRAPH = graph


def get_default_graph() -> InMemoryKnowledgeGraph:
    if _DEFAULT_GRAPH is None:
        raise RuntimeError(
            "No default knowledge graph has been built. "
            "Call build_graph_from_enterprise_support_dataset(dataset) first."
        )
    return _DEFAULT_GRAPH


def get_neighbors(node_id: str, depth: int = 1) -> list[KGNode]:
    return get_default_graph().get_neighbors(node_id, depth=depth)
