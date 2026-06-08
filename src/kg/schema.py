from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

NodeType = Literal[
    "Customer",
    "Account",
    "Product",
    "Ticket",
    "TicketMessage",
    "Policy",
    "Service",
    "GitHubIssue",
    "RiskEvent",
    "Team",
]

EdgeType = Literal[
    "HAS_ACCOUNT",
    "CREATED_TICKET",
    "HAS_MESSAGE",
    "HAS_RESOLUTION",
    "MENTIONS_PRODUCT",
    "REFERENCES_POLICY",
    "AFFECTS_SERVICE",
    "OWNED_BY_TEAM",
    "RELATED_TO_ISSUE",
    "HAS_RISK_EVENT",
]

NODE_TYPES: tuple[NodeType, ...] = (
    "Customer",
    "Account",
    "Product",
    "Ticket",
    "TicketMessage",
    "Policy",
    "Service",
    "GitHubIssue",
    "RiskEvent",
    "Team",
)

EDGE_TYPES: tuple[EdgeType, ...] = (
    "HAS_ACCOUNT",
    "CREATED_TICKET",
    "HAS_MESSAGE",
    "HAS_RESOLUTION",
    "MENTIONS_PRODUCT",
    "REFERENCES_POLICY",
    "AFFECTS_SERVICE",
    "OWNED_BY_TEAM",
    "RELATED_TO_ISSUE",
    "HAS_RISK_EVENT",
)


@dataclass(frozen=True)
class KGNode:
    id: str
    type: NodeType
    label: str
    properties: dict[str, Any] = field(default_factory=dict)
    text: str = ""


@dataclass(frozen=True)
class KGEdge:
    source_id: str
    target_id: str
    type: EdgeType
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphContext:
    query: str
    text: str
    matched_nodes: list[KGNode]
    context_nodes: list[KGNode]
    context_edges: list[KGEdge]
