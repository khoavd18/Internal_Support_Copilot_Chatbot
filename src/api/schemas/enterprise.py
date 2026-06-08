from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class EnterpriseContextItem(BaseModel):
    source_type: str
    entity_id: str
    title: str = ""
    summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphRAGAskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Enterprise support question")
    debug: bool = Field(default=False, description="Include enterprise retrieval debug metadata")
    use_agentic_retrieval: bool = Field(
        default=False,
        description="Use bounded rule-based retrieval planning and one retry when evidence is weak",
    )
    top_k: int = Field(default=5, ge=1, le=10, description="Vector retrieval top_k")
    graph_depth: int = Field(default=2, ge=1, le=4, description="Knowledge Graph traversal depth")
    source_type: str | None = Field(
        default=None, description="Optional enterprise source_type filter"
    )
    customer_id: str | None = Field(default=None, description="Optional customer_id filter")
    ticket_id: str | None = Field(default=None, description="Optional ticket_id filter")
    service_id: str | None = Field(default=None, description="Optional service_id filter")
    product_id: str | None = Field(default=None, description="Optional product_id filter")


class GraphRAGEvidenceItem(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    context_source: Literal["vector", "graph", "both"]
    source_type: str = ""
    title: str = ""


class GraphRAGAskResponse(BaseModel):
    answer: str
    confidence: Literal["high", "medium", "low"] = "low"
    vector_evidence: list[GraphRAGEvidenceItem] = Field(default_factory=list)
    graph_evidence: list[GraphRAGEvidenceItem] = Field(default_factory=list)
    merged_context: list[GraphRAGEvidenceItem] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    stats: dict[str, Any] = Field(default_factory=dict)
