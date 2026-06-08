from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field
from src.api.schemas.crm import CustomerSummaryRequest, CustomerSummaryResponse
from src.api.schemas.enterprise import (
    EnterpriseContextItem,
    GraphRAGAskRequest,
    GraphRAGAskResponse,
    GraphRAGEvidenceItem,
)
from src.api.schemas.support import (
    SlaCheckResponse,
    SuggestedReplyResponse,
    TicketAutomationRequest,
    TicketTriageResponse,
)

__all__ = [
    "AgentAskRequest",
    "AgentMeta",
    "AgentResponse",
    "AgentToolCall",
    "AskRequest",
    "AskResponse",
    "CommitRequest",
    "CreateIssueRequest",
    "CreateRepoRequest",
    "CustomerSummaryRequest",
    "CustomerSummaryResponse",
    "EnterpriseContextItem",
    "GraphRAGAskRequest",
    "GraphRAGAskResponse",
    "GraphRAGEvidenceItem",
    "SlaCheckResponse",
    "SourceItem",
    "SuggestedReplyResponse",
    "TicketAutomationRequest",
    "TicketTriageResponse",
]


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="End-user question")
    debug: bool = Field(default=False, description="Whether to include retrieval debug metadata")
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional top_k override",
    )
    session_id: str | None = Field(
        default=None,
        description="Chat session identifier for short-term conversation context",
    )


class SourceItem(BaseModel):
    index: int
    title: str
    source: str
    path: str = ""
    url: str = ""
    doc_id: str | None = None
    source_type: str | None = None
    rerank_score: float | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    debug: list[dict[str, Any]] = Field(default_factory=list)


class AgentToolCall(BaseModel):
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "error", "skipped"] = "ok"
    note: str = ""


class AgentMeta(BaseModel):
    route: Literal["answer_from_kb", "retrieve_only", "clarify", "action"]
    reason: str
    tool_calls: list[AgentToolCall] = Field(default_factory=list)


class AgentAskRequest(AskRequest):
    mode: Literal["auto", "answer", "search"] = Field(
        default="auto",
        description="auto=agent decides route, answer=force answer generation, search=return sources only",
    )
    confirmed: bool = Field(
        default=False,
        description="Confirmation flag for write actions such as repository creation or commit",
    )
    idempotency_key: str | None = Field(
        default=None,
        description="Optional idempotency key for write actions detected from chat input",
    )


class CreateRepoRequest(BaseModel):
    org: str = Field(..., min_length=1, description="Target GitHub organization")
    name: str = Field(..., min_length=1, description="Repository name")
    description: str = Field(default="", description="Short repository description")
    private: bool = Field(default=True, description="Whether the repository should be private")
    auto_init: bool = Field(
        default=False, description="Whether to initialize the repository with a first commit"
    )
    confirmed: bool = Field(default=False, description="Confirmation flag for repository creation")
    idempotency_key: str | None = Field(
        default=None,
        description="Optional idempotency key for safe retries of repository creation",
    )


class CreateIssueRequest(BaseModel):
    repo_full_name: str = Field(
        ..., min_length=3, description="Target repository in owner/repo format"
    )
    title: str = Field(..., min_length=1, description="Issue title")
    body: str = Field(..., min_length=1, description="Issue body")
    labels: list[str] = Field(
        default_factory=list, description="Optional labels to attach to the issue"
    )
    assignees: list[str] = Field(
        default_factory=list, description="Optional assignees for the issue"
    )
    confirmed: bool = Field(default=False, description="Confirmation flag for issue creation")
    idempotency_key: str | None = Field(
        default=None,
        description="Optional idempotency key for safe retries of issue creation",
    )


class CommitRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Commit message")
    repo_path: str | None = Field(
        default=None, description="Local repository path, defaults to the project root"
    )
    paths: list[str] = Field(
        default_factory=list, description="Specific paths to stage before committing"
    )
    stage_all: bool = Field(
        default=False,
        description="Stage all tracked changes, or all files when include_untracked is also true",
    )
    include_untracked: bool = Field(
        default=False, description="Whether stage_all should include new files"
    )
    confirmed: bool = Field(default=False, description="Confirmation flag for commit creation")
    idempotency_key: str | None = Field(
        default=None,
        description="Optional idempotency key for safe retries of commit creation",
    )


class AgentResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)
    debug: list[dict[str, Any]] = Field(default_factory=list)
    agent: AgentMeta
