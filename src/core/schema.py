from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="End-user question")
    debug: bool = Field(default=False, description="Whether to include retrieval debug metadata")
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Optional top_k override",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Chat session identifier for short-term conversation context",
    )


class SourceItem(BaseModel):
    index: int
    title: str
    source: str
    path: str = ""
    url: str = ""
    doc_id: Optional[str] = None
    source_type: Optional[str] = None
    rerank_score: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)
    debug: List[Dict[str, Any]] = Field(default_factory=list)


class AgentToolCall(BaseModel):
    tool_name: str
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    status: Literal["ok", "error", "skipped"] = "ok"
    note: str = ""


class AgentMeta(BaseModel):
    route: Literal["answer_from_kb", "retrieve_only", "clarify", "action"]
    reason: str
    tool_calls: List[AgentToolCall] = Field(default_factory=list)


class AgentAskRequest(AskRequest):
    mode: Literal["auto", "answer", "search"] = Field(
        default="auto",
        description="auto=agent decides route, answer=force answer generation, search=return sources only",
    )
    confirmed: bool = Field(
        default=False,
        description="Confirmation flag for write actions such as repository creation or commit",
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        description="Optional idempotency key for write actions detected from chat input",
    )


class CreateRepoRequest(BaseModel):
    org: str = Field(..., min_length=1, description="Target GitHub organization")
    name: str = Field(..., min_length=1, description="Repository name")
    description: str = Field(default="", description="Short repository description")
    private: bool = Field(default=True, description="Whether the repository should be private")
    auto_init: bool = Field(default=False, description="Whether to initialize the repository with a first commit")
    confirmed: bool = Field(default=False, description="Confirmation flag for repository creation")
    idempotency_key: Optional[str] = Field(
        default=None,
        description="Optional idempotency key for safe retries of repository creation",
    )


class CreateIssueRequest(BaseModel):
    repo_full_name: str = Field(..., min_length=3, description="Target repository in owner/repo format")
    title: str = Field(..., min_length=1, description="Issue title")
    body: str = Field(..., min_length=1, description="Issue body")
    labels: List[str] = Field(default_factory=list, description="Optional labels to attach to the issue")
    assignees: List[str] = Field(default_factory=list, description="Optional assignees for the issue")
    confirmed: bool = Field(default=False, description="Confirmation flag for issue creation")
    idempotency_key: Optional[str] = Field(
        default=None,
        description="Optional idempotency key for safe retries of issue creation",
    )


class CommitRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Commit message")
    repo_path: Optional[str] = Field(default=None, description="Local repository path, defaults to the project root")
    paths: List[str] = Field(default_factory=list, description="Specific paths to stage before committing")
    stage_all: bool = Field(
        default=False,
        description="Stage all tracked changes, or all files when include_untracked is also true",
    )
    include_untracked: bool = Field(default=False, description="Whether stage_all should include new files")
    confirmed: bool = Field(default=False, description="Confirmation flag for commit creation")
    idempotency_key: Optional[str] = Field(
        default=None,
        description="Optional idempotency key for safe retries of commit creation",
    )


class AgentResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)
    debug: List[Dict[str, Any]] = Field(default_factory=list)
    agent: AgentMeta
