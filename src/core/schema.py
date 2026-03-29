from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Câu hỏi từ người dùng")
    debug: bool = Field(default=False, description="Có trả thêm debug retrieval hay không")
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=10,
        description="Override top_k nếu muốn",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="ID phiên chat để giữ ngữ cảnh hội thoại ngắn hạn",
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
    route: Literal["answer_from_kb", "retrieve_only", "clarify"]
    reason: str
    tool_calls: List[AgentToolCall] = Field(default_factory=list)


class AgentAskRequest(AskRequest):
    mode: Literal["auto", "answer", "search"] = Field(
        default="auto",
        description="auto=agent tự chọn route, answer=ép trả lời, search=ép chỉ trả sources",
    )


class AgentResponse(BaseModel):
    answer: str
    sources: List[SourceItem] = Field(default_factory=list)
    stats: Dict[str, Any] = Field(default_factory=dict)
    debug: List[Dict[str, Any]] = Field(default_factory=list)
    agent: AgentMeta