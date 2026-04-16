from __future__ import annotations

from typing import Any, Dict, List, Literal
from typing_extensions import NotRequired, TypedDict


class WorkerResult(TypedDict, total=False):
    agent_name: str
    status: str
    answer: str
    sources: List[Dict[str, Any]]
    documents_raw: List[Any]
    documents: List[Dict[str, Any]]
    debug: List[Dict[str, Any]]
    stats: Dict[str, Any]


class SupervisorState(TypedDict):
    question: str
    session_id: NotRequired[str]
    mode: NotRequired[Literal["auto", "answer", "search"]]
    debug: NotRequired[bool]
    top_k: NotRequired[int]

    history: NotRequired[List[Dict[str, str]]]
    effective_question: NotRequired[str]

    selected_agents: NotRequired[List[str]]
    route_reason: NotRequired[str]
    response_route: NotRequired[Literal["answer_from_kb", "retrieve_only", "clarify"]]
    route_scores: NotRequired[Dict[str, int]]
    route_matches: NotRequired[Dict[str, List[str]]]

    github_docs_result: NotRequired[WorkerResult]
    gitlab_result: NotRequired[WorkerResult]
    issues_result: NotRequired[WorkerResult]

    merged_documents: NotRequired[List[Any]]
    merged_sources: NotRequired[List[Dict[str, Any]]]
    merged_debug: NotRequired[List[Dict[str, Any]]]

    answer: NotRequired[str]
    stats: NotRequired[Dict[str, Any]]
    agent: NotRequired[Dict[str, Any]]
