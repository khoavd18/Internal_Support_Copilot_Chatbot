from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from src.agent.tools import search_github_issues


class IssuesState(TypedDict):
    question: str
    top_k: NotRequired[int]
    debug: NotRequired[bool]

    result: NotRequired[Dict[str, Any]]


def retrieve_issues_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = (state.get("question") or "").strip()
    top_k = int(state.get("top_k", 4))

    result = search_github_issues(
        query=question,
        top_k=top_k,
    )

    return {"result": result}


@lru_cache(maxsize=1)
def get_issues_graph():
    builder = StateGraph(IssuesState)

    builder.add_node("retrieve_issues", retrieve_issues_node)

    builder.add_edge(START, "retrieve_issues")
    builder.add_edge("retrieve_issues", END)

    return builder.compile()