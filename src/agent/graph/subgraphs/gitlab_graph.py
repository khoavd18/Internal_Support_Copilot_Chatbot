from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from src.agent.tools import search_gitlab_handbook


class GitLabState(TypedDict):
    question: str
    top_k: NotRequired[int]
    debug: NotRequired[bool]

    result: NotRequired[Dict[str, Any]]


def retrieve_gitlab_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = (state.get("question") or "").strip()
    top_k = int(state.get("top_k", 4))

    result = search_gitlab_handbook(
        query=question,
        top_k=top_k,
    )

    return {"result": result}


@lru_cache(maxsize=1)
def get_gitlab_graph():
    builder = StateGraph(GitLabState)

    builder.add_node("retrieve_gitlab", retrieve_gitlab_node)

    builder.add_edge(START, "retrieve_gitlab")
    builder.add_edge("retrieve_gitlab", END)

    return builder.compile()