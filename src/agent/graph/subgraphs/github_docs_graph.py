from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph
from typing_extensions import NotRequired, TypedDict

from src.agent.tools import search_github_docs


class GitHubDocsState(TypedDict):
    question: str
    top_k: NotRequired[int]
    debug: NotRequired[bool]

    result: NotRequired[Dict[str, Any]]


def retrieve_github_docs_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = (state.get("question") or "").strip()
    top_k = int(state.get("top_k", 4))

    result = search_github_docs(
        query=question,
        top_k=top_k,
    )

    return {"result": result}


@lru_cache(maxsize=1)
def get_github_docs_graph():
    builder = StateGraph(GitHubDocsState)

    builder.add_node("retrieve_github_docs", retrieve_github_docs_node)

    builder.add_edge(START, "retrieve_github_docs")
    builder.add_edge("retrieve_github_docs", END)

    return builder.compile()