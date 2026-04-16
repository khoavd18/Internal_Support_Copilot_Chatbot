from __future__ import annotations

from functools import lru_cache
import logging
from typing import Any, Dict

from langgraph.graph import END, START, StateGraph

from src.core.logging_utils import bind_log_context
from src.agent.graph.state import SupervisorState
from src.agent.graph.nodes.route import route_supervisor_node
from src.agent.graph.nodes.synthesize import merge_results_node, synthesize_answer_node
from src.agent.graph.subgraphs.github_docs_graph import get_github_docs_graph
from src.agent.graph.subgraphs.gitlab_graph import get_gitlab_graph
from src.agent.graph.subgraphs.issues_graph import get_issues_graph
from src.persistence.checkpoints import get_graph_checkpointer

logger = logging.getLogger(__name__)


def run_github_docs_node(state: Dict[str, Any]) -> Dict[str, Any]:
    selected_agents = state.get("selected_agents", [])
    if "github_docs" not in selected_agents:
        return {}

    with bind_log_context(agent_name="github_docs"):
        logger.info(
            "Invoking GitHub Docs subgraph",
            extra={"event": "supervisor.subgraph.started", "subgraph": "github_docs"},
        )
        graph = get_github_docs_graph()
        sub_result = graph.invoke(
            {
                "question": state.get("effective_question") or state.get("question") or "",
                "top_k": state.get("top_k", 4),
                "debug": state.get("debug", False),
            }
        )

        result = sub_result.get("result", {})
        logger.info(
            "GitHub Docs subgraph completed",
            extra={
                "event": "supervisor.subgraph.completed",
                "subgraph": "github_docs",
                "retrieved_docs": len(result.get("documents_raw", []) or []),
            },
        )

        return {
            "github_docs_result": {
                "agent_name": "github_docs",
                "status": "ok",
                "documents_raw": result.get("documents_raw", []),
                "documents": result.get("documents", []),
                "sources": result.get("sources", []),
                "debug": result.get("debug", []),
                "stats": result.get("stats", {}),
            }
        }


def run_gitlab_node(state: Dict[str, Any]) -> Dict[str, Any]:
    selected_agents = state.get("selected_agents", [])
    if "gitlab" not in selected_agents:
        return {}

    with bind_log_context(agent_name="gitlab"):
        logger.info(
            "Invoking GitLab subgraph",
            extra={"event": "supervisor.subgraph.started", "subgraph": "gitlab"},
        )
        graph = get_gitlab_graph()
        sub_result = graph.invoke(
            {
                "question": state.get("effective_question") or state.get("question") or "",
                "top_k": state.get("top_k", 4),
                "debug": state.get("debug", False),
            }
        )

        result = sub_result.get("result", {})
        logger.info(
            "GitLab subgraph completed",
            extra={
                "event": "supervisor.subgraph.completed",
                "subgraph": "gitlab",
                "retrieved_docs": len(result.get("documents_raw", []) or []),
            },
        )

        return {
            "gitlab_result": {
                "agent_name": "gitlab",
                "status": "ok",
                "documents_raw": result.get("documents_raw", []),
                "documents": result.get("documents", []),
                "sources": result.get("sources", []),
                "debug": result.get("debug", []),
                "stats": result.get("stats", {}),
            }
        }


def run_issues_node(state: Dict[str, Any]) -> Dict[str, Any]:
    selected_agents = state.get("selected_agents", [])
    if "issues" not in selected_agents:
        return {}

    with bind_log_context(agent_name="issues"):
        logger.info(
            "Invoking Issues subgraph",
            extra={"event": "supervisor.subgraph.started", "subgraph": "issues"},
        )
        graph = get_issues_graph()
        sub_result = graph.invoke(
            {
                "question": state.get("effective_question") or state.get("question") or "",
                "top_k": state.get("top_k", 4),
                "debug": state.get("debug", False),
            }
        )

        result = sub_result.get("result", {})
        logger.info(
            "Issues subgraph completed",
            extra={
                "event": "supervisor.subgraph.completed",
                "subgraph": "issues",
                "retrieved_docs": len(result.get("documents_raw", []) or []),
            },
        )

        return {
            "issues_result": {
                "agent_name": "issues",
                "status": "ok",
                "documents_raw": result.get("documents_raw", []),
                "documents": result.get("documents", []),
                "sources": result.get("sources", []),
                "debug": result.get("debug", []),
                "stats": result.get("stats", {}),
            }
        }


@lru_cache(maxsize=1)
def get_supervisor_graph():
    builder = StateGraph(SupervisorState)

    builder.add_node("route", route_supervisor_node)
    builder.add_node("run_github_docs", run_github_docs_node)
    builder.add_node("run_gitlab", run_gitlab_node)
    builder.add_node("run_issues", run_issues_node)
    builder.add_node("merge_results", merge_results_node)
    builder.add_node("synthesize_answer", synthesize_answer_node)

    builder.add_edge(START, "route")
    builder.add_edge("route", "run_github_docs")
    builder.add_edge("route", "run_gitlab")
    builder.add_edge("route", "run_issues")
    builder.add_edge(["run_github_docs", "run_gitlab", "run_issues"], "merge_results")
    builder.add_edge("merge_results", "synthesize_answer")
    builder.add_edge("synthesize_answer", END)

    checkpointer = get_graph_checkpointer()
    return builder.compile(checkpointer=checkpointer)
