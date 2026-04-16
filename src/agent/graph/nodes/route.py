from __future__ import annotations

import logging
from typing import Any, Dict

from src.agent.router import decide_supervisor_dispatch, select_supervisor_agents

logger = logging.getLogger(__name__)


def _log_route_decision(
    *,
    selected_agents: list[str],
    response_route: str,
    route_reason: str,
    question_length: int,
    route_scores: Dict[str, int],
    route_matches: Dict[str, list[str]],
) -> None:
    logger.info(
        "Supervisor route decided",
        extra={
            "event": "supervisor.route.decided",
            "selected_agents": selected_agents,
            "response_route": response_route,
            "route_reason": route_reason,
            "question_length": question_length,
            "route_scores": route_scores,
            "route_matches": route_matches,
        },
    )


def route_supervisor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    question = (state.get("effective_question") or state.get("question") or "").strip()
    mode = str(state.get("mode") or "auto").strip().lower()

    selection = select_supervisor_agents(question)
    dispatch = decide_supervisor_dispatch(
        question=question,
        mode=mode,
        selection=selection,
    )

    result = {
        "selected_agents": selection.selected_agents,
        "route_reason": dispatch.route_reason,
        "response_route": dispatch.response_route,
        "route_scores": selection.scores,
        "route_matches": selection.matched_keywords,
    }
    _log_route_decision(
        selected_agents=result["selected_agents"],
        response_route=result["response_route"],
        route_reason=result["route_reason"],
        question_length=len(question),
        route_scores=result["route_scores"],
        route_matches=result["route_matches"],
    )
    return result
