from __future__ import annotations

from src.agent.graph.nodes.route import route_supervisor_node
from src.agent.router import (
    decide_route,
    decide_route_for_mode,
    decide_supervisor_dispatch,
    match_route_rules,
    select_supervisor_agents,
)


def test_single_agent_route_detects_retrieve_only_signal():
    matches = match_route_rules("Show sources for passkey docs")
    decision = decide_route("Show sources for passkey docs")

    assert matches
    assert matches[0].route == "retrieve_only"
    assert "show sources" in matches[0].matched_terms
    assert decision.route == "retrieve_only"
    assert "show sources" in decision.matched_signals


def test_single_agent_route_prefers_clarify_for_short_query_even_with_signal():
    decision = decide_route("source?")

    assert decision.route == "clarify"
    assert "too short" in decision.reason.lower()


def test_single_agent_route_mode_override_preserves_behavior():
    assert decide_route_for_mode("How do I sign in with a passkey?", "answer").route == "answer_from_kb"
    assert decide_route_for_mode("How do I sign in with a passkey?", "search").route == "retrieve_only"
    assert decide_route_for_mode("How do I sign in with a passkey?", "auto").route == "answer_from_kb"


def test_supervisor_selection_handles_ambiguous_query_with_stable_order():
    selection = select_supervisor_agents(
        "GitHub auth handbook issue for repository onboarding approval"
    )

    assert selection.selected_agents == ["github_docs", "gitlab", "issues"]
    assert selection.scores["github_docs"] >= 1
    assert selection.scores["gitlab"] >= 1
    assert selection.scores["issues"] >= 1
    assert "github" in selection.matched_keywords["github_docs"]
    assert "handbook" in selection.matched_keywords["gitlab"]
    assert "issue" in selection.matched_keywords["issues"]


def test_supervisor_dispatch_defaults_and_short_query_edge_cases():
    empty_selection = select_supervisor_agents("")
    empty_dispatch = decide_supervisor_dispatch(
        question="",
        mode="auto",
        selection=empty_selection,
    )

    default_selection = select_supervisor_agents("need help")
    default_dispatch = decide_supervisor_dispatch(
        question="need help",
        mode="auto",
        selection=default_selection,
    )

    short_selection = select_supervisor_agents("gitlab?")
    short_dispatch = decide_supervisor_dispatch(
        question="gitlab?",
        mode="auto",
        selection=short_selection,
    )

    assert empty_selection.selected_agents == ["github_docs"]
    assert empty_dispatch.response_route == "clarify"
    assert default_selection.selected_agents == ["github_docs"]
    assert default_dispatch.response_route == "answer_from_kb"
    assert short_selection.selected_agents == ["gitlab"]
    assert short_dispatch.response_route == "clarify"


def test_route_supervisor_node_exposes_scores_and_matches():
    result = route_supervisor_node(
        {
            "question": "GitHub auth handbook issue for repository onboarding approval",
            "mode": "auto",
        }
    )

    assert result["selected_agents"] == ["github_docs", "gitlab", "issues"]
    assert result["response_route"] == "answer_from_kb"
    assert result["route_scores"]["github_docs"] >= 1
    assert "github" in result["route_matches"]["github_docs"]


def test_route_supervisor_node_respects_forced_search_mode():
    result = route_supervisor_node(
        {
            "question": "How do I sign in with a passkey?",
            "mode": "search",
        }
    )

    assert result["selected_agents"] == ["github_docs"]
    assert result["response_route"] == "retrieve_only"
    assert "forced to search" in result["route_reason"].lower()
