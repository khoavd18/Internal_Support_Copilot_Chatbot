from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Sequence

from src.agent.route_config import (
    DEFAULT_ROUTE_REASON,
    DEFAULT_SUPERVISOR_AGENT,
    MIN_QUESTION_COMPACT_LENGTH,
    RETRIEVE_ONLY_ROUTE_RULES,
    SUPERVISOR_DOMAIN_ROUTE_DEFINITIONS,
    RouteName,
    SupervisorAgentName,
)


@dataclass(frozen=True)
class NormalizedQuestion:
    raw: str
    lower: str
    compact: str


@dataclass(frozen=True)
class RouteMatch:
    route: RouteName
    rule_name: str
    score: int
    matched_terms: list[str] = field(default_factory=list)
    reason: str = ""
    priority: int = 100


@dataclass(frozen=True)
class RouteDecision:
    route: RouteName
    reason: str
    score: int = 0
    matched_signals: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SupervisorAgentMatch:
    agent_name: SupervisorAgentName
    score: int
    matched_keywords: list[str] = field(default_factory=list)
    description: str = ""
    priority: int = 100


@dataclass(frozen=True)
class SupervisorSelection:
    selected_agents: list[SupervisorAgentName]
    reason: str
    scores: Dict[SupervisorAgentName, int] = field(default_factory=dict)
    matched_keywords: Dict[SupervisorAgentName, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class SupervisorDispatchDecision:
    response_route: RouteName
    route_reason: str


RETRIEVE_ONLY_SIGNALS = [
    signal
    for rule in RETRIEVE_ONLY_ROUTE_RULES
    for signal in rule.signals
]


def normalize_question(question: str) -> NormalizedQuestion:
    raw = (question or "").strip()
    return NormalizedQuestion(
        raw=raw,
        lower=raw.lower(),
        compact=re.sub(r"\s+", "", raw),
    )


def _match_terms(question_lower: str, terms: Sequence[str]) -> list[str]:
    matched: list[str] = []
    for term in terms:
        normalized_term = str(term or "").strip().lower()
        if not normalized_term:
            continue
        escaped_term = re.escape(normalized_term).replace(r"\ ", r"\s+")
        pattern = rf"(?<!\w){escaped_term}(?!\w)"
        if re.search(pattern, question_lower) and normalized_term not in matched:
            matched.append(normalized_term)
    return matched


def match_route_rules(question: str) -> list[RouteMatch]:
    normalized = normalize_question(question)
    matches: list[RouteMatch] = []

    for rule in RETRIEVE_ONLY_ROUTE_RULES:
        matched_terms = _match_terms(normalized.lower, rule.signals)
        if not matched_terms:
            continue
        matches.append(
            RouteMatch(
                route=rule.route,
                rule_name=rule.name,
                score=len(matched_terms),
                matched_terms=matched_terms,
                reason=rule.reason,
                priority=rule.priority,
            )
        )

    return matches


def score_route_matches(matches: Sequence[RouteMatch]) -> list[RouteMatch]:
    return sorted(
        matches,
        key=lambda item: (-item.score, item.priority, item.rule_name),
    )


def decide_route(question: str) -> RouteDecision:
    normalized = normalize_question(question)

    if not normalized.raw:
        return RouteDecision(
            route="clarify",
            reason="Question is empty.",
        )

    if len(normalized.compact) < MIN_QUESTION_COMPACT_LENGTH:
        return RouteDecision(
            route="clarify",
            reason="Question is too short to retrieve evidence reliably.",
        )

    ranked_matches = score_route_matches(match_route_rules(normalized.raw))
    if ranked_matches:
        best = ranked_matches[0]
        return RouteDecision(
            route=best.route,
            reason=best.reason,
            score=best.score,
            matched_signals=best.matched_terms,
        )

    return RouteDecision(
        route="answer_from_kb",
        reason=DEFAULT_ROUTE_REASON,
    )


def decide_route_for_mode(question: str, mode: str) -> RouteDecision:
    normalized_mode = str(mode or "auto").strip().lower()
    if normalized_mode == "answer":
        return RouteDecision(
            route="answer_from_kb",
            reason="Mode was explicitly forced to answer.",
        )
    if normalized_mode == "search":
        return RouteDecision(
            route="retrieve_only",
            reason="Mode was explicitly forced to search.",
        )
    return decide_route(question)


def match_supervisor_agents(question: str) -> list[SupervisorAgentMatch]:
    normalized = normalize_question(question)
    matches: list[SupervisorAgentMatch] = []

    for definition in SUPERVISOR_DOMAIN_ROUTE_DEFINITIONS:
        matched_keywords = _match_terms(normalized.lower, definition.keywords)
        if not matched_keywords:
            continue
        matches.append(
            SupervisorAgentMatch(
                agent_name=definition.agent_name,
                score=len(matched_keywords),
                matched_keywords=matched_keywords,
                description=definition.description,
                priority=definition.priority,
            )
        )

    return matches


def score_supervisor_agent_matches(
    matches: Sequence[SupervisorAgentMatch],
) -> list[SupervisorAgentMatch]:
    return sorted(
        matches,
        key=lambda item: (-item.score, item.priority, item.agent_name),
    )


def select_supervisor_agents(question: str) -> SupervisorSelection:
    normalized = normalize_question(question)
    ranked_matches = score_supervisor_agent_matches(match_supervisor_agents(normalized.raw))

    if not ranked_matches:
        if not normalized.raw:
            reason = "Question is empty, so no domain could be selected reliably."
        else:
            reason = "No clear domain matched; defaulting to GitHub Docs."
        return SupervisorSelection(
            selected_agents=[DEFAULT_SUPERVISOR_AGENT],
            reason=reason,
        )

    selected_agents = [match.agent_name for match in ranked_matches]
    return SupervisorSelection(
        selected_agents=selected_agents,
        reason=f"Supervisor selected agents: {', '.join(selected_agents)}",
        scores={match.agent_name: match.score for match in ranked_matches},
        matched_keywords={
            match.agent_name: list(match.matched_keywords) for match in ranked_matches
        },
    )


def decide_supervisor_dispatch(
    *,
    question: str,
    mode: str,
    selection: SupervisorSelection,
) -> SupervisorDispatchDecision:
    normalized = normalize_question(question)
    normalized_mode = str(mode or "auto").strip().lower()

    if not normalized.raw:
        return SupervisorDispatchDecision(
            response_route="clarify",
            route_reason="Question is empty.",
        )

    if normalized_mode == "search":
        return SupervisorDispatchDecision(
            response_route="retrieve_only",
            route_reason=f"Mode was explicitly forced to search. {selection.reason}",
        )

    if normalized_mode == "answer":
        return SupervisorDispatchDecision(
            response_route="answer_from_kb",
            route_reason=f"Mode was explicitly forced to answer. {selection.reason}",
        )

    if len(normalized.compact) < MIN_QUESTION_COMPACT_LENGTH:
        return SupervisorDispatchDecision(
            response_route="clarify",
            route_reason="Question is too short to retrieve evidence reliably.",
        )

    return SupervisorDispatchDecision(
        response_route="answer_from_kb",
        route_reason=selection.reason,
    )


__all__ = [
    "DEFAULT_ROUTE_REASON",
    "MIN_QUESTION_COMPACT_LENGTH",
    "RETRIEVE_ONLY_SIGNALS",
    "RouteDecision",
    "RouteMatch",
    "RouteName",
    "SupervisorAgentMatch",
    "SupervisorDispatchDecision",
    "SupervisorSelection",
    "decide_route",
    "decide_route_for_mode",
    "decide_supervisor_dispatch",
    "match_route_rules",
    "match_supervisor_agents",
    "normalize_question",
    "score_route_matches",
    "score_supervisor_agent_matches",
    "select_supervisor_agents",
]
