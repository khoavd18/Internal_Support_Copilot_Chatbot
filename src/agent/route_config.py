from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


RouteName = Literal["answer_from_kb", "retrieve_only", "clarify"]
SupervisorAgentName = Literal["github_docs", "gitlab", "issues"]

MIN_QUESTION_COMPACT_LENGTH = 8
DEFAULT_ROUTE_REASON = "Defaulting to a synthesized answer from the knowledge base."
DEFAULT_SUPERVISOR_AGENT: SupervisorAgentName = "github_docs"


@dataclass(frozen=True)
class RetrieveOnlyRouteRule:
    name: str
    route: RouteName
    signals: tuple[str, ...]
    reason: str
    priority: int = 100


@dataclass(frozen=True)
class SupervisorDomainRouteDefinition:
    agent_name: SupervisorAgentName
    keywords: tuple[str, ...]
    description: str
    priority: int = 100


RETRIEVE_ONLY_ROUTE_RULES: tuple[RetrieveOnlyRouteRule, ...] = (
    RetrieveOnlyRouteRule(
        name="source_listing",
        route="retrieve_only",
        signals=(
            "cho tôi source",
            "cho tôi nguồn",
            "liệt kê tài liệu",
            "liệt kê source",
            "cho toi source",
            "cho toi nguon",
            "liet ke tai lieu",
            "liet ke source",
            "show source",
            "show sources",
            "related docs",
            "tài liệu liên quan",
            "nguồn liên quan",
            "tai lieu lien quan",
            "nguon lien quan",
            "debug retrieval",
        ),
        reason="The user appears to want related sources instead of a synthesized answer.",
        priority=10,
    ),
)


SUPERVISOR_DOMAIN_ROUTE_DEFINITIONS: tuple[SupervisorDomainRouteDefinition, ...] = (
    SupervisorDomainRouteDefinition(
        agent_name="github_docs",
        keywords=(
            "github",
            "git",
            "passkey",
            "ssh",
            "https",
            "push",
            "pull request",
            "repo",
            "repository",
            "actions",
            "workflow",
            "auth",
            "authentication",
        ),
        description="GitHub Docs and platform how-to guidance.",
        priority=10,
    ),
    SupervisorDomainRouteDefinition(
        agent_name="gitlab",
        keywords=(
            "gitlab",
            "handbook",
            "policy",
            "process",
            "guideline",
            "onboarding",
            "internal",
            "team",
            "approval",
        ),
        description="GitLab handbook and internal process guidance.",
        priority=20,
    ),
    SupervisorDomainRouteDefinition(
        agent_name="issues",
        keywords=(
            "issue",
            "bug",
            "error",
            "lỗi",
            "không chạy",
            "workaround",
            "known issue",
            "discussion",
            "problem",
        ),
        description="Issue-style troubleshooting, incidents, and known problems.",
        priority=30,
    ),
)
