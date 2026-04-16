from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from src.agent import actions as action_handlers
from src.persistence.action_store import ActionStore


ActionMatcher = Callable[[str, str], "ActionIntent | None"]
ActionExecutor = Callable[..., Dict[str, Any]]
ConfirmationRequirement = Callable[[], bool]


@dataclass(frozen=True)
class ActionIntent:
    action: str
    payload: Dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    needs_clarification: bool = False
    clarify_answer: str = ""


@dataclass(frozen=True)
class ActionDefinition:
    name: str
    matcher: ActionMatcher
    executor: ActionExecutor
    permission_name: str
    tool_name: str
    confirmation_required: ConfirmationRequirement
    description: str = ""


def _normalize_text(value: str) -> str:
    lowered = str(value or "").strip().lower()
    decomposed = unicodedata.normalize("NFD", lowered)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


CONFIRM_SIGNALS = {
    "y",
    "yes",
    "ok",
    "okay",
    "confirm",
    "confirmed",
    "xac nhan",
    "dong y",
    "thuc hien",
    "lam di",
    "run it",
    "do it",
}

CANCEL_SIGNALS = {
    "cancel",
    "stop",
    "abort",
    "khong",
    "khong dong y",
    "huy",
    "bo qua",
}


def normalize_action_text(value: str) -> str:
    return _normalize_text(value)


def is_confirmation_message(question: str) -> bool:
    normalized = _normalize_text(question)
    return normalized in CONFIRM_SIGNALS


def is_cancellation_message(question: str) -> bool:
    normalized = _normalize_text(question)
    return normalized in CANCEL_SIGNALS


def _extract_prefixed_quoted_value(question: str, prefixes: list[str]) -> str:
    for prefix in prefixes:
        pattern = re.compile(rf"{prefix}\s*[:=]\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
        match = pattern.search(question)
        if match:
            return match.group(1).strip()
    return ""


def _extract_prefixed_csv_values(question: str, prefixes: list[str]) -> List[str]:
    for prefix in prefixes:
        pattern = re.compile(
            rf"{prefix}\s*[:=]\s*([^\n]+?)(?=\s+[A-Za-z_][A-Za-z0-9_]*\s*[:=]|$)",
            re.IGNORECASE,
        )
        match = pattern.search(question)
        if match:
            return [item.strip() for item in match.group(1).split(",") if item.strip()]
    return []


def _extract_repo_full_name(question: str) -> str:
    prefixed = _extract_prefixed_quoted_value(question, ["repo", "repository"])
    if prefixed:
        return prefixed

    prefixed_pattern = re.compile(
        r"\b(?:repo|repository)\s*[:=]\s*([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
        re.IGNORECASE,
    )
    prefixed_match = prefixed_pattern.search(question)
    if prefixed_match:
        return prefixed_match.group(1).strip()

    repo_match = re.search(r"\b([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)\b", question)
    if not repo_match:
        return ""

    return f"{repo_match.group(1)}/{repo_match.group(2)}"


def _looks_like_commit_request(normalized: str) -> bool:
    if not any(signal in normalized for signal in ["commit", "tao commit"]):
        return False

    if normalized.startswith("commit"):
        return True

    return any(
        signal in normalized
        for signal in [
            "commit message",
            "commit msg",
            "message:",
            "msg:",
            "files:",
            "paths:",
            "stage all",
            "include untracked",
            "hay commit",
        ]
    )


def _strip_trailing_commit_flags(value: str) -> str:
    cleaned = str(value or "").strip()
    markers = [
        " stage all",
        " all changes",
        " include untracked",
        " tat ca thay doi",
        " kem file moi",
    ]
    lowered = cleaned.lower()
    cut_positions = [lowered.find(marker) for marker in markers if lowered.find(marker) >= 0]
    if not cut_positions:
        return cleaned
    return cleaned[: min(cut_positions)].strip()


def _match_create_repo(raw_question: str, normalized: str) -> ActionIntent | None:
    if not any(
        signal in normalized
        for signal in ["create repo", "create repository", "tao repo", "tao repository"]
    ):
        return None

    repo_full_name = _extract_repo_full_name(raw_question)
    if not repo_full_name:
        return ActionIntent(
            action="create_repo",
            needs_clarification=True,
            reason="Missing owner/repo pair for repository creation.",
            clarify_answer=(
                "De tao repo, hay dung cu phap ro rang nhu: "
                "`create repo my-org/my-new-repo private`."
            ),
        )

    description = _extract_prefixed_quoted_value(raw_question, ["description", "mo ta"])
    visibility_public = " public" in f" {normalized} "
    auto_init = any(signal in normalized for signal in ["auto init", "auto_init", "khoi tao"])

    return ActionIntent(
        action="create_repo",
        payload={
            "org": repo_full_name.split("/", 1)[0],
            "name": repo_full_name.split("/", 1)[1],
            "description": description,
            "private": not visibility_public,
            "auto_init": auto_init,
        },
        reason="Detected repository creation request.",
    )


def _match_create_issue(raw_question: str, normalized: str) -> ActionIntent | None:
    if not any(
        signal in normalized
        for signal in [
            "create issue",
            "open issue",
            "file issue",
            "tao issue",
            "mo issue",
        ]
    ):
        return None

    repo_full_name = _extract_repo_full_name(raw_question)
    title = _extract_prefixed_quoted_value(raw_question, ["title", "tieu de"])
    body = _extract_prefixed_quoted_value(raw_question, ["body", "mo ta", "description"])
    labels = _extract_prefixed_csv_values(raw_question, ["labels", "label"])
    assignees = _extract_prefixed_csv_values(raw_question, ["assignees", "assignee"])

    missing_fields = []
    if not repo_full_name:
        missing_fields.append("repo")
    if not title:
        missing_fields.append("title")
    if not body:
        missing_fields.append("body")

    if missing_fields:
        return ActionIntent(
            action="create_issue",
            needs_clarification=True,
            reason=f"Missing fields for issue creation: {', '.join(missing_fields)}.",
            clarify_answer=(
                "De tao issue, hay dung cu phap ro rang nhu: "
                "`create issue repo:owner/repo title:\"Bug login\" body:\"Mo ta loi\"`."
            ),
        )

    return ActionIntent(
        action="create_issue",
        payload={
            "repo_full_name": repo_full_name,
            "title": title,
            "body": body,
            "labels": labels,
            "assignees": assignees,
        },
        reason="Detected issue creation request.",
    )


def _match_commit(raw_question: str, normalized: str) -> ActionIntent | None:
    if not _looks_like_commit_request(normalized):
        return None

    message = _extract_prefixed_quoted_value(raw_question, ["message", "msg"])
    if not message:
        direct_commit = re.search(r"\bcommit\b\s*['\"]([^'\"]+)['\"]", raw_question, re.IGNORECASE)
        if direct_commit:
            message = direct_commit.group(1).strip()

    if not message:
        return ActionIntent(
            action="commit",
            needs_clarification=True,
            reason="Missing commit message.",
            clarify_answer=(
                "De commit, hay dung cu phap ro rang nhu: "
                "`commit message:\"feat: add repo automation\" files:src/api/main.py,src/agent/actions.py`."
            ),
        )

    raw_paths = _extract_prefixed_csv_values(raw_question, ["files", "paths"])
    paths = []
    if raw_paths:
        combined_paths = _strip_trailing_commit_flags(",".join(raw_paths))
        paths = [item.strip() for item in combined_paths.split(",") if item.strip()]

    stage_all = any(signal in normalized for signal in ["stage all", "all changes", "tat ca thay doi"])
    include_untracked = any(
        signal in normalized for signal in ["include untracked", "kem file moi"]
    )

    return ActionIntent(
        action="commit",
        payload={
            "message": message,
            "paths": paths,
            "stage_all": stage_all,
            "include_untracked": include_untracked,
        },
        reason="Detected commit request.",
    )


def _create_repo_requires_confirmation() -> bool:
    return bool(action_handlers.GITHUB_REQUIRE_CONFIRM_FOR_WRITE)


def _create_issue_requires_confirmation() -> bool:
    return bool(action_handlers.GITHUB_REQUIRE_CONFIRM_FOR_WRITE)


def _commit_requires_confirmation() -> bool:
    return bool(action_handlers.LOCAL_GIT_REQUIRE_CONFIRM_FOR_WRITE)


def _execute_create_repo(
    *,
    payload: Dict[str, Any],
    confirmed: bool,
    session_id: str | None = None,
    idempotency_key: str | None = None,
    action_store: ActionStore | None = None,
) -> Dict[str, Any]:
    return action_handlers.create_repo_action(
        org=str(payload.get("org") or "").strip(),
        name=str(payload.get("name") or "").strip(),
        description=str(payload.get("description") or "").strip(),
        private=bool(payload.get("private", True)),
        auto_init=bool(payload.get("auto_init", False)),
        confirmed=confirmed,
        session_id=session_id,
        idempotency_key=idempotency_key,
        action_store=action_store,
    )


def _execute_create_issue(
    *,
    payload: Dict[str, Any],
    confirmed: bool,
    session_id: str | None = None,
    idempotency_key: str | None = None,
    action_store: ActionStore | None = None,
) -> Dict[str, Any]:
    return action_handlers.create_issue_action(
        repo_full_name=str(payload.get("repo_full_name") or "").strip(),
        title=str(payload.get("title") or "").strip(),
        body=str(payload.get("body") or "").strip(),
        labels=list(payload.get("labels") or []),
        assignees=list(payload.get("assignees") or []),
        confirmed=confirmed,
        session_id=session_id,
        idempotency_key=idempotency_key,
        action_store=action_store,
    )


def _execute_commit(
    *,
    payload: Dict[str, Any],
    confirmed: bool,
    session_id: str | None = None,
    idempotency_key: str | None = None,
    action_store: ActionStore | None = None,
) -> Dict[str, Any]:
    return action_handlers.commit_changes_action(
        message=str(payload.get("message") or "").strip(),
        repo_path=payload.get("repo_path"),
        paths=list(payload.get("paths") or []),
        stage_all=bool(payload.get("stage_all", False)),
        include_untracked=bool(payload.get("include_untracked", False)),
        confirmed=confirmed,
        session_id=session_id,
        idempotency_key=idempotency_key,
        action_store=action_store,
    )


ACTION_DEFINITIONS: tuple[ActionDefinition, ...] = (
    ActionDefinition(
        name="create_repo",
        matcher=_match_create_repo,
        executor=_execute_create_repo,
        permission_name="create_repo",
        tool_name=action_handlers.ACTION_TOOL_NAMES["create_repo"],
        confirmation_required=_create_repo_requires_confirmation,
        description="Create a GitHub repository.",
    ),
    ActionDefinition(
        name="create_issue",
        matcher=_match_create_issue,
        executor=_execute_create_issue,
        permission_name="create_issue",
        tool_name=action_handlers.ACTION_TOOL_NAMES["create_issue"],
        confirmation_required=_create_issue_requires_confirmation,
        description="Create a GitHub issue.",
    ),
    ActionDefinition(
        name="commit",
        matcher=_match_commit,
        executor=_execute_commit,
        permission_name="commit",
        tool_name=action_handlers.ACTION_TOOL_NAMES["commit"],
        confirmation_required=_commit_requires_confirmation,
        description="Create a local git commit.",
    ),
)

ACTION_DEFINITIONS_BY_NAME: Mapping[str, ActionDefinition] = {
    definition.name: definition for definition in ACTION_DEFINITIONS
}


def list_action_definitions() -> tuple[ActionDefinition, ...]:
    return ACTION_DEFINITIONS


def get_action_definition(action_name: str) -> ActionDefinition:
    normalized = str(action_name or "").strip().lower()
    definition = ACTION_DEFINITIONS_BY_NAME.get(normalized)
    if definition is None:
        raise KeyError(f"Unknown action '{action_name}'.")
    return definition


def get_action_permission_name(action_name: str) -> str:
    return get_action_definition(action_name).permission_name


def get_action_tool_name(action_name: str) -> str:
    return get_action_definition(action_name).tool_name


def action_requires_confirmation(action_name: str) -> bool:
    return bool(get_action_definition(action_name).confirmation_required())


def detect_action_request(question: str) -> ActionIntent | None:
    raw_question = str(question or "").strip()
    if not raw_question:
        return None

    normalized = _normalize_text(raw_question)
    for definition in ACTION_DEFINITIONS:
        intent = definition.matcher(raw_question, normalized)
        if intent is not None:
            return intent
    return None


def execute_registered_action(
    action_name: str,
    *,
    payload: Dict[str, Any],
    confirmed: bool,
    session_id: Optional[str] = None,
    idempotency_key: str | None = None,
    action_store: ActionStore | None = None,
) -> Dict[str, Any]:
    definition = get_action_definition(action_name)
    return definition.executor(
        payload=dict(payload or {}),
        confirmed=confirmed,
        session_id=session_id,
        idempotency_key=idempotency_key,
        action_store=action_store,
    )


__all__ = [
    "ACTION_DEFINITIONS",
    "ACTION_DEFINITIONS_BY_NAME",
    "ActionDefinition",
    "ActionIntent",
    "CANCEL_SIGNALS",
    "CONFIRM_SIGNALS",
    "action_requires_confirmation",
    "detect_action_request",
    "execute_registered_action",
    "get_action_definition",
    "get_action_permission_name",
    "get_action_tool_name",
    "is_cancellation_message",
    "is_confirmation_message",
    "list_action_definitions",
    "normalize_action_text",
]
