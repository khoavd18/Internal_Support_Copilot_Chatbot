from __future__ import annotations

import copy
import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

from src.core.settings import (
    GITHUB_REQUIRE_CONFIRM_FOR_WRITE,
    LOCAL_GIT_ACTIONS_ENABLED,
    LOCAL_GIT_REQUIRE_CONFIRM_FOR_WRITE,
)
from src.core.logging_utils import bind_log_context
from src.core.observability import increment_counter, observe_duration
from src.core.security import sanitize_error_text
from src.integrations.github_client import GitHubClient, GitHubClientError
from src.integrations.local_git_client import LocalGitClient, LocalGitClientError
from src.persistence.action_store import ActionRecord, ActionStore, get_action_store

logger = logging.getLogger(__name__)


ACTION_TOOL_NAMES = {
    "create_repo": "create_organization_repository",
    "create_issue": "create_issue",
    "commit": "commit_local_changes",
}

TERMINAL_ACTION_STATUSES = {"succeeded", "failed", "cancelled"}
UNORDERED_LIST_FIELDS = {"assignees", "labels", "paths"}
ALLOWED_TRANSITIONS = {
    "pending": {"confirmed", "cancelled"},
    "confirmed": {"running", "cancelled"},
    "running": {"succeeded", "failed"},
}


class AgentActionError(RuntimeError):
    """Raised when an action request should be rejected or clarified."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _tool_name_for_action(action_name: str) -> str:
    return ACTION_TOOL_NAMES.get(action_name, action_name)


def _action_log_fields(record: ActionRecord) -> Dict[str, Any]:
    payload = record.payload or {}
    fields: Dict[str, Any] = {
        "action_name": record.action_name,
        "status": record.status,
        "attempt_count": record.attempt_count,
        "confirmation_required": record.confirmation_required,
    }
    if record.action_name == "create_repo":
        fields["org"] = payload.get("org", "")
        fields["repo_name"] = payload.get("name", "")
    elif record.action_name == "create_issue":
        fields["repo"] = payload.get("repo_full_name", "")
        fields["title_length"] = len(str(payload.get("title") or ""))
        fields["labels_count"] = len(payload.get("labels", []) or [])
        fields["assignees_count"] = len(payload.get("assignees", []) or [])
    elif record.action_name == "commit":
        fields["repo_path"] = payload.get("repo_path", "")
        fields["paths_count"] = len(payload.get("paths", []) or [])
        fields["stage_all"] = bool(payload.get("stage_all", False))
        fields["include_untracked"] = bool(payload.get("include_untracked", False))
    return fields


def _build_action_response(
    *,
    answer: str,
    reason: str,
    tool_name: str,
    tool_input: Dict[str, Any],
    stats: Dict[str, Any],
    tool_status: str = "ok",
    tool_note: str = "Action executed successfully.",
    action_status: str = "ok",
) -> Dict[str, Any]:
    return {
        "answer": answer,
        "sources": [],
        "stats": {
            "backend_mode": "multi_agent",
            "action_status": action_status,
            **stats,
        },
        "debug": [],
        "agent": {
            "route": "action",
            "reason": reason,
            "tool_calls": [
                {
                    "tool_name": tool_name,
                    "tool_input": tool_input,
                    "status": tool_status,
                    "note": tool_note,
                }
            ],
        },
    }


def _normalize_payload_value(key: str, value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        normalized: Dict[str, Any] = {}
        for child_key, child_value in value.items():
            cleaned = _normalize_payload_value(str(child_key), child_value)
            if cleaned in (None, "", [], {}):
                continue
            normalized[str(child_key)] = cleaned
        return normalized

    if isinstance(value, (list, tuple)):
        normalized_items = []
        for item in value:
            cleaned = _normalize_payload_value(key, item)
            if cleaned in (None, "", [], {}):
                continue
            normalized_items.append(cleaned)
        if key in UNORDERED_LIST_FIELDS:
            unique_items = []
            seen = set()
            for item in normalized_items:
                item_key = json.dumps(item, ensure_ascii=False, sort_keys=True)
                if item_key in seen:
                    continue
                seen.add(item_key)
                unique_items.append(item)
            return sorted(
                unique_items,
                key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
            )
        return normalized_items

    return value


def normalize_action_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in dict(payload or {}).items():
        cleaned = _normalize_payload_value(str(key), value)
        if cleaned in (None, "", [], {}):
            continue
        normalized[str(key)] = cleaned
    return normalized


def build_action_idempotency_key(
    *,
    action_name: str,
    payload: Dict[str, Any],
    session_id: str | None = None,
    idempotency_key: str | None = None,
) -> str:
    scope = f"session:{str(session_id or '').strip()}" if session_id else "global"
    prefix = f"{scope}:{action_name}:"
    explicit_key = str(idempotency_key or "").strip()
    if explicit_key:
        if explicit_key.startswith(prefix) or explicit_key.startswith("session:") or explicit_key.startswith("global:"):
            return explicit_key
        base_key = explicit_key
    else:
        fingerprint = json.dumps(
            {
                "action": action_name,
                "payload": normalize_action_payload(payload),
                "scope": scope,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        base_key = hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
    return f"{prefix}{base_key}"


def _clone_record(record: ActionRecord) -> ActionRecord:
    return ActionRecord.from_dict(record.to_dict())


def _with_action_record_metadata(
    result: Dict[str, Any],
    record: ActionRecord,
    *,
    idempotent_replay: bool = False,
) -> Dict[str, Any]:
    normalized = copy.deepcopy(result)
    stats = dict(normalized.get("stats", {}))
    stats.update(
        {
            "action_id": record.action_id,
            "idempotency_key": record.idempotency_key,
            "action_record_status": record.status,
            "action_attempt_count": record.attempt_count,
            "idempotent_replay": idempotent_replay,
        }
    )
    normalized["stats"] = stats
    return normalized


def prepare_action_record(
    *,
    action_name: str,
    payload: Dict[str, Any],
    reason: str = "",
    session_id: str | None = None,
    idempotency_key: str | None = None,
    confirmation_required: bool = False,
    action_store: Optional[ActionStore] = None,
) -> ActionRecord:
    store = action_store or get_action_store()
    normalized_payload = normalize_action_payload(payload)
    resolved_idempotency_key = build_action_idempotency_key(
        action_name=action_name,
        payload=normalized_payload,
        session_id=session_id,
        idempotency_key=idempotency_key,
    )
    record = ActionRecord(
        action_id=uuid4().hex,
        action_name=action_name,
        idempotency_key=resolved_idempotency_key,
        payload=normalized_payload,
        session_id=str(session_id or "").strip(),
        reason=str(reason or "").strip(),
        confirmation_required=confirmation_required,
    )
    resolved = store.create_or_get(record)
    logger.info(
        "Prepared action record",
        extra={
            "event": "action.record.prepared",
            "action_id": resolved.action_id,
            "session_id": resolved.session_id,
            "idempotency_reused": resolved.action_id != record.action_id,
            **_action_log_fields(resolved),
        },
    )
    return resolved


def get_action_record(
    action_id: str,
    *,
    action_store: Optional[ActionStore] = None,
) -> Optional[ActionRecord]:
    store = action_store or get_action_store()
    return store.get_action(action_id)


def _transition_action_record(
    record: ActionRecord,
    new_status: str,
    *,
    action_store: Optional[ActionStore] = None,
    last_error: str = "",
    result: Optional[Dict[str, Any]] = None,
    side_effect: Optional[Dict[str, Any]] = None,
) -> ActionRecord:
    store = action_store or get_action_store()
    previous_status = record.status
    if new_status == record.status:
        updated = _clone_record(record)
    else:
        allowed = ALLOWED_TRANSITIONS.get(record.status, set())
        if new_status not in allowed:
            raise AgentActionError(
                f"Invalid action state transition: {record.status} -> {new_status}."
            )
        updated = _clone_record(record)
        updated.status = new_status

    now = _utc_now()
    updated.updated_at = now

    if new_status == "confirmed":
        updated.confirmed_at = updated.confirmed_at or now
    elif new_status == "running":
        updated.confirmed_at = updated.confirmed_at or now
        updated.started_at = now
        if record.status != "running":
            updated.attempt_count = max(updated.attempt_count, 0) + 1

    if new_status in TERMINAL_ACTION_STATUSES:
        updated.completed_at = now

    if new_status == "succeeded":
        updated.last_error = ""
    elif last_error:
        updated.last_error = str(last_error).strip()

    if result is not None:
        updated.result = copy.deepcopy(result)
    if side_effect is not None:
        updated.side_effect = copy.deepcopy(side_effect)

    saved = store.save(updated)
    logger.info(
        "Action state transitioned",
        extra={
            "event": "action.state.transition",
            "action_id": saved.action_id,
            "from_status": previous_status,
            "to_status": saved.status,
            "session_id": saved.session_id,
            **_action_log_fields(saved),
        },
    )
    return saved


def cancel_action_record(
    action_id: str,
    *,
    reason: str = "Cancelled by user.",
    action_store: Optional[ActionStore] = None,
) -> Optional[ActionRecord]:
    record = get_action_record(action_id, action_store=action_store)
    if record is None:
        return None
    if record.status in {"pending", "confirmed"}:
        return _transition_action_record(
            record,
            "cancelled",
            action_store=action_store,
            last_error=reason,
        )
    return record


def replay_action_record(record: ActionRecord) -> Dict[str, Any]:
    if record.result:
        return _with_action_record_metadata(record.result, record, idempotent_replay=True)
    return build_cancelled_action_response(record)


def build_in_progress_action_response(record: ActionRecord) -> Dict[str, Any]:
    return _with_action_record_metadata(
        _build_action_response(
            answer=(
                f"Action `{record.action_name}` is already running or resuming from a previously "
                "confirmed request. It will not be re-executed automatically for the same "
                "idempotency key."
            ),
            reason="Idempotency guard prevented duplicate execution.",
            tool_name=_tool_name_for_action(record.action_name),
            tool_input=dict(record.payload),
            stats={
                "action": record.action_name,
            },
            tool_status="skipped",
            tool_note="Existing action record is already in progress.",
            action_status="in_progress",
        ),
        record,
    )


def build_failed_action_response(record: ActionRecord) -> Dict[str, Any]:
    return _with_action_record_metadata(
        _build_action_response(
            answer=(
                f"Action `{record.action_name}` previously failed and will not be retried "
                "automatically for the same idempotency key."
            ),
            reason="Previous action attempt failed.",
            tool_name=_tool_name_for_action(record.action_name),
            tool_input=dict(record.payload),
            stats={
                "action": record.action_name,
            },
            tool_status="error",
            tool_note="Previous attempt failed. Review server logs using the action_id for details.",
            action_status="error",
        ),
        record,
    )


def build_cancelled_action_response(record: ActionRecord) -> Dict[str, Any]:
    return _with_action_record_metadata(
        _build_action_response(
            answer=f"Action `{record.action_name}` was cancelled and will not be executed.",
            reason="Pending action was cancelled.",
            tool_name=_tool_name_for_action(record.action_name),
            tool_input=dict(record.payload),
            stats={
                "action": record.action_name,
            },
            tool_status="skipped",
            tool_note="Action record is cancelled.",
            action_status="cancelled",
        ),
        record,
    )


def _begin_action_execution(
    *,
    action_name: str,
    payload: Dict[str, Any],
    confirmed: bool,
    confirmation_required: bool,
    session_id: str | None = None,
    idempotency_key: str | None = None,
    reason: str = "",
    action_store: Optional[ActionStore] = None,
) -> tuple[str, ActionRecord]:
    record = prepare_action_record(
        action_name=action_name,
        payload=payload,
        reason=reason,
        session_id=session_id,
        idempotency_key=idempotency_key,
        confirmation_required=confirmation_required,
        action_store=action_store,
    )

    if record.status == "succeeded":
        logger.info(
            "Action replay requested for succeeded record",
            extra={"event": "action.execution.replay", "action_id": record.action_id, **_action_log_fields(record)},
        )
        return "replay", record
    if record.status == "running":
        logger.info(
            "Action execution skipped because record is already running",
            extra={"event": "action.execution.in_progress", "action_id": record.action_id, **_action_log_fields(record)},
        )
        return "in_progress", record
    if record.status == "failed":
        logger.warning(
            "Action execution skipped because previous attempt failed",
            extra={"event": "action.execution.failed_existing", "action_id": record.action_id, **_action_log_fields(record)},
        )
        return "failed", record
    if record.status == "cancelled":
        logger.info(
            "Action execution skipped because record is cancelled",
            extra={"event": "action.execution.cancelled_existing", "action_id": record.action_id, **_action_log_fields(record)},
        )
        return "cancelled", record

    if record.status == "pending":
        if confirmation_required and not confirmed:
            logger.info(
                "Action execution blocked pending confirmation",
                extra={"event": "action.execution.awaiting_confirmation", "action_id": record.action_id, **_action_log_fields(record)},
            )
            raise AgentActionError(
                f"Action '{action_name}' requires write confirmation. "
                "Send confirmed=true to allow execution."
            )
        record = _transition_action_record(record, "confirmed", action_store=action_store)

    if record.status == "confirmed":
        record = _transition_action_record(record, "running", action_store=action_store)
        logger.info(
            "Action execution starting",
            extra={"event": "action.execution.started", "action_id": record.action_id, **_action_log_fields(record)},
        )
        return "execute", record

    raise AgentActionError(f"Unsupported action record state '{record.status}'.")


def _finalize_action_success(
    record: ActionRecord,
    response: Dict[str, Any],
    *,
    side_effect: Dict[str, Any],
    action_store: Optional[ActionStore] = None,
) -> Dict[str, Any]:
    store = action_store or get_action_store()
    succeeded = _transition_action_record(
        record,
        "succeeded",
        action_store=store,
        result=response,
        side_effect=side_effect,
    )
    annotated = _with_action_record_metadata(succeeded.result or response, succeeded)
    succeeded.result = copy.deepcopy(annotated)
    store.save(succeeded)
    logger.info(
        "Action execution succeeded",
        extra={
            "event": "action.execution.succeeded",
            "action_id": succeeded.action_id,
            **_action_log_fields(succeeded),
            "side_effect": side_effect,
        },
    )
    increment_counter(
        "action.execution.total",
        attributes={
            "action_name": succeeded.action_name,
            "status": "succeeded",
        },
    )
    return annotated


def _finalize_action_failure(
    record: ActionRecord,
    error_message: str,
    *,
    action_store: Optional[ActionStore] = None,
) -> ActionRecord:
    safe_error_message = sanitize_error_text(error_message, max_length=240)
    failed = _transition_action_record(
        record,
        "failed",
        action_store=action_store,
        last_error=safe_error_message,
    )
    logger.warning(
        "Action execution failed",
        extra={
            "event": "action.execution.failed",
            "action_id": failed.action_id,
            "error_message": safe_error_message,
            **_action_log_fields(failed),
        },
    )
    increment_counter(
        "action.execution.total",
        attributes={
            "action_name": failed.action_name,
            "status": "failed",
        },
    )
    return failed


def _failed_action_message(record: ActionRecord) -> str:
    return (
        f"Action '{record.action_name}' already failed for this idempotency key and will not "
        f"be retried automatically. action_id={record.action_id}. "
        "Review server logs for detailed diagnostics."
    )


def _cancelled_action_message(record: ActionRecord) -> str:
    return (
        f"Action '{record.action_name}' was cancelled for this idempotency key and will not "
        f"be executed automatically. action_id={record.action_id}."
    )


def create_repo_action(
    *,
    org: str,
    name: str,
    description: str = "",
    private: bool = True,
    auto_init: bool = False,
    confirmed: bool = False,
    session_id: str | None = None,
    idempotency_key: str | None = None,
    github_client: Optional[GitHubClient] = None,
    action_store: Optional[ActionStore] = None,
) -> Dict[str, Any]:
    payload = {
        "org": org,
        "name": name,
        "description": description,
        "private": bool(private),
        "auto_init": bool(auto_init),
    }
    state, record = _begin_action_execution(
        action_name="create_repo",
        payload=payload,
        confirmed=confirmed,
        confirmation_required=GITHUB_REQUIRE_CONFIRM_FOR_WRITE,
        session_id=session_id,
        idempotency_key=idempotency_key,
        reason="Repository creation requested.",
        action_store=action_store,
    )

    if state == "replay":
        return replay_action_record(record)
    if state == "in_progress":
        return build_in_progress_action_response(record)
    if state == "failed":
        raise AgentActionError(_failed_action_message(record))
    if state == "cancelled":
        raise AgentActionError(_cancelled_action_message(record))

    client = github_client or GitHubClient()
    with bind_log_context(action_id=record.action_id, session_id=record.session_id):
        with observe_duration(
            "action.execute",
            metric_name="action.execution.duration_ms",
            metric_attributes={"action_name": "create_repo"},
            span_attributes={"action_name": "create_repo"},
        ) as observation:
            try:
                result = client.create_organization_repository(
                    org=org,
                    name=name,
                    description=description,
                    private=private,
                    auto_init=auto_init,
                )
            except GitHubClientError as exc:
                observation.record_exception(exc)
                observation.finish(status="failed")
                updated = _finalize_action_failure(record, str(exc), action_store=action_store)
                raise AgentActionError(_failed_action_message(updated)) from exc
            repo_full_name = result.get("full_name") or f"{org}/{name}"
            answer = f"Created repository {repo_full_name}. URL: {result.get('html_url', '')}".strip()
            response = _build_action_response(
                answer=answer,
                reason="Executed repository creation action.",
                tool_name="create_organization_repository",
                tool_input={
                    "org": org,
                    "name": name,
                    "private": private,
                    "auto_init": auto_init,
                },
                stats={
                    "action": "create_repo",
                    "repo": repo_full_name,
                    "repo_url": result.get("html_url"),
                    "repo_private": result.get("private"),
                    "default_branch": result.get("default_branch"),
                },
            )
            observation.set_attribute("repo", repo_full_name)
            observation.finish(status="succeeded")
            return _finalize_action_success(
                record,
                response,
                side_effect={
                    "repo": repo_full_name,
                    "repo_url": result.get("html_url"),
                },
                action_store=action_store,
            )


def create_issue_action(
    *,
    repo_full_name: str,
    title: str,
    body: str,
    labels: Optional[Iterable[str]] = None,
    assignees: Optional[Iterable[str]] = None,
    confirmed: bool = False,
    session_id: str | None = None,
    idempotency_key: str | None = None,
    github_client: Optional[GitHubClient] = None,
    action_store: Optional[ActionStore] = None,
) -> Dict[str, Any]:
    normalized_labels = [str(item).strip() for item in (labels or []) if str(item).strip()]
    normalized_assignees = [str(item).strip() for item in (assignees or []) if str(item).strip()]
    payload = {
        "repo_full_name": repo_full_name,
        "title": title,
        "body": body,
        "labels": normalized_labels,
        "assignees": normalized_assignees,
    }
    state, record = _begin_action_execution(
        action_name="create_issue",
        payload=payload,
        confirmed=confirmed,
        confirmation_required=GITHUB_REQUIRE_CONFIRM_FOR_WRITE,
        session_id=session_id,
        idempotency_key=idempotency_key,
        reason="Issue creation requested.",
        action_store=action_store,
    )

    if state == "replay":
        return replay_action_record(record)
    if state == "in_progress":
        return build_in_progress_action_response(record)
    if state == "failed":
        raise AgentActionError(_failed_action_message(record))
    if state == "cancelled":
        raise AgentActionError(_cancelled_action_message(record))

    client = github_client or GitHubClient()
    with bind_log_context(action_id=record.action_id, session_id=record.session_id):
        with observe_duration(
            "action.execute",
            metric_name="action.execution.duration_ms",
            metric_attributes={"action_name": "create_issue"},
            span_attributes={"action_name": "create_issue", "repo": repo_full_name},
        ) as observation:
            try:
                result = client.create_issue(
                    repo_full_name=repo_full_name,
                    title=title,
                    body=body,
                    labels=normalized_labels,
                    assignees=normalized_assignees,
                )
            except GitHubClientError as exc:
                observation.record_exception(exc)
                observation.finish(status="failed")
                updated = _finalize_action_failure(record, str(exc), action_store=action_store)
                raise AgentActionError(_failed_action_message(updated)) from exc
            issue_number = result.get("issue_number")
            answer = (
                f"Created issue #{issue_number} in {repo_full_name}. "
                f"URL: {result.get('html_url', '')}"
            ).strip()
            response = _build_action_response(
                answer=answer,
                reason="Executed GitHub issue creation action.",
                tool_name="create_issue",
                tool_input={
                    "repo_full_name": repo_full_name,
                    "title": title,
                    "labels": normalized_labels,
                    "assignees": normalized_assignees,
                },
                stats={
                    "action": "create_issue",
                    "repo": repo_full_name,
                    "issue_number": issue_number,
                    "issue_title": result.get("title"),
                    "issue_url": result.get("html_url"),
                    "issue_state": result.get("state"),
                },
            )
            observation.set_attribute("issue_number", issue_number)
            observation.finish(status="succeeded")
            return _finalize_action_success(
                record,
                response,
                side_effect={
                    "repo": repo_full_name,
                    "issue_number": issue_number,
                    "issue_url": result.get("html_url"),
                },
                action_store=action_store,
            )


def commit_changes_action(
    *,
    message: str,
    repo_path: Optional[str] = None,
    paths: Optional[Iterable[str]] = None,
    stage_all: bool = False,
    include_untracked: bool = False,
    confirmed: bool = False,
    session_id: str | None = None,
    idempotency_key: str | None = None,
    git_client: Optional[LocalGitClient] = None,
    action_store: Optional[ActionStore] = None,
) -> Dict[str, Any]:
    if not LOCAL_GIT_ACTIONS_ENABLED:
        raise AgentActionError("LOCAL_GIT_ACTIONS_ENABLED is disabled.")

    payload = {
        "message": message,
        "repo_path": repo_path,
        "paths": list(paths or []),
        "stage_all": bool(stage_all),
        "include_untracked": bool(include_untracked),
    }
    state, record = _begin_action_execution(
        action_name="commit",
        payload=payload,
        confirmed=confirmed,
        confirmation_required=LOCAL_GIT_REQUIRE_CONFIRM_FOR_WRITE,
        session_id=session_id,
        idempotency_key=idempotency_key,
        reason="Local git commit requested.",
        action_store=action_store,
    )

    if state == "replay":
        return replay_action_record(record)
    if state == "in_progress":
        return build_in_progress_action_response(record)
    if state == "failed":
        raise AgentActionError(_failed_action_message(record))
    if state == "cancelled":
        raise AgentActionError(_cancelled_action_message(record))

    client = git_client or LocalGitClient()
    with bind_log_context(action_id=record.action_id, session_id=record.session_id):
        with observe_duration(
            "action.execute",
            metric_name="action.execution.duration_ms",
            metric_attributes={"action_name": "commit"},
            span_attributes={"action_name": "commit"},
        ) as observation:
            try:
                result = client.commit(
                    message=message,
                    repo_path=repo_path,
                    paths=paths,
                    stage_all=stage_all,
                    include_untracked=include_untracked,
                )
            except LocalGitClientError as exc:
                observation.record_exception(exc)
                observation.finish(status="failed")
                updated = _finalize_action_failure(record, str(exc), action_store=action_store)
                raise AgentActionError(_failed_action_message(updated)) from exc
            answer = (
                f"Created commit {result.get('commit_sha', '')} on branch "
                f"{result.get('branch', '')}."
            )
            response = _build_action_response(
                answer=answer,
                reason="Executed local git commit action.",
                tool_name="commit_local_changes",
                tool_input={
                    "repo_path": repo_path or result.get("repo_path"),
                    "paths": list(paths or []),
                    "stage_all": stage_all,
                    "include_untracked": include_untracked,
                },
                stats={
                    "action": "commit",
                    "repo_path": result.get("repo_path"),
                    "branch": result.get("branch"),
                    "commit_sha": result.get("commit_sha"),
                    "commit_message": result.get("message"),
                    "staged_paths": result.get("staged_paths", []),
                },
            )
            observation.set_attribute("commit_sha", result.get("commit_sha"))
            observation.finish(status="succeeded")
            return _finalize_action_success(
                record,
                response,
                side_effect={
                    "repo_path": result.get("repo_path"),
                    "commit_sha": result.get("commit_sha"),
                    "branch": result.get("branch"),
                },
                action_store=action_store,
            )


__all__ = [
    "ACTION_TOOL_NAMES",
    "AgentActionError",
    "build_action_idempotency_key",
    "build_cancelled_action_response",
    "build_failed_action_response",
    "build_in_progress_action_response",
    "cancel_action_record",
    "commit_changes_action",
    "create_issue_action",
    "create_repo_action",
    "get_action_record",
    "normalize_action_payload",
    "prepare_action_record",
    "replay_action_record",
]
