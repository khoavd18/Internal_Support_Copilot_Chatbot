from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from src.agent.action_registry import (
    ActionIntent,
    action_requires_confirmation,
    detect_action_request,
    execute_registered_action,
    get_action_tool_name,
    is_cancellation_message,
    is_confirmation_message,
)
from src.agent.actions import (
    AgentActionError,
    build_cancelled_action_response,
    build_failed_action_response,
    build_in_progress_action_response,
    cancel_action_record,
    get_action_record,
    prepare_action_record,
    replay_action_record,
)
from src.agent.memory import clear_pending_action, get_pending_action, set_pending_action
from src.core.security import sanitize_error_text
from src.persistence.action_store import ActionRecord, ActionStore

logger = logging.getLogger(__name__)


def _stringify_payload(payload: Dict[str, Any]) -> str:
    lines = []
    for key, value in payload.items():
        if value in (None, "", [], {}):
            continue
        lines.append(f"- {key}: {value}")
    return "\n".join(lines)


def _attach_action_record_metadata(
    response: Dict[str, Any],
    record: ActionRecord | None,
    *,
    idempotent_replay: bool = False,
) -> Dict[str, Any]:
    if record is None:
        return response

    normalized = dict(response)
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


def _build_response(
    *,
    answer: str,
    action_name: str,
    tool_input: Dict[str, Any],
    tool_status: str,
    reason: str,
    backend_mode: str,
    action_status: str,
    note: str,
    action_record: ActionRecord | None = None,
    idempotent_replay: bool = False,
) -> Dict[str, Any]:
    return _attach_action_record_metadata(
        {
            "answer": answer,
            "sources": [],
            "stats": {
                "backend_mode": backend_mode,
                "action": action_name,
                "action_status": action_status,
                "requires_confirmation": action_requires_confirmation(action_name),
            },
            "debug": [],
            "agent": {
                "route": "action",
                "reason": reason,
                "tool_calls": [
                    {
                        "tool_name": get_action_tool_name(action_name),
                        "tool_input": tool_input,
                        "status": tool_status,
                        "note": note,
                    }
                ],
            },
        },
        action_record,
        idempotent_replay=idempotent_replay,
    )


def _build_clarify_response(intent: ActionIntent, backend_mode: str) -> Dict[str, Any]:
    answer = intent.clarify_answer or (
        "I detected a write action request, but the input is still incomplete."
    )
    return _build_response(
        answer=answer,
        action_name=intent.action,
        tool_input=intent.payload,
        tool_status="skipped",
        reason=intent.reason or "Action request needs clarification.",
        backend_mode=backend_mode,
        action_status="clarify",
        note="Action was not executed because required fields are missing.",
    )


def _build_confirmation_response(
    intent: ActionIntent,
    backend_mode: str,
    *,
    action_record: ActionRecord,
    restored_from_pending: bool = False,
    idempotent_replay: bool = False,
) -> Dict[str, Any]:
    payload_preview = _stringify_payload(intent.payload)
    status_note = (
        "Pending action restored from persisted session state."
        if restored_from_pending
        else "Action request captured from chat."
    )
    answer = (
        "I can do that, but this is a write action and needs confirmation before execution.\n\n"
        f"Planned action: {intent.action}\n"
        f"{payload_preview}\n\n"
        "Reply with `yes`/`xac nhan` in the same session, or resend the request with `confirmed=true`."
    ).strip()
    return _build_response(
        answer=answer,
        action_name=intent.action,
        tool_input=intent.payload,
        tool_status="skipped",
        reason=intent.reason or "Write action requires confirmation.",
        backend_mode=backend_mode,
        action_status="requires_confirmation",
        note=status_note,
        action_record=action_record,
        idempotent_replay=idempotent_replay,
    )


def _build_cancel_response(
    pending_action: Dict[str, Any],
    backend_mode: str,
    *,
    action_record: ActionRecord | None = None,
) -> Dict[str, Any]:
    action_name = str(pending_action.get("action") or "action")
    return _build_response(
        answer=f"Cancelled pending action `{action_name}` for this session.",
        action_name=action_name,
        tool_input=dict(pending_action.get("payload") or {}),
        tool_status="skipped",
        reason="User cancelled the pending action.",
        backend_mode=backend_mode,
        action_status="cancelled",
        note="Pending action was cleared from persisted session state.",
        action_record=action_record,
    )


def _build_error_response(
    *,
    intent: ActionIntent,
    error_message: str,
    backend_mode: str,
    action_record: ActionRecord | None = None,
) -> Dict[str, Any]:
    safe_error_message = sanitize_error_text(error_message, max_length=240)
    return _build_response(
        answer=f"The requested action could not be completed: {safe_error_message}",
        action_name=intent.action,
        tool_input=intent.payload,
        tool_status="error",
        reason=intent.reason or "Action execution failed.",
        backend_mode=backend_mode,
        action_status="error",
        note=safe_error_message,
        action_record=action_record,
    )


def _execute_intent(
    intent: ActionIntent,
    *,
    confirmed: bool,
    session_id: str | None = None,
    idempotency_key: str | None = None,
    action_store: ActionStore | None = None,
) -> Dict[str, Any]:
    try:
        return execute_registered_action(
            intent.action,
            payload=intent.payload,
            confirmed=confirmed,
            session_id=session_id,
            idempotency_key=idempotency_key,
            action_store=action_store,
        )
    except KeyError as exc:
        raise AgentActionError(f"Unsupported action '{intent.action}'.") from exc


def _intent_from_pending_action(pending_action: Dict[str, Any]) -> ActionIntent:
    return ActionIntent(
        action=str(pending_action.get("action") or "").strip(),
        payload=dict(pending_action.get("payload") or {}),
        reason=str(pending_action.get("reason") or "").strip(),
    )


def _annotate_execution_result(
    result: Dict[str, Any],
    *,
    backend_mode: str,
    original_question: str,
    action_name: str,
) -> Dict[str, Any]:
    normalized = dict(result)
    stats = dict(normalized.get("stats", {}))
    stats.update(
        {
            "backend_mode": backend_mode,
            "action": action_name,
            "action_status": stats.get("action_status", "ok"),
            "original_question": original_question,
        }
    )
    normalized["stats"] = stats
    return normalized


def _build_existing_record_response(
    record: ActionRecord,
    *,
    backend_mode: str,
    original_question: str,
    action_name: str,
) -> Dict[str, Any]:
    if record.status == "succeeded":
        result = replay_action_record(record)
    elif record.status in {"confirmed", "running"}:
        result = build_in_progress_action_response(record)
    elif record.status == "failed":
        result = build_failed_action_response(record)
    elif record.status == "cancelled":
        result = build_cancelled_action_response(record)
    else:
        raise AgentActionError(f"Unsupported action record state '{record.status}'.")

    return _annotate_execution_result(
        result,
        backend_mode=backend_mode,
        original_question=original_question,
        action_name=action_name,
    )


def maybe_handle_chat_action(
    *,
    question: str,
    confirmed: bool,
    session_id: Optional[str],
    backend_mode: str,
    idempotency_key: str | None = None,
    action_store: ActionStore | None = None,
) -> Optional[Dict[str, Any]]:
    cleaned_question = str(question or "").strip()
    if not cleaned_question:
        return None

    pending_action = get_pending_action(session_id)
    if pending_action and is_cancellation_message(cleaned_question):
        action_record = None
        action_id = str(pending_action.get("action_id") or "").strip()
        if action_id:
            action_record = cancel_action_record(
                action_id,
                reason="Cancelled by user from chat confirmation flow.",
                action_store=action_store,
            )
        logger.info(
            "Pending chat action cancelled",
            extra={
                "event": "chat_action.cancelled",
                "action_id": action_id,
                "session_id": session_id or "",
            },
        )
        clear_pending_action(session_id)
        return _annotate_execution_result(
            _build_cancel_response(
                pending_action,
                backend_mode,
                action_record=action_record,
            ),
            backend_mode=backend_mode,
            original_question=cleaned_question,
            action_name=str(pending_action.get("action") or "action"),
        )

    if pending_action and is_confirmation_message(cleaned_question):
        action_id = str(pending_action.get("action_id") or "").strip()
        record = get_action_record(action_id, action_store=action_store) if action_id else None
        logger.info(
            "Chat confirmation received for pending action",
            extra={
                "event": "chat_action.confirmed",
                "action_id": action_id,
                "session_id": session_id or "",
                "existing_status": record.status if record else "",
            },
        )

        if record and record.status in {"succeeded", "running", "confirmed", "failed", "cancelled"}:
            clear_pending_action(session_id)
            return _build_existing_record_response(
                record,
                backend_mode=backend_mode,
                original_question=cleaned_question,
                action_name=record.action_name,
            )

        intent = _intent_from_pending_action(pending_action)
        try:
            result = _execute_intent(
                intent,
                confirmed=True,
                session_id=session_id,
                idempotency_key=(record.idempotency_key if record else idempotency_key),
                action_store=action_store,
            )
        except AgentActionError as exc:
            refreshed = get_action_record(action_id, action_store=action_store) if action_id else None
            clear_pending_action(session_id)
            if refreshed and refreshed.status in {"failed", "cancelled"}:
                return _build_existing_record_response(
                    refreshed,
                    backend_mode=backend_mode,
                    original_question=cleaned_question,
                    action_name=refreshed.action_name,
                )
            return _build_error_response(
                intent=intent,
                error_message=str(exc),
                backend_mode=backend_mode,
                action_record=refreshed,
            )

        clear_pending_action(session_id)
        return _annotate_execution_result(
            result,
            backend_mode=backend_mode,
            original_question=cleaned_question,
            action_name=intent.action,
        )

    intent = detect_action_request(cleaned_question)
    if intent is None:
        return None

    if intent.needs_clarification:
        clear_pending_action(session_id)
        return _build_clarify_response(intent, backend_mode)

    action_record = prepare_action_record(
        action_name=intent.action,
        payload=intent.payload,
        reason=intent.reason,
        session_id=session_id,
        idempotency_key=idempotency_key,
        confirmation_required=action_requires_confirmation(intent.action),
        action_store=action_store,
    )
    logger.info(
        "Chat action intent detected",
        extra={
            "event": "chat_action.detected",
            "action_id": action_record.action_id,
            "action_name": intent.action,
            "session_id": session_id or "",
            "requires_confirmation": action_requires_confirmation(intent.action),
        },
    )

    if action_record.status in {"succeeded", "confirmed", "running", "failed", "cancelled"}:
        clear_pending_action(session_id)
        return _build_existing_record_response(
            action_record,
            backend_mode=backend_mode,
            original_question=cleaned_question,
            action_name=intent.action,
        )

    if action_requires_confirmation(intent.action) and not confirmed:
        set_pending_action(
            session_id,
            {
                "action_id": action_record.action_id,
                "action": intent.action,
                "payload": intent.payload,
                "reason": intent.reason,
            },
        )
        return _build_confirmation_response(
            intent,
            backend_mode,
            action_record=action_record,
            idempotent_replay=False,
        )

    try:
        result = _execute_intent(
            intent,
            confirmed=confirmed,
            session_id=session_id,
            idempotency_key=action_record.idempotency_key,
            action_store=action_store,
        )
    except AgentActionError as exc:
        clear_pending_action(session_id)
        refreshed = get_action_record(action_record.action_id, action_store=action_store)
        if refreshed and refreshed.status in {"failed", "cancelled"}:
            return _build_existing_record_response(
                refreshed,
                backend_mode=backend_mode,
                original_question=cleaned_question,
                action_name=intent.action,
            )
        return _build_error_response(
            intent=intent,
            error_message=str(exc),
            backend_mode=backend_mode,
            action_record=refreshed or action_record,
        )

    clear_pending_action(session_id)
    return _annotate_execution_result(
        result,
        backend_mode=backend_mode,
        original_question=cleaned_question,
        action_name=intent.action,
    )
