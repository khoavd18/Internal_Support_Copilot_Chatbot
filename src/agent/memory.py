"""
Session-scoped chat history and pending confirmation state.

This module keeps the existing function-based API for the rest of the application,
but delegates storage to a pluggable persistence backend.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.persistence.session_store import (
    InMemorySessionStateStore,
    RedisSessionStateStore,
    SessionStateStore,
    configure_session_state_store,
    get_session_state_store,
    reset_session_state_store,
)


def get_history(session_id: str | None) -> List[Dict[str, str]]:
    return get_session_state_store().get_history(session_id)


def append_turn(session_id: str | None, role: str, content: str) -> None:
    get_session_state_store().append_turn(session_id, role, content)


def clear_history(session_id: str | None) -> None:
    get_session_state_store().clear_history(session_id)


def get_pending_action(session_id: str | None) -> Optional[Dict[str, Any]]:
    return get_session_state_store().get_pending_action(session_id)


def set_pending_action(session_id: str | None, payload: Dict[str, Any]) -> None:
    get_session_state_store().set_pending_action(session_id, payload)


def clear_pending_action(session_id: str | None) -> None:
    get_session_state_store().clear_pending_action(session_id)


__all__ = [
    "SessionStateStore",
    "InMemorySessionStateStore",
    "RedisSessionStateStore",
    "append_turn",
    "clear_history",
    "clear_pending_action",
    "configure_session_state_store",
    "get_history",
    "get_pending_action",
    "get_session_state_store",
    "reset_session_state_store",
    "set_pending_action",
]
