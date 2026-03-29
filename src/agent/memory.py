from __future__ import annotations

from collections import defaultdict
from threading import Lock
from typing import DefaultDict, Dict, List

MAX_HISTORY_MESSAGES = 6

_MEMORY_LOCK = Lock()
_MEMORY_STORE: DefaultDict[str, List[Dict[str, str]]] = defaultdict(list)


def get_history(session_id: str | None) -> List[Dict[str, str]]:
    if not session_id:
        return []

    with _MEMORY_LOCK:
        return [dict(item) for item in _MEMORY_STORE.get(session_id, [])]


def append_turn(session_id: str | None, role: str, content: str) -> None:
    if not session_id:
        return

    cleaned_role = (role or "").strip()
    cleaned_content = (content or "").strip()

    if not cleaned_role or not cleaned_content:
        return

    with _MEMORY_LOCK:
        _MEMORY_STORE[session_id].append(
            {
                "role": cleaned_role,
                "content": cleaned_content,
            }
        )
        _MEMORY_STORE[session_id] = _MEMORY_STORE[session_id][-MAX_HISTORY_MESSAGES:]


def clear_history(session_id: str | None) -> None:
    if not session_id:
        return

    with _MEMORY_LOCK:
        _MEMORY_STORE.pop(session_id, None)