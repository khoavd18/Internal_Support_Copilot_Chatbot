from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections import defaultdict
from threading import Lock
from typing import Any, DefaultDict, Dict, List, Optional
from urllib.parse import quote

from src.core.security import sanitize_error_text
from src.core.settings import (
    REDIS_KEY_PREFIX,
    REDIS_URL,
    SESSION_HISTORY_MAX_MESSAGES,
    SESSION_STATE_TTL_SECONDS,
    SESSION_STORE_BACKEND,
)

try:
    import redis
except ImportError:  # pragma: no cover - exercised indirectly when redis isn't installed
    redis = None  # type: ignore[assignment]


def _normalize_session_id(session_id: str) -> str:
    return quote(str(session_id or "").strip(), safe="")


def _load_json_dict(raw: Any) -> Dict[str, Any]:
    if raw in (None, b"", ""):
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    return data if isinstance(data, dict) else {}


def _load_json_list(raw_items: List[Any]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for raw in raw_items:
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        if isinstance(data, dict):
            items.append(
                {
                    "role": str(data.get("role") or "").strip(),
                    "content": str(data.get("content") or "").strip(),
                }
            )
    return [item for item in items if item["role"] and item["content"]]


class SessionStateStore(ABC):
    @abstractmethod
    def get_history(self, session_id: str | None) -> List[Dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def append_turn(self, session_id: str | None, role: str, content: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear_history(self, session_id: str | None) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_pending_action(self, session_id: str | None) -> Optional[Dict[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def set_pending_action(self, session_id: str | None, payload: Dict[str, Any]) -> None:
        raise NotImplementedError

    @abstractmethod
    def clear_pending_action(self, session_id: str | None) -> None:
        raise NotImplementedError

    def healthcheck(self) -> Dict[str, Any]:
        return {"backend": "unknown", "ok": True}


class InMemorySessionStateStore(SessionStateStore):
    def __init__(self, *, max_history_messages: int = SESSION_HISTORY_MAX_MESSAGES) -> None:
        self.max_history_messages = max_history_messages
        self._lock = Lock()
        self._history: DefaultDict[str, List[Dict[str, str]]] = defaultdict(list)
        self._pending: Dict[str, Dict[str, Any]] = {}

    def get_history(self, session_id: str | None) -> List[Dict[str, str]]:
        if not session_id:
            return []

        with self._lock:
            return [dict(item) for item in self._history.get(session_id, [])]

    def append_turn(self, session_id: str | None, role: str, content: str) -> None:
        if not session_id:
            return

        cleaned_role = (role or "").strip()
        cleaned_content = (content or "").strip()
        if not cleaned_role or not cleaned_content:
            return

        with self._lock:
            self._history[session_id].append(
                {
                    "role": cleaned_role,
                    "content": cleaned_content,
                }
            )
            self._history[session_id] = self._history[session_id][-self.max_history_messages :]

    def clear_history(self, session_id: str | None) -> None:
        if not session_id:
            return

        with self._lock:
            self._history.pop(session_id, None)

    def get_pending_action(self, session_id: str | None) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None

        with self._lock:
            pending = self._pending.get(session_id)
            return dict(pending) if pending else None

    def set_pending_action(self, session_id: str | None, payload: Dict[str, Any]) -> None:
        if not session_id:
            return

        normalized_payload = dict(payload or {})
        if not normalized_payload:
            return

        with self._lock:
            self._pending[session_id] = normalized_payload

    def clear_pending_action(self, session_id: str | None) -> None:
        if not session_id:
            return

        with self._lock:
            self._pending.pop(session_id, None)

    def healthcheck(self) -> Dict[str, Any]:
        return {"backend": "memory", "ok": True}


class RedisSessionStateStore(SessionStateStore):
    def __init__(
        self,
        client: Any,
        *,
        key_prefix: str = REDIS_KEY_PREFIX,
        max_history_messages: int = SESSION_HISTORY_MAX_MESSAGES,
        ttl_seconds: int = SESSION_STATE_TTL_SECONDS,
    ) -> None:
        self.client = client
        self.key_prefix = (key_prefix or "internal_support_copilot").strip()
        self.max_history_messages = max_history_messages
        self.ttl_seconds = max(0, int(ttl_seconds))

    @classmethod
    def from_url(
        cls,
        url: str = REDIS_URL,
        *,
        key_prefix: str = REDIS_KEY_PREFIX,
        max_history_messages: int = SESSION_HISTORY_MAX_MESSAGES,
        ttl_seconds: int = SESSION_STATE_TTL_SECONDS,
    ) -> "RedisSessionStateStore":
        if redis is None:
            raise RuntimeError(
                "SESSION_STORE_BACKEND=redis requires the `redis` package to be installed."
            )

        client = redis.from_url(url, decode_responses=False)
        return cls(
            client,
            key_prefix=key_prefix,
            max_history_messages=max_history_messages,
            ttl_seconds=ttl_seconds,
        )

    def _history_key(self, session_id: str) -> str:
        return f"{self.key_prefix}:session:{_normalize_session_id(session_id)}:history"

    def _pending_key(self, session_id: str) -> str:
        return f"{self.key_prefix}:session:{_normalize_session_id(session_id)}:pending_action"

    def _apply_ttl(self, pipe: Any, *keys: str) -> None:
        if self.ttl_seconds <= 0:
            return
        for key in keys:
            pipe.expire(key, self.ttl_seconds)

    def get_history(self, session_id: str | None) -> List[Dict[str, str]]:
        if not session_id:
            return []
        raw_items = self.client.lrange(self._history_key(session_id), 0, -1)
        return _load_json_list(raw_items)

    def append_turn(self, session_id: str | None, role: str, content: str) -> None:
        if not session_id:
            return

        cleaned_role = (role or "").strip()
        cleaned_content = (content or "").strip()
        if not cleaned_role or not cleaned_content:
            return

        history_key = self._history_key(session_id)
        payload = json.dumps(
            {"role": cleaned_role, "content": cleaned_content},
            ensure_ascii=False,
        )

        pipe = self.client.pipeline()
        pipe.rpush(history_key, payload)
        pipe.ltrim(history_key, -self.max_history_messages, -1)
        self._apply_ttl(pipe, history_key)
        pipe.execute()

    def clear_history(self, session_id: str | None) -> None:
        if not session_id:
            return
        self.client.delete(self._history_key(session_id))

    def get_pending_action(self, session_id: str | None) -> Optional[Dict[str, Any]]:
        if not session_id:
            return None

        raw = self.client.get(self._pending_key(session_id))
        data = _load_json_dict(raw)
        return data or None

    def set_pending_action(self, session_id: str | None, payload: Dict[str, Any]) -> None:
        if not session_id:
            return

        normalized_payload = dict(payload or {})
        if not normalized_payload:
            return

        value = json.dumps(normalized_payload, ensure_ascii=False)
        pending_key = self._pending_key(session_id)

        if self.ttl_seconds > 0:
            self.client.set(pending_key, value, ex=self.ttl_seconds)
        else:
            self.client.set(pending_key, value)

    def clear_pending_action(self, session_id: str | None) -> None:
        if not session_id:
            return
        self.client.delete(self._pending_key(session_id))

    def healthcheck(self) -> Dict[str, Any]:
        payload = {
            "backend": "redis",
            "ok": False,
            "key_prefix": self.key_prefix,
        }
        try:
            payload["ok"] = bool(self.client.ping())
        except Exception as exc:  # pragma: no cover - validated via runtime checks
            payload["error"] = sanitize_error_text(exc, max_length=240)
        return payload


_SESSION_STORE: SessionStateStore | None = None


def build_session_state_store() -> SessionStateStore:
    if SESSION_STORE_BACKEND == "redis":
        return RedisSessionStateStore.from_url()
    return InMemorySessionStateStore()


def get_session_state_store() -> SessionStateStore:
    global _SESSION_STORE
    if _SESSION_STORE is None:
        _SESSION_STORE = build_session_state_store()
    return _SESSION_STORE


def configure_session_state_store(store: SessionStateStore) -> None:
    global _SESSION_STORE
    _SESSION_STORE = store


def reset_session_state_store() -> None:
    global _SESSION_STORE
    _SESSION_STORE = None
