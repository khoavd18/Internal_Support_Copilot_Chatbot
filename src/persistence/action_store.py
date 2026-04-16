from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Literal, Optional
from urllib.parse import quote

from src.core.security import sanitize_error_text
from src.core.settings import (
    ACTION_STATE_TTL_SECONDS,
    ACTION_STORE_BACKEND,
    REDIS_KEY_PREFIX,
    REDIS_URL,
)

try:
    import redis
except ImportError:  # pragma: no cover - exercised indirectly when redis isn't installed
    redis = None  # type: ignore[assignment]


ActionStatus = Literal["pending", "confirmed", "running", "succeeded", "failed", "cancelled"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _normalize_key(value: str) -> str:
    return quote(str(value or "").strip(), safe="")


def _load_json_dict(raw: Any) -> Dict[str, Any]:
    if raw in (None, b"", ""):
        return {}
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    data = json.loads(raw)
    return data if isinstance(data, dict) else {}


@dataclass
class ActionRecord:
    action_id: str
    action_name: str
    idempotency_key: str
    payload: Dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    reason: str = ""
    status: ActionStatus = "pending"
    confirmation_required: bool = False
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    confirmed_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    attempt_count: int = 0
    last_error: str = ""
    result: Dict[str, Any] = field(default_factory=dict)
    side_effect: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionRecord":
        return cls(
            action_id=str(data.get("action_id") or "").strip(),
            action_name=str(data.get("action_name") or "").strip(),
            idempotency_key=str(data.get("idempotency_key") or "").strip(),
            payload=dict(data.get("payload") or {}),
            session_id=str(data.get("session_id") or "").strip(),
            reason=str(data.get("reason") or "").strip(),
            status=str(data.get("status") or "pending").strip() or "pending",  # type: ignore[arg-type]
            confirmation_required=bool(data.get("confirmation_required", False)),
            created_at=str(data.get("created_at") or _utc_now()).strip(),
            updated_at=str(data.get("updated_at") or _utc_now()).strip(),
            confirmed_at=str(data.get("confirmed_at") or "").strip(),
            started_at=str(data.get("started_at") or "").strip(),
            completed_at=str(data.get("completed_at") or "").strip(),
            attempt_count=int(data.get("attempt_count") or 0),
            last_error=str(data.get("last_error") or "").strip(),
            result=dict(data.get("result") or {}),
            side_effect=dict(data.get("side_effect") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_name": self.action_name,
            "idempotency_key": self.idempotency_key,
            "payload": dict(self.payload),
            "session_id": self.session_id,
            "reason": self.reason,
            "status": self.status,
            "confirmation_required": self.confirmation_required,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "confirmed_at": self.confirmed_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "attempt_count": self.attempt_count,
            "last_error": self.last_error,
            "result": dict(self.result),
            "side_effect": dict(self.side_effect),
        }


class ActionStore(ABC):
    @abstractmethod
    def get_action(self, action_id: str) -> Optional[ActionRecord]:
        raise NotImplementedError

    @abstractmethod
    def get_by_idempotency_key(self, idempotency_key: str) -> Optional[ActionRecord]:
        raise NotImplementedError

    @abstractmethod
    def create_or_get(self, record: ActionRecord) -> ActionRecord:
        raise NotImplementedError

    @abstractmethod
    def save(self, record: ActionRecord) -> ActionRecord:
        raise NotImplementedError

    def healthcheck(self) -> Dict[str, Any]:
        return {"backend": "unknown", "ok": True}


class InMemoryActionStore(ActionStore):
    def __init__(self) -> None:
        self._lock = Lock()
        self._records: Dict[str, ActionRecord] = {}
        self._idempotency_index: Dict[str, str] = {}

    def get_action(self, action_id: str) -> Optional[ActionRecord]:
        if not action_id:
            return None
        with self._lock:
            record = self._records.get(action_id)
            return ActionRecord.from_dict(record.to_dict()) if record else None

    def get_by_idempotency_key(self, idempotency_key: str) -> Optional[ActionRecord]:
        if not idempotency_key:
            return None
        with self._lock:
            action_id = self._idempotency_index.get(idempotency_key)
            record = self._records.get(action_id or "")
            return ActionRecord.from_dict(record.to_dict()) if record else None

    def create_or_get(self, record: ActionRecord) -> ActionRecord:
        with self._lock:
            existing_id = self._idempotency_index.get(record.idempotency_key)
            if existing_id:
                existing = self._records.get(existing_id)
                if existing:
                    return ActionRecord.from_dict(existing.to_dict())

            cloned = ActionRecord.from_dict(record.to_dict())
            self._records[cloned.action_id] = cloned
            self._idempotency_index[cloned.idempotency_key] = cloned.action_id
            return ActionRecord.from_dict(cloned.to_dict())

    def save(self, record: ActionRecord) -> ActionRecord:
        with self._lock:
            cloned = ActionRecord.from_dict(record.to_dict())
            self._records[cloned.action_id] = cloned
            self._idempotency_index[cloned.idempotency_key] = cloned.action_id
            return ActionRecord.from_dict(cloned.to_dict())

    def healthcheck(self) -> Dict[str, Any]:
        return {"backend": "memory", "ok": True}


class RedisActionStore(ActionStore):
    def __init__(
        self,
        client: Any,
        *,
        key_prefix: str = REDIS_KEY_PREFIX,
        ttl_seconds: int = ACTION_STATE_TTL_SECONDS,
    ) -> None:
        self.client = client
        self.key_prefix = (key_prefix or "internal_support_copilot").strip()
        self.ttl_seconds = max(0, int(ttl_seconds))

    @classmethod
    def from_url(
        cls,
        url: str = REDIS_URL,
        *,
        key_prefix: str = REDIS_KEY_PREFIX,
        ttl_seconds: int = ACTION_STATE_TTL_SECONDS,
    ) -> "RedisActionStore":
        if redis is None:
            raise RuntimeError(
                "ACTION_STORE_BACKEND=redis requires the `redis` package to be installed."
            )

        client = redis.from_url(url, decode_responses=False)
        return cls(client, key_prefix=key_prefix, ttl_seconds=ttl_seconds)

    def _record_key(self, action_id: str) -> str:
        return f"{self.key_prefix}:action:{_normalize_key(action_id)}"

    def _idempotency_key(self, idempotency_key: str) -> str:
        return f"{self.key_prefix}:action-idempotency:{_normalize_key(idempotency_key)}"

    def _apply_ttl(self, *keys: str) -> None:
        if self.ttl_seconds <= 0:
            return
        pipe = self.client.pipeline()
        for key in keys:
            pipe.expire(key, self.ttl_seconds)
        pipe.execute()

    def _write_record(self, record: ActionRecord) -> ActionRecord:
        record_key = self._record_key(record.action_id)
        idem_key = self._idempotency_key(record.idempotency_key)
        payload = json.dumps(record.to_dict(), ensure_ascii=False)
        pipe = self.client.pipeline()
        pipe.set(record_key, payload)
        pipe.set(idem_key, record.action_id)
        if self.ttl_seconds > 0:
            pipe.expire(record_key, self.ttl_seconds)
            pipe.expire(idem_key, self.ttl_seconds)
        pipe.execute()
        return record

    def get_action(self, action_id: str) -> Optional[ActionRecord]:
        if not action_id:
            return None
        data = _load_json_dict(self.client.get(self._record_key(action_id)))
        return ActionRecord.from_dict(data) if data else None

    def get_by_idempotency_key(self, idempotency_key: str) -> Optional[ActionRecord]:
        if not idempotency_key:
            return None
        raw = self.client.get(self._idempotency_key(idempotency_key))
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        action_id = str(raw or "").strip()
        if not action_id:
            return None
        record = self.get_action(action_id)
        if record is None:
            self.client.delete(self._idempotency_key(idempotency_key))
        return record

    def create_or_get(self, record: ActionRecord) -> ActionRecord:
        existing = self.get_by_idempotency_key(record.idempotency_key)
        if existing is not None:
            return existing

        idem_key = self._idempotency_key(record.idempotency_key)
        if self.ttl_seconds > 0:
            claimed = self.client.set(idem_key, record.action_id, nx=True, ex=self.ttl_seconds)
        else:
            claimed = self.client.set(idem_key, record.action_id, nx=True)

        if not claimed:
            existing = self.get_by_idempotency_key(record.idempotency_key)
            if existing is not None:
                return existing
            self.client.delete(idem_key)
            if self.ttl_seconds > 0:
                self.client.set(idem_key, record.action_id, nx=True, ex=self.ttl_seconds)
            else:
                self.client.set(idem_key, record.action_id, nx=True)

        return ActionRecord.from_dict(self._write_record(record).to_dict())

    def save(self, record: ActionRecord) -> ActionRecord:
        return ActionRecord.from_dict(self._write_record(record).to_dict())

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


_ACTION_STORE: ActionStore | None = None


def build_action_store() -> ActionStore:
    if ACTION_STORE_BACKEND == "redis":
        return RedisActionStore.from_url()
    return InMemoryActionStore()


def get_action_store() -> ActionStore:
    global _ACTION_STORE
    if _ACTION_STORE is None:
        _ACTION_STORE = build_action_store()
    return _ACTION_STORE


def configure_action_store(store: ActionStore) -> None:
    global _ACTION_STORE
    _ACTION_STORE = store


def reset_action_store() -> None:
    global _ACTION_STORE
    _ACTION_STORE = None
