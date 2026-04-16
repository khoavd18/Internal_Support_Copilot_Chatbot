from __future__ import annotations

import json
import logging
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict, Iterator

from src.core.security import (
    EXTENDED_TEXT_LIMIT,
    SENSITIVE_KEY_FRAGMENTS,
    sanitize_error_text,
)
from src.core.settings import LOG_FORMAT, LOG_LEVEL


CORRELATION_FIELDS = ("request_id", "session_id", "user_id", "action_id", "agent_name")
TEXT_VALUE_LIMIT = 240
LIST_VALUE_LIMIT = 10
_LOG_CONTEXT: ContextVar[Dict[str, str]] = ContextVar("log_context", default={})
_STANDARD_LOG_RECORD_FIELDS = frozenset(
    logging.LogRecord(
        name="",
        level=0,
        pathname="",
        lineno=0,
        msg="",
        args=(),
        exc_info=None,
    ).__dict__.keys()
)


def _is_sensitive_key(key: str) -> bool:
    normalized = str(key or "").strip().lower()
    return any(fragment in normalized for fragment in SENSITIVE_KEY_FRAGMENTS)


def _truncate_text(value: str, max_length: int = TEXT_VALUE_LIMIT) -> str:
    return sanitize_error_text(value, max_length=max_length)


def sanitize_log_value(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if _is_sensitive_key(key):
        return "[REDACTED]"

    if value is None or isinstance(value, (bool, int, float)):
        return value

    if isinstance(value, str):
        return sanitize_error_text(value, max_length=TEXT_VALUE_LIMIT)

    if depth >= 3:
        return sanitize_error_text(repr(value), max_length=TEXT_VALUE_LIMIT)

    if isinstance(value, dict):
        return {
            str(child_key): sanitize_log_value(child_value, key=str(child_key), depth=depth + 1)
            for child_key, child_value in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        sanitized = [
            sanitize_log_value(item, key=key, depth=depth + 1)
            for item in items[:LIST_VALUE_LIMIT]
        ]
        if len(items) > LIST_VALUE_LIMIT:
            sanitized.append(f"...(+{len(items) - LIST_VALUE_LIMIT} more)")
        return sanitized

    return sanitize_error_text(repr(value), max_length=TEXT_VALUE_LIMIT)


def get_log_context() -> Dict[str, str]:
    return dict(_LOG_CONTEXT.get())


@contextmanager
def bind_log_context(**kwargs: Any) -> Iterator[Dict[str, str]]:
    current = dict(_LOG_CONTEXT.get())
    for key, value in kwargs.items():
        if value in (None, ""):
            continue
        current[str(key)] = str(value)
    token = _LOG_CONTEXT.set(current)
    try:
        yield current
    finally:
        _LOG_CONTEXT.reset(token)


def clear_log_context() -> None:
    _LOG_CONTEXT.set({})


class StructuredContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        context = get_log_context()
        for field in CORRELATION_FIELDS:
            if getattr(record, field, None):
                continue
            value = context.get(field)
            if value:
                setattr(record, field, value)
        return True


class StructuredJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        context = get_log_context()
        payload: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": sanitize_error_text(record.getMessage(), max_length=TEXT_VALUE_LIMIT),
        }

        event = getattr(record, "event", None)
        if event:
            payload["event"] = str(event)

        for field in CORRELATION_FIELDS:
            value = getattr(record, field, None) or context.get(field)
            if value not in (None, ""):
                payload[field] = str(value)

        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_RECORD_FIELDS or key in CORRELATION_FIELDS or key == "event":
                continue
            payload[key] = sanitize_log_value(value, key=key)

        if record.exc_info:
            payload["exception"] = sanitize_error_text(
                self.formatException(record.exc_info),
                max_length=EXTENDED_TEXT_LIMIT,
            )
        if record.stack_info:
            payload["stack_info"] = sanitize_error_text(
                self.formatStack(record.stack_info),
                max_length=EXTENDED_TEXT_LIMIT,
            )

        return json.dumps(payload, ensure_ascii=False, default=str)


class StructuredTextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        context = get_log_context()
        fields = [
            f"timestamp={datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat().replace('+00:00', 'Z')}",
            f"level={record.levelname}",
            f"logger={record.name}",
        ]
        event = getattr(record, "event", None)
        if event:
            fields.append(f"event={event}")

        for field in CORRELATION_FIELDS:
            value = getattr(record, field, None) or context.get(field)
            if value not in (None, ""):
                fields.append(f"{field}={value}")

        fields.append(
            f"message={json.dumps(sanitize_error_text(record.getMessage(), max_length=TEXT_VALUE_LIMIT), ensure_ascii=False)}"
        )

        for key, value in record.__dict__.items():
            if key in _STANDARD_LOG_RECORD_FIELDS or key in CORRELATION_FIELDS or key == "event":
                continue
            sanitized = sanitize_log_value(value, key=key)
            fields.append(f"{key}={json.dumps(sanitized, ensure_ascii=False, default=str)}")

        if record.exc_info:
            fields.append(
                f"exception={json.dumps(sanitize_error_text(self.formatException(record.exc_info), max_length=EXTENDED_TEXT_LIMIT), ensure_ascii=False)}"
            )

        return " ".join(fields)


def configure_logging(*, force: bool = False) -> None:
    root_logger = logging.getLogger()
    if getattr(configure_logging, "_configured", False) and not force:
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(StructuredContextFilter())
    if str(LOG_FORMAT).strip().lower() == "text":
        handler.setFormatter(StructuredTextFormatter())
    else:
        handler.setFormatter(StructuredJsonFormatter())

    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, str(LOG_LEVEL).upper(), logging.INFO))
    root_logger.propagate = False

    for noisy_logger in ("httpx", "urllib3", "uvicorn.access"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    configure_logging._configured = True  # type: ignore[attr-defined]
