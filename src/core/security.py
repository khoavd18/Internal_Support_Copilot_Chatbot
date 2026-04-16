from __future__ import annotations

import os
import re
from typing import Any, Iterable


SENSITIVE_KEY_FRAGMENTS = (
    "token",
    "secret",
    "password",
    "authorization",
    "private_key",
    "api_key",
    "jwt",
    "credential",
)
SENSITIVE_TEXT_FIELDS = (
    "access_token",
    "refresh_token",
    "client_secret",
    "api_key",
    "authorization",
    "private_key_path",
    "private_key",
    "password",
    "secret",
    "token",
    "jwt",
)
SECRET_ENV_VAR_NAMES = frozenset(
    {
        "QDRANT_API_KEY",
        "GITHUB_PRIVATE_KEY_PATH",
        "GITHUB_TOKEN",
        "GITLAB_TOKEN",
        "OPENAI_API_KEY",
        "HF_TOKEN",
    }
)
DEFAULT_TEXT_LIMIT = 240
EXTENDED_TEXT_LIMIT = 8000
REDACTION_PLACEHOLDER = "[REDACTED]"

_SENSITIVE_FIELD_PATTERN = "|".join(
    re.escape(item) for item in sorted(SENSITIVE_TEXT_FIELDS, key=len, reverse=True)
)
_SENSITIVE_JSON_STRING_RE = re.compile(
    rf'(?i)((?:"|\')?(?:{_SENSITIVE_FIELD_PATTERN})(?:"|\')?\s*:\s*)(["\'])(.*?)(\2)'
)
_SENSITIVE_JSON_BARE_RE = re.compile(
    rf'(?i)((?:"|\')?(?:{_SENSITIVE_FIELD_PATTERN})(?:"|\')?\s*:\s*)([^,\s\}}\]]+)'
)
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    rf'(?i)(\b(?:{_SENSITIVE_FIELD_PATTERN})\b\s*=\s*)(\S+)'
)
_AUTH_BEARER_RE = re.compile(r"(?i)(\bAuthorization\b\s*[:=]\s*Bearer\s+)(\S+)")
_URL_USERINFO_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9+.-]*://)([^/\s@]+@)")


def read_env(name: str, default: Any = "") -> str:
    value = os.getenv(name)
    if value is None:
        return "" if default is None else str(default)
    return str(value)


def read_secret_env(name: str, default: Any = "") -> str:
    return read_env(name, default)


def redact_url_credentials(value: Any) -> str:
    text = str(value or "")
    return _URL_USERINFO_RE.sub(r"\1[REDACTED]@", text)


def _redact_sensitive_assignments(text: str) -> str:
    text = _AUTH_BEARER_RE.sub(r"\1[REDACTED]", text)
    text = _SENSITIVE_JSON_STRING_RE.sub(r"\1\2[REDACTED]\4", text)
    text = _SENSITIVE_JSON_BARE_RE.sub(r"\1[REDACTED]", text)
    text = _SENSITIVE_ASSIGNMENT_RE.sub(r"\1[REDACTED]", text)
    return text


def _iter_secret_variants(secret: str) -> list[str]:
    variants = {secret}
    escaped = (
        secret.replace("\\", "\\\\")
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    variants.add(escaped)
    return sorted((item for item in variants if item), key=len, reverse=True)


def _configured_secret_values(extra_secret_values: Iterable[Any] | None = None) -> list[str]:
    values: set[str] = set()
    for name in SECRET_ENV_VAR_NAMES:
        raw = read_secret_env(name, "")
        cleaned = raw.strip()
        if cleaned:
            values.add(cleaned)

    for item in extra_secret_values or ():
        cleaned = str(item or "").strip()
        if cleaned:
            values.add(cleaned)

    return sorted(values, key=len, reverse=True)


def sanitize_error_text(
    value: Any,
    *,
    extra_secret_values: Iterable[Any] | None = None,
    max_length: int | None = DEFAULT_TEXT_LIMIT,
) -> str:
    text = redact_url_credentials(str(value or ""))
    text = _redact_sensitive_assignments(text)

    for secret in _configured_secret_values(extra_secret_values):
        for variant in _iter_secret_variants(secret):
            text = text.replace(variant, REDACTION_PLACEHOLDER)

    if max_length is not None and len(text) > max_length:
        return f"{text[:max_length]}..."
    return text


__all__ = [
    "DEFAULT_TEXT_LIMIT",
    "EXTENDED_TEXT_LIMIT",
    "REDACTION_PLACEHOLDER",
    "SECRET_ENV_VAR_NAMES",
    "SENSITIVE_KEY_FRAGMENTS",
    "SENSITIVE_TEXT_FIELDS",
    "read_env",
    "read_secret_env",
    "redact_url_credentials",
    "sanitize_error_text",
]
