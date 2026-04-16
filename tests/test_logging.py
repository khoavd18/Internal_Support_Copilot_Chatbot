from __future__ import annotations

import json
import logging
import sys

from fastapi.testclient import TestClient

from src.api.main import app
from src.core.logging_utils import StructuredJsonFormatter, bind_log_context, clear_log_context


def test_structured_json_formatter_includes_context_and_redacts_secrets():
    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello",
        args=(),
        exc_info=None,
    )
    record.event = "unit.test"
    record.api_key = "super-secret-key"
    record.payload = {"token": "abc123", "repo": "my-org/demo-repo"}

    with bind_log_context(
        request_id="req-1",
        session_id="sess-1",
        user_id="user-1",
        action_id="act-1",
        agent_name="agent-a",
    ):
        payload = json.loads(formatter.format(record))

    clear_log_context()

    assert payload["request_id"] == "req-1"
    assert payload["session_id"] == "sess-1"
    assert payload["user_id"] == "user-1"
    assert payload["action_id"] == "act-1"
    assert payload["agent_name"] == "agent-a"
    assert payload["event"] == "unit.test"
    assert payload["api_key"] == "[REDACTED]"
    assert payload["payload"]["token"] == "[REDACTED]"
    assert payload["payload"]["repo"] == "my-org/demo-repo"


def test_request_logging_middleware_sets_request_id_header():
    with TestClient(app) as client:
        response = client.get("/health", headers={"X-Request-ID": "req-test-123"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "req-test-123"


def test_structured_json_formatter_redacts_message_and_exception_strings(monkeypatch):
    secret_path = r"C:\secrets\github-app.pem"
    monkeypatch.setenv("GITHUB_PRIVATE_KEY_PATH", secret_path)

    try:
        raise RuntimeError(
            f"token=abc123 Authorization: Bearer bearer-456 private_key_path={secret_path}"
        )
    except RuntimeError:
        exc_info = sys.exc_info()

    formatter = StructuredJsonFormatter()
    record = logging.LogRecord(
        name="test.logger",
        level=logging.ERROR,
        pathname=__file__,
        lineno=42,
        msg="request failed with redis://:pw@localhost:6379/0 and token=abc123",
        args=(),
        exc_info=exc_info,
    )

    payload = json.loads(formatter.format(record))

    assert "abc123" not in payload["message"]
    assert "bearer-456" not in payload["exception"]
    assert secret_path not in payload["exception"]
    assert "redis://[REDACTED]@localhost:6379/0" in payload["message"]
