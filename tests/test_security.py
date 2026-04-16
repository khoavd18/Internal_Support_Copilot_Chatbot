from __future__ import annotations

from typing import Any, Dict

import pytest

from src.core import runtime_checks
from src.core.security import sanitize_error_text
from src.integrations import github_client as github_client_module
from src.integrations.github_app_auth import GitHubAppAuth
from src.integrations.github_client import GitHubClient, GitHubClientError
from src.persistence.session_store import RedisSessionStateStore


class _FakeResponse:
    def __init__(self, status_code: int, text: str, payload: Dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}

    def json(self) -> Dict[str, Any]:
        return self._payload


class _RepoAuth:
    def get_repo_installation(self, owner: str, repo: str) -> Dict[str, Any]:
        assert owner == "my-org"
        assert repo == "demo-repo"
        return {"id": 123}

    def get_installation_token(self, installation_id: str) -> str:
        assert installation_id == "123"
        return "install-token"

    def auth_health(self) -> Dict[str, Any]:
        return {"configured": True}


def test_sanitize_error_text_redacts_assignments_and_url_credentials(monkeypatch):
    secret_path = r"C:\secrets\github-app.pem"
    monkeypatch.setenv("QDRANT_API_KEY", "qdrant-secret")
    monkeypatch.setenv("GITHUB_PRIVATE_KEY_PATH", secret_path)

    text = (
        'token=abc123 Authorization: Bearer bearer-456 '
        '{"api_key":"xyz","private_key_path":"C:\\\\secrets\\\\github-app.pem"} '
        "redis://:redis-pass@localhost:6379/0 qdrant-secret"
    )

    sanitized = sanitize_error_text(text, max_length=None)

    assert "abc123" not in sanitized
    assert "bearer-456" not in sanitized
    assert "xyz" not in sanitized
    assert "github-app.pem" not in sanitized
    assert "qdrant-secret" not in sanitized
    assert "redis://[REDACTED]@localhost:6379/0" in sanitized


def test_github_app_auth_health_redacts_private_key_path(tmp_path):
    private_key_path = tmp_path / "missing-github-app.pem"
    auth = GitHubAppAuth(client_id="client-id", private_key_path=str(private_key_path))

    payload = auth.auth_health()

    assert payload["configured"] is True
    assert payload["jwt_ready"] is False
    assert str(private_key_path) not in payload["error"]


def test_github_client_error_messages_redact_sensitive_response_text(monkeypatch):
    monkeypatch.setattr(github_client_module, "GITHUB_ACTIONS_ENABLED", True)

    client = GitHubClient(auth=_RepoAuth(), allowed_repos={"my-org/demo-repo"})
    client._session.request = lambda **kwargs: _FakeResponse(
        401,
        'token=abc123 {"api_key":"xyz"} redis://:redis-pass@localhost:6379/0',
    )

    with pytest.raises(GitHubClientError) as exc_info:
        client.create_issue(
            repo_full_name="my-org/demo-repo",
            title="Bug login",
            body="Steps to reproduce",
        )

    message = str(exc_info.value)
    assert "abc123" not in message
    assert "xyz" not in message
    assert "redis-pass" not in message
    assert "redis://[REDACTED]@localhost:6379/0" in message


def test_qdrant_readiness_redacts_url_credentials(monkeypatch):
    secured_url = "http://user:secret-pass@localhost:6333"
    monkeypatch.setattr(runtime_checks, "QDRANT_URL", secured_url)
    monkeypatch.setattr(
        runtime_checks,
        "_build_qdrant_client",
        lambda: (_ for _ in ()).throw(
            RuntimeError(f"cannot connect to {secured_url} token=abc123")
        ),
    )

    payload = runtime_checks.check_qdrant_readiness()

    assert payload["ok"] is False
    assert payload["url"] == "http://[REDACTED]@localhost:6333"
    assert "secret-pass" not in payload["error"]
    assert "abc123" not in payload["error"]


def test_redis_healthcheck_redacts_sensitive_error_details():
    class _BrokenRedisClient:
        def ping(self) -> bool:
            raise RuntimeError("password=abc123 redis://:redis-pass@localhost:6379/0 refused")

    payload = RedisSessionStateStore(_BrokenRedisClient()).healthcheck()

    assert payload["ok"] is False
    assert "abc123" not in payload["error"]
    assert "redis-pass" not in payload["error"]
    assert "redis://[REDACTED]@localhost:6379/0" in payload["error"]
