from __future__ import annotations

from typing import Any, Dict

import pytest

from src.integrations import github_client as github_client_module
from src.integrations.github_client import GitHubClient, GitHubClientError


class _FakeAuth:
    def get_installation_token(self, installation_id: str) -> str:
        assert installation_id == "123"
        return "token-123"

    def get_org_installation(self, org: str) -> Dict[str, Any]:
        assert org == "my-org"
        return {"id": 123}

    def auth_health(self) -> Dict[str, Any]:
        return {"configured": True}


class _FakeResponse:
    def __init__(self, status_code: int, payload: Dict[str, Any]) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = str(payload)

    def json(self) -> Dict[str, Any]:
        return self._payload


def test_create_organization_repository_uses_org_installation(monkeypatch):
    monkeypatch.setattr(github_client_module, "GITHUB_ACTIONS_ENABLED", True)
    captured: Dict[str, Any] = {}

    def _fake_request(**kwargs):
        captured.update(kwargs)
        return _FakeResponse(
            201,
            {
                "id": 99,
                "name": "demo-repo",
                "full_name": "my-org/demo-repo",
                "private": False,
                "html_url": "https://github.com/my-org/demo-repo",
                "default_branch": "main",
            },
        )

    client = GitHubClient(auth=_FakeAuth(), allowed_orgs={"my-org"})
    monkeypatch.setattr(client._session, "request", _fake_request)

    result = client.create_organization_repository(
        org="my-org",
        name="demo-repo",
        description="created by agent",
        private=False,
        auto_init=True,
    )

    assert result["full_name"] == "my-org/demo-repo"
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/orgs/my-org/repos")
    assert captured["json"] == {
        "name": "demo-repo",
        "description": "created by agent",
        "private": False,
        "auto_init": True,
    }
    assert captured["headers"]["Authorization"] == "Bearer token-123"


def test_create_issue_uses_repo_installation(monkeypatch):
    monkeypatch.setattr(github_client_module, "GITHUB_ACTIONS_ENABLED", True)
    captured: Dict[str, Any] = {}

    class _RepoAuth(_FakeAuth):
        def get_repo_installation(self, owner: str, repo: str) -> Dict[str, Any]:
            assert owner == "my-org"
            assert repo == "demo-repo"
            return {"id": 123}

    def _fake_request(**kwargs):
        captured.update(kwargs)
        return _FakeResponse(
            201,
            {
                "id": 555,
                "number": 42,
                "title": "Bug login",
                "html_url": "https://github.com/my-org/demo-repo/issues/42",
                "state": "open",
            },
        )

    client = GitHubClient(auth=_RepoAuth(), allowed_repos={"my-org/demo-repo"})
    monkeypatch.setattr(client._session, "request", _fake_request)

    result = client.create_issue(
        repo_full_name="my-org/demo-repo",
        title="Bug login",
        body="Steps to reproduce",
        labels=["bug"],
        assignees=["khoa"],
    )

    assert result["issue_number"] == 42
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/repos/my-org/demo-repo/issues")
    assert captured["json"] == {
        "title": "Bug login",
        "body": "Steps to reproduce",
        "labels": ["bug"],
        "assignees": ["khoa"],
    }
    assert captured["headers"]["Authorization"] == "Bearer token-123"


def test_create_organization_repository_requires_allowed_org():
    client = GitHubClient(auth=_FakeAuth(), allowed_orgs={"another-org"})

    with pytest.raises(GitHubClientError, match="GITHUB_ALLOWED_ORGS"):
        client.create_organization_repository(org="my-org", name="demo")
