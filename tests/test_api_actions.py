from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)
OPERATOR_HEADERS = {
    "X-User-ID": "operator-1",
    "X-User-Role": "operator",
}


def test_create_repo_endpoint_returns_agent_response(monkeypatch):
    monkeypatch.setattr(
        "src.api.main.execute_registered_action",
        lambda action_name, **kwargs: {
            "answer": "created",
            "sources": [],
            "stats": {"action": action_name},
            "debug": [],
            "agent": {
                "route": "action",
                "reason": "ok",
                "tool_calls": [],
            },
        },
    )

    response = client.post(
        "/multi-agent/actions/create-repo",
        json={
            "org": "my-org",
            "name": "demo-repo",
            "confirmed": True,
        },
        headers=OPERATOR_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "created"
    assert body["agent"]["route"] == "action"


def test_create_issue_endpoint_returns_agent_response(monkeypatch):
    monkeypatch.setattr(
        "src.api.main.execute_registered_action",
        lambda action_name, **kwargs: {
            "answer": "issue created",
            "sources": [],
            "stats": {"action": action_name},
            "debug": [],
            "agent": {
                "route": "action",
                "reason": "ok",
                "tool_calls": [],
            },
        },
    )

    response = client.post(
        "/multi-agent/actions/create-issue",
        json={
            "repo_full_name": "my-org/demo-repo",
            "title": "Bug login",
            "body": "Steps to reproduce",
            "confirmed": True,
        },
        headers=OPERATOR_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "issue created"
    assert body["stats"]["action"] == "create_issue"


def test_commit_endpoint_returns_agent_response(monkeypatch):
    monkeypatch.setattr(
        "src.api.main.execute_registered_action",
        lambda action_name, **kwargs: {
            "answer": "committed",
            "sources": [],
            "stats": {"action": action_name},
            "debug": [],
            "agent": {
                "route": "action",
                "reason": "ok",
                "tool_calls": [],
            },
        },
    )

    response = client.post(
        "/multi-agent/actions/commit",
        json={
            "message": "feat: test",
            "confirmed": True,
        },
        headers=OPERATOR_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "committed"
    assert body["stats"]["action"] == "commit"


def test_write_endpoint_rejects_viewer_role(monkeypatch):
    monkeypatch.setattr(
        "src.api.main.execute_registered_action",
        lambda action_name, **kwargs: {
            "answer": "issue created",
            "sources": [],
            "stats": {"action": action_name},
            "debug": [],
            "agent": {
                "route": "action",
                "reason": "ok",
                "tool_calls": [],
            },
        },
    )

    response = client.post(
        "/multi-agent/actions/create-issue",
        json={
            "repo_full_name": "my-org/demo-repo",
            "title": "Bug login",
            "body": "Steps to reproduce",
            "confirmed": True,
        },
    )

    assert response.status_code == 403
    assert "Operator role is required" in response.json()["detail"]
