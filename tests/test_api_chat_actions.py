from __future__ import annotations

from fastapi.testclient import TestClient

from src.agent.memory import clear_history, clear_pending_action
from src.api.main import app


client = TestClient(app)
OPERATOR_HEADERS = {
    "X-User-ID": "operator-1",
    "X-User-Role": "operator",
}


def _cleanup_session(session_id: str) -> None:
    clear_pending_action(session_id)
    clear_history(session_id)


def _fail_if_called():
    raise AssertionError("retrieval graph should not be called for action requests")


def test_multi_agent_ask_returns_issue_confirmation_proposal(monkeypatch):
    session_id = "api-create-issue-proposal"
    _cleanup_session(session_id)
    monkeypatch.setattr("src.agent.actions.GITHUB_REQUIRE_CONFIRM_FOR_WRITE", True)
    monkeypatch.setattr("src.api.main.get_supervisor_graph", _fail_if_called)

    response = client.post(
        "/multi-agent/ask",
        json={
            "question": 'create issue repo:my-org/demo-repo title:"Bug login" body:"Steps to reproduce"',
            "session_id": session_id,
        },
        headers=OPERATOR_HEADERS,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["agent"]["route"] == "action"
    assert body["stats"]["action"] == "create_issue"
    assert body["stats"]["action_status"] == "requires_confirmation"

    _cleanup_session(session_id)


def test_multi_agent_ask_executes_issue_after_yes(monkeypatch):
    session_id = "api-create-issue-confirm"
    _cleanup_session(session_id)
    monkeypatch.setattr("src.agent.actions.GITHUB_REQUIRE_CONFIRM_FOR_WRITE", True)
    monkeypatch.setattr("src.api.main.get_supervisor_graph", _fail_if_called)

    def _fake_create_issue_action(**kwargs):
        assert kwargs["repo_full_name"] == "my-org/demo-repo"
        assert kwargs["confirmed"] is True
        return {
            "answer": "issue created",
            "sources": [],
            "stats": {"action": "create_issue"},
            "debug": [],
            "agent": {
                "route": "action",
                "reason": "ok",
                "tool_calls": [],
            },
        }

    monkeypatch.setattr("src.agent.actions.create_issue_action", _fake_create_issue_action)

    proposal = client.post(
        "/multi-agent/ask",
        json={
            "question": 'create issue repo:my-org/demo-repo title:"Bug login" body:"Steps to reproduce"',
            "session_id": session_id,
        },
        headers=OPERATOR_HEADERS,
    )
    execution = client.post(
        "/multi-agent/ask",
        json={
            "question": "yes",
            "session_id": session_id,
        },
        headers=OPERATOR_HEADERS,
    )

    assert proposal.status_code == 200
    assert execution.status_code == 200
    body = execution.json()
    assert body["answer"] == "issue created"
    assert body["stats"]["action_status"] == "ok"

    _cleanup_session(session_id)


def test_multi_agent_ask_rejects_write_intent_for_viewer(monkeypatch):
    session_id = "api-create-issue-denied"
    _cleanup_session(session_id)
    monkeypatch.setattr("src.agent.actions.GITHUB_REQUIRE_CONFIRM_FOR_WRITE", True)
    monkeypatch.setattr("src.api.main.get_supervisor_graph", _fail_if_called)

    response = client.post(
        "/multi-agent/ask",
        json={
            "question": 'create issue repo:my-org/demo-repo title:"Bug login" body:"Steps to reproduce"',
            "session_id": session_id,
        },
    )

    assert response.status_code == 403
    assert "Operator role is required" in response.json()["detail"]

    _cleanup_session(session_id)
