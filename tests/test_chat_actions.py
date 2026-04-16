from __future__ import annotations

from types import SimpleNamespace

from src.agent.action_registry import (
    action_requires_confirmation,
    detect_action_request,
    execute_registered_action,
    get_action_definition,
)
from src.agent.chat_actions import maybe_handle_chat_action
from src.agent.memory import clear_history, clear_pending_action
from src.persistence.action_store import InMemoryActionStore


def _cleanup_session(session_id: str) -> None:
    clear_pending_action(session_id)
    clear_history(session_id)


def test_detect_action_request_for_create_issue():
    intent = detect_action_request(
        'create issue repo:my-org/demo-repo title:"Bug login" body:"Steps to reproduce" labels:bug,auth assignees:khoa'
    )

    assert intent is not None
    assert intent.action == "create_issue"
    assert intent.payload["repo_full_name"] == "my-org/demo-repo"
    assert intent.payload["labels"] == ["bug", "auth"]
    assert intent.payload["assignees"] == ["khoa"]


def test_detect_action_request_ignores_commit_how_to_question():
    intent = detect_action_request("How do I commit changes to GitHub?")
    assert intent is None


def test_detect_action_request_returns_clarification_for_incomplete_issue():
    intent = detect_action_request('create issue repo:my-org/demo-repo title:"Bug login"')

    assert intent is not None
    assert intent.action == "create_issue"
    assert intent.needs_clarification is True
    assert "body" in intent.reason


def test_detect_action_request_parses_commit_flags_and_paths():
    intent = detect_action_request(
        'commit message:"feat: add automation" files:src/api/main.py,src/agent/actions.py stage all include untracked'
    )

    assert intent is not None
    assert intent.action == "commit"
    assert intent.payload["message"] == "feat: add automation"
    assert intent.payload["paths"] == ["src/api/main.py", "src/agent/actions.py"]
    assert intent.payload["stage_all"] is True
    assert intent.payload["include_untracked"] is True


def test_chat_action_returns_confirmation_proposal(monkeypatch):
    session_id = "session-create-issue-proposal"
    _cleanup_session(session_id)
    monkeypatch.setattr("src.agent.actions.GITHUB_REQUIRE_CONFIRM_FOR_WRITE", True)

    result = maybe_handle_chat_action(
        question='create issue repo:my-org/demo-repo title:"Bug login" body:"Steps to reproduce"',
        confirmed=False,
        session_id=session_id,
        backend_mode="multi_agent",
    )

    assert result is not None
    assert result["agent"]["route"] == "action"
    assert result["stats"]["action_status"] == "requires_confirmation"
    assert result["agent"]["tool_calls"][0]["status"] == "skipped"
    assert "confirmed=true" in result["answer"]

    _cleanup_session(session_id)


def test_chat_action_executes_pending_issue_after_confirmation(monkeypatch):
    session_id = "session-create-issue-confirm"
    _cleanup_session(session_id)
    monkeypatch.setattr("src.agent.actions.GITHUB_REQUIRE_CONFIRM_FOR_WRITE", True)

    executed = {}

    def _fake_create_issue_action(**kwargs):
        executed.update(kwargs)
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

    first = maybe_handle_chat_action(
        question='create issue repo:my-org/demo-repo title:"Bug login" body:"Steps to reproduce"',
        confirmed=False,
        session_id=session_id,
        backend_mode="multi_agent",
    )
    second = maybe_handle_chat_action(
        question="yes",
        confirmed=False,
        session_id=session_id,
        backend_mode="multi_agent",
    )

    assert first is not None
    assert second is not None
    assert executed["repo_full_name"] == "my-org/demo-repo"
    assert executed["confirmed"] is True
    assert second["answer"] == "issue created"
    assert second["stats"]["action_status"] == "ok"

    _cleanup_session(session_id)


def test_chat_action_reuses_action_record_for_duplicate_requests(monkeypatch):
    session_id = "session-create-issue-idempotent"
    _cleanup_session(session_id)
    action_store = InMemoryActionStore()
    monkeypatch.setattr("src.agent.actions.GITHUB_REQUIRE_CONFIRM_FOR_WRITE", True)

    client = SimpleNamespace(calls=0)

    def _create_issue(**kwargs):
        client.calls += 1
        return {
            "issue_number": 77,
            "title": kwargs["title"],
            "html_url": "https://github.com/my-org/demo-repo/issues/77",
            "state": "open",
            "id": 7700,
        }

    monkeypatch.setattr(
        "src.agent.actions.GitHubClient",
        lambda: SimpleNamespace(create_issue=_create_issue),
    )

    first = maybe_handle_chat_action(
        question='create issue repo:my-org/demo-repo title:"Bug login" body:"Steps to reproduce"',
        confirmed=False,
        session_id=session_id,
        backend_mode="multi_agent",
        action_store=action_store,
    )
    duplicate_request = maybe_handle_chat_action(
        question='create issue repo:my-org/demo-repo title:"Bug login" body:"Steps to reproduce"',
        confirmed=False,
        session_id=session_id,
        backend_mode="multi_agent",
        action_store=action_store,
    )
    confirmation = maybe_handle_chat_action(
        question="yes",
        confirmed=False,
        session_id=session_id,
        backend_mode="multi_agent",
        action_store=action_store,
    )
    replay = maybe_handle_chat_action(
        question='create issue repo:my-org/demo-repo title:"Bug login" body:"Steps to reproduce"',
        confirmed=False,
        session_id=session_id,
        backend_mode="multi_agent",
        action_store=action_store,
    )

    assert first is not None
    assert duplicate_request is not None
    assert confirmation is not None
    assert replay is not None
    assert client.calls == 1
    assert first["stats"]["action_record_status"] == "pending"
    assert duplicate_request["stats"]["action_record_status"] == "pending"
    assert confirmation["stats"]["action_record_status"] == "succeeded"
    assert replay["stats"]["idempotent_replay"] is True
    assert replay["stats"]["action_record_status"] == "succeeded"

    _cleanup_session(session_id)


def test_action_registry_exposes_metadata_and_runtime_confirmation(monkeypatch):
    definition = get_action_definition("create_issue")

    monkeypatch.setattr("src.agent.actions.GITHUB_REQUIRE_CONFIRM_FOR_WRITE", True)
    assert definition.permission_name == "create_issue"
    assert definition.tool_name == "create_issue"
    assert action_requires_confirmation("create_issue") is True


def test_execute_registered_action_routes_to_registered_executor(monkeypatch):
    executed = {}

    def _fake_create_issue_action(**kwargs):
        executed.update(kwargs)
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

    result = execute_registered_action(
        "create_issue",
        payload={
            "repo_full_name": "my-org/demo-repo",
            "title": "Bug login",
            "body": "Steps to reproduce",
            "labels": ["bug"],
            "assignees": ["khoa"],
        },
        confirmed=True,
        session_id="chat-registry-session",
        idempotency_key="registry-issue-1",
    )

    assert result["answer"] == "issue created"
    assert executed["repo_full_name"] == "my-org/demo-repo"
    assert executed["confirmed"] is True
    assert executed["session_id"] == "chat-registry-session"
