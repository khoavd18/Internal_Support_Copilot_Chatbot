from __future__ import annotations

import fakeredis
from langgraph.checkpoint.base import empty_checkpoint

from src.agent.chat_actions import maybe_handle_chat_action
from src.agent.memory import (
    clear_history,
    clear_pending_action,
    configure_session_state_store,
    reset_session_state_store,
)
from src.agent.actions import create_issue_action
from src.persistence.action_store import RedisActionStore
from src.persistence.checkpoints import RedisCheckpointSaver
from src.persistence.session_store import RedisSessionStateStore


def _build_fake_redis_pair():
    server = fakeredis.FakeServer()
    return (
        fakeredis.FakeRedis(server=server, decode_responses=False),
        fakeredis.FakeRedis(server=server, decode_responses=False),
    )


def _cleanup_session(session_id: str) -> None:
    clear_pending_action(session_id)
    clear_history(session_id)


def test_redis_session_store_persists_history_across_instances():
    client1, client2 = _build_fake_redis_pair()
    store1 = RedisSessionStateStore(client1, key_prefix="test-history", max_history_messages=3)
    store2 = RedisSessionStateStore(client2, key_prefix="test-history", max_history_messages=3)

    store1.append_turn("session-a", "user", "first")
    store1.append_turn("session-a", "assistant", "second")
    store1.append_turn("session-a", "user", "third")
    store1.append_turn("session-a", "assistant", "fourth")

    assert store2.get_history("session-a") == [
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
        {"role": "assistant", "content": "fourth"},
    ]


def test_confirmation_flow_continues_after_store_recreation(monkeypatch):
    client1, client2 = _build_fake_redis_pair()
    session_id = "redis-confirmation-session"
    store1 = RedisSessionStateStore(client1, key_prefix="test-confirm")
    store2 = RedisSessionStateStore(client2, key_prefix="test-confirm")

    try:
        configure_session_state_store(store1)
        _cleanup_session(session_id)
        monkeypatch.setattr("src.agent.actions.GITHUB_REQUIRE_CONFIRM_FOR_WRITE", True)

        first = maybe_handle_chat_action(
            question='create issue repo:my-org/demo-repo title:"Bug login" body:"Steps to reproduce"',
            confirmed=False,
            session_id=session_id,
            backend_mode="multi_agent",
        )

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
        configure_session_state_store(store2)

        second = maybe_handle_chat_action(
            question="yes",
            confirmed=False,
            session_id=session_id,
            backend_mode="multi_agent",
        )

        assert first is not None
        assert first["stats"]["action_status"] == "requires_confirmation"
        assert second is not None
        assert second["answer"] == "issue created"
        assert second["stats"]["action_status"] == "ok"
        assert executed["repo_full_name"] == "my-org/demo-repo"
        assert executed["confirmed"] is True
    finally:
        _cleanup_session(session_id)
        reset_session_state_store()


def test_redis_checkpoint_saver_persists_checkpoint_and_writes_across_instances():
    client1, client2 = _build_fake_redis_pair()
    saver1 = RedisCheckpointSaver(client1, key_prefix="test-checkpoints")
    saver2 = RedisCheckpointSaver(client2, key_prefix="test-checkpoints")

    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"history": [{"role": "user", "content": "hello"}]}
    checkpoint["channel_versions"] = {"history": "00000000000000000000000000000001.0000000000000001"}
    config = {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}

    stored_config = saver1.put(
        config,
        checkpoint,
        {"source": "unit-test"},
        checkpoint["channel_versions"],
    )
    saver1.put_writes(
        stored_config,
        [("__resume__", {"confirmed": True}), ("result", {"answer": "done"})],
        task_id="task-1",
    )

    restored = saver2.get_tuple({"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}})

    assert restored is not None
    assert restored.metadata["source"] == "unit-test"
    assert restored.checkpoint["channel_values"]["history"][0]["content"] == "hello"
    assert ("task-1", "__resume__", {"confirmed": True}) in restored.pending_writes
    assert ("task-1", "result", {"answer": "done"}) in restored.pending_writes


def test_redis_action_store_replays_duplicate_execution_across_instances(monkeypatch):
    client1, client2 = _build_fake_redis_pair()
    store1 = RedisActionStore(client1, key_prefix="test-actions")
    store2 = RedisActionStore(client2, key_prefix="test-actions")

    issued = {"calls": 0}

    class _FakeGitHubIssueClient:
        def create_issue(self, **kwargs):
            issued["calls"] += 1
            return {
                "issue_number": 88,
                "title": kwargs["title"],
                "html_url": "https://github.com/my-org/demo-repo/issues/88",
                "state": "open",
                "id": 8800,
            }

    monkeypatch.setattr("src.agent.actions.GITHUB_REQUIRE_CONFIRM_FOR_WRITE", False)

    first = create_issue_action(
        repo_full_name="my-org/demo-repo",
        title="Bug login",
        body="Steps to reproduce",
        labels=["bug"],
        assignees=["khoa"],
        confirmed=False,
        github_client=_FakeGitHubIssueClient(),
        action_store=store1,
        session_id="redis-idempotent-session",
    )
    second = create_issue_action(
        repo_full_name="my-org/demo-repo",
        title="Bug login",
        body="Steps to reproduce",
        labels=["bug"],
        assignees=["khoa"],
        confirmed=False,
        github_client=_FakeGitHubIssueClient(),
        action_store=store2,
        session_id="redis-idempotent-session",
    )

    assert issued["calls"] == 1
    assert first["stats"]["action_id"] == second["stats"]["action_id"]
    assert second["stats"]["idempotent_replay"] is True
    assert second["stats"]["action_record_status"] == "succeeded"
