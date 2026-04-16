from __future__ import annotations

import pytest

from src.agent import actions
from src.agent.actions import (
    AgentActionError,
    build_failed_action_response,
    commit_changes_action,
    create_issue_action,
    create_repo_action,
    prepare_action_record,
)
from src.integrations.github_client import GitHubClientError
from src.integrations.local_git_client import LocalGitClient, LocalGitClientError
from src.persistence.action_store import InMemoryActionStore


class _FakeGitHubClient:
    def create_organization_repository(self, **kwargs):
        assert kwargs["org"] == "my-org"
        assert kwargs["name"] == "demo-repo"
        return {
            "full_name": "my-org/demo-repo",
            "html_url": "https://github.com/my-org/demo-repo",
            "private": True,
            "default_branch": "main",
        }


class _FakeLocalGitClient:
    def commit(self, **kwargs):
        assert kwargs["message"] == "feat: agent commit"
        return {
            "repo_path": "D:/code/rag/Internal_Support_Copilot",
            "branch": "main",
            "commit_sha": "abc123",
            "message": kwargs["message"],
            "staged_paths": ["src/api/main.py"],
        }


class _FakeGitHubIssueClient:
    def create_issue(self, **kwargs):
        assert kwargs["repo_full_name"] == "my-org/demo-repo"
        assert kwargs["title"] == "Bug login"
        assert kwargs["body"] == "Steps to reproduce"
        assert kwargs["labels"] == ["bug"]
        assert kwargs["assignees"] == ["khoa"]
        return {
            "issue_number": 42,
            "title": kwargs["title"],
            "html_url": "https://github.com/my-org/demo-repo/issues/42",
            "state": "open",
            "id": 4242,
        }


class _CountingGitHubIssueClient:
    def __init__(self):
        self.calls = 0

    def create_issue(self, **kwargs):
        self.calls += 1
        return {
            "issue_number": 100 + self.calls,
            "title": kwargs["title"],
            "html_url": f"https://github.com/{kwargs['repo_full_name']}/issues/{100 + self.calls}",
            "state": "open",
            "id": 5000 + self.calls,
        }


class _FailingGitHubIssueClient:
    def create_issue(self, **kwargs):
        raise GitHubClientError("Repository 'my-org/demo-repo' is not listed in GITHUB_ALLOWED_REPOS")


def test_create_repo_action_requires_confirmation(monkeypatch):
    monkeypatch.setattr(actions, "GITHUB_REQUIRE_CONFIRM_FOR_WRITE", True)

    with pytest.raises(AgentActionError, match="confirmed=true"):
        create_repo_action(
            org="my-org",
            name="demo-repo",
            confirmed=False,
            github_client=_FakeGitHubClient(),
        )


def test_create_repo_action_returns_agent_response(monkeypatch):
    monkeypatch.setattr(actions, "GITHUB_REQUIRE_CONFIRM_FOR_WRITE", False)

    result = create_repo_action(
        org="my-org",
        name="demo-repo",
        confirmed=False,
        github_client=_FakeGitHubClient(),
    )

    assert result["agent"]["route"] == "action"
    assert result["stats"]["action"] == "create_repo"
    assert result["stats"]["repo"] == "my-org/demo-repo"


def test_create_issue_action_requires_confirmation(monkeypatch):
    monkeypatch.setattr(actions, "GITHUB_REQUIRE_CONFIRM_FOR_WRITE", True)

    with pytest.raises(AgentActionError, match="confirmed=true"):
        create_issue_action(
            repo_full_name="my-org/demo-repo",
            title="Bug login",
            body="Steps to reproduce",
            confirmed=False,
            github_client=_FakeGitHubIssueClient(),
        )


def test_create_issue_action_returns_agent_response(monkeypatch):
    monkeypatch.setattr(actions, "GITHUB_REQUIRE_CONFIRM_FOR_WRITE", False)

    result = create_issue_action(
        repo_full_name="my-org/demo-repo",
        title="Bug login",
        body="Steps to reproduce",
        labels=["bug"],
        assignees=["khoa"],
        confirmed=False,
        github_client=_FakeGitHubIssueClient(),
    )

    assert result["agent"]["route"] == "action"
    assert result["stats"]["action"] == "create_issue"
    assert result["stats"]["issue_number"] == 42


def test_create_issue_action_wraps_github_client_errors(monkeypatch):
    monkeypatch.setattr(actions, "GITHUB_REQUIRE_CONFIRM_FOR_WRITE", False)

    with pytest.raises(AgentActionError, match="Review server logs") as exc_info:
        create_issue_action(
            repo_full_name="my-org/demo-repo",
            title="Bug login",
            body="Steps to reproduce",
            confirmed=False,
            github_client=_FailingGitHubIssueClient(),
        )

    assert "GITHUB_ALLOWED_REPOS" not in str(exc_info.value)


def test_create_issue_action_replays_duplicate_execution(monkeypatch):
    monkeypatch.setattr(actions, "GITHUB_REQUIRE_CONFIRM_FOR_WRITE", False)
    action_store = InMemoryActionStore()
    github_client = _CountingGitHubIssueClient()

    first = create_issue_action(
        repo_full_name="my-org/demo-repo",
        title="Bug login",
        body="Steps to reproduce",
        labels=["bug"],
        assignees=["khoa"],
        confirmed=False,
        github_client=github_client,
        action_store=action_store,
    )
    second = create_issue_action(
        repo_full_name="my-org/demo-repo",
        title="Bug login",
        body="Steps to reproduce",
        labels=["bug"],
        assignees=["khoa"],
        confirmed=False,
        github_client=github_client,
        action_store=action_store,
    )

    assert github_client.calls == 1
    assert first["stats"]["action_id"] == second["stats"]["action_id"]
    assert second["stats"]["idempotent_replay"] is True
    assert second["stats"]["action_record_status"] == "succeeded"
    assert second["stats"]["issue_number"] == first["stats"]["issue_number"]


def test_create_issue_action_does_not_rerun_running_record(monkeypatch):
    monkeypatch.setattr(actions, "GITHUB_REQUIRE_CONFIRM_FOR_WRITE", False)
    action_store = InMemoryActionStore()
    github_client = _CountingGitHubIssueClient()

    record = prepare_action_record(
        action_name="create_issue",
        payload={
            "repo_full_name": "my-org/demo-repo",
            "title": "Bug login",
            "body": "Steps to reproduce",
            "labels": ["bug"],
            "assignees": ["khoa"],
        },
        confirmation_required=False,
        action_store=action_store,
    )
    record.status = "running"
    record.confirmed_at = record.confirmed_at or record.created_at
    record.started_at = record.started_at or record.created_at
    record.attempt_count = 1
    action_store.save(record)

    result = create_issue_action(
        repo_full_name="my-org/demo-repo",
        title="Bug login",
        body="Steps to reproduce",
        labels=["bug"],
        assignees=["khoa"],
        confirmed=False,
        github_client=github_client,
        action_store=action_store,
    )

    assert github_client.calls == 0
    assert result["stats"]["action_status"] == "in_progress"
    assert result["stats"]["action_record_status"] == "running"


def test_failed_action_response_hides_internal_error_details():
    record = prepare_action_record(
        action_name="create_issue",
        payload={
            "repo_full_name": "my-org/demo-repo",
            "title": "Bug login",
            "body": "Steps to reproduce",
        },
        confirmation_required=True,
        action_store=InMemoryActionStore(),
    )
    record.status = "failed"
    record.last_error = "Repository 'my-org/demo-repo' is not listed in GITHUB_ALLOWED_REPOS"

    result = build_failed_action_response(record)

    assert "action_last_error" not in result["stats"]
    assert "GITHUB_ALLOWED_REPOS" not in result["agent"]["tool_calls"][0]["note"]


def test_commit_changes_action_requires_feature_flag(monkeypatch):
    monkeypatch.setattr(actions, "LOCAL_GIT_ACTIONS_ENABLED", False)

    with pytest.raises(AgentActionError, match="LOCAL_GIT_ACTIONS_ENABLED"):
        commit_changes_action(
            message="feat: agent commit",
            confirmed=True,
            git_client=_FakeLocalGitClient(),
        )


def test_commit_changes_action_returns_agent_response(monkeypatch):
    monkeypatch.setattr(actions, "LOCAL_GIT_ACTIONS_ENABLED", True)
    monkeypatch.setattr(actions, "LOCAL_GIT_REQUIRE_CONFIRM_FOR_WRITE", False)

    result = commit_changes_action(
        message="feat: agent commit",
        paths=["src/api/main.py"],
        confirmed=False,
        git_client=_FakeLocalGitClient(),
    )

    assert result["agent"]["route"] == "action"
    assert result["stats"]["action"] == "commit"
    assert result["stats"]["commit_sha"] == "abc123"


def test_local_git_client_rejects_paths_outside_repo(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    def _runner(command, cwd, text, capture_output):
        if command[1:] == ["rev-parse", "--show-toplevel"]:
            return _completed(stdout=str(repo_root) + "\n")
        return _completed()

    client = LocalGitClient(default_repo_path=repo_root, allowed_roots=[repo_root], runner=_runner)

    with pytest.raises(LocalGitClientError, match="allowed roots"):
        client.commit(
            message="feat: test",
            paths=[str(tmp_path / "outside.txt")],
        )


def test_local_git_client_commits_requested_paths(tmp_path):
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    target_file = repo_root / "src" / "api" / "main.py"
    target_file.parent.mkdir(parents=True)
    target_file.write_text("print('ok')\n", encoding="utf-8")
    commands = []

    def _runner(command, cwd, text, capture_output):
        commands.append(command)
        args = command[1:]
        if args == ["rev-parse", "--show-toplevel"]:
            return _completed(stdout=str(repo_root) + "\n")
        if args[:2] == ["add", "--"]:
            return _completed()
        if args == ["diff", "--cached", "--name-only"]:
            return _completed(stdout="src/api/main.py\n")
        if args == ["status", "--short"]:
            return _completed(stdout="M  src/api/main.py\n")
        if args[:2] == ["commit", "-m"]:
            return _completed(stdout="[main abc123] feat: test\n")
        if args == ["rev-parse", "HEAD"]:
            return _completed(stdout="abc123\n")
        if args == ["rev-parse", "--abbrev-ref", "HEAD"]:
            return _completed(stdout="main\n")
        raise AssertionError(f"Unexpected command: {command}")

    client = LocalGitClient(default_repo_path=repo_root, allowed_roots=[repo_root], runner=_runner)
    result = client.commit(message="feat: test", paths=["src/api/main.py"])

    assert result["commit_sha"] == "abc123"
    assert result["staged_paths"] == ["src/api/main.py"]
    assert ["git", "add", "--", "src/api/main.py"] in commands


def _completed(*, stdout: str = "", stderr: str = "", returncode: int = 0):
    import subprocess

    return subprocess.CompletedProcess(
        args=["git"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )
