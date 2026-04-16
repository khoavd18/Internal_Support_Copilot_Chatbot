from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.api.main import app
from src.core import runtime_checks


def test_validate_environment_reports_missing_github_configuration(monkeypatch, tmp_path):
    monkeypatch.setattr(runtime_checks, "GITHUB_ACTIONS_ENABLED", True)
    monkeypatch.setattr(runtime_checks, "GITHUB_APP_ID", "")
    monkeypatch.setattr(runtime_checks, "GITHUB_CLIENT_ID", "")
    monkeypatch.setattr(runtime_checks, "GITHUB_PRIVATE_KEY_PATH", "")
    monkeypatch.setattr(runtime_checks, "GITHUB_ALLOWED_REPOS", set())
    monkeypatch.setattr(runtime_checks, "GITHUB_ALLOWED_ORGS", set())
    monkeypatch.setattr(runtime_checks, "DOCUMENTS_PATH", tmp_path / "documents.jsonl")
    monkeypatch.setattr(runtime_checks, "TICKETS_PATH", tmp_path / "tickets.jsonl")

    report = runtime_checks.validate_environment_settings()

    assert report["ok"] is False
    assert any("GITHUB_APP_ID or GITHUB_CLIENT_ID" in message for message in report["errors"])
    assert any("GITHUB_PRIVATE_KEY_PATH" in message for message in report["errors"])
    assert any("GITHUB_ALLOWED_REPOS" in message for message in report["errors"])


def test_build_readiness_report_includes_qdrant_diagnostics(monkeypatch, tmp_path):
    documents_path = tmp_path / "documents.jsonl"
    documents_path.write_text('{"id":"doc-1"}\n', encoding="utf-8")

    class _FakeQdrantClient:
        def get_collections(self):
            return SimpleNamespace(collections=[SimpleNamespace(name="other_collection")])

    monkeypatch.setattr(runtime_checks, "_build_qdrant_client", lambda: _FakeQdrantClient())
    monkeypatch.setattr(runtime_checks, "DOCUMENTS_PATH", documents_path)
    monkeypatch.setattr(runtime_checks, "TICKETS_PATH", tmp_path / "tickets.jsonl")
    monkeypatch.setattr(runtime_checks, "INCLUDE_TICKETS", False)

    report = runtime_checks.build_readiness_report()

    assert report["ready"] is False
    assert report["checks"]["qdrant"]["server_reachable"] is True
    assert report["checks"]["qdrant"]["collection_exists"] is False
    assert "ingest-data" in report["checks"]["qdrant"]["hint"]


def test_health_endpoint_returns_dependency_summary(monkeypatch):
    monkeypatch.setattr(
        "src.api.main.build_readiness_report",
        lambda: {
            "ready": True,
            "status": "ready",
            "checks": {
                "qdrant": {"required": True, "ok": True, "points_count": 42},
                "processed_data": {"required": False, "ok": True},
                "github_actions": {"required": False, "ok": True, "status": "disabled"},
                "local_git": {"required": False, "ok": True, "status": "disabled"},
            },
        },
    )

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["dependency_ready"] is True
    assert body["dependencies"]["qdrant"]["points_count"] == 42
    assert body["authorization"]["provider"] == "header"


def test_ready_endpoint_returns_503_when_dependencies_are_not_ready(monkeypatch):
    monkeypatch.setattr(
        "src.api.main.build_readiness_report",
        lambda: {
            "ready": False,
            "status": "not_ready",
            "checks": {
                "qdrant": {
                    "required": True,
                    "ok": False,
                    "error": "Collection missing",
                },
            },
        },
    )

    with TestClient(app) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["qdrant"]["error"] == "Collection missing"


def test_health_endpoint_sanitizes_internal_error_strings(monkeypatch):
    monkeypatch.setenv("GITHUB_PRIVATE_KEY_PATH", r"C:\secret\github-app.pem")
    monkeypatch.setattr(
        "src.api.main.build_readiness_report",
        lambda: {
            "ready": True,
            "status": "ready",
            "checks": {
                "qdrant": {"required": True, "ok": True},
                "processed_data": {"required": False, "ok": True},
                "github_actions": {"required": False, "ok": True},
                "local_git": {"required": False, "ok": True},
            },
        },
    )

    def _raise_agent():
        raise RuntimeError("token=abc123 private_key_path=C:\\secret\\github-app.pem")

    class _FakePipelineFactory:
        @staticmethod
        def cache_info():
            return SimpleNamespace(currsize=1)

        def __call__(self):
            raise RuntimeError("Authorization: Bearer bearer-456")

    monkeypatch.setattr("src.api.main.get_agent", _raise_agent)
    monkeypatch.setattr("src.api.main.get_default_pipeline", _FakePipelineFactory())

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert "abc123" not in body["agent_error"]
    assert "github-app.pem" not in body["agent_error"]
    assert "bearer-456" not in body["llm_error"]
