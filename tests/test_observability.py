from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.agent import actions
from src.agent.actions import AgentActionError, create_issue_action
from src.api.main import app
from src.core.observability import (
    InMemoryObservabilityBackend,
    get_observability_backend,
    set_observability_backend,
)
from src.integrations.github_client import GitHubClientError
from src.persistence.action_store import InMemoryActionStore
from src.pipeline import LocalRAGPipeline
from src.rag.retrieval import retriever


def _find_histogram(snapshot, name: str, expected_attributes: dict) -> dict:
    for item in snapshot["histograms"]:
        if item["name"] != name:
            continue
        attributes = item["attributes"]
        if all(attributes.get(key) == value for key, value in expected_attributes.items()):
            return item
    raise AssertionError(f"Histogram not found: {name} {expected_attributes}")


def _find_counter(snapshot, name: str, expected_attributes: dict) -> dict:
    for item in snapshot["counters"]:
        if item["name"] != name:
            continue
        attributes = item["attributes"]
        if all(attributes.get(key) == value for key, value in expected_attributes.items()):
            return item
    raise AssertionError(f"Counter not found: {name} {expected_attributes}")


@pytest.fixture
def observability_backend():
    previous = get_observability_backend()
    backend = InMemoryObservabilityBackend(trace_history_limit=50)
    set_observability_backend(backend)
    try:
        yield backend
    finally:
        set_observability_backend(previous)


def test_request_observability_records_latency_and_span(monkeypatch, observability_backend):
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
        response = client.get("/health", headers={"X-Request-ID": "req-observe-1"})

    assert response.status_code == 200
    snapshot = observability_backend.snapshot()
    histogram = _find_histogram(
        snapshot,
        "http.server.request.duration_ms",
        {
            "method": "GET",
            "path": "/health",
            "status_code": 200,
            "status": "ok",
        },
    )
    assert histogram["count"] == 1
    assert any(
        span["name"] == "http.request" and span["attributes"].get("request_id") == "req-observe-1"
        for span in snapshot["spans"]
    )


def test_retrieval_observability_records_retrieval_and_rerank_metrics(monkeypatch, observability_backend):
    docs = [
        Document(page_content="passkey setup", metadata={"doc_id": "doc-1", "source": "github_docs"}),
        Document(page_content="passkey recovery", metadata={"doc_id": "doc-2", "source": "github_docs"}),
    ]

    class _FakeVectorStore:
        def similarity_search(self, query, k, filter=None):
            return list(docs)

    monkeypatch.setattr(retriever, "get_vector_store", lambda rebuild=False: _FakeVectorStore())
    monkeypatch.setattr(retriever, "_load_parent_map_if_available", lambda: {})
    monkeypatch.setattr(retriever, "rerank_documents", lambda query, docs, top_k: list(docs)[:top_k])
    monkeypatch.setattr(
        retriever,
        "rerank_with_cross_encoder",
        lambda query, docs, top_k: list(docs)[:top_k],
    )
    monkeypatch.setattr(retriever, "USE_CROSS_ENCODER", True)

    result = retriever.retrieve_documents("How do I use a passkey?", top_k=2)

    assert len(result) == 2
    snapshot = observability_backend.snapshot()
    assert _find_histogram(
        snapshot,
        "rag.retrieval.duration_ms",
        {"top_k": 2, "rebuild": False, "has_filter": False, "status": "ok"},
    )["count"] == 1
    assert _find_histogram(
        snapshot,
        "rag.rerank.duration_ms",
        {"stage": "stage1", "strategy": "heuristic", "status": "ok"},
    )["count"] == 1
    assert _find_histogram(
        snapshot,
        "rag.rerank.duration_ms",
        {"stage": "final", "strategy": "cross_encoder", "status": "ok"},
    )["count"] == 1
    assert any(span["name"] == "rag.retrieve" for span in snapshot["spans"])


def test_llm_observability_records_latency(observability_backend):
    class _FakeLLM:
        def invoke(self, prompt):
            return "answer"

    pipeline = LocalRAGPipeline(top_k=4, rebuild=False)
    pipeline.llm = _FakeLLM()

    result = pipeline._invoke_llm("hello")

    assert result == "answer"
    snapshot = observability_backend.snapshot()
    assert _find_histogram(
        snapshot,
        "llm.call.duration_ms",
        {"backend": "local", "entrypoint": "invoke", "status": "ok"},
    )["count"] == 1
    assert any(span["name"] == "llm.call" for span in snapshot["spans"])


def test_action_observability_records_latency_and_counts(monkeypatch, observability_backend):
    class _SuccessGitHubClient:
        def create_issue(self, **kwargs):
            return {
                "issue_number": 42,
                "title": kwargs["title"],
                "html_url": "https://github.com/my-org/demo-repo/issues/42",
                "state": "open",
                "id": 4200,
            }

    class _FailingGitHubClient:
        def create_issue(self, **kwargs):
            raise GitHubClientError("simulated create_issue failure")

    monkeypatch.setattr(actions, "GITHUB_REQUIRE_CONFIRM_FOR_WRITE", False)

    create_issue_action(
        repo_full_name="my-org/demo-repo",
        title="Bug login",
        body="Steps to reproduce",
        confirmed=False,
        github_client=_SuccessGitHubClient(),
        action_store=InMemoryActionStore(),
        idempotency_key="observe-success",
    )

    with pytest.raises(AgentActionError):
        create_issue_action(
            repo_full_name="my-org/demo-repo",
            title="Bug login failed",
            body="Steps to reproduce",
            confirmed=False,
            github_client=_FailingGitHubClient(),
            action_store=InMemoryActionStore(),
            idempotency_key="observe-failed",
        )

    snapshot = observability_backend.snapshot()
    assert _find_histogram(
        snapshot,
        "action.execution.duration_ms",
        {"action_name": "create_issue", "status": "succeeded"},
    )["count"] == 1
    assert _find_histogram(
        snapshot,
        "action.execution.duration_ms",
        {"action_name": "create_issue", "status": "failed"},
    )["count"] == 1
    assert _find_counter(
        snapshot,
        "action.execution.total",
        {"action_name": "create_issue", "status": "succeeded"},
    )["value"] == 1
    assert _find_counter(
        snapshot,
        "action.execution.total",
        {"action_name": "create_issue", "status": "failed"},
    )["value"] == 1
    assert len([span for span in snapshot["spans"] if span["name"] == "action.execute"]) == 2
