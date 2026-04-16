from __future__ import annotations

from langchain_core.documents import Document

from src.agent import tools


def _make_doc(doc_id: str, source: str, source_type: str) -> Document:
    return Document(
        page_content=f"content for {doc_id}",
        metadata={
            "doc_id": doc_id,
            "title": doc_id,
            "path": f"/docs/{doc_id}",
            "source": source,
            "source_type": source_type,
            "rerank_score": 7.5,
        },
    )


def test_search_github_docs_only_keeps_github_source(monkeypatch):
    docs = [
        _make_doc("gh-1", "github_docs", "document"),
        _make_doc("gh-2", "github_docs", "document"),
        _make_doc("gl-1", "gitlab_handbook", "document"),
    ]
    captured = {}

    def _fake_retrieve_documents(**kwargs):
        captured["qdrant_filter"] = kwargs.get("qdrant_filter")
        return docs[:2]

    monkeypatch.setattr(tools, "retrieve_documents", _fake_retrieve_documents)

    result = tools.search_github_docs(query="passkey", top_k=2)

    assert len(result["documents_raw"]) == 2
    assert {doc.metadata["source"] for doc in result["documents_raw"]} == {"github_docs"}
    assert result["stats"]["query_filter_applied"] is True
    source_condition = captured["qdrant_filter"].must[0]
    assert source_condition.key == "metadata.source"
    assert source_condition.match.any == ["github_docs"]


def test_search_github_issues_can_fallback_to_issue_source_type(monkeypatch):
    docs = [
        _make_doc("issue-1", "", "issue"),
        _make_doc("issue-2", "github_issues", "github_issue"),
    ]
    captured = {}

    def _fake_retrieve_documents(**kwargs):
        captured["qdrant_filter"] = kwargs.get("qdrant_filter")
        return docs

    monkeypatch.setattr(tools, "retrieve_documents", _fake_retrieve_documents)

    result = tools.search_github_issues(query="passkey error", top_k=2)

    assert len(result["documents_raw"]) == 2
    assert {doc.metadata["doc_id"] for doc in result["documents_raw"]} == {"issue-1", "issue-2"}
    assert result["stats"]["query_filter_applied"] is True
    assert len(captured["qdrant_filter"].should) == 2
