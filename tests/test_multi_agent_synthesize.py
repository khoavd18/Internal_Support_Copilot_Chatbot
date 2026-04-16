from __future__ import annotations

from langchain_core.documents import Document

from src.agent.graph.nodes.synthesize import merge_results_node, synthesize_answer_node


def _make_doc(doc_id: str, source: str, score: float) -> Document:
    return Document(
        page_content=f"content for {doc_id}",
        metadata={
            "doc_id": doc_id,
            "title": doc_id,
            "path": f"/docs/{doc_id}",
            "source": source,
            "source_type": "document",
            "rerank_score": score,
        },
    )


def test_merge_results_dedupes_and_sorts_by_score():
    weaker = _make_doc("shared", "github_docs", 4.2)
    stronger = _make_doc("shared", "github_docs", 8.5)
    other = _make_doc("gitlab", "gitlab_handbook", 7.1)

    state = {
        "github_docs_result": {
            "documents_raw": [weaker],
            "sources": [
                {
                    "doc_id": "shared",
                    "title": "shared",
                    "path": "/docs/shared",
                    "source": "github_docs",
                    "rerank_score": 4.2,
                }
            ],
            "debug": [{"doc_id": "shared"}],
        },
        "gitlab_result": {
            "documents_raw": [stronger, other],
            "sources": [
                {
                    "doc_id": "shared",
                    "title": "shared",
                    "path": "/docs/shared",
                    "source": "github_docs",
                    "rerank_score": 8.5,
                },
                {
                    "doc_id": "gitlab",
                    "title": "gitlab",
                    "path": "/docs/gitlab",
                    "source": "gitlab_handbook",
                    "rerank_score": 7.1,
                },
            ],
            "debug": [{"doc_id": "gitlab"}],
        },
        "issues_result": {},
    }

    merged = merge_results_node(state)

    assert [doc.metadata["doc_id"] for doc in merged["merged_documents"]] == ["shared", "gitlab"]
    assert [item["doc_id"] for item in merged["merged_sources"]] == ["shared", "gitlab"]
    assert merged["merged_sources"][0]["index"] == 1
    assert len(merged["merged_debug"]) == 2


def test_synthesize_answer_falls_back_to_search_when_scores_are_weak():
    doc1 = _make_doc("gh-low", "github_docs", 4.8)
    doc2 = _make_doc("gl-low", "gitlab_handbook", 4.6)

    result = synthesize_answer_node(
        {
            "question": "passkey error",
            "effective_question": "passkey error",
            "top_k": 4,
            "debug": True,
            "selected_agents": ["github_docs", "gitlab"],
            "route_reason": "Supervisor selected github_docs and gitlab",
            "response_route": "answer_from_kb",
            "merged_documents": [doc1, doc2],
            "merged_sources": [
                {
                    "index": 1,
                    "doc_id": "gh-low",
                    "title": "gh-low",
                    "path": "/docs/gh-low",
                    "source": "github_docs",
                    "rerank_score": 4.8,
                },
                {
                    "index": 2,
                    "doc_id": "gl-low",
                    "title": "gl-low",
                    "path": "/docs/gl-low",
                    "source": "gitlab_handbook",
                    "rerank_score": 4.6,
                },
            ],
            "merged_debug": [{"doc_id": "gh-low"}, {"doc_id": "gl-low"}],
            "github_docs_result": {"documents_raw": [doc1]},
            "gitlab_result": {"documents_raw": [doc2]},
        }
    )

    assert result["agent"]["route"] == "retrieve_only"
    assert result["stats"]["guardrail_action"] == "fallback_to_search"
    assert result["debug"] == [{"doc_id": "gh-low"}, {"doc_id": "gl-low"}]
