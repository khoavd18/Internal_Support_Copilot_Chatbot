from __future__ import annotations

from langchain_core.documents import Document
from langchain_qdrant import RetrievalMode
from src.rag.retrieval import enterprise_hybrid as hybrid


def _doc(
    entity_id: str,
    *,
    source_type: str = "ticket",
    customer_id: str = "cust_001",
    text: str = "API timeout evidence for Northstar",
) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": "enterprise_support",
            "source_type": source_type,
            "entity_id": entity_id,
            "ticket_id": entity_id if source_type == "ticket" else "",
            "customer_id": customer_id,
            "title": f"{entity_id} title",
        },
    )


def test_build_enterprise_qdrant_filter_includes_source_and_metadata_filters() -> None:
    qdrant_filter = hybrid.build_enterprise_qdrant_filter(
        {
            "source_type": ["ticket", "risk_event"],
            "customer_id": "cust_001",
            "ignored": "not-supported",
        }
    )

    conditions = {condition.key: condition.match for condition in qdrant_filter.must}

    assert conditions["metadata.source"].value == "enterprise_support"
    assert conditions["metadata.source_type"].any == ["ticket", "risk_event"]
    assert conditions["metadata.customer_id"].value == "cust_001"
    assert "metadata.ignored" not in conditions


def test_fuse_enterprise_search_results_merges_scores_and_debug_metadata() -> None:
    dense_hit = hybrid.HybridSearchHit(
        document=_doc("tkt_001"),
        rank=1,
        score=0.91,
        channel="dense",
    )
    lexical_hit = hybrid.HybridSearchHit(
        document=_doc("tkt_001", text="Exact lexical API timeout ticket evidence"),
        rank=1,
        score=8.0,
        channel="lexical",
    )
    policy_hit = hybrid.HybridSearchHit(
        document=_doc(
            "pol_api_timeout",
            source_type="knowledge_base",
            text="API timeout runbook",
        ),
        rank=2,
        score=5.0,
        channel="lexical",
    )

    documents, debug = hybrid.fuse_enterprise_search_results(
        dense_hits=[dense_hit],
        sparse_or_lexical_hits=[lexical_hit, policy_hit],
        top_k=2,
    )

    assert len(documents) == 2
    assert documents[0].metadata["entity_id"] == "tkt_001"
    assert documents[0].metadata["dense_score"] == 0.91
    assert documents[0].metadata["lexical_score"] == 8.0
    assert documents[0].metadata["fused_score"] > documents[1].metadata["fused_score"]
    assert documents[0].metadata["retrieval_channels"] == ["dense", "lexical"]
    assert documents[0].metadata["matched_metadata"]["customer_id"] == "cust_001"
    assert debug[0]["dense_score"] == 0.91
    assert debug[0]["lexical_score"] == 8.0
    assert debug[0]["fused_score"] == documents[0].metadata["fused_score"]


def test_retrieve_enterprise_hybrid_documents_uses_lexical_fallback_when_sparse_fails(
    monkeypatch,
) -> None:
    captured = []

    def _fake_search_qdrant(*, query, mode, limit, filters, channel):
        captured.append((mode, filters, channel))
        if mode == RetrievalMode.DENSE:
            return [
                hybrid.HybridSearchHit(
                    document=_doc("tkt_001", text="Dense semantic API timeout evidence"),
                    rank=1,
                    score=0.87,
                    channel=channel,
                )
            ]
        raise RuntimeError("sparse vectors are not configured")

    monkeypatch.setattr(hybrid, "_search_qdrant", _fake_search_qdrant)
    monkeypatch.setattr(
        hybrid,
        "_load_enterprise_documents",
        lambda data_dir: (
            _doc("tkt_001", text="Ticket ID tkt_001 API timeout customer cust_001"),
            _doc("tkt_002", customer_id="cust_002", text="Different customer refund issue"),
        ),
    )

    result = hybrid.retrieve_enterprise_hybrid_documents(
        "tkt_001 API timeout",
        top_k=2,
        filters={"customer_id": "cust_001"},
    )

    assert [item[0] for item in captured] == [RetrievalMode.DENSE, RetrievalMode.SPARSE]
    assert captured[0][1] == {"customer_id": ["cust_001"]}
    assert result["stats"]["sparse_mode"] == "local_lexical_fallback"
    assert "sparse vectors are not configured" in result["stats"]["sparse_error"]
    assert result["documents"][0].metadata["entity_id"] == "tkt_001"
    assert result["documents"][0].metadata["dense_score"] == 0.87
    assert result["documents"][0].metadata["lexical_score"] > 0
    assert result["documents"][0].metadata["fused_score"] > 0
    assert result["debug"][0]["matched_metadata"]["customer_id"] == "cust_001"


def test_metadata_matches_filter_supports_enterprise_filter_fields() -> None:
    metadata = {
        "source_type": "ticket",
        "customer_id": "cust_001",
        "service_id": "svc_api_gateway",
    }

    assert hybrid.metadata_matches_filter(metadata, {"customer_id": "cust_001"})
    assert hybrid.metadata_matches_filter(
        metadata,
        {"source_type": ["ticket", "risk_event"], "service_id": "svc_api_gateway"},
    )
    assert not hybrid.metadata_matches_filter(metadata, {"customer_id": "cust_999"})
