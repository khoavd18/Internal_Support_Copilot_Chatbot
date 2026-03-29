from __future__ import annotations

from src.core.settings import CROSS_ENCODER_TOP_K, USE_CROSS_ENCODER
from src.rag.indexing.store_manager import get_vector_store
from src.rag.ingest.hierarchical_loader import PARENT_PATH, load_parent_map
from src.rag.retrieval.cross_encoder import rerank_with_cross_encoder
from src.rag.retrieval.parent_merge import merge_leaf_results_with_parents
from src.rag.retrieval.query_analyzer import extract_keyword_groups
from src.rag.retrieval.reranker import rerank_documents


def _load_parent_map_if_available() -> dict[str, dict]:
    if PARENT_PATH.exists():
        return load_parent_map()
    return {}


def _looks_like_leaf(doc) -> bool:
    metadata = doc.metadata or {}
    return (
        str(metadata.get("node_type") or "").lower() == "leaf"
        and bool(metadata.get("parent_id"))
    )


def _origin_key(doc) -> str:
    md = doc.metadata or {}
    return (
        md.get("origin_doc_id")
        or md.get("doc_id")
        or md.get("path")
        or md.get("url")
        or ""
    )


def _node_priority(doc) -> int:
    md = doc.metadata or {}
    node_type = str(md.get("node_type") or "").lower()

    if node_type == "parent_pack":
        return 3
    if node_type == "parent":
        return 2
    if node_type == "leaf":
        return 1
    return 0


def _final_dedupe_by_origin(docs: list, top_k: int) -> list:
    """
    Nếu đã có parent_pack cho một origin_doc_id thì bỏ parent/leaf cùng origin đó.
    Nếu chưa có pack thì giữ doc có priority cao nhất và xuất hiện sớm nhất.
    """
    kept = []
    seen = {}

    for doc in docs:
        key = _origin_key(doc)
        if not key:
            kept.append(doc)
            continue

        if key not in seen:
            seen[key] = doc
            continue

        old_doc = seen[key]
        if _node_priority(doc) > _node_priority(old_doc):
            seen[key] = doc

    used_keys = set()
    for doc in docs:
        key = _origin_key(doc)
        if not key:
            if len(kept) < top_k:
                kept.append(doc)
            continue

        if key in used_keys:
            continue

        chosen = seen[key]
        if chosen is doc:
            kept.append(doc)
            used_keys.add(key)

        if len(kept) >= top_k:
            break

    return kept[:top_k]


def _prefilter_candidates_for_auth_query(query: str, docs: list, top_k: int) -> list:
    """
    Nếu query nghiêng mạnh về GitHub authentication/passkey,
    ưu tiên candidate từ github_docs có auth signal rõ ràng.
    """
    keyword_groups = extract_keyword_groups(query)
    intent_labels = set(keyword_groups.get("intent_labels", []))

    if "github_authentication" not in intent_labels:
        return docs

    filtered = []
    for doc in docs:
        md = doc.metadata or {}
        source = str(md.get("source") or "").lower()
        title = str(md.get("title") or "").lower()
        path = str(md.get("path") or "").lower()
        content_head = doc.page_content[:1200].lower()

        auth_hit = (
            "passkey" in title
            or "passkey" in path
            or "authentication" in title
            or "authentication" in path
            or "sign in" in title
            or "login" in title
            or "passkey" in content_head
            or "authentication" in content_head
        )

        if source == "github_docs" and auth_hit:
            filtered.append(doc)

    # chỉ dùng filter khi còn đủ candidate để tránh lọc quá tay
    if len(filtered) >= max(top_k, 4):
        return filtered

    return docs


def retrieve_documents(query: str, top_k: int = 4, rebuild: bool = False):
    """
    Retrieval v3 - Qdrant native hybrid:
    1) Qdrant hybrid similarity search
    2) prefilter nhẹ theo intent nếu cần
    3) heuristic rerank
    4) merge cùng parent / sibling sections cùng doc
    5) heuristic rerank lần 2
    6) cross-encoder rerank final candidates
    7) final dedupe theo origin_doc_id
    """
    vector_store = get_vector_store(rebuild=rebuild)

    candidate_k = max(top_k * 6, 18)
    rerank_k = max(top_k * 3, 8)
    ce_k = max(CROSS_ENCODER_TOP_K, top_k)

    # QdrantVectorStore đã chạy hybrid ở backend
    candidate_docs = vector_store.similarity_search(
        query,
        k=candidate_k,
    )

    if not candidate_docs:
        return []

    # Prefilter nhẹ cho auth/passkey intent
    candidate_docs = _prefilter_candidates_for_auth_query(
        query=query,
        docs=candidate_docs,
        top_k=top_k,
    )

    # Stage 1: heuristic rerank leaf candidates
    stage1_docs = rerank_documents(
        query=query,
        docs=candidate_docs,
        top_k=rerank_k,
    )

    # Fallback mềm nếu heuristic quá gắt
    if not stage1_docs:
        stage1_docs = candidate_docs[:rerank_k]

    parent_map = _load_parent_map_if_available()

    # Nếu không có hierarchical data hoặc docs không phải leaf -> fallback
    if not parent_map or not any(_looks_like_leaf(doc) for doc in stage1_docs):
        if USE_CROSS_ENCODER:
            final_docs = rerank_with_cross_encoder(
                query=query,
                docs=stage1_docs,
                top_k=max(top_k * 2, top_k),
            )
            return _final_dedupe_by_origin(final_docs, top_k=top_k)

        final_docs = rerank_documents(
            query=query,
            docs=stage1_docs,
            top_k=top_k,
        )
        return _final_dedupe_by_origin(final_docs, top_k=top_k)

    # Stage 2: merge context
    merged_docs = merge_leaf_results_with_parents(
        retrieved_docs=stage1_docs,
        parent_map=parent_map,
        min_children_to_merge=2,
        min_sibling_parents_to_merge=2,
        max_results=rerank_k,
    )

    # Stage 3: heuristic rerank lại sau merge
    stage3_docs = rerank_documents(
        query=query,
        docs=merged_docs,
        top_k=ce_k,
    )

    if not stage3_docs:
        stage3_docs = merged_docs[:ce_k]

    # Stage 4: cross-encoder final rerank
    if USE_CROSS_ENCODER:
        final_docs = rerank_with_cross_encoder(
            query=query,
            docs=stage3_docs,
            top_k=max(top_k * 2, top_k),
        )
        return _final_dedupe_by_origin(final_docs, top_k=top_k)

    return _final_dedupe_by_origin(stage3_docs, top_k=top_k)