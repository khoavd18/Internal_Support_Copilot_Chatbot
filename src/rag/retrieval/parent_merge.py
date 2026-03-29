from __future__ import annotations

import hashlib
from collections import defaultdict

from langchain_core.documents import Document


def _stable_hash(text: str, n: int = 16) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _build_parent_document(parent_row: dict, merged_from_leaf_count: int) -> Document:
    parent_text = parent_row.get("text", "")
    parent_metadata = {k: v for k, v in parent_row.items() if k != "text"}
    parent_metadata["node_type"] = "parent"
    parent_metadata["merged_from_leaf_count"] = merged_from_leaf_count
    parent_metadata["merge_strategy"] = "same_parent"

    return Document(
        page_content=parent_text,
        metadata=parent_metadata,
    )


def _build_parent_pack_document(
    origin_doc_id: str,
    parent_rows: list[dict],
    merged_leaf_count: int,
) -> Document:
    ordered_rows = sorted(
        parent_rows,
        key=lambda row: _safe_int(row.get("parent_index"), 0),
    )

    texts = []
    parent_ids = []

    for row in ordered_rows:
        text = str(row.get("text") or "").strip()
        if not text:
            continue
        texts.append(text)
        parent_ids.append(str(row.get("parent_id") or "").strip())

    packed_text = "\n\n---\n\n".join(texts).strip()

    first = ordered_rows[0] if ordered_rows else {}
    pack_id = f"pack_{_stable_hash(origin_doc_id + '::' + '||'.join(parent_ids))}"

    metadata = {k: v for k, v in first.items() if k != "text"}
    metadata["node_type"] = "parent_pack"
    metadata["pack_id"] = pack_id
    metadata["doc_id"] = f"{origin_doc_id}::pack"
    metadata["origin_doc_id"] = origin_doc_id
    metadata["parent_ids"] = parent_ids
    metadata["merged_from_parent_count"] = len(parent_ids)
    metadata["merged_from_leaf_count"] = merged_leaf_count
    metadata["merge_strategy"] = "sibling_sections"

    return Document(
        page_content=packed_text,
        metadata=metadata,
    )


def merge_leaf_results_with_parents(
    retrieved_docs: list[Document],
    parent_map: dict[str, dict],
    min_children_to_merge: int = 2,
    min_sibling_parents_to_merge: int = 2,
    max_results: int | None = None,
) -> list[Document]:
    """
    v2.1 heuristic:
    1) nếu >= N leaf cùng parent_id -> thay bằng parent
    2) nếu >= N parent khác nhau nhưng cùng origin_doc_id -> thay bằng parent_pack
    """
    ranked_docs = list(enumerate(retrieved_docs))

    parent_groups = defaultdict(list)
    doc_groups = defaultdict(list)
    passthrough = []

    for rank, doc in ranked_docs:
        metadata = doc.metadata or {}
        parent_id = str(metadata.get("parent_id") or "").strip()
        origin_doc_id = str(metadata.get("origin_doc_id") or "").strip()

        if parent_id and parent_id in parent_map:
            parent_groups[parent_id].append((rank, doc))
            if origin_doc_id:
                doc_groups[origin_doc_id].append((rank, doc))
        else:
            passthrough.append((rank, doc))

    merged_results = []
    consumed_parent_ids = set()

    # Phase 1: merge leaf cùng parent_id
    for parent_id, items in parent_groups.items():
        best_rank = min(rank for rank, _ in items)

        if len(items) >= min_children_to_merge:
            parent_row = parent_map[parent_id]
            merged_doc = _build_parent_document(
                parent_row=parent_row,
                merged_from_leaf_count=len(items),
            )
            merged_results.append((best_rank, merged_doc))
            consumed_parent_ids.add(parent_id)

    # Phase 2: merge sibling sections cùng origin_doc_id
    for origin_doc_id, items in doc_groups.items():
        remaining_parent_ids = []
        merged_leaf_count = 0

        for _, doc in items:
            parent_id = str(doc.metadata.get("parent_id") or "").strip()
            if not parent_id or parent_id in consumed_parent_ids:
                continue
            if parent_id not in remaining_parent_ids:
                remaining_parent_ids.append(parent_id)
            merged_leaf_count += 1

        if len(remaining_parent_ids) < min_sibling_parents_to_merge:
            continue

        parent_rows = []
        for pid in remaining_parent_ids[:3]:
            row = parent_map.get(pid)
            if row:
                parent_rows.append(row)

        if not parent_rows:
            continue

        best_rank = min(
            rank
            for rank, doc in items
            if str(doc.metadata.get("parent_id") or "").strip() in remaining_parent_ids
        )

        pack_doc = _build_parent_pack_document(
            origin_doc_id=origin_doc_id,
            parent_rows=parent_rows,
            merged_leaf_count=merged_leaf_count,
        )
        merged_results.append((best_rank, pack_doc))

        for pid in remaining_parent_ids:
            consumed_parent_ids.add(pid)

    # Phase 3: passthrough leaf chưa bị consume
    for rank, doc in ranked_docs:
        parent_id = str(doc.metadata.get("parent_id") or "").strip()

        if parent_id and parent_id in consumed_parent_ids:
            continue

        if not parent_id or parent_id not in parent_map:
            continue

        merged_results.append((rank, doc))

    merged_results.extend(passthrough)
    merged_results.sort(key=lambda x: x[0])

    docs = [doc for _, doc in merged_results]

    if max_results is not None:
        docs = docs[:max_results]

    return docs