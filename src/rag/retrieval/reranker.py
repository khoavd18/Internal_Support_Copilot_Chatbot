from __future__ import annotations

from typing import Any, Dict, List, Tuple

from src.rag.retrieval.query_analyzer import (
    PLATFORM_TERMS,
    extract_keyword_groups,
    normalize_text,
    unique_keep_order,
)


def _expand_term_variants(term: str) -> list[str]:
    """
    Nới nhẹ matching cho các biến thể phổ biến:
    - manage -> managing
    - passkey -> passkeys
    - methods -> method
    """
    term = normalize_text(term)
    variants = {term}

    if term.endswith("e") and len(term) > 3:
        variants.add(term[:-1] + "ing")

    if term.endswith("ing") and len(term) > 5:
        variants.add(term[:-3])
        if not term[:-3].endswith("e"):
            variants.add(term[:-3] + "e")

    if not term.endswith("s") and len(term) > 3:
        variants.add(term + "s")

    if term.endswith("s") and len(term) > 4:
        variants.add(term[:-1])

    return [v for v in variants if v]


def get_doc_identity(doc) -> str:
    metadata = doc.metadata or {}
    node_type = str(metadata.get("node_type") or "").lower()

    if node_type == "leaf":
        return (
            metadata.get("leaf_id")
            or metadata.get("chunk_id")
            or metadata.get("doc_id")
            or metadata.get("path")
            or doc.page_content[:80]
        )

    if node_type == "parent":
        return (
            metadata.get("parent_id")
            or metadata.get("doc_id")
            or metadata.get("path")
            or doc.page_content[:80]
        )

    if node_type == "parent_pack":
        return (
            metadata.get("pack_id")
            or metadata.get("origin_doc_id")
            or metadata.get("doc_id")
            or doc.page_content[:80]
        )

    return (
        metadata.get("chunk_id")
        or metadata.get("doc_id")
        or metadata.get("title")
        or metadata.get("url")
        or metadata.get("path")
        or doc.page_content[:80]
    )


def get_doc_fields(doc) -> Dict[str, str]:
    metadata = doc.metadata or {}

    title = normalize_text(str(metadata.get("title", "")))
    doc_id = normalize_text(str(metadata.get("doc_id", "")))
    source = normalize_text(str(metadata.get("source", "")))
    source_type = normalize_text(str(metadata.get("source_type", "")))
    path = normalize_text(str(metadata.get("path", "")))
    url = normalize_text(str(metadata.get("url", "")))
    content_head = normalize_text(doc.page_content[:1800])

    return {
        "title": title,
        "doc_id": doc_id,
        "source": source,
        "source_type": source_type,
        "path": path,
        "url": url,
        "content_head": content_head,
    }


def term_hit_locations(term: str, fields: Dict[str, str]) -> Dict[str, bool]:
    variants = _expand_term_variants(term)

    title_blob = " ".join([
        fields["title"],
        fields["source"],
        fields["path"],
        fields["url"],
    ])
    id_blob = fields["doc_id"]
    content_blob = fields["content_head"]

    return {
        "title": any(v in title_blob for v in variants),
        "doc_id": any(v in id_blob for v in variants),
        "content": any(v in content_blob for v in variants),
    }


def detect_opposite_platform_hit(
    query_platforms: List[str],
    fields: Dict[str, str],
) -> bool:
    if not query_platforms:
        return False

    blob = " ".join([
        fields["title"],
        fields["doc_id"],
        fields["source"],
        fields["path"],
        fields["url"],
        fields["content_head"][:500],
    ])

    opposites = PLATFORM_TERMS.difference(set(query_platforms))
    return any(platform in blob for platform in opposites)


def score_document(doc, keyword_groups: Dict[str, List[str]]) -> Tuple[float, Dict[str, Any]]:
    fields = get_doc_fields(doc)
    metadata = doc.metadata or {}

    score = 0.0
    platform_hits: List[str] = []
    strong_hits: List[str] = []
    weak_hits: List[str] = []

    for kw in keyword_groups["platform_keywords"]:
        hit = term_hit_locations(kw, fields)
        if hit["title"]:
            score += 7.0
            platform_hits.append(kw)
        elif hit["doc_id"]:
            score += 5.0
            platform_hits.append(kw)
        elif hit["content"]:
            score += 2.5
            platform_hits.append(kw)

    opposite_platform_hit = detect_opposite_platform_hit(
        keyword_groups["platform_keywords"],
        fields,
    )

    if keyword_groups["platform_keywords"] and not platform_hits:
        score -= 4.0

    if opposite_platform_hit:
        score -= 4.0

    for kw in keyword_groups["strong_keywords"]:
        hit = term_hit_locations(kw, fields)
        if hit["title"]:
            score += 8.0
            strong_hits.append(kw)
        elif hit["doc_id"]:
            score += 5.0
            strong_hits.append(kw)
        elif hit["content"]:
            score += 2.5
            strong_hits.append(kw)

    for kw in keyword_groups["weak_keywords"]:
        hit = term_hit_locations(kw, fields)
        if hit["title"]:
            score += 2.0
            weak_hits.append(kw)
        elif hit["doc_id"]:
            score += 1.5
            weak_hits.append(kw)
        elif hit["content"]:
            score += 0.8
            weak_hits.append(kw)

    source_type = str(metadata.get("source_type", "")).lower()
    source = str(metadata.get("source", "")).lower()
    node_type = str(metadata.get("node_type", "")).lower()
    preferred_sources = set(keyword_groups.get("preferred_sources", []))
    penalized_sources = set(keyword_groups.get("penalized_sources", []))
    phrase_keywords = keyword_groups.get("phrase_keywords", [])
    intent_labels = set(keyword_groups.get("intent_labels", []))

    if source in preferred_sources:
        score += 4.0

    if source in penalized_sources:
        score -= 4.5

    for phrase in phrase_keywords:
        phrase = normalize_text(phrase)
        if not phrase:
            continue

        if phrase in fields["title"]:
            score += 6.0
        elif phrase in fields["doc_id"] or phrase in fields["path"]:
            score += 4.0
        elif phrase in fields["content_head"]:
            score += 2.0

    if "github_authentication" in intent_labels:
        if source == "github_docs":
            score += 3.0

        if source == "gitlab_handbook":
            score -= 5.0

        if any(term in fields["title"] for term in ["passkey", "authentication", "sign in", "login"]):
            score += 4.0

    if source in {"github_docs", "gitlab_handbook"}:
        score += 0.5

    if source_type in {"issue", "ticket"} or source in {"github_issues"}:
        score -= 0.5

    if node_type == "parent":
        score += 0.4

    if node_type == "parent_pack":
        score += 0.8

    if keyword_groups["strong_keywords"] and not strong_hits:
        score -= 3.0

    if not strong_hits and any(
        term in fields["title"] for term in ["about", "overview", "getting started"]
    ):
        score -= 1.0

    info = {
        "platform_hits": unique_keep_order(platform_hits),
        "strong_hits": unique_keep_order(strong_hits),
        "weak_hits": unique_keep_order(weak_hits),
        "opposite_platform_hit": opposite_platform_hit,
    }
    return score, info


def rerank_documents(query: str, docs: list, top_k: int) -> list:
    keyword_groups = extract_keyword_groups(query)

    if not docs:
        return []

    scored_docs = []
    for doc in docs:
        rerank_score, info = score_document(doc, keyword_groups)
        doc.metadata["rerank_score"] = rerank_score
        doc.metadata["platform_hits"] = info["platform_hits"]
        doc.metadata["strong_hits"] = info["strong_hits"]
        doc.metadata["weak_hits"] = info["weak_hits"]
        doc.metadata["opposite_platform_hit"] = info["opposite_platform_hit"]
        scored_docs.append((doc, rerank_score, info))

    best_by_identity = {}
    for doc, score, info in scored_docs:
        identity = get_doc_identity(doc)
        if identity not in best_by_identity or score > best_by_identity[identity][1]:
            best_by_identity[identity] = (doc, score, info)

    deduped = list(best_by_identity.values())
    deduped.sort(
        key=lambda x: (
            x[1],
            len(x[2]["platform_hits"]),
            len(x[2]["strong_hits"]),
            len(x[0].page_content),
        ),
        reverse=True,
    )

    if not deduped:
        return []

    best_doc, best_score, best_info = deduped[0]

    # Gate mềm hơn: chỉ chặn khi thật sự quá tệ
    if keyword_groups["platform_keywords"] and len(best_info["platform_hits"]) == 0 and best_score < 1.5:
        return []

    if best_info["opposite_platform_hit"] and best_score < 2.0:
        return []

    if best_score < 0.8:
        return []

    score_threshold = max(best_score * 0.45, 0.8)

    filtered = [
        (doc, score, info)
        for doc, score, info in deduped
        if score >= score_threshold
        and not (info["opposite_platform_hit"] and score < 2.5)
    ]

    if not filtered:
        # fallback mềm: giữ vài candidate tốt nhất thay vì trả rỗng
        return [doc for doc, _, _ in deduped[:top_k]]

    return [doc for doc, _, _ in filtered[:top_k]]