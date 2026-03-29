from __future__ import annotations

import re
from typing import Any, Dict, List

from langchain_core.documents import Document


def extract_llm_text(response: Any) -> str:
    if response is None:
        return ""

    if isinstance(response, str):
        return response

    if hasattr(response, "content"):
        content = getattr(response, "content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(str(x) for x in content)

    if isinstance(response, dict):
        for key in ("text", "generated_text", "content", "answer", "output"):
            if key in response:
                return str(response[key])

    return str(response)


def _clean_non_code_block(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _fix_common_command_and_url_issues(text: str) -> str:
    replacements = {
        "https:// /": "https://",
        "http:// /": "http://",
        "https:// github.com/": "https://github.com/",
        "http:// github.com/": "http://github.com/",
        "git@ github.com:": "git@github.com:",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)

    text = re.sub(r"https://\s+github\.com/", "https://github.com/", text, flags=re.IGNORECASE)
    text = re.sub(r"http://\s+github\.com/", "http://github.com/", text, flags=re.IGNORECASE)
    text = re.sub(r"git@\s+github\.com:", "git@github.com:", text, flags=re.IGNORECASE)

    text = re.sub(
        r"git remote set-url origin https:///?\s*/OWNER/REPOSITORY\.git",
        "git remote set-url origin https://github.com/OWNER/REPOSITORY.git",
        text,
        flags=re.IGNORECASE,
    )
    return text


def clean_answer(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""

    prefixes_to_remove = [
        "Dựa trên context được cung cấp,",
        "Dựa trên tài liệu được cung cấp,",
        "Theo context được cung cấp,",
        "Theo tài liệu được cung cấp,",
        "Dựa trên các nguồn được cung cấp,",
        "Theo các nguồn được cung cấp,",
    ]
    for prefix in prefixes_to_remove:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()

    parts = re.split(r"(```.*?```)", text, flags=re.DOTALL)
    cleaned_parts = []
    for part in parts:
        if not part:
            continue
        if part.startswith("```") and part.endswith("```"):
            cleaned_parts.append(part.strip())
        else:
            cleaned_parts.append(_clean_non_code_block(part))

    text = "\n\n".join(part for part in cleaned_parts if part).strip()
    text = _fix_common_command_and_url_issues(text)
    text = text.replace("[trong CONTEXT]", "trong tài liệu tham khảo")
    text = text.replace("trong CONTEXT", "trong tài liệu tham khảo")
    return text.strip()


def build_safe_fallback_answer(question: str) -> str:
    question = (question or "").strip()

    base = (
        "Trong dữ liệu hiện có, tôi chưa tìm được tài liệu đủ liên quan để trả lời chắc chắn câu hỏi này."
    )

    suggestions = [
        "Bạn có thể hỏi lại cụ thể hơn theo đúng thao tác muốn làm.",
        "Ví dụ: “cách connect local repo với GitHub bằng SSH”",
        "hoặc: “cách push project local lên GitHub bằng command line”.",
    ]

    q = question.lower()
    if "gitbash" in q or "git bash" in q:
        suggestions = [
            "Bạn có thể hỏi lại cụ thể hơn theo đúng thao tác muốn làm trong Git Bash.",
            "Ví dụ: “cách push project local lên GitHub bằng SSH trong Git Bash”",
            "hoặc: “cách thêm remote origin và push branch main lên GitHub”.",
        ]

    return base + "\n\n" + "\n".join(f"- {s}" for s in suggestions)


def get_title(doc: Document) -> str:
    metadata = doc.metadata or {}
    return (
        metadata.get("title")
        or metadata.get("doc_id")
        or metadata.get("path")
        or "Unknown"
    )


def get_source(doc: Document) -> str:
    metadata = doc.metadata or {}
    return metadata.get("source") or "unknown"


def get_path(doc: Document) -> str:
    metadata = doc.metadata or {}
    return metadata.get("path") or ""


def get_url(doc: Document) -> str:
    metadata = doc.metadata or {}
    return metadata.get("url") or ""


def build_sources(documents: List[Document]) -> List[Dict[str, Any]]:
    sources = []
    for idx, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        sources.append(
            {
                "index": idx,
                "title": get_title(doc),
                "source": get_source(doc),
                "path": get_path(doc),
                "url": get_url(doc),
                "doc_id": metadata.get("doc_id"),
                "source_type": metadata.get("source_type"),
                "rerank_score": metadata.get("rerank_score"),
            }
        )
    return sources


def build_debug_rows(documents: List[Document]) -> List[Dict[str, Any]]:
    debug_rows = []

    for idx, doc in enumerate(documents, start=1):
        metadata = doc.metadata or {}
        debug_rows.append(
            {
                "rank": idx,
                "title": get_title(doc),
                "source": get_source(doc),
                "path": get_path(doc),
                "url": get_url(doc),
                "doc_id": metadata.get("doc_id"),
                "source_type": metadata.get("source_type"),
                "rerank_score": metadata.get("rerank_score"),
                "strong_hits": metadata.get("strong_hits"),
                "platform_hits": metadata.get("platform_hits"),
                "chunk_id": metadata.get("chunk_id"),
                "chunk_index": metadata.get("chunk_index"),
                "chunk_len": metadata.get("chunk_len"),
                "preview": doc.page_content[:250],
            }
        )

    return debug_rows