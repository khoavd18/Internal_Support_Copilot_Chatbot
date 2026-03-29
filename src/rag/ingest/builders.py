from __future__ import annotations

from typing import Any

from src.rag.ingest.cleaners import (
    clean_text,
    first_non_empty,
    join_non_empty,
    normalize_text,
)


def _extract_path_from_doc_id(doc_id: str) -> str:
    """
    Ví dụ:
      github_docs::authentication/authenticating-with-a-passkey/signing-in-with-a-passkey.md
      -> authentication/authenticating-with-a-passkey/signing-in-with-a-passkey.md
    """
    if not doc_id:
        return ""

    if "::" in doc_id:
        return doc_id.split("::", 1)[1].strip()

    return doc_id.strip()


def _build_github_docs_url(path: str) -> str:
    """
    Dựng URL tham khảo từ path markdown của github docs.
    Đây chỉ là fallback để trace source đẹp hơn.
    """
    if not path:
        return ""

    normalized = path.strip().lstrip("/")

    if normalized.endswith(".md"):
        normalized = normalized[:-3]

    if normalized.startswith("content/"):
        normalized = normalized[len("content/") :]

    return f"https://docs.github.com/en/{normalized}"


def build_document_text(record: dict[str, Any]) -> str:
    """Tạo page_content cho docs."""
    title = first_non_empty(record, ["title", "name", "path", "slug"])
    summary = normalize_text(record.get("summary"))
    content = normalize_text(
        record.get("content")
        or record.get("text")
        or record.get("markdown")
        or record.get("body")
    )

    text = join_non_empty(
        [
            f"Title: {title}" if title else "",
            f"Summary: {summary}" if summary else "",
            content,
        ]
    )
    return clean_text(text)


def build_ticket_text(record: dict[str, Any]) -> str:
    """Tạo page_content cho tickets/issues."""
    ticket_id = first_non_empty(record, ["ticket_id", "issue_number", "number", "id"])
    title = first_non_empty(record, ["title", "issue_title"])
    body = normalize_text(
        record.get("body")
        or record.get("content")
        or record.get("text")
        or record.get("description")
    )
    comments = normalize_text(record.get("comments"))
    resolution = normalize_text(
        record.get("resolution")
        or record.get("answer")
        or record.get("accepted_answer")
        or record.get("closing_note")
    )

    labels = record.get("labels", [])
    labels_text = ""
    if isinstance(labels, list):
        clean_labels = []
        for label in labels:
            if isinstance(label, dict):
                name = first_non_empty(label, ["name"])
                if name:
                    clean_labels.append(name)
            else:
                clean_labels.append(str(label).strip())
        labels_text = ", ".join([x for x in clean_labels if x])

    text = join_non_empty(
        [
            f"Ticket ID: {ticket_id}" if ticket_id else "",
            f"Title: {title}" if title else "",
            f"Labels: {labels_text}" if labels_text else "",
            f"Body:\n{body}" if body else "",
            f"Comments:\n{comments}" if comments else "",
            f"Resolution:\n{resolution}" if resolution else "",
        ]
    )
    return clean_text(text)


def build_metadata(record: dict[str, Any], source_type: str) -> dict[str, Any]:
    """Metadata để truy vết source sau này."""
    doc_id = first_non_empty(record, ["doc_id", "ticket_id", "issue_number", "number", "id"])
    source = first_non_empty(record, ["source", "repo", "origin"]) or source_type

    raw_path = first_non_empty(record, ["path", "file_path"])
    path = raw_path or _extract_path_from_doc_id(doc_id)

    raw_url = first_non_empty(record, ["url", "html_url", "source_url"])
    url = raw_url

    if not url and source == "github_docs":
        url = _build_github_docs_url(path)

    return {
        "source_type": source_type,
        "doc_id": doc_id,
        "title": first_non_empty(record, ["title", "name", "issue_title", "path"]),
        "url": url,
        "source": source,
        "path": path,
    }