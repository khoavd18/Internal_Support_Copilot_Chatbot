from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from src.core.settings import DOCUMENTS_PATH, INCLUDE_TICKETS, TICKETS_PATH
from src.rag.ingest.builders import (
    build_document_text,
    build_metadata,
    build_ticket_text,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Đọc file jsonl: mỗi dòng là 1 object json."""
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")

    rows: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Lỗi JSON ở {path}, dòng {line_no}: {e}") from e

    return rows


def load_documents(
    documents_path: Path = DOCUMENTS_PATH,
    tickets_path: Path = TICKETS_PATH,
    include_tickets: bool = INCLUDE_TICKETS,
) -> list[Document]:
    """Load toàn bộ documents + tickets thành list[Document]."""
    all_docs: list[Document] = []

    raw_docs = read_jsonl(documents_path)
    for row in raw_docs:
        text = build_document_text(row)
        if not text.strip():
            continue

        all_docs.append(
            Document(
                page_content=text,
                metadata=build_metadata(row, source_type="document"),
            )
        )

    if include_tickets:
        if tickets_path.exists():
            raw_tickets = read_jsonl(tickets_path)
            for row in raw_tickets:
                text = build_ticket_text(row)
                if not text.strip():
                    continue

                all_docs.append(
                    Document(
                        page_content=text,
                        metadata=build_metadata(row, source_type="ticket"),
                    )
                )
        else:
            print(
                f"[WARN] tickets file not found: {tickets_path}. Continue without tickets."
            )

    return all_docs


if __name__ == "__main__":
    docs = load_documents()

    print(f"[DONE] Loaded {len(docs)} LangChain documents")
    print("-" * 80)
    print("First metadata:")
    print(docs[0].metadata)
    print("-" * 80)
    print("First content preview:")
    print(docs[0].page_content[:500])