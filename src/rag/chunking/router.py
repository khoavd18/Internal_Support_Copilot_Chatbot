from __future__ import annotations

from langchain_core.documents import Document

from src.rag.chunking.base import (
    choose_chunk_config,
    enforce_hard_token_limit,
    finalize_chunks,
    get_path,
    get_source,
    get_source_type,
    make_recursive_splitter,
)
from src.rag.chunking.github_docs_chunker import chunk_github_docs
from src.rag.chunking.github_issues_chunker import chunk_github_issues
from src.rag.chunking.gitlab_chunker import chunk_gitlab_handbook


def _chunk_generic(doc: Document) -> list[Document]:
    cfg = choose_chunk_config(doc)
    splitter = make_recursive_splitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents([doc])


def _route_single_document(doc: Document) -> list[Document]:
    source = get_source(doc)
    source_type = get_source_type(doc)
    path = get_path(doc)

    if source == "github_docs" or path.endswith(".md") or source_type == "markdown":
        return chunk_github_docs(doc)

    if source == "gitlab_handbook" or source_type in {"gitlab", "html"}:
        return chunk_gitlab_handbook(doc)

    if source == "github_issues" or source_type in {"github_issue", "github_issues", "ticket", "issue"}:
        return chunk_github_issues(doc)

    return _chunk_generic(doc)


def split_documents(documents) -> list[Document]:
    all_chunks: list[Document] = []

    for doc in documents:
        cfg = choose_chunk_config(doc)

        raw_chunks = _route_single_document(doc)
        bounded_chunks = enforce_hard_token_limit(
            chunks=raw_chunks,
            max_tokens=cfg["hard_max_tokens"],
            chunk_overlap=cfg["chunk_overlap"],
        )
        final_chunks = finalize_chunks(bounded_chunks)

        all_chunks.extend(final_chunks)

    return all_chunks