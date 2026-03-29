from __future__ import annotations

from pathlib import Path
import hashlib
import json

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from src.rag.chunking.base import (
    choose_chunk_config,
    enforce_hard_token_limit,
    finalize_chunks,
    get_path,
    get_source,
    get_source_type,
    get_title,
    make_recursive_splitter,
    token_length,
)
from src.rag.ingest.loader import load_documents


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "data_source" / "processed" / "hierarchical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARENT_PATH = OUTPUT_DIR / "parent_nodes.jsonl"
LEAF_PATH = OUTPUT_DIR / "leaf_nodes.jsonl"


def save_jsonl(records: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def doc_to_record(doc: Document) -> dict:
    record = dict(doc.metadata)
    record["text"] = doc.page_content
    return record


def stable_hash(text: str, n: int = 16) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:n]


def build_parent_id(source: str, base_doc_id: str, parent_index: int) -> str:
    raw = f"{source}::{base_doc_id}::{parent_index}"
    return f"parent_{stable_hash(raw)}"


def make_parent_document(
    base_doc: Document,
    parent_text: str,
    parent_index: int,
    parent_level: str,
    extra_metadata: dict | None = None,
) -> Document:
    extra_metadata = extra_metadata or {}

    source = get_source(base_doc) or "unknown"
    original_doc_id = str(base_doc.metadata.get("doc_id") or "").strip()
    parent_id = build_parent_id(source, original_doc_id or "unknown", parent_index)

    metadata = dict(base_doc.metadata)
    metadata["origin_doc_id"] = original_doc_id
    metadata["doc_id"] = f"{original_doc_id}::parent::{parent_index}" if original_doc_id else parent_id
    metadata["parent_id"] = parent_id
    metadata["parent_index"] = parent_index
    metadata["parent_level"] = parent_level
    metadata["node_type"] = "parent"
    metadata["parent_len_chars"] = len(parent_text)
    metadata["parent_len_tokens_est"] = token_length(parent_text)
    metadata.update(extra_metadata)

    return Document(page_content=parent_text, metadata=metadata)


def build_github_docs_parents(doc: Document) -> list[Document]:
    """
    Với markdown docs:
    - parent = section theo markdown headers
    """
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
            ("####", "h4"),
        ],
        strip_headers=False,
    )

    sections = header_splitter.split_text(doc.page_content)

    if not sections:
        return [make_parent_document(doc, doc.page_content, 0, "document")]

    parents: list[Document] = []

    for idx, sec in enumerate(sections):
        merged_metadata = dict(doc.metadata)
        merged_metadata.update(sec.metadata)

        temp_doc = Document(page_content=sec.page_content, metadata=merged_metadata)

        extra = {}
        for key in ["h1", "h2", "h3", "h4"]:
            if key in sec.metadata:
                extra[key] = sec.metadata[key]

        parent_doc = make_parent_document(
            base_doc=temp_doc,
            parent_text=sec.page_content,
            parent_index=idx,
            parent_level="section",
            extra_metadata=extra,
        )
        parents.append(parent_doc)

    return parents


def augment_issue_text(doc: Document) -> str:
    title = get_title(doc)
    text = doc.page_content.strip()

    if title and not text.lower().startswith("title:"):
        return f"Title: {title}\n\n{text}"

    return text


def build_gitlab_parents(doc: Document) -> list[Document]:
    """
    v2-lite:
    - mỗi handbook doc = 1 parent
    Sau này nếu muốn chuẩn hơn nữa, tách parent từ HTML sections ngay ở prepare step.
    """
    return [make_parent_document(doc, doc.page_content, 0, "document")]


def build_issue_parents(doc: Document) -> list[Document]:
    text = augment_issue_text(doc)
    temp_doc = Document(page_content=text, metadata=dict(doc.metadata))
    return [make_parent_document(temp_doc, text, 0, "document")]


def build_generic_parents(doc: Document) -> list[Document]:
    return [make_parent_document(doc, doc.page_content, 0, "document")]


def build_parents_for_document(doc: Document) -> list[Document]:
    source = get_source(doc)
    source_type = get_source_type(doc)
    path = get_path(doc)

    if source == "github_docs" or path.endswith(".md") or source_type == "markdown":
        return build_github_docs_parents(doc)

    if source == "gitlab_handbook" or source_type in {"gitlab", "html"}:
        return build_gitlab_parents(doc)

    if source == "github_issues" or source_type in {"ticket", "issue", "github_issue", "github_issues"}:
        return build_issue_parents(doc)

    return build_generic_parents(doc)


def split_parent_to_leaves(parent_doc: Document) -> list[Document]:
    cfg = choose_chunk_config(parent_doc)
    source = get_source(parent_doc)

    if source == "github_docs":
        separators = [
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ]
    elif source == "github_issues":
        separators = [
            "\n\nComments:",
            "\n\nComment ",
            "\n\nBody:",
            "\n\nDescription:",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ]
    else:
        separators = [
            "\n## ",
            "\n### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ]

    splitter = make_recursive_splitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=separators,
    )

    raw_leaf_docs = splitter.split_documents([parent_doc])

    bounded_leaf_docs = enforce_hard_token_limit(
        chunks=raw_leaf_docs,
        max_tokens=cfg["hard_max_tokens"],
        chunk_overlap=cfg["chunk_overlap"],
    )

    final_leaf_docs = finalize_chunks(bounded_leaf_docs)

    for leaf in final_leaf_docs:
        leaf.metadata["node_type"] = "leaf"
        leaf.metadata["leaf_id"] = leaf.metadata["chunk_id"]
        leaf.metadata["parent_id"] = parent_doc.metadata["parent_id"]
        leaf.metadata["origin_doc_id"] = parent_doc.metadata.get("origin_doc_id", "")
        leaf.metadata["parent_level"] = parent_doc.metadata.get("parent_level", "document")

    return final_leaf_docs


def main():
    docs = load_documents()

    parent_nodes: list[Document] = []
    leaf_nodes: list[Document] = []

    for doc in docs:
        parents = build_parents_for_document(doc)
        parent_nodes.extend(parents)

        for parent_doc in parents:
            leaves = split_parent_to_leaves(parent_doc)
            leaf_nodes.extend(leaves)

    parent_records = [doc_to_record(doc) for doc in parent_nodes]
    leaf_records = [doc_to_record(doc) for doc in leaf_nodes]

    save_jsonl(parent_records, PARENT_PATH)
    save_jsonl(leaf_records, LEAF_PATH)

    print(f"Original docs: {len(docs)}")
    print(f"Parent nodes: {len(parent_nodes)}")
    print(f"Leaf nodes: {len(leaf_nodes)}")
    print(f"[DONE] Saved parent nodes -> {PARENT_PATH}")
    print(f"[DONE] Saved leaf nodes -> {LEAF_PATH}")

    if leaf_nodes:
        lengths = [len(doc.page_content) for doc in leaf_nodes]
        token_lengths = [doc.metadata.get('chunk_len_tokens_est', 0) for doc in leaf_nodes]

        print("\nLeaf stats (chars):")
        print(f"Min length: {min(lengths)}")
        print(f"Max length: {max(lengths)}")
        print(f"Average length: {sum(lengths) / len(lengths):.2f}")

        print("\nLeaf stats (tokens est):")
        print(f"Min tokens: {min(token_lengths)}")
        print(f"Max tokens: {max(token_lengths)}")
        print(f"Average tokens: {sum(token_lengths) / len(token_lengths):.2f}")


if __name__ == "__main__":
    main()