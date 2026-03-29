from __future__ import annotations

import hashlib
import os
import re
from functools import lru_cache

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


MIN_CHUNK_CHARS = 80

DEFAULT_TOKENIZER_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)

SOURCE_CHUNK_CONFIG = {
    "github_docs": {
        "chunk_size": 320,
        "chunk_overlap": 40,
        "hard_max_tokens": 350,
    },
    "gitlab_handbook": {
        "chunk_size": 280,
        "chunk_overlap": 40,
        "hard_max_tokens": 320,
    },
    "github_issues": {
        "chunk_size": 220,
        "chunk_overlap": 30,
        "hard_max_tokens": 260,
    },
    "default": {
        "chunk_size": 300,
        "chunk_overlap": 40,
        "hard_max_tokens": 320,
    },
}


@lru_cache(maxsize=1)
def get_tokenizer():
    """
    Load tokenizer để đếm token.
    Dùng tokenize() thay vì encode() để tránh warning kiểu 742 > 512
    chỉ khi ta đang muốn đếm độ dài text.
    """
    if AutoTokenizer is None:
        return None

    try:
        return AutoTokenizer.from_pretrained(DEFAULT_TOKENIZER_NAME)
    except Exception:
        return None


def token_length(text: str) -> int:
    """
    Ưu tiên đếm theo tokenizer thật.
    Nếu không load được tokenizer thì fallback về len(text).
    """
    if not text:
        return 0

    tokenizer = get_tokenizer()
    if tokenizer is None:
        return len(text)

    try:
        return len(tokenizer.tokenize(text))
    except Exception:
        return len(text)


def get_source(doc: Document) -> str:
    return str(doc.metadata.get("source") or "").strip().lower()


def get_source_type(doc: Document) -> str:
    return str(doc.metadata.get("source_type") or "").strip().lower()


def get_path(doc: Document) -> str:
    return str(doc.metadata.get("path") or "").strip().lower()


def get_title(doc: Document) -> str:
    return str(doc.metadata.get("title") or "").strip()


def choose_chunk_config(doc: Document) -> dict:
    source = get_source(doc)
    source_type = get_source_type(doc)
    path = get_path(doc)

    if source == "github_docs" or path.endswith(".md") or source_type == "markdown":
        return SOURCE_CHUNK_CONFIG["github_docs"]

    if source == "gitlab_handbook" or source_type in {"gitlab", "html"}:
        return SOURCE_CHUNK_CONFIG["gitlab_handbook"]

    if source == "github_issues" or source_type in {"github_issue", "github_issues", "ticket", "issue"}:
        return SOURCE_CHUNK_CONFIG["github_issues"]

    return SOURCE_CHUNK_CONFIG["default"]


def build_source_key(metadata: dict) -> str:
    """
    Khóa ổn định của document gốc.
    """
    doc_id = str(metadata.get("doc_id") or "").strip()
    if doc_id:
        return doc_id

    issue_number = str(metadata.get("issue_number") or metadata.get("number") or "").strip()
    if issue_number:
        return f"issue:{issue_number}"

    source = str(metadata.get("source") or "").strip()
    path = str(metadata.get("path") or "").strip()
    title = str(metadata.get("title") or "").strip()

    raw = f"{source}||{path}||{title}"
    if raw.strip("|"):
        return raw

    return "unknown-source"


def build_chunk_id(metadata: dict, chunk_index: int, page_content: str) -> str:
    source_key = build_source_key(metadata)
    source_digest = hashlib.md5(source_key.encode("utf-8")).hexdigest()[:10]

    start_index = metadata.get("start_index", -1)
    content_digest = hashlib.md5(page_content.encode("utf-8")).hexdigest()[:8]

    return f"{source_digest}:{chunk_index}:{start_index}:{content_digest}"


def is_meaningful_chunk(text: str) -> bool:
    if not text:
        return False

    cleaned = text.strip()
    if len(cleaned) < MIN_CHUNK_CHARS:
        return False

    if len(re.findall(r"\w", cleaned, flags=re.UNICODE)) < 20:
        return False

    return True


def make_recursive_splitter(
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_length,
        is_separator_regex=False,
        separators=separators,
        add_start_index=True,
    )


def enforce_hard_token_limit(
    chunks: list[Document],
    max_tokens: int,
    chunk_overlap: int = 30,
) -> list[Document]:
    """
    Nếu chunk nào vẫn quá to sau split chính, cắt thêm 1 vòng cuối.
    """
    hard_splitter = make_recursive_splitter(
        chunk_size=max_tokens,
        chunk_overlap=min(chunk_overlap, max_tokens // 5),
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    final_chunks: list[Document] = []

    for chunk in chunks:
        if token_length(chunk.page_content) <= max_tokens:
            final_chunks.append(chunk)
        else:
            sub_chunks = hard_splitter.split_documents([chunk])
            final_chunks.extend(sub_chunks)

    return final_chunks


def finalize_chunks(chunks: list[Document]) -> list[Document]:
    """
    Lọc chunk rác và gắn metadata chuẩn hóa.
    Đánh lại chunk_index sau khi filter để index liên tục 0,1,2,...
    """
    cleaned: list[Document] = []

    for chunk in chunks:
        if not is_meaningful_chunk(chunk.page_content):
            continue
        cleaned.append(chunk)

    for idx, chunk in enumerate(cleaned):
        chunk.metadata["chunk_index"] = idx
        chunk.metadata["chunk_len_chars"] = len(chunk.page_content)
        chunk.metadata["chunk_len_tokens_est"] = token_length(chunk.page_content)
        chunk.metadata["parent_source_key"] = build_source_key(chunk.metadata)
        chunk.metadata["chunk_id"] = build_chunk_id(
            metadata=chunk.metadata,
            chunk_index=idx,
            page_content=chunk.page_content,
        )

    return cleaned