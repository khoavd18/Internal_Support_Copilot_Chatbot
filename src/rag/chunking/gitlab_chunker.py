from __future__ import annotations

from langchain_core.documents import Document

from src.rag.chunking.base import choose_chunk_config, get_title, make_recursive_splitter


def _augment_gitlab_text(doc: Document) -> str:
    """
    Prefix title để retrieval đỡ bị mờ ngữ cảnh ở chunk đầu.
    """
    title = get_title(doc)
    text = doc.page_content.strip()

    if title and title.lower() not in text[:200].lower():
        return f"{title}\n\n{text}"

    return text


def chunk_gitlab_handbook(doc: Document) -> list[Document]:
    cfg = choose_chunk_config(doc)

    working_doc = Document(
        page_content=_augment_gitlab_text(doc),
        metadata=dict(doc.metadata),
    )

    splitter = make_recursive_splitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=[
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    return splitter.split_documents([working_doc])