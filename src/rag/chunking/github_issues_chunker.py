from __future__ import annotations

from langchain_core.documents import Document

from src.rag.chunking.base import choose_chunk_config, get_title, make_recursive_splitter


def _augment_issue_text(doc: Document) -> str:
    """
    Thêm title lên đầu để chunk đầu và embedding giữ được chủ đề issue.
    """
    title = get_title(doc)
    text = doc.page_content.strip()

    if title and not text.lower().startswith("title:"):
        return f"Title: {title}\n\n{text}"

    return text


def chunk_github_issues(doc: Document) -> list[Document]:
    """
    Ưu tiên giữ title/body/comments liền ngữ nghĩa hơn.
    """
    cfg = choose_chunk_config(doc)

    working_doc = Document(
        page_content=_augment_issue_text(doc),
        metadata=dict(doc.metadata),
    )

    splitter = make_recursive_splitter(
        chunk_size=cfg["chunk_size"],
        chunk_overlap=cfg["chunk_overlap"],
        separators=[
            "\n\nComments:",
            "\n\nComment ",
            "\n\nBody:",
            "\n\nDescription:",
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    return splitter.split_documents([working_doc])