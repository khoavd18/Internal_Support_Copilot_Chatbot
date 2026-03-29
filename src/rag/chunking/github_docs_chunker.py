from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter

from src.rag.chunking.base import choose_chunk_config, make_recursive_splitter


def chunk_github_docs(doc: Document) -> list[Document]:
    """
    GitHub Docs (.md):
    1) Tách theo markdown headers
    2) Recursive split trong từng section
    """
    cfg = choose_chunk_config(doc)

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

    recursive_splitter = make_recursive_splitter(
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

    final_chunks: list[Document] = []

    if not sections:
        return recursive_splitter.split_documents([doc])

    for sec in sections:
        merged_metadata = dict(doc.metadata)
        merged_metadata.update(sec.metadata)

        section_doc = Document(
            page_content=sec.page_content,
            metadata=merged_metadata,
        )

        sub_chunks = recursive_splitter.split_documents([section_doc])
        final_chunks.extend(sub_chunks)

    return final_chunks