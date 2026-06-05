from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "sample_enterprise_support"
DEFAULT_COLLECTION_NAME = "enterprise_support_copilot_qdrant"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _source_type_counts(documents: list[dict[str, Any]]) -> Counter[str]:
    return Counter(
        str(document.get("metadata", {}).get("source_type") or "unknown") for document in documents
    )


def load_validate_build_documents(data_dir: Path, limit: int | None = None) -> list[dict[str, Any]]:
    from scripts.validate_enterprise_support_data import format_report, validate_dataset
    from src.data.enterprise_support_documents import build_enterprise_support_documents
    from src.data.enterprise_support_loader import load_enterprise_support_dataset

    report = validate_dataset(data_dir)
    if not report.is_valid:
        raise ValueError(format_report(report))

    dataset = load_enterprise_support_dataset(data_dir)
    documents = build_enterprise_support_documents(dataset)
    if limit is not None:
        return documents[:limit]
    return documents


def print_dry_run_summary(
    documents: list[dict[str, Any]],
    *,
    collection_name: str,
    stream: TextIO,
) -> None:
    print("Dry run: no documents written to Qdrant.", file=stream)
    print(f"Target collection: {collection_name}", file=stream)
    print(f"Total documents: {len(documents)}", file=stream)
    print("", file=stream)
    print("Document count by source_type:", file=stream)
    for source_type, count in sorted(_source_type_counts(documents).items()):
        print(f"  {source_type}: {count}", file=stream)

    print("", file=stream)
    print("Sample documents:", file=stream)
    for index, document in enumerate(documents[:3], start=1):
        text = str(document.get("text") or "").replace("\n", " ")
        preview = text[:500] + ("..." if len(text) > 500 else "")
        print(f"  Sample #{index}", file=stream)
        print(f"    id: {document.get('id')}", file=stream)
        print(f"    metadata: {document.get('metadata')}", file=stream)
        print(f"    text: {preview}", file=stream)


def _ensure_collection(
    *,
    client: Any,
    collection_name: str,
    embeddings: Any,
    vector_name: str,
    sparse_vector_name: str,
) -> None:
    from qdrant_client import models
    from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

    try:
        client.get_collection(collection_name)
        return
    except Exception:
        pass

    dense_size = len(embeddings.embed_query("test embedding size"))
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            vector_name: VectorParams(
                size=dense_size,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config={
            sparse_vector_name: SparseVectorParams(index=models.SparseIndexParams(on_disk=False))
        },
    )


def _to_langchain_documents(
    documents: list[dict[str, Any]],
) -> tuple[list[Any], list[str]]:
    from langchain_core.documents import Document
    from src.rag.indexing.qdrant_store import _to_qdrant_point_id

    langchain_documents: list[Document] = []
    ids: list[str] = []

    for document in documents:
        document_id = str(document["id"])
        text = str(document.get("text") or "").strip()
        if not text:
            continue

        metadata = dict(document.get("metadata") or {})
        metadata["doc_id"] = document_id
        metadata["source_chunk_id"] = document_id

        langchain_documents.append(Document(page_content=text, metadata=metadata))
        ids.append(_to_qdrant_point_id(document_id))

    return langchain_documents, ids


def ingest_documents_into_qdrant(
    documents: list[dict[str, Any]],
    *,
    collection_name: str,
) -> int:
    from langchain_qdrant import QdrantVectorStore, RetrievalMode
    from qdrant_client import QdrantClient
    from src.core.settings import (
        QDRANT_API_KEY,
        QDRANT_PREFER_GRPC,
        QDRANT_SPARSE_VECTOR_NAME,
        QDRANT_URL,
        QDRANT_VECTOR_NAME,
        USE_QDRANT_HYBRID,
    )
    from src.rag.embedding.embeddings import get_embedding_model
    from src.rag.indexing.qdrant_store import _get_sparse_embeddings

    embeddings = get_embedding_model()
    sparse_embeddings = _get_sparse_embeddings()
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
        prefer_grpc=QDRANT_PREFER_GRPC,
    )
    _ensure_collection(
        client=client,
        collection_name=collection_name,
        embeddings=embeddings,
        vector_name=QDRANT_VECTOR_NAME,
        sparse_vector_name=QDRANT_SPARSE_VECTOR_NAME,
    )

    langchain_documents, ids = _to_langchain_documents(documents)
    if not langchain_documents:
        return 0

    retrieval_mode = RetrievalMode.HYBRID if USE_QDRANT_HYBRID else RetrievalMode.DENSE
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=retrieval_mode,
        vector_name=QDRANT_VECTOR_NAME,
        sparse_vector_name=QDRANT_SPARSE_VECTOR_NAME,
    )
    vector_store.add_documents(documents=langchain_documents, ids=ids)
    return len(langchain_documents)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest the synthetic enterprise support dataset into Qdrant."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Enterprise support dataset root.",
    )
    parser.add_argument(
        "--collection-name",
        default=DEFAULT_COLLECTION_NAME,
        help=(
            "Qdrant collection to write. Defaults to a separate enterprise support "
            "collection so the existing GitHub demo collection is not changed."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and build documents, but do not write to Qdrant.",
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=None,
        help="Limit the number of built documents before dry-run output or ingestion.",
    )
    return parser


def run(args: argparse.Namespace, *, stream: TextIO = sys.stdout) -> int:
    data_dir = Path(args.data_dir)
    documents = load_validate_build_documents(data_dir, limit=args.limit)

    if args.dry_run:
        print_dry_run_summary(
            documents,
            collection_name=args.collection_name,
            stream=stream,
        )
        return 0

    ingested_count = ingest_documents_into_qdrant(
        documents,
        collection_name=args.collection_name,
    )
    print(
        f"Ingested {ingested_count} enterprise support documents into "
        f"Qdrant collection '{args.collection_name}'.",
        file=stream,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
