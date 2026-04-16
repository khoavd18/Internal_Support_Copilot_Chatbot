from __future__ import annotations

from functools import lru_cache
import logging

from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, SparseVectorParams, VectorParams

from src.core.settings import (
    QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME,
    QDRANT_PREFER_GRPC,
    QDRANT_SPARSE_VECTOR_NAME,
    QDRANT_URL,
    QDRANT_VECTOR_NAME,
    USE_QDRANT_HYBRID,
)
from src.rag.embedding.embeddings import get_embedding_model
from src.rag.ingest.hierarchical_loader import LEAF_PATH, load_leaf_documents
from src.rag.ingest.loader import load_documents
from src.rag.chunking.chunking import split_documents

import uuid

logger = logging.getLogger(__name__)

def _to_qdrant_point_id(raw_id: str) -> str:
    """
    Chuyển internal id của project sang UUID ổn định để Qdrant chấp nhận.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))

def _get_client() -> QdrantClient:
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
        prefer_grpc=QDRANT_PREFER_GRPC,
    )


@lru_cache(maxsize=1)
def _get_sparse_embeddings() -> FastEmbedSparse:
    return FastEmbedSparse(model_name="Qdrant/bm25")


def _prepare_indexable_documents():
    if LEAF_PATH.exists():
        logger.info(
            "Loading hierarchical leaf nodes for indexing",
            extra={"event": "indexing.documents.load", "indexing_mode": "hierarchical_leaf"},
        )
        docs = load_leaf_documents()
        chunks = docs
        indexing_mode = "hierarchical_leaf"
    else:
        logger.info(
            "Loading regular documents for indexing",
            extra={"event": "indexing.documents.load", "indexing_mode": "regular_split"},
        )
        docs = load_documents()
        chunks = split_documents(docs)
        indexing_mode = "regular_split"

    return docs, chunks, indexing_mode


def _ensure_collection(client: QdrantClient, embeddings, rebuild: bool):
    dense_size = len(embeddings.embed_query("test embedding size"))
    logger.info(
        "Resolved dense vector size",
        extra={"event": "indexing.collection.dimension", "dense_size": dense_size},
    )

    if rebuild:
        try:
            client.delete_collection(QDRANT_COLLECTION_NAME)
            logger.info(
                "Deleted existing Qdrant collection",
                extra={"event": "indexing.collection.deleted", "collection": QDRANT_COLLECTION_NAME},
            )
        except Exception:
            pass

    try:
        client.get_collection(QDRANT_COLLECTION_NAME)
        logger.info(
            "Qdrant collection already exists",
            extra={"event": "indexing.collection.exists", "collection": QDRANT_COLLECTION_NAME},
        )
        return
    except Exception:
        pass

    sparse_vectors_config = {
        QDRANT_SPARSE_VECTOR_NAME: SparseVectorParams(
            index=models.SparseIndexParams(on_disk=False)
        )
    }

    client.create_collection(
        collection_name=QDRANT_COLLECTION_NAME,
        vectors_config={
            QDRANT_VECTOR_NAME: VectorParams(
                size=dense_size,
                distance=Distance.COSINE,
            )
        },
        sparse_vectors_config=sparse_vectors_config,
    )
    logger.info(
        "Created Qdrant collection",
        extra={"event": "indexing.collection.created", "collection": QDRANT_COLLECTION_NAME},
    )


def build_qdrant_store(rebuild: bool = False):
    docs, chunks, indexing_mode = _prepare_indexable_documents()
    embeddings = get_embedding_model()
    sparse_embeddings = _get_sparse_embeddings()
    client = _get_client()

    _ensure_collection(client, embeddings, rebuild=rebuild)

    filtered_chunks = []
    ids = []

    for i, doc in enumerate(chunks):
        text = doc.page_content.strip()
        if len(text) < 50:
            continue

        md = doc.metadata or {}
        raw_id = (
            md.get("leaf_id")
            or md.get("chunk_id")
            or md.get("parent_id")
            or md.get("doc_id")
            or f"doc-{i}"
        )
        raw_id = str(raw_id)

        # giữ lại id gốc để debug / trace
        doc.metadata["source_chunk_id"] = raw_id

        qdrant_id = _to_qdrant_point_id(raw_id)

        filtered_chunks.append(doc)
        ids.append(qdrant_id)

    retrieval_mode = RetrievalMode.HYBRID if USE_QDRANT_HYBRID else RetrievalMode.DENSE

    logger.info(
        "Building Qdrant store",
        extra={
            "event": "indexing.build.started",
            "indexing_mode": indexing_mode,
            "final_chunks": len(filtered_chunks),
            "retrieval_mode": str(retrieval_mode),
            "rebuild": rebuild,
        },
    )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=retrieval_mode,
        vector_name=QDRANT_VECTOR_NAME,
        sparse_vector_name=QDRANT_SPARSE_VECTOR_NAME,
    )

    vector_store.add_documents(documents=filtered_chunks, ids=ids)

    logger.info(
        "Qdrant store ready",
        extra={"event": "indexing.build.completed", "collection": QDRANT_COLLECTION_NAME},
    )
    return vector_store


def load_qdrant_store():
    embeddings = get_embedding_model()
    sparse_embeddings = _get_sparse_embeddings()
    client = _get_client()

    retrieval_mode = RetrievalMode.HYBRID if USE_QDRANT_HYBRID else RetrievalMode.DENSE

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=QDRANT_COLLECTION_NAME,
        embedding=embeddings,
        sparse_embedding=sparse_embeddings,
        retrieval_mode=retrieval_mode,
        vector_name=QDRANT_VECTOR_NAME,
        sparse_vector_name=QDRANT_SPARSE_VECTOR_NAME,
    )

    logger.info(
        "Loaded Qdrant store",
        extra={
            "event": "indexing.store.loaded",
            "collection": QDRANT_COLLECTION_NAME,
            "retrieval_mode": str(retrieval_mode),
        },
    )
    return vector_store


def main():
    store = build_qdrant_store(rebuild=True)
    docs = store.similarity_search("how to sign in with a passkey", k=3)

    print(f"[INFO] Retrieved {len(docs)} docs")
    for i, doc in enumerate(docs, start=1):
        print("=" * 80)
        print(f"Rank #{i}")
        print(doc.metadata)
        print(doc.page_content[:500])


if __name__ == "__main__":
    main()
