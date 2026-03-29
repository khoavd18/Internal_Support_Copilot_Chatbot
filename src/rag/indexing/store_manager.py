from __future__ import annotations

_VECTOR_STORE = None


def get_vector_store(rebuild: bool = False):
    """
    Singleton vector store chỉ dùng Qdrant.
    """
    global _VECTOR_STORE

    if rebuild or _VECTOR_STORE is None:
        from src.rag.indexing.qdrant_store import (
            build_qdrant_store,
            load_qdrant_store,
        )

        if rebuild:
            _VECTOR_STORE = build_qdrant_store(rebuild=True)
        else:
            _VECTOR_STORE = load_qdrant_store()

    return _VECTOR_STORE