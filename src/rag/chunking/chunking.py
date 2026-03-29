from __future__ import annotations

from collections import Counter

from src.rag.chunking.router import split_documents
from src.rag.ingest.loader import load_documents


def main():
    docs = load_documents()
    chunks = split_documents(docs)

    print(f"Original docs: {len(docs)}")
    print(f"Total chunks: {len(chunks)}")

    if docs:
        source_counter = Counter(str(doc.metadata.get("source") or "unknown") for doc in docs)
        print("\nSource breakdown (docs):")
        for source, count in source_counter.items():
            print(f"- {source}: {count}")

    if chunks:
        print("\nFirst chunk metadata:")
        print(chunks[0].metadata)

        print("\nFirst chunk preview:")
        print(chunks[0].page_content[:500])

    lengths = [len(chunk.page_content) for chunk in chunks]
    token_lengths = [chunk.metadata.get("chunk_len_tokens_est", 0) for chunk in chunks]

    if lengths:
        print("\nChunk stats (chars):")
        print(f"Min length: {min(lengths)}")
        print(f"Max length: {max(lengths)}")
        print(f"Average length: {sum(lengths) / len(lengths):.2f}")

    if token_lengths:
        print("\nChunk stats (tokens est):")
        print(f"Min tokens: {min(token_lengths)}")
        print(f"Max tokens: {max(token_lengths)}")
        print(f"Average tokens: {sum(token_lengths) / len(token_lengths):.2f}")


if __name__ == "__main__":
    main()