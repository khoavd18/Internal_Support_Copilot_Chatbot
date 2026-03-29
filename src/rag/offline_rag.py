from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.pipeline import build_pipeline


def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def pretty_print_sources(sources: list[dict]):
    if not sources:
        print("[INFO] No sources")
        return

    for src in sources:
        print("-" * 100)
        print(f"[{src.get('index')}] {src.get('title')}")
        print(f"source      : {src.get('source')}")
        print(f"path        : {src.get('path')}")
        print(f"url         : {src.get('url')}")
        print(f"doc_id      : {src.get('doc_id')}")
        print(f"source_type : {src.get('source_type')}")
        print(f"rerank_score: {src.get('rerank_score')}")


def pretty_print_debug(debug_rows: list[dict]):
    if not debug_rows:
        print("[INFO] No debug rows")
        return

    for row in debug_rows:
        print("-" * 100)
        print(f"rank         : {row.get('rank')}")
        print(f"title        : {row.get('title')}")
        print(f"source       : {row.get('source')}")
        print(f"path         : {row.get('path')}")
        print(f"url          : {row.get('url')}")
        print(f"doc_id       : {row.get('doc_id')}")
        print(f"source_type  : {row.get('source_type')}")
        print(f"rerank_score : {row.get('rerank_score')}")
        print(f"strong_hits  : {row.get('strong_hits')}")
        print(f"platform_hits: {row.get('platform_hits')}")
        print(f"chunk_id     : {row.get('chunk_id')}")
        print(f"chunk_index  : {row.get('chunk_index')}")
        print(f"chunk_len    : {row.get('chunk_len')}")
        print(f"preview      : {row.get('preview')}")


def save_result_if_needed(result: dict, output_path: str | None):
    if not output_path:
        return

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[DONE] Saved result -> {path}")


def main():
    parser = argparse.ArgumentParser(description="Offline RAG CLI for Internal Support Copilot")
    parser.add_argument("--query", required=True, help="User question")
    parser.add_argument("--top_k", type=int, default=4, help="Number of retrieved docs")
    parser.add_argument("--debug", action="store_true", help="Show retrieval debug rows")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild vector store before retrieval")
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save full result as JSON",
    )

    args = parser.parse_args()

    print_section("OFFLINE RAG")
    print(f"[INFO] query   = {args.query}")
    print(f"[INFO] top_k   = {args.top_k}")
    print(f"[INFO] debug   = {args.debug}")
    print(f"[INFO] rebuild = {args.rebuild}")

    pipeline = build_pipeline(
        top_k=args.top_k,
        rebuild=args.rebuild,
    )

    result = pipeline.ask(
        question=args.query,
        debug=args.debug,
    )

    print_section("ANSWER")
    print(result.get("answer", ""))

    print_section("STATS")
    print(json.dumps(result.get("stats", {}), ensure_ascii=False, indent=2))

    print_section("SOURCES")
    pretty_print_sources(result.get("sources", []))

    if args.debug:
        print_section("DEBUG")
        pretty_print_debug(result.get("debug", []))

        prompt = result.get("prompt")
        if prompt:
            print_section("PROMPT")
            print(prompt)

    save_result_if_needed(result, args.save)


if __name__ == "__main__":
    main()