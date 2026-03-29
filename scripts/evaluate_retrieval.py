from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json

from src.rag.retrieval.retriever import retrieve_documents


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUERIES_PATH = PROJECT_ROOT / "eval" / "queries" / "evaluation_queries.txt"
RUNS_DIR = PROJECT_ROOT / "eval" / "runs"
REPORTS_DIR = PROJECT_ROOT / "eval" / "reports"

RUNS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_queries(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file queries: {path}")

    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        queries.append(line)
    return queries


def doc_to_row(rank: int, doc) -> dict:
    md = doc.metadata or {}
    return {
        "rank": rank,
        "node_type": md.get("node_type"),
        "merge_strategy": md.get("merge_strategy"),
        "source": md.get("source"),
        "source_type": md.get("source_type"),
        "title": md.get("title"),
        "doc_id": md.get("doc_id"),
        "origin_doc_id": md.get("origin_doc_id"),
        "parent_id": md.get("parent_id"),
        "rerank_score": md.get("rerank_score"),
        "preview": doc.page_content[:220],
    }


def main():
    queries = load_queries(QUERIES_PATH)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = RUNS_DIR / f"{timestamp}_retrieval_eval.jsonl"
    summary_path = REPORTS_DIR / "latest_retrieval_summary.txt"

    total_queries = 0
    total_results = 0
    total_parent = 0
    total_parent_pack = 0
    total_leaf = 0

    with run_path.open("w", encoding="utf-8") as f:
        for query in queries:
            total_queries += 1
            docs = retrieve_documents(query=query, top_k=5)

            rows = []
            for idx, doc in enumerate(docs, start=1):
                row = doc_to_row(idx, doc)
                rows.append(row)

                node_type = row.get("node_type")
                if node_type == "parent":
                    total_parent += 1
                elif node_type == "parent_pack":
                    total_parent_pack += 1
                elif node_type == "leaf":
                    total_leaf += 1

            total_results += len(rows)

            record = {
                "query": query,
                "result_count": len(rows),
                "results": rows,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print("=" * 100)
            print(f"QUERY: {query}")
            print(f"RESULTS: {len(rows)}")
            for row in rows:
                print(
                    f"[{row['rank']}] "
                    f"{row['node_type']} | "
                    f"{row['merge_strategy']} | "
                    f"{row['source']} | "
                    f"{row['title']}"
                )

    avg_results = (total_results / total_queries) if total_queries else 0.0

    summary = (
        f"Run file          : {run_path}\n"
        f"Total queries     : {total_queries}\n"
        f"Total results     : {total_results}\n"
        f"Average results   : {avg_results:.2f}\n"
        f"Parent count      : {total_parent}\n"
        f"Parent pack count : {total_parent_pack}\n"
        f"Leaf count        : {total_leaf}\n"
    )

    summary_path.write_text(summary, encoding="utf-8")

    print("\n" + "=" * 100)
    print("SUMMARY")
    print(summary)
    print(f"[DONE] Saved run    -> {run_path}")
    print(f"[DONE] Saved report -> {summary_path}")


if __name__ == "__main__":
    main()