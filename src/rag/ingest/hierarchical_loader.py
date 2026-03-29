from __future__ import annotations

from pathlib import Path
import json

from langchain_core.documents import Document


PROJECT_ROOT = Path(__file__).resolve().parents[3]
HIER_DIR = PROJECT_ROOT / "data_source" / "processed" / "hierarchical"

PARENT_PATH = HIER_DIR / "parent_nodes.jsonl"
LEAF_PATH = HIER_DIR / "leaf_nodes.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_leaf_documents() -> list[Document]:
    rows = read_jsonl(LEAF_PATH)
    documents: list[Document] = []

    for row in rows:
        text = row.pop("text", "")
        documents.append(Document(page_content=text, metadata=row))

    return documents


def load_parent_map() -> dict[str, dict]:
    rows = read_jsonl(PARENT_PATH)
    parent_map: dict[str, dict] = {}

    for row in rows:
        parent_id = str(row.get("parent_id") or "").strip()
        if not parent_id:
            continue
        parent_map[parent_id] = row

    return parent_map