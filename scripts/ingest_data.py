from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_data import main as prepare_data_main
from src.rag.indexing.qdrant_store import build_qdrant_store


def _ensure_raw_data_present() -> None:
    raw_dir = PROJECT_ROOT / "data_source" / "raw"
    if not raw_dir.exists():
        raise SystemExit(
            "Missing data_source/raw. Add your source documents there before running ingest."
        )

    if not any(path.is_file() for path in raw_dir.rglob("*")):
        raise SystemExit(
            "No source files found under data_source/raw. Add your source documents first, then rerun ingest."
        )


def main() -> int:
    _ensure_raw_data_present()
    prepare_data_main()
    build_qdrant_store(rebuild=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
