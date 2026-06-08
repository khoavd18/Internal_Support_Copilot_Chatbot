from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any, TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ingest_enterprise_support_data import ingest_documents_into_qdrant  # noqa: E402
from scripts.validate_domain_data import validate_domain_data  # noqa: E402
from src.domains.base import DomainAdapter, format_validation_result  # noqa: E402
from src.domains.registry import get_domain_adapter  # noqa: E402


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _source_type_counts(documents: list[dict[str, Any]]) -> Counter[str]:
    return Counter(
        str(document.get("metadata", {}).get("source_type") or "unknown") for document in documents
    )


def load_validate_build_documents(
    adapter: DomainAdapter,
    *,
    data_dir: Path,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    validation = validate_domain_data(adapter, data_dir)
    if not validation.is_valid:
        raise ValueError(format_validation_result(validation))

    dataset = adapter.load_dataset(data_dir)
    dataset_validation = adapter.validate_dataset(dataset)
    if not dataset_validation.is_valid:
        dataset_validation.data_dir = data_dir.resolve()
        raise ValueError(format_validation_result(dataset_validation))

    documents = adapter.build_documents(dataset)
    if limit is not None:
        return documents[:limit]
    return documents


def print_dry_run_summary(
    documents: list[dict[str, Any]],
    *,
    domain: str,
    collection_name: str,
    stream: TextIO,
) -> None:
    print("Dry run: no documents written to Qdrant.", file=stream)
    print(f"Domain: {domain}", file=stream)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest a domain dataset into Qdrant.")
    parser.add_argument("--domain", required=True, help="Domain adapter name.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Dataset root. Defaults to the domain adapter data directory.",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help="Qdrant collection. Defaults to the domain adapter collection name.",
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
        help="Limit built documents before dry-run output or ingestion.",
    )
    return parser


def run(args: argparse.Namespace, *, stream: TextIO = sys.stdout) -> int:
    adapter = get_domain_adapter(args.domain)
    data_dir = Path(args.data_dir) if args.data_dir is not None else adapter.default_data_dir
    collection_name = args.collection_name or adapter.default_collection_name
    documents = load_validate_build_documents(
        adapter,
        data_dir=data_dir,
        limit=args.limit,
    )

    if args.dry_run:
        print_dry_run_summary(
            documents,
            domain=adapter.name,
            collection_name=collection_name,
            stream=stream,
        )
        return 0

    ingested_count = ingest_documents_into_qdrant(
        documents,
        collection_name=collection_name,
    )
    print(
        f"Ingested {ingested_count} {adapter.name} documents into "
        f"Qdrant collection '{collection_name}'.",
        file=stream,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        return run(parser.parse_args(argv))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
