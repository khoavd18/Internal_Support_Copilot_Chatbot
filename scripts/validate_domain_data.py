from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domains.base import (  # noqa: E402
    DomainAdapter,
    DomainValidationResult,
    format_validation_result,
)
from src.domains.registry import get_domain_adapter  # noqa: E402


def validate_domain_data(
    adapter: DomainAdapter,
    data_dir: Path | None = None,
) -> DomainValidationResult:
    resolved_data_dir = Path(data_dir) if data_dir is not None else adapter.default_data_dir
    validate_data_dir = getattr(adapter, "validate_data_dir", None)
    if callable(validate_data_dir):
        return validate_data_dir(resolved_data_dir)

    dataset = adapter.load_dataset(resolved_data_dir)
    result = adapter.validate_dataset(dataset)
    result.data_dir = resolved_data_dir.resolve()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a domain dataset.")
    parser.add_argument("--domain", required=True, help="Domain adapter name.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Dataset root. Defaults to the domain adapter data directory.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    adapter = get_domain_adapter(args.domain)
    result = validate_domain_data(adapter, args.data_dir)
    stream = sys.stdout if result.is_valid else sys.stderr
    print(format_validation_result(result), file=stream)
    return 0 if result.is_valid else 1


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        return run(parser.parse_args(argv))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
