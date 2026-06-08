from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.domains.registry import get_domain_adapter  # noqa: E402


def run_domain_evaluation(
    *,
    domain: str,
    dry_run: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    adapter = get_domain_adapter(domain)
    run_evaluation = getattr(adapter, "run_evaluation", None)
    if not callable(run_evaluation):
        raise ValueError(f"Domain '{adapter.name}' does not expose an evaluation runner.")
    result = run_evaluation(
        data_dir=adapter.default_data_dir,
        dry_run=dry_run,
        limit=limit,
    )
    result["summary"]["domain"] = adapter.name
    return result


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def print_summary(result: dict[str, Any]) -> None:
    summary = result["summary"]
    metrics = summary["metrics"]
    print("Domain Evaluation")
    print(f"Domain               : {summary['domain']}")
    print(f"Mode                 : {summary['mode']}")
    print(f"Queries              : {summary['query_count']}")
    print(f"Recall@5             : {_format_metric(metrics['recall_at_5'])}")
    print(f"Source type hit rate : {_format_metric(metrics['source_type_hit_rate'])}")
    print(f"Groundedness proxy   : {_format_metric(metrics['groundedness_rate'])}")
    print(f"Vector errors        : {summary['vector_error_count']}")


def write_json_output(path_value: str, payload: dict[str, Any]) -> None:
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    if path_value == "-":
        print(text)
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a configured copilot domain.")
    parser.add_argument("--domain", required=True, help="Domain adapter name.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use the domain evaluator dry-run mode when available.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N cases.",
    )
    parser.add_argument(
        "--json-output",
        nargs="?",
        const="-",
        default=None,
        help="Write machine-readable evaluation output to a path, or stdout if no path is supplied.",
    )
    return parser


def run_cli(args: argparse.Namespace) -> int:
    result = run_domain_evaluation(
        domain=args.domain,
        dry_run=args.dry_run,
        limit=args.limit,
    )
    payload = {
        "summary": result["summary"],
        "paths": result["paths"],
        "records": result["records"],
    }
    if args.json_output:
        write_json_output(str(args.json_output), payload)
    if args.json_output != "-":
        print_summary(result)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        return run_cli(parser.parse_args(argv))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
