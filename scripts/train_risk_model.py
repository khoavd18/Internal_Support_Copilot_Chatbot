from __future__ import annotations

import argparse
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "sample_enterprise_support"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "artifacts" / "risk" / "isolation_forest.pkl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an optional IsolationForest baseline for synthetic customer risk scoring."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Path to the synthetic enterprise support dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path for the pickled model artifact.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data and print feature-table details without training or writing a model.",
    )
    return parser.parse_args(argv)


def run_training(
    *,
    data_dir: Path,
    output_path: Path,
    dry_run: bool,
    stdout: TextIO = sys.stdout,
    stderr: TextIO = sys.stderr,
) -> int:
    from scripts.validate_enterprise_support_data import format_report, validate_dataset
    from src.data.enterprise_support_loader import load_enterprise_support_dataset
    from src.ml.anomaly import (
        OptionalMLDependencyError,
        build_customer_feature_table,
        train_isolation_forest,
    )
    from src.ml.schemas import FEATURE_NAMES

    report = validate_dataset(data_dir)
    if not report.is_valid:
        print(format_report(report), file=stderr)
        return 1

    dataset = load_enterprise_support_dataset(data_dir)
    feature_table = build_customer_feature_table(dataset)

    print(f"Data directory: {data_dir}", file=stdout)
    print(f"Customer rows: {len(feature_table)}", file=stdout)
    print(f"Features: {', '.join(FEATURE_NAMES)}", file=stdout)
    if feature_table:
        sample = feature_table[0]
        preview = {name: sample[name] for name in FEATURE_NAMES}
        print(f"Sample row: customer_id={sample['customer_id']} features={preview}", file=stdout)

    if dry_run:
        print("Dry run: no model trained or written.", file=stdout)
        return 0

    try:
        model = train_isolation_forest(feature_table)
    except OptionalMLDependencyError as exc:
        print(str(exc), file=stderr)
        print(
            "Install scikit-learn in your local environment to train the optional ML baseline.",
            file=stderr,
        )
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "model_type": "isolation_forest",
        "feature_names": list(FEATURE_NAMES),
        "trained_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "data_dir": str(data_dir),
        "customer_count": len(feature_table),
    }
    with output_path.open("wb") as file:
        pickle.dump(artifact, file)

    print(f"Wrote model artifact: {output_path}", file=stdout)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run_training(
        data_dir=args.data_dir,
        output_path=args.output_path,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
