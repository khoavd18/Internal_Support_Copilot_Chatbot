from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> None:
    print(f"[run] {' '.join(command)}")
    subprocess.run(command, check=True, cwd=ROOT_DIR)


def _ensure_raw_data_present() -> None:
    raw_dir = ROOT_DIR / "data_source" / "raw"
    if not raw_dir.exists():
        raise SystemExit(
            "Missing data_source/raw. Add your source documents there before running ingest-data."
        )

    if not any(path.is_file() for path in raw_dir.rglob("*")):
        raise SystemExit(
            "No source files found under data_source/raw. Add your source documents first, then rerun ingest-data."
        )


def install(_: list[str]) -> None:
    _run([sys.executable, "-m", "pip", "install", "-r", "requirements-dev.txt"])


def run_api(extra_args: list[str]) -> None:
    _run([sys.executable, "-m", "uvicorn", "src.api.main:app", "--reload", *extra_args])


def run_ui(extra_args: list[str]) -> None:
    _run([sys.executable, "-m", "streamlit", "run", "src/ui/chatbot.py", *extra_args])


def run_tests(extra_args: list[str]) -> None:
    args = extra_args or ["-q"]
    _run([sys.executable, "-m", "pytest", *args])


def benchmark_retrieval(extra_args: list[str]) -> None:
    _run([sys.executable, "scripts/evaluate_retrieval.py", *extra_args])


def benchmark_answers(extra_args: list[str]) -> None:
    _run([sys.executable, "scripts/evaluate_answer_quality.py", *extra_args])


def ingest_data(_: list[str]) -> None:
    _ensure_raw_data_present()
    _run([sys.executable, "scripts/ingest_data.py"])


COMMANDS = {
    "benchmark-answers": benchmark_answers,
    "benchmark-retrieval": benchmark_retrieval,
    "install": install,
    "run-api": run_api,
    "run-ui": run_ui,
    "run-tests": run_tests,
    "ingest-data": ingest_data,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cross-platform helper for local development tasks."
    )
    parser.add_argument("command", choices=sorted(COMMANDS))
    parser.add_argument(
        "args",
        nargs=argparse.REMAINDER,
        help="Extra arguments to pass to the underlying command after '--'.",
    )
    parsed = parser.parse_args()

    extra_args = parsed.args
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]

    COMMANDS[parsed.command](extra_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
