from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CI gate for LLM regression promotion policy.",
    )
    parser.add_argument(
        "--prod",
        required=True,
        help="Path to the production model JSON file.",
    )
    parser.add_argument(
        "--cand",
        required=True,
        help="Path to the candidate model JSON file.",
    )
    parser.add_argument(
        "--dataset",
        default="data/eval_dataset.json",
        help="Path to the frozen evaluation dataset.",
    )
    parser.add_argument(
        "--history",
        default="artifacts/decision_history.json",
        help="Path to append decision history records.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write the comparison result JSON.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_regression_check.py"),
        "--prod",
        args.prod,
        "--cand",
        args.cand,
        "--dataset",
        args.dataset,
        "--history",
        args.history,
    ]
    if args.output:
        command += ["--output", args.output]

    result = subprocess.run(command)
    if result.returncode != 0:
        print("CI regression gate: REJECT — candidate model failed promotion policy.")
        return 1

    print("CI regression gate: GO — candidate model passed promotion policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
