from __future__ import annotations

import argparse
import json

from llm_eval.dataset import load_records
from llm_eval.metrics import compute_behavior_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze a manually labeled LLM behavior dataset.",
    )
    parser.add_argument("--input", required=True, help="Labeled JSON or JSONL dataset.")
    parser.add_argument("--output", help="Optional path to write the JSON summary.")
    parser.add_argument(
        "--require-complete-labels",
        action="store_true",
        help="Fail if any record is missing tone, cultural, or type labels.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    summary = compute_behavior_summary(
        load_records(args.input),
        require_complete_labels=args.require_complete_labels,
    )
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")

    print(rendered)


if __name__ == "__main__":
    main()
