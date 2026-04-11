from __future__ import annotations

import argparse

from llm_eval.dataset import bootstrap_records_from_file, load_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert existing eval result files into manual-labeling datasets.",
    )
    parser.add_argument("--input", required=True, help="Source JSON or JSONL file.")
    parser.add_argument("--output", required=True, help="Destination JSONL or JSON file.")
    parser.add_argument("--experiment-id", help="Optional experiment identifier override.")
    parser.add_argument(
        "--default-prompt-type",
        help="Fallback prompt_type when the source file does not define one.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    output_path = bootstrap_records_from_file(
        args.input,
        args.output,
        experiment_id=args.experiment_id,
        default_prompt_type=args.default_prompt_type,
    )
    record_count = len(load_records(output_path))
    print(f"Wrote {record_count} records to {output_path}")


if __name__ == "__main__":
    main()
