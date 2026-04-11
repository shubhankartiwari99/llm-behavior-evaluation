from __future__ import annotations

import argparse

from llm_eval.sampling import load_experiment_spec, run_experiment_sampling


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one of the probabilistic LLM behavior experiments and write unlabeled samples.",
    )
    parser.add_argument("--spec", required=True, help="Experiment JSON spec.")
    parser.add_argument("--output", required=True, help="Destination JSONL or JSON file.")
    parser.add_argument("--model-dir", help="Optional model directory override.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    spec = load_experiment_spec(args.spec)
    output_path = run_experiment_sampling(
        spec,
        args.output,
        model_dir=args.model_dir,
    )
    print(f"Wrote sampled responses to {output_path}")


if __name__ == "__main__":
    main()
