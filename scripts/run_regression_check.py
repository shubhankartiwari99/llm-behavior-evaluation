from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from scripts.regression_engine import (
    append_decision_history,
    decide_promotion,
    evaluate_model,
    load_dataset,
    load_model,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a simple model regression check on a frozen dataset.",
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
        "--output",
        default=None,
        help="Optional path to write the comparison result JSON.",
    )
    parser.add_argument(
        "--history",
        default="artifacts/decision_history.json",
        help="Path to append decision history records.",
    )
    return parser.parse_args()


def build_record(
    prod_model: dict[str, Any],
    cand_model: dict[str, Any],
    prod_metrics: Any,
    cand_metrics: Any,
    decision: Any,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "prod_model": prod_model["model_id"],
        "cand_model": cand_model["model_id"],
        "production_metrics": prod_metrics.to_dict(),
        "candidate_metrics": cand_metrics.to_dict(),
        "decision": decision.decision,
        "reason": decision.reason,
        "deltas": decision.deltas,
    }


def main() -> int:
    args = _parse_args()
    dataset = load_dataset(args.dataset)
    prod_model = load_model(args.prod)
    cand_model = load_model(args.cand)

    prod_metrics = evaluate_model(prod_model, dataset)
    cand_metrics = evaluate_model(cand_model, dataset)
    decision = decide_promotion(prod_metrics, cand_metrics)

    print(f"Production model: {prod_model['model_id']}")
    print(f"Candidate model:  {cand_model['model_id']}\n")
    print("Production:")
    print(f"  accuracy: {prod_metrics.accuracy:.4f}")
    print(f"  consistency: {prod_metrics.consistency_score:.4f}")
    print("\nCandidate:")
    print(f"  accuracy: {cand_metrics.accuracy:.4f}")
    print(f"  consistency: {cand_metrics.consistency_score:.4f}")
    print(f"\nΔ accuracy: {decision.deltas['delta_accuracy']:+.4f}")
    print(f"Δ consistency: {decision.deltas['delta_consistency']:+.4f}")
    print(f"\nDecision: {decision.decision} ({decision.reason})")


    result_record = build_record(prod_model, cand_model, prod_metrics, cand_metrics, decision)
    append_decision_history(args.history, result_record)
    print(f"\nAppended decision history to {args.history}")

    if args.output:
        output_data = {
            "production_model": prod_model["model_id"],
            "candidate_model": cand_model["model_id"],
            "production_metrics": prod_metrics.to_dict(),
            "candidate_metrics": cand_metrics.to_dict(),
            "decision": decision.to_dict(),
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
        print(f"Persisted regression comparison to {output_path}")

    return 1 if decision.decision == "reject" else 0


if __name__ == "__main__":
    raise SystemExit(main())
