from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class EvaluationMetrics:
    accuracy: float
    hallucination_rate: float
    format_error_rate: float

    def to_dict(self) -> dict[str, float]:
        return {
            "accuracy": round(self.accuracy, 6),
            "hallucination_rate": round(self.hallucination_rate, 6),
            "format_error_rate": round(self.format_error_rate, 6),
        }


def load_dataset(path: str) -> list[dict[str, Any]]:
    dataset_path = Path(path)
    raw = json.loads(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Evaluation dataset must be a JSON array of examples.")
    return raw


def load_model(path: str) -> dict[str, Any]:
    model_path = Path(path)
    raw = json.loads(model_path.read_text(encoding="utf-8"))
    if "model_id" not in raw or "responses" not in raw:
        raise ValueError("Model JSON must contain 'model_id' and 'responses'.")
    return raw


def normalize_text(value: str) -> str:
    return " ".join(value.strip().lower().split())


def is_json_array(value: str) -> bool:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return False
    return isinstance(parsed, list)


def evaluate_model(model: dict[str, Any], dataset: list[dict[str, Any]]) -> EvaluationMetrics:
    responses = model["responses"]
    total_examples = len(dataset)
    if total_examples == 0:
        return EvaluationMetrics(accuracy=0.0, hallucination_rate=0.0, format_error_rate=0.0)

    correct = 0
    hallucinations = 0
    format_errors = 0
    format_examples = 0

    for item in dataset:
        prompt = item["prompt"]
        expected_type = item["expected_type"]
        expected_answer = item.get("expected_answer", "")
        output = normalize_text(str(responses.get(prompt, "")))
        normalized_expected = normalize_text(str(expected_answer))

        if output and normalized_expected and output == normalized_expected:
            correct += 1
        else:
            hallucinations += 1

        if expected_type == "format":
            format_examples += 1
            if not is_json_array(responses.get(prompt, "")):
                format_errors += 1

    accuracy = correct / total_examples
    hallucination_rate = hallucinations / total_examples
    format_error_rate = format_errors / max(1, format_examples)

    return EvaluationMetrics(
        accuracy=accuracy,
        hallucination_rate=hallucination_rate,
        format_error_rate=format_error_rate,
    )


def compare_models(prod_metrics: EvaluationMetrics, cand_metrics: EvaluationMetrics) -> dict[str, float]:
    return {
        "delta_accuracy": round(cand_metrics.accuracy - prod_metrics.accuracy, 6),
        "delta_hallucination": round(cand_metrics.hallucination_rate - prod_metrics.hallucination_rate, 6),
        "delta_format_error": round(cand_metrics.format_error_rate - prod_metrics.format_error_rate, 6),
    }


@dataclass
class Decision:
    decision: str
    reason: str
    deltas: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "reason": self.reason,
            "deltas": {k: round(v, 6) for k, v in self.deltas.items()},
        }


def decide_promotion(prod_metrics: EvaluationMetrics, cand_metrics: EvaluationMetrics) -> Decision:
    deltas = compare_models(prod_metrics, cand_metrics)
    if deltas["delta_hallucination"] > 0:
        return Decision(
            decision="reject",
            reason="hallucination_increased",
            deltas=deltas,
        )
    if deltas["delta_format_error"] > 0:
        return Decision(
            decision="reject",
            reason="format_error_increased",
            deltas=deltas,
        )
    if deltas["delta_accuracy"] < 0:
        return Decision(
            decision="reject",
            reason="accuracy_decreased",
            deltas=deltas,
        )
    return Decision(
        decision="promote",
        reason="all_metrics_stable_or_improved",
        deltas=deltas,
    )


def append_decision_history(path: str, record: dict[str, Any]) -> None:
    history_path = Path(path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        existing = json.loads(history_path.read_text(encoding="utf-8"))
        if not isinstance(existing, list):
            existing = []
    else:
        existing = []
    existing.append(record)
    history_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
