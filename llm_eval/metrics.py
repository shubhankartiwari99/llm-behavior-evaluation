from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import mean, pvariance
from typing import Iterable, Optional

from llm_eval.schema import LabeledResponseRecord

DISPLAY_FIELDS = ("tone", "cultural", "type")
SCORE_MAPS = {
    "tone": {
        "informal": 0.0,
        "neutral": 1.0,
        "formal": 2.0,
    },
    "cultural": {
        "none": 0.0,
        "weak_indian_context": 1.0,
        "strong_indian_context": 2.0,
    },
    "type": {
        "generic": 0.0,
        "specific": 1.0,
        "example_driven": 2.0,
    },
}


def _label_value(record: LabeledResponseRecord, field_name: str) -> Optional[str]:
    if field_name == "type":
        value = record.response_type
    else:
        value = getattr(record, field_name)
    if value is None:
        return None
    return value.value if hasattr(value, "value") else str(value)


def _pattern_key(record: LabeledResponseRecord) -> Optional[str]:
    pattern = record.label_pattern()
    if pattern is None:
        return None
    return " | ".join(pattern)


def _distribution(values: Iterable[str]) -> dict[str, dict[str, float | int]]:
    counter = Counter(values)
    total = sum(counter.values())
    if total == 0:
        return {}
    return {
        label: {
            "count": count,
            "probability": count / total,
        }
        for label, count in sorted(counter.items())
    }


def _majority_ratio(values: list[str]) -> tuple[str, float]:
    counter = Counter(values)
    majority_label, majority_count = counter.most_common(1)[0]
    return majority_label, majority_count / len(values)


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _safe_variance(values: list[float]) -> float:
    return pvariance(values) if len(values) >= 2 else 0.0


def frequency_distribution(records: list[LabeledResponseRecord], field_name: str) -> dict[str, dict[str, float | int]]:
    values = [
        value
        for value in (_label_value(record, field_name) for record in records)
        if value is not None
    ]
    return _distribution(values)


def joint_pattern_distribution(records: list[LabeledResponseRecord]) -> dict[str, dict[str, float | int]]:
    values = [
        value
        for value in (_pattern_key(record) for record in records)
        if value is not None
    ]
    return _distribution(values)


def consistency_summary(records: list[LabeledResponseRecord]) -> dict[str, object]:
    grouped: dict[str, list[LabeledResponseRecord]] = defaultdict(list)
    for record in records:
        grouped[record.prompt_id].append(record)

    field_majority_ratios: dict[str, list[float]] = {
        field_name: [] for field_name in DISPLAY_FIELDS
    }
    joint_majority_ratios: list[float] = []
    prompt_reports: dict[str, object] = {}

    for prompt_id, prompt_records in grouped.items():
        prompt_report: dict[str, object] = {
            "sample_count": len(prompt_records),
            "prompt_type": prompt_records[0].prompt_type,
        }

        for field_name in DISPLAY_FIELDS:
            values = [_label_value(record, field_name) for record in prompt_records]
            filtered = [value for value in values if value is not None]
            if not filtered:
                continue
            majority_label, ratio = _majority_ratio(filtered)
            field_majority_ratios[field_name].append(ratio)
            prompt_report[field_name] = {
                "majority_label": majority_label,
                "majority_ratio": ratio,
                "distribution": _distribution(filtered),
            }

        joint_values = [
            value
            for value in (_pattern_key(record) for record in prompt_records)
            if value is not None
        ]
        if joint_values:
            majority_pattern, ratio = _majority_ratio(joint_values)
            joint_majority_ratios.append(ratio)
            prompt_report["joint_pattern"] = {
                "majority_pattern": majority_pattern,
                "majority_ratio": ratio,
                "distribution": _distribution(joint_values),
            }

        prompt_reports[prompt_id] = prompt_report

    return {
        "prompt_count": len(grouped),
        "field_consistency": {
            field_name: {
                "mean_majority_ratio": _safe_mean(field_majority_ratios[field_name]),
                "variance": _safe_variance(field_majority_ratios[field_name]),
                "prompt_count": len(field_majority_ratios[field_name]),
            }
            for field_name in DISPLAY_FIELDS
        },
        "joint_pattern_consistency": {
            "mean_majority_ratio": _safe_mean(joint_majority_ratios),
            "variance": _safe_variance(joint_majority_ratios),
            "prompt_count": len(joint_majority_ratios),
        },
        "prompt_groups": prompt_reports,
    }


def diversity_summary(records: list[LabeledResponseRecord]) -> dict[str, float | int]:
    pattern_keys = [
        value
        for value in (_pattern_key(record) for record in records)
        if value is not None
    ]
    pattern_counter = Counter(pattern_keys)
    total_patterns = sum(pattern_counter.values())

    pattern_entropy = 0.0
    if total_patterns:
        for count in pattern_counter.values():
            probability = count / total_patterns
            pattern_entropy -= probability * math.log2(probability)

    normalized_pattern_entropy = 0.0
    if len(pattern_counter) > 1:
        normalized_pattern_entropy = pattern_entropy / math.log2(len(pattern_counter))

    normalized_texts = {" ".join(record.response.lower().split()) for record in records}
    return {
        "unique_label_patterns": len(pattern_counter),
        "pattern_entropy": pattern_entropy,
        "normalized_pattern_entropy": normalized_pattern_entropy,
        "unique_response_texts": len(normalized_texts),
        "unique_response_ratio": (len(normalized_texts) / len(records)) if records else 0.0,
    }


def cultural_bias_strength(records: list[LabeledResponseRecord]) -> dict[str, float | int]:
    cultural_values = [_label_value(record, "cultural") for record in records]
    filtered = [value for value in cultural_values if value is not None]
    total = len(filtered)
    strong_count = sum(1 for value in filtered if value == "strong_indian_context")
    weak_count = sum(1 for value in filtered if value == "weak_indian_context")
    context_count = strong_count + weak_count
    return {
        "responses_with_indian_context": context_count,
        "indian_context_rate": (context_count / total) if total else 0.0,
        "strong_indian_context_rate": (strong_count / total) if total else 0.0,
        "weak_indian_context_rate": (weak_count / total) if total else 0.0,
    }


def conditional_probability_table(
    records: list[LabeledResponseRecord],
    *,
    target_field: str,
    given_field: str,
) -> dict[str, dict[str, object]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for record in records:
        given_value = getattr(record, given_field, None)
        if given_value is None:
            continue
        if not isinstance(given_value, str):
            given_value = str(given_value)
        target_value = _label_value(record, target_field)
        if target_value is None:
            continue
        grouped[given_value].append(target_value)

    return {
        given_value: {
            "count": len(values),
            "distribution": _distribution(values),
        }
        for given_value, values in sorted(grouped.items())
    }


def expectation_summary(records: list[LabeledResponseRecord]) -> dict[str, float]:
    word_lengths = [len(record.response.split()) for record in records]
    char_lengths = [len(record.response) for record in records]

    expectation = {
        "avg_response_length_words": _safe_mean([float(value) for value in word_lengths]),
        "avg_response_length_chars": _safe_mean([float(value) for value in char_lengths]),
    }

    for field_name, score_map in SCORE_MAPS.items():
        scores = [
            score_map[value]
            for value in (_label_value(record, field_name) for record in records)
            if value in score_map
        ]
        expectation[f"avg_{field_name}_score"] = _safe_mean(scores)

    return expectation


def variance_summary(records: list[LabeledResponseRecord]) -> dict[str, float]:
    word_lengths = [float(len(record.response.split())) for record in records]
    variance = {
        "response_length_variance_words": _safe_variance(word_lengths),
        "response_length_std_dev_words": math.sqrt(_safe_variance(word_lengths)),
    }

    for field_name, score_map in SCORE_MAPS.items():
        scores = [
            score_map[value]
            for value in (_label_value(record, field_name) for record in records)
            if value in score_map
        ]
        variance[f"{field_name}_score_variance"] = _safe_variance(scores)

    return variance


def compute_behavior_summary(
    records: list[LabeledResponseRecord],
    *,
    require_complete_labels: bool = False,
) -> dict[str, object]:
    complete_records = [record for record in records if record.is_fully_labeled()]
    incomplete_count = len(records) - len(complete_records)

    if require_complete_labels and incomplete_count:
        raise ValueError(
            f"Found {incomplete_count} incomplete records. Finish manual labeling "
            "before analysis."
        )
    if not complete_records:
        raise ValueError("At least one fully labeled record is required for analysis.")

    return {
        "total_records": len(records),
        "fully_labeled_records": len(complete_records),
        "incomplete_records": incomplete_count,
        "distributions": {
            "tone": frequency_distribution(complete_records, "tone"),
            "cultural": frequency_distribution(complete_records, "cultural"),
            "type": frequency_distribution(complete_records, "type"),
            "joint_pattern": joint_pattern_distribution(complete_records),
        },
        "consistency": consistency_summary(complete_records),
        "diversity": diversity_summary(complete_records),
        "cultural_bias_strength": cultural_bias_strength(complete_records),
        "conditional_probabilities": {
            "P(cultural|prompt_type)": conditional_probability_table(
                complete_records,
                target_field="cultural",
                given_field="prompt_type",
            )
        },
        "expectation": expectation_summary(complete_records),
        "variance": variance_summary(complete_records),
        "stat_110_mapping": {
            "basic_probability": "Use the per-label frequency distributions as empirical P(category).",
            "conditional_probability": "Use P(cultural|prompt_type) to compare prompt conditions.",
            "expectation": "Use average response length and average ordinal label scores as empirical expectations.",
            "variance": "Use response-length variance and label-score variance to quantify stability.",
        },
    }
