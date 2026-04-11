from __future__ import annotations

import pytest

from llm_eval.experiment_templates import default_experiment_specs
from llm_eval.metrics import compute_behavior_summary
from llm_eval.schema import LabeledResponseRecord


def _record(
    prompt_id: str,
    prompt_type: str,
    response: str,
    tone: str,
    cultural: str,
    response_type: str,
) -> LabeledResponseRecord:
    return LabeledResponseRecord(
        prompt_id=prompt_id,
        prompt_type=prompt_type,
        response=response,
        tone=tone,
        cultural=cultural,
        type=response_type,
    )


def test_compute_behavior_summary_returns_expected_probabilities():
    records = [
        _record("p1", "neutral_prompt", "A practical answer for most people.", "neutral", "strong_indian_context", "specific"),
        _record("p1", "neutral_prompt", "Another practical answer with detail.", "neutral", "strong_indian_context", "specific"),
        _record("p1", "neutral_prompt", "For example, use a simple monthly budget.", "formal", "weak_indian_context", "example_driven"),
        _record("p2", "india_specific_prompt", "Keep your spending list basic and short.", "informal", "none", "generic"),
        _record("p2", "india_specific_prompt", "Track your needs before your wants.", "informal", "none", "generic"),
        _record("p2", "india_specific_prompt", "Use a rent cap and transport budget.", "neutral", "strong_indian_context", "specific"),
    ]

    summary = compute_behavior_summary(records, require_complete_labels=True)

    assert summary["total_records"] == 6
    assert summary["distributions"]["cultural"]["strong_indian_context"]["probability"] == pytest.approx(0.5)
    assert summary["distributions"]["type"]["generic"]["count"] == 2
    assert summary["consistency"]["joint_pattern_consistency"]["mean_majority_ratio"] == pytest.approx(2 / 3)
    assert summary["cultural_bias_strength"]["indian_context_rate"] == pytest.approx(4 / 6)
    assert summary["expectation"]["avg_tone_score"] == pytest.approx(5 / 6)
    assert summary["variance"]["tone_score_variance"] == pytest.approx(17 / 36)
    conditional = summary["conditional_probabilities"]["P(cultural|prompt_type)"]
    assert conditional["neutral_prompt"]["distribution"]["strong_indian_context"]["probability"] == pytest.approx(2 / 3)
    assert conditional["india_specific_prompt"]["distribution"]["none"]["probability"] == pytest.approx(2 / 3)


def test_compute_behavior_summary_ignores_incomplete_records_when_not_strict():
    records = [
        _record("p1", "neutral_prompt", "A neutral answer.", "neutral", "none", "generic"),
        LabeledResponseRecord(
            prompt_id="p1",
            prompt_type="neutral_prompt",
            response="Unlabeled answer for manual review.",
        ),
    ]

    summary = compute_behavior_summary(records)

    assert summary["total_records"] == 2
    assert summary["fully_labeled_records"] == 1
    assert summary["incomplete_records"] == 1


def test_default_experiment_specs_cover_three_starter_experiments():
    specs = default_experiment_specs()

    assert len(specs) == 3
    assert specs[0].experiment_id == "exp_01_single_prompt_stability"
    assert specs[1].runs_per_prompt == 20
    assert specs[2].prompts[1].condition == "india_specific"
