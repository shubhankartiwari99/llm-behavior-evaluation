from __future__ import annotations

import json
from pathlib import Path
from typing import Union

from llm_eval.schema import ExperimentPrompt, ExperimentSpec

DEFAULT_INFERENCE_PARAMS = {
    "temperature": 0.7,
    "top_p": 0.9,
    "max_new_tokens": 128,
    "do_sample": True,
    "monte_carlo_samples": 5,
    "emotional_lang": "en",
    "mode": "factual",
}


def default_experiment_specs() -> list[ExperimentSpec]:
    return [
        ExperimentSpec(
            experiment_id="exp_01_single_prompt_stability",
            title="Single Prompt Stability",
            objective=(
                "Estimate how often the model repeats the same labeled behavior "
                "for one fixed prompt."
            ),
            hypothesis=(
                "If the model is stable, one joint label pattern should "
                "dominate repeated runs."
            ),
            description=(
                "Run one prompt 30 times, manually label each response, and "
                "compare the empirical distribution over tone, cultural "
                "signal, and response type."
            ),
            runs_per_prompt=30,
            inference_params=DEFAULT_INFERENCE_PARAMS,
            prompts=[
                ExperimentPrompt(
                    prompt_id="p1",
                    prompt_type="single_prompt_stability",
                    condition="baseline",
                    prompt="Explain the importance of discipline for long-term success.",
                    notes="Keep the prompt fixed across all 30 runs.",
                )
            ],
            metrics=[
                "P(tone)",
                "P(cultural)",
                "P(type)",
                "joint pattern consistency",
                "response length variance",
            ],
        ),
        ExperimentSpec(
            experiment_id="exp_02_prompt_variation",
            title="Prompt Variation Sensitivity",
            objective="Measure how small wording changes shift the output distribution.",
            hypothesis=(
                "If the model is wording-sensitive, prompt variants will "
                "produce noticeably different cultural and response-type "
                "distributions."
            ),
            description=(
                "Use three semantically similar prompts with different "
                "phrasing. Compare their empirical distributions after "
                "repeated sampling and manual labeling."
            ),
            runs_per_prompt=20,
            inference_params=DEFAULT_INFERENCE_PARAMS,
            prompts=[
                ExperimentPrompt(
                    prompt_id="p1",
                    prompt_type="prompt_variation",
                    condition="baseline_wording",
                    prompt="Explain how discipline helps a person improve over time.",
                ),
                ExperimentPrompt(
                    prompt_id="p2",
                    prompt_type="prompt_variation",
                    condition="casual_wording",
                    prompt=(
                        "In simple words, how does discipline help someone get "
                        "better over time?"
                    ),
                ),
                ExperimentPrompt(
                    prompt_id="p3",
                    prompt_type="prompt_variation",
                    condition="instructional_wording",
                    prompt=(
                        "Give a short explanation of why discipline matters "
                        "for steady long-term progress."
                    ),
                ),
            ],
            metrics=[
                "P(category|condition)",
                "distribution shift across prompt variants",
                "joint pattern diversity",
            ],
        ),
        ExperimentSpec(
            experiment_id="exp_03_cultural_triggering",
            title="Cultural Triggering",
            objective=(
                "Measure how much an India-specific cue changes the "
                "probability of Indian cultural context in the response."
            ),
            hypothesis=(
                "Adding an India-specific cue should increase "
                "P(strong_indian_context) and overall Indian-context rate."
            ),
            description=(
                "Compare a neutral prompt and an India-specific prompt that "
                "ask for the same kind of answer."
            ),
            runs_per_prompt=20,
            inference_params=DEFAULT_INFERENCE_PARAMS,
            prompts=[
                ExperimentPrompt(
                    prompt_id="p1",
                    prompt_type="neutral_prompt",
                    condition="neutral",
                    prompt=(
                        "What are some practical ways a young professional can "
                        "manage living expenses?"
                    ),
                ),
                ExperimentPrompt(
                    prompt_id="p2",
                    prompt_type="india_specific_prompt",
                    condition="india_specific",
                    prompt=(
                        "What are some practical ways a young professional in "
                        "India can manage living expenses?"
                    ),
                ),
            ],
            metrics=[
                "P(cultural|prompt_type)",
                "indian_context_rate delta",
                "response type shift",
            ],
        ),
    ]


def write_default_experiment_specs(output_dir: Union[str, Path]) -> list[Path]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    for spec in default_experiment_specs():
        filename = f"{spec.experiment_id}.json"
        path = target_dir / filename
        path.write_text(
            json.dumps(spec.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        output_paths.append(path)

    return output_paths
