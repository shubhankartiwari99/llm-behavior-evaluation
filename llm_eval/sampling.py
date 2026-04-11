from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

from llm_eval.dataset import write_records
from llm_eval.schema import ExperimentSpec, LabeledResponseRecord


def load_experiment_spec(path: Union[str, Path]) -> ExperimentSpec:
    import json

    spec_path = Path(path)
    return ExperimentSpec.model_validate(json.loads(spec_path.read_text(encoding="utf-8")))


def _resolve_model_dir(model_dir: Optional[str] = None) -> str:
    if model_dir:
        return model_dir

    import os

    from app.engine_config import MODEL_BACKEND

    env_model_dir = os.environ.get("MODEL_DIR")
    if env_model_dir:
        return env_model_dir
    if MODEL_BACKEND == "gguf":
        return "model_gguf"
    if MODEL_BACKEND == "remote":
        return "."
    raise RuntimeError("MODEL_DIR environment variable must be set for local experiment sampling.")


def run_experiment_sampling(
    spec: ExperimentSpec,
    output_path: Union[str, Path],
    *,
    model_dir: Optional[str] = None,
) -> Path:
    from app.api import run_inference_pipeline
    from app.inference import InferenceEngine

    engine = InferenceEngine(_resolve_model_dir(model_dir))
    records: list[LabeledResponseRecord] = []

    for prompt_spec in spec.prompts:
        for run_index in range(1, spec.runs_per_prompt + 1):
            validated: dict[str, Any] = {
                **spec.inference_params,
                "prompt": prompt_spec.prompt,
            }
            result = run_inference_pipeline(engine, validated)
            records.append(
                LabeledResponseRecord(
                    record_id=f"{prompt_spec.prompt_id}_r{run_index}",
                    experiment_id=spec.experiment_id,
                    prompt_id=prompt_spec.prompt_id,
                    run_id=f"{prompt_spec.prompt_id}_r{run_index}",
                    prompt=prompt_spec.prompt,
                    prompt_type=prompt_spec.prompt_type,
                    condition=prompt_spec.condition,
                    response=result["response_text"],
                    confidence=result.get("confidence"),
                    instability=result.get("instability"),
                    latency_ms=result.get("latency_ms"),
                    metadata={
                        "sample_count": result.get("sample_count"),
                        "cluster_count": result.get("cluster_count"),
                        "source": "llm_eval_sampling",
                    },
                )
            )

    return write_records(output_path, records)
