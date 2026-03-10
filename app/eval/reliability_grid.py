import asyncio
import itertools
from typing import Any, Dict

# Updated from grid sweep empirical results (400 inferences, Qwen 2.5-7B).
# T=0.1 and top_p=0.5 are the optimal corner; both are now included so the
# live heatmap surfaces the best-performing configuration.
TEMPS = [0.1, 0.3, 0.7, 0.9]
TOP_PS = [0.5, 0.7, 0.85, 0.95]


async def run_reliability_grid(prompt: str, engine: Any, base_params: Dict[str, Any]):
    """
    Runs a grid search over temperature and top_p for a single prompt.

    Parameter space mirrors the published Kaggle grid sweep so the live
    dashboard heatmap is directly comparable to the research results.
    """
    from app.api import run_inference_pipeline

    results = []

    for temperature, top_p in itertools.product(TEMPS, TOP_PS):
        eval_payload = {
            "prompt": prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_new_tokens": base_params.get("max_new_tokens", 80),
            "monte_carlo_samples": base_params.get("monte_carlo_samples", 5),
        }

        result = await asyncio.to_thread(run_inference_pipeline, engine, eval_payload)

        results.append(
            {
                "temperature": temperature,
                "top_p": top_p,
                "instability": result.get("instability", 0),
                "confidence": result.get("confidence", 0),
                "uncertainty": result.get("uncertainty", 0),
                "resampled": result.get("resampled", False),
            }
        )

    return results
