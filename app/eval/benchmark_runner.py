import asyncio
import datetime
import statistics
from typing import Any, Dict, List


async def run_benchmark(
    prompts: List[Dict[str, str]],
    engine: Any,
    validated_params: Dict[str, Any],
):
    """
    Sequentially runs reliability evaluation for each prompt in the list.
    """
    from app.api import run_inference_pipeline

    results = []
    total = len(prompts)

    # Per-prompt timeout: prevents a slow/looping model generation from
    # hanging the entire batch request until FastAPI's server timeout fires.
    prompt_timeout_seconds = 180

    for i, item in enumerate(prompts):
        prompt_text = item.get("prompt", "")
        if not prompt_text:
            continue

        eval_payload = {
            **validated_params,
            "prompt": prompt_text,
        }

        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(run_inference_pipeline, engine, eval_payload),
                timeout=prompt_timeout_seconds,
            )
            results.append(result)
        except asyncio.TimeoutError:
            import logging

            logging.warning(
                "Prompt %d/%d timed out after %ds - skipping: %.80r",
                i + 1,
                total,
                prompt_timeout_seconds,
                prompt_text,
            )
        except Exception as exc:
            import logging

            logging.warning("Prompt %d/%d failed (%s) - skipping.", i + 1, total, exc)

    from app.engine_identity import ENGINE_NAME

    summary = summarize_benchmark(results)
    summary["model"] = ENGINE_NAME
    return {
        "summary": summary,
        "results": results,
    }


def summarize_benchmark(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes aggregate stats from a list of evaluation results.
    """
    if not results:
        return {}

    total = len(results)

    confidence_scores = [r["confidence"] for r in results]
    instability_scores = [r["instability"] for r in results]
    uncertainty_scores = [r.get("uncertainty", 0.0) for r in results]
    entropy_scores = [r.get("entropy", 0.0) for r in results]
    latency_scores = [r["latency_ms"] for r in results]
    tokens_out = [r["output_tokens"] for r in results]

    escalations = sum(1 for r in results if r.get("escalate", False))
    guard_triggers = sum(1 for r in results if r.get("resampled", False))

    guard_deltas = []
    for result in results:
        mc = result.get("trace", {}).get("monte_carlo_analysis", {})
        rg = mc.get("reliability_guard", {}) if isinstance(mc, dict) else {}
        if rg.get("triggered") and isinstance(rg.get("instability_delta"), (int, float)):
            guard_deltas.append(rg["instability_delta"])

    # Hallucination proxy: fraction of results with detected instability or divergence
    hallucination_tags = {"stochastic_instability", "semantic_divergence"}
    hallucination_count = sum(
        1 for r in results
        if hallucination_tags.intersection(r.get("failures") or [])
    )

    def mean_safe(data):
        return statistics.mean(data) if data else 0.0

    def stdev_safe(data):
        return statistics.stdev(data) if len(data) >= 2 else 0.0

    def percentile(data, pct):
        if not data:
            return 0.0
        s = sorted(data)
        k = max(0, min(int(len(s) * pct / 100), len(s) - 1))
        return s[k]

    summary = {
        "total": total,
        "mean_confidence": mean_safe(confidence_scores),
        "mean_instability": mean_safe(instability_scores),
        "mean_uncertainty": mean_safe(uncertainty_scores),
        "mean_entropy": mean_safe(entropy_scores),
        "entropy_std_dev": stdev_safe(entropy_scores),
        "confidence_p90": percentile(confidence_scores, 90),
        "hallucination_proxy_rate": hallucination_count / total if total > 0 else 0.0,
        "escalation_rate": escalations / total if total > 0 else 0.0,
        "guard_trigger_rate": guard_triggers / total if total > 0 else 0.0,
        "mean_guard_instability_delta": mean_safe(guard_deltas),
        "avg_latency": mean_safe(latency_scores),
        "avg_output_tokens": mean_safe(tokens_out),
        "distributions": {
            "confidence": confidence_scores,
            "instability": instability_scores,
            "entropy": entropy_scores,
        },
        "timestamp": datetime.datetime.now().isoformat(),
    }

    return summary
