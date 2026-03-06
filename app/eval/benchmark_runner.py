import asyncio
import statistics
import datetime
from typing import Any, List, Dict
from app.intelligence.dual_plane import evaluate_dual_plane

async def run_benchmark(prompts: List[Dict[str, str]], engine: Any, validated_params: Dict[str, Any]):
    """
    Sequentially runs reliability evaluation for each prompt in the list.
    """
    from app.api import run_inference_pipeline  # Lazy import to avoid circular dependency
    
    results = []
    total = len(prompts)
    
    # Per-prompt timeout: prevents a slow/looping model generation from
    # hanging the entire batch request until FastAPI's server timeout fires.
    PROMPT_TIMEOUT_SECONDS = 180

    for i, item in enumerate(prompts):
        prompt_text = item.get("prompt", "")
        if not prompt_text:
            continue
            
        # Prepare evaluation parameters for this prompt
        eval_payload = {
            **validated_params,
            "prompt": prompt_text
        }
        
        try:
            # Run the full reliability pipeline in a thread, with a timeout guard
            result = await asyncio.wait_for(
                asyncio.to_thread(run_inference_pipeline, engine, eval_payload),
                timeout=PROMPT_TIMEOUT_SECONDS,
            )
            results.append(result)
        except asyncio.TimeoutError:
            import logging
            logging.warning(
                "Prompt %d/%d timed out after %ds — skipping: %.80r",
                i + 1, total, PROMPT_TIMEOUT_SECONDS, prompt_text,
            )
        except Exception as exc:
            import logging
            logging.warning("Prompt %d/%d failed (%s) — skipping.", i + 1, total, exc)
        
    from app.engine_identity import ENGINE_NAME
    summary = summarize_benchmark(results)
    summary["model"] = ENGINE_NAME
    return {
        "summary": summary,
        "results": results
    }

def summarize_benchmark(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Computes aggregate stats from a list of evaluation results.
    """
    if not results:
        return {}
        
    total = len(results)
    
    # Extract metric lists
    confidence_scores = [r["confidence"] for r in results]
    instability_scores = [r["instability"] for r in results]
    uncertainty_scores = [r.get("uncertainty", 0.0) for r in results]
    entropy_scores = [r.get("entropy", 0.0) for r in results]
    latency_scores = [r["latency_ms"] for r in results]
    tokens_out = [r["output_tokens"] for r in results]
    
    escalations = sum(1 for r in results if r.get("escalate", False))
    
    def mean_safe(data):
        return statistics.mean(data) if data else 0.0

    summary = {
        "total": total,
        "mean_confidence": mean_safe(confidence_scores),
        "mean_instability": mean_safe(instability_scores),
        "mean_uncertainty": mean_safe(uncertainty_scores),
        "mean_entropy": mean_safe(entropy_scores),
        "escalation_rate": escalations / total if total > 0 else 0.0,
        "avg_latency": mean_safe(latency_scores),
        "avg_output_tokens": mean_safe(tokens_out),
        # Histograms/Distributions for UI
        "distributions": {
            "confidence": confidence_scores,
            "instability": instability_scores
        },
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    return summary
