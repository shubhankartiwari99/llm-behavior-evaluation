import asyncio
import itertools
from typing import Any, List, Dict

TEMPS = [0.2, 0.4, 0.6, 0.8]
TOP_PS = [0.8, 0.9, 0.95]

async def run_reliability_grid(prompt: str, engine: Any, base_params: Dict[str, Any]):
    """
    Runs a grid search over temperature and top_p for a single prompt.
    """
    from app.api import run_inference_pipeline
    
    results = []
    
    for t, p in itertools.product(TEMPS, TOP_PS):
        eval_payload = {
            "prompt": prompt,
            "temperature": t,
            "top_p": p,
            "max_new_tokens": base_params.get("max_new_tokens", 80),
            "monte_carlo_samples": base_params.get("monte_carlo_samples", 5)
        }
        
        # Run inference in a thread to keep it non-blocking for the event loop
        res = await asyncio.to_thread(run_inference_pipeline, engine, eval_payload)
        
        results.append({
            "temperature": t,
            "top_p": p,
            "instability": res.get("instability", 0),
            "confidence": res.get("confidence", 0),
            "uncertainty": res.get("uncertainty", 0)
        })
        
    return results
