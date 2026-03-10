"""
Grid-sweep-informed reliability fallback layer.

Empirical basis (Qwen 2.5-7B, 400-run parameter grid sweep):
  - Optimal stability: T=0.1 + top_p=0.5  -> instability 0.056
  - Worst observed:    T=0.9 + top_p=1.0  -> instability 0.290
  - Temperature is the dominant lever (variance 0.003712 vs 0.001192 for top_p)
  - Instability threshold 0.25 sits at the empirical inflection point between
    moderate and high stochastic instability across all category types.

When a request's initial entropy pass exceeds INSTABILITY_THRESHOLD, this guard
intercepts, switches to FALLBACK params, resamples, and returns an enriched
result with full before/after telemetry embedded in the trace.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from app.inference import InferenceEngine

# Empirically derived constants (grid sweep, 400 inferences)
INSTABILITY_THRESHOLD: float = 0.25
FALLBACK_TEMPERATURE: float = 0.1
FALLBACK_TOP_P: float = 0.5
FALLBACK_MIN_SAMPLES: int = 4
FALLBACK_STABILITY_THRESHOLD: float = 0.03

logger = logging.getLogger(__name__)


def evaluate_dual_plane(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazy wrapper to keep the dual-plane dependency patchable in tests."""
    from app.intelligence.dual_plane import evaluate_dual_plane as _evaluate_dual_plane

    return _evaluate_dual_plane(*args, **kwargs)


def needs_fallback(instability: float) -> bool:
    """Returns True if instability exceeds the empirically derived threshold."""
    return instability > INSTABILITY_THRESHOLD


def run_fallback_sampling(
    engine: "InferenceEngine",
    structured_prompt: str,
    det_response_text: str,
    det_token_count: int,
    validated: dict[str, Any],
    stop_tokens: list[str],
) -> dict[str, Any]:
    """
    Resamples using grid-sweep optimal params (T=0.1, top_p=0.5).

    Returns a dict with:
      - ent_outputs: list of sampled texts
      - ent_metas: list of per-run metadata
      - analysis: evaluate_dual_plane result
      - samples_used: how many MC samples were taken
    """
    fallback_kwargs = {
        "temperature": FALLBACK_TEMPERATURE,
        "top_p": FALLBACK_TOP_P,
        "do_sample": True,
        "max_new_tokens": validated["max_new_tokens"],
        "stop": stop_tokens,
        "include_token_entropy": True,
    }

    ent_outputs: list[str] = []
    ent_metas: list[dict[str, Any]] = []
    instability_history: list[float] = []
    final_analysis: dict[str, Any] | None = None
    max_samples = validated["monte_carlo_samples"]

    while len(ent_outputs) < max_samples:
        res_text, res_meta = engine.generate(
            structured_prompt,
            return_meta=True,
            **fallback_kwargs,
        )
        ent_outputs.append(res_text)
        ent_metas.append(res_meta)

        if len(ent_outputs) >= 2:
            current_token_counts = [
                meta.get("output_tokens", len(output.split()))
                for output, meta in zip(ent_outputs, ent_metas)
            ]
            analysis = evaluate_dual_plane(
                det_response_text,
                ent_outputs,
                det_token_count,
                current_token_counts,
            )
            instability_history.append(analysis["instability"])

            if len(ent_outputs) >= FALLBACK_MIN_SAMPLES and len(instability_history) >= 2:
                delta = abs(instability_history[-1] - instability_history[-2])
                if delta < FALLBACK_STABILITY_THRESHOLD:
                    final_analysis = analysis
                    break

            final_analysis = analysis

    if final_analysis is None:
        token_counts = [
            meta.get("output_tokens", len(output.split()))
            for output, meta in zip(ent_outputs, ent_metas)
        ]
        final_analysis = evaluate_dual_plane(
            det_response_text,
            ent_outputs,
            det_token_count,
            token_counts,
        )

    return {
        "ent_outputs": ent_outputs,
        "ent_metas": ent_metas,
        "analysis": final_analysis,
        "samples_used": len(ent_outputs),
    }


def apply_reliability_guard(
    engine: "InferenceEngine",
    structured_prompt: str,
    det_response_text: str,
    det_token_count: int,
    validated: dict[str, Any],
    stop_tokens: list[str],
    initial_analysis: dict[str, Any],
    initial_ent_outputs: list[str],
    initial_ent_metas: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Main entry point for the reliability guard.

    If the initial instability is within threshold, returns a pass-through
    result with resampled=False. Otherwise triggers the fallback pass and
    returns enriched telemetry.
    """
    initial_instability = initial_analysis["instability"]

    if not needs_fallback(initial_instability):
        return {
            "resampled": False,
            "ent_outputs": initial_ent_outputs,
            "ent_metas": list(initial_ent_metas or []),
            "analysis": initial_analysis,
            "samples_used": len(initial_ent_outputs),
            "reliability_guard": {
                "triggered": False,
                "initial_instability": round(initial_instability, 4),
                "threshold": INSTABILITY_THRESHOLD,
            },
        }

    logger.info(
        "ReliabilityGuard triggered - instability %.3f > threshold %.3f. "
        "Switching to T=%.1f top_p=%.1f.",
        initial_instability,
        INSTABILITY_THRESHOLD,
        FALLBACK_TEMPERATURE,
        FALLBACK_TOP_P,
    )

    fallback_result = run_fallback_sampling(
        engine=engine,
        structured_prompt=structured_prompt,
        det_response_text=det_response_text,
        det_token_count=det_token_count,
        validated=validated,
        stop_tokens=stop_tokens,
    )

    final_instability = fallback_result["analysis"]["instability"]
    instability_delta = round(initial_instability - final_instability, 4)
    improved = final_instability < initial_instability

    logger.info(
        "ReliabilityGuard fallback complete - instability %.3f -> %.3f "
        "(delta %.4f, improved=%s).",
        initial_instability,
        final_instability,
        instability_delta,
        improved,
    )

    return {
        "resampled": True,
        "ent_outputs": fallback_result["ent_outputs"],
        "ent_metas": list(fallback_result.get("ent_metas", [])),
        "analysis": fallback_result["analysis"],
        "samples_used": fallback_result["samples_used"],
        "reliability_guard": {
            "triggered": True,
            "initial_instability": round(initial_instability, 4),
            "final_instability": round(final_instability, 4),
            "instability_delta": instability_delta,
            "improved": improved,
            "threshold": INSTABILITY_THRESHOLD,
            "fallback_temperature": FALLBACK_TEMPERATURE,
            "fallback_top_p": FALLBACK_TOP_P,
            "fallback_samples_used": fallback_result["samples_used"],
        },
    }
