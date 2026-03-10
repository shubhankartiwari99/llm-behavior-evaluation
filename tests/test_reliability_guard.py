"""
Unit tests for the grid-sweep-informed reliability fallback layer.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.intelligence.reliability_guard import (
    FALLBACK_TEMPERATURE,
    FALLBACK_TOP_P,
    INSTABILITY_THRESHOLD,
    apply_reliability_guard,
    needs_fallback,
)


def _make_analysis(instability: float) -> dict:
    return {
        "instability": instability,
        "confidence": round(1.0 - instability, 3),
        "entropy": 0.3,
        "uncertainty": instability * 0.8,
        "escalate": instability > 0.4,
        "sample_count": 4,
        "semantic_dispersion": 0.1,
        "cluster_count": 1,
        "cluster_entropy": 0.2,
        "dominant_cluster_ratio": 0.8,
        "self_consistency": 0.75,
        "det_entropy_similarity": 0.85,
        "entropy_consistency": 0.80,
        "entropy_variance": 0.05,
        "pairwise_disagreement_entropy": 0.15,
        "uncertainty_level": "low",
        "ambiguity": 0.1,
        "length_delta": 10.0,
        "token_delta": 5.0,
        "ambiguities": [0.1, 0.1, 0.1, 0.1],
        "cluster_labels": [0, 0, 0, 0],
    }


def _make_engine(response: str = "test response") -> MagicMock:
    engine = MagicMock()
    engine.generate.return_value = (response, {"output_tokens": 10})
    return engine


def _make_validated() -> dict:
    return {
        "prompt": "test prompt",
        "temperature": 0.7,
        "top_p": 0.95,
        "max_new_tokens": 128,
        "do_sample": True,
        "monte_carlo_samples": 6,
        "emotional_lang": "en",
        "mode": "factual",
    }


def test_needs_fallback_below_threshold():
    assert needs_fallback(0.10) is False
    assert needs_fallback(0.24) is False


def test_needs_fallback_at_threshold():
    assert needs_fallback(INSTABILITY_THRESHOLD) is False


def test_needs_fallback_above_threshold():
    assert needs_fallback(0.26) is True
    assert needs_fallback(0.50) is True
    assert needs_fallback(1.0) is True


def test_guard_passthrough_when_stable():
    engine = _make_engine()
    initial_analysis = _make_analysis(0.15)
    initial_outputs = ["resp_a", "resp_b", "resp_c", "resp_d"]

    result = apply_reliability_guard(
        engine=engine,
        structured_prompt="test",
        det_response_text="det",
        det_token_count=10,
        validated=_make_validated(),
        stop_tokens=[],
        initial_analysis=initial_analysis,
        initial_ent_outputs=initial_outputs,
    )

    assert result["resampled"] is False
    assert result["ent_outputs"] is initial_outputs
    assert result["analysis"] is initial_analysis
    assert result["reliability_guard"]["triggered"] is False
    assert result["reliability_guard"]["threshold"] == INSTABILITY_THRESHOLD

    engine.generate.assert_not_called()


def test_guard_triggers_on_high_instability():
    engine = _make_engine("fallback_response")
    initial_analysis = _make_analysis(0.30)
    initial_outputs = ["resp_a", "resp_b", "resp_c", "resp_d"]

    fallback_analysis = _make_analysis(0.10)

    with patch(
        "app.intelligence.reliability_guard.run_fallback_sampling",
        return_value={
            "ent_outputs": ["fb_a", "fb_b", "fb_c", "fb_d"],
            "analysis": fallback_analysis,
            "samples_used": 4,
        },
    ) as mock_fallback:
        result = apply_reliability_guard(
            engine=engine,
            structured_prompt="test",
            det_response_text="det",
            det_token_count=10,
            validated=_make_validated(),
            stop_tokens=[],
            initial_analysis=initial_analysis,
            initial_ent_outputs=initial_outputs,
        )

    assert result["resampled"] is True
    assert result["ent_outputs"] == ["fb_a", "fb_b", "fb_c", "fb_d"]
    assert result["analysis"] is fallback_analysis
    assert result["samples_used"] == 4

    guard = result["reliability_guard"]
    assert guard["triggered"] is True
    assert guard["initial_instability"] == round(0.30, 4)
    assert guard["final_instability"] == round(0.10, 4)
    assert guard["instability_delta"] == round(0.30 - 0.10, 4)
    assert guard["improved"] is True
    assert guard["fallback_temperature"] == FALLBACK_TEMPERATURE
    assert guard["fallback_top_p"] == FALLBACK_TOP_P

    mock_fallback.assert_called_once()


def test_guard_reports_not_improved_when_fallback_worse():
    engine = _make_engine()
    initial_analysis = _make_analysis(0.30)
    initial_outputs = ["a", "b", "c", "d"]
    fallback_analysis = _make_analysis(0.35)

    with patch(
        "app.intelligence.reliability_guard.run_fallback_sampling",
        return_value={
            "ent_outputs": ["x", "y", "z", "w"],
            "analysis": fallback_analysis,
            "samples_used": 4,
        },
    ):
        result = apply_reliability_guard(
            engine=engine,
            structured_prompt="test",
            det_response_text="det",
            det_token_count=10,
            validated=_make_validated(),
            stop_tokens=[],
            initial_analysis=initial_analysis,
            initial_ent_outputs=initial_outputs,
        )

    assert result["reliability_guard"]["improved"] is False
    assert result["reliability_guard"]["instability_delta"] < 0


def test_fallback_sampling_uses_optimal_params():
    engine = _make_engine()
    validated = _make_validated()
    validated["monte_carlo_samples"] = 4

    fallback_analysis = _make_analysis(0.05)

    with patch(
        "app.intelligence.reliability_guard.evaluate_dual_plane",
        return_value=fallback_analysis,
    ):
        from app.intelligence.reliability_guard import run_fallback_sampling

        run_fallback_sampling(
            engine=engine,
            structured_prompt="test",
            det_response_text="det",
            det_token_count=10,
            validated=validated,
            stop_tokens=[],
        )

    for call in engine.generate.call_args_list:
        kwargs = call.kwargs if call.kwargs else call[1]
        assert kwargs.get("temperature") == FALLBACK_TEMPERATURE
        assert kwargs.get("top_p") == FALLBACK_TOP_P
