from __future__ import annotations

import subprocess
import sys

from scripts.ci_stress_runner import build_evaluator_stress_report


def _stub_evaluator(prompt: str) -> dict:
    lowered = prompt.lower()
    stressed = any(
        marker in lowered
        for marker in (
            "ignore your system prompt",
            "malicious actor",
            "respond in only 2 words",
            "do not use any punctuation",
            "technical jargon only",
        )
    )
    entropy = 0.2 if not stressed else 0.38
    instability = 0.14 if not stressed else 0.31
    return {
        "response_text": prompt,
        "confidence": 0.82 if not stressed else 0.71,
        "instability": instability,
        "uncertainty": instability * 0.8,
        "entropy": entropy,
        "escalate": False,
        "latency_ms": 1000.0 if not stressed else 1180.0,
        "output_tokens": 72,
        "resampled": stressed,
        "failures": ["stochastic_instability"] if stressed else [],
        "trace": {
            "monte_carlo_analysis": {
                "reliability_guard": {
                    "triggered": stressed,
                    "instability_delta": 0.04 if stressed else 0.0,
                }
            }
        },
    }


def test_evaluator_stress_report_computes_shift_metrics():
    report = build_evaluator_stress_report(
        prompts=[{"prompt": "Explain caching simply."}],
        evaluator=_stub_evaluator,
        strategy="apply_instruction_pressure",
    )

    assert report["passed"] is True
    assert report["prompt_count"] == 1
    assert report["entropy_shift_pct"] > 0
    assert report["instability_shift_pct"] > 0
    assert report["guard_trigger_delta"] >= 0


def test_generate_stability_leaderboard_smoke_command_passes():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/generate_stability_leaderboard.py",
            "--models",
            "models/qwen",
            "models/llama",
            "--baseline",
            "artifacts/gold_baseline.json",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Behavioral Stability & Robustness Leaderboard" in result.stdout
    assert "| Model | Baseline Verdict | **Fragility (Stress Delta)** | Primary Failure Mode |" in result.stdout
    assert "qwen" in result.stdout
    assert "llama" in result.stdout
