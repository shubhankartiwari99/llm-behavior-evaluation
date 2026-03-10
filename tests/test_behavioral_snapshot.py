"""
Unit tests for the Behavioral Snapshot & DeltaCalculator system.

All tests are pure-Python — no inference engine, no HTTP server.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from app.eval.behavioral_snapshot import (
    BehavioralSnapshot,
    SnapshotCapturer,
)
from app.eval.delta_calculator import DeltaCalculator, RegressionReport


# ─────────────────────────────────────────────────────────────────────────────
# Test fixtures
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_DICT = {
    "snapshot_metadata": {
        "schema_version": "1.0.0",
        "model_id": "test-model-baseline",
        "timestamp": "2026-01-01T00:00:00Z",
        "eval_prompt_count": 50,
        "quantization": "none",
        "deployment_env": "test",
    },
    "behavioral_invariants": {
        "safety": {
            "escalation_rate": 0.04,
            "guard_trigger_rate": 0.06,
        },
        "reasoning": {
            "mean_entropy": 0.28,
            "entropy_std_dev": 0.10,
            "mean_instability": 0.18,
        },
        "reliability": {
            "mean_confidence": 0.82,
            "confidence_p90": 0.91,
            "hallucination_proxy_rate": 0.06,
        },
    },
    "system_perf": {
        "avg_latency_ms": 1850.0,
        "avg_output_tokens": 112.0,
    },
}


def _baseline_file() -> Path:
    """Write baseline to a temp file and return its path."""
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(BASELINE_DICT, tmp)
    tmp.flush()
    return Path(tmp.name)


def _snapshot_from_dict(d: dict) -> BehavioralSnapshot:
    return BehavioralSnapshot.from_dict(d)


def _clean_current() -> BehavioralSnapshot:
    """A current snapshot within all thresholds — should produce GO."""
    d = json.loads(json.dumps(BASELINE_DICT))
    d["snapshot_metadata"]["model_id"] = "test-model-current"
    return _snapshot_from_dict(d)


def _make_summary(**overrides) -> dict:
    """Minimal benchmark summary dict, optionally overridden."""
    base = {
        "total": 10,
        "model": "test-model",
        "mean_confidence": 0.82,
        "mean_instability": 0.18,
        "mean_uncertainty": 0.12,
        "mean_entropy": 0.28,
        "escalation_rate": 0.04,
        "guard_trigger_rate": 0.06,
        "mean_guard_instability_delta": 0.01,
        "avg_latency": 1850.0,
        "avg_output_tokens": 112.0,
        "hallucination_proxy_rate": 0.06,
        "distributions": {
            "confidence": [0.75, 0.80, 0.82, 0.85, 0.90, 0.91, 0.92, 0.88, 0.79, 0.83],
            "instability": [0.15, 0.18, 0.20, 0.17, 0.19, 0.16, 0.18, 0.20, 0.15, 0.22],
            "entropy": [0.25, 0.28, 0.30, 0.27, 0.29, 0.26, 0.28, 0.30, 0.25, 0.32],
        },
        "timestamp": "2026-03-10T12:00:00",
    }
    base.update(overrides)
    return base


# ─────────────────────────────────────────────────────────────────────────────
# SnapshotCapturer tests
# ─────────────────────────────────────────────────────────────────────────────

class TestSnapshotCapturer:
    def test_from_summary_has_all_required_keys(self):
        summary = _make_summary()
        snap = SnapshotCapturer.from_benchmark_summary(summary)
        d = snap.to_dict()
        assert "snapshot_metadata" in d
        assert "behavioral_invariants" in d
        assert "system_perf" in d
        assert "safety" in d["behavioral_invariants"]
        assert "reasoning" in d["behavioral_invariants"]
        assert "reliability" in d["behavioral_invariants"]

    def test_from_summary_passthrough_scalars(self):
        summary = _make_summary(mean_entropy=0.55, mean_instability=0.33)
        snap = SnapshotCapturer.from_benchmark_summary(summary)
        assert snap.behavioral_invariants.reasoning.mean_entropy == pytest.approx(0.55)
        assert snap.behavioral_invariants.reasoning.mean_instability == pytest.approx(0.33)

    def test_from_summary_computes_entropy_std_dev(self):
        entropy_vals = [0.10, 0.50]   # stdev = 0.2828...
        summary = _make_summary()
        summary["distributions"]["entropy"] = entropy_vals
        snap = SnapshotCapturer.from_benchmark_summary(summary)
        assert snap.behavioral_invariants.reasoning.entropy_std_dev == pytest.approx(
            0.2828, abs=0.001
        )

    def test_from_summary_computes_confidence_p90(self):
        # 10 values sorted: p90 = index 9 = max
        conf_vals = list(range(1, 11))  # 1‥10 scaled
        conf_floats = [v / 10.0 for v in conf_vals]     # 0.1 … 1.0
        summary = _make_summary()
        summary["distributions"]["confidence"] = conf_floats
        snap = SnapshotCapturer.from_benchmark_summary(summary)
        # nearest-rank p90 of 10 items → index 9 → 1.0
        assert snap.behavioral_invariants.reliability.confidence_p90 == pytest.approx(1.0)

    def test_from_summary_metadata_override(self):
        summary = _make_summary()
        snap = SnapshotCapturer.from_benchmark_summary(
            summary, metadata={"model_id": "custom-model", "deployment_env": "staging"}
        )
        assert snap.snapshot_metadata.model_id == "custom-model"
        assert snap.snapshot_metadata.deployment_env == "staging"

    def test_roundtrip_json(self):
        summary = _make_summary()
        snap = SnapshotCapturer.from_benchmark_summary(summary)
        restored = BehavioralSnapshot.from_dict(json.loads(snap.to_json()))
        assert restored.snapshot_metadata.model_id == snap.snapshot_metadata.model_id
        assert restored.behavioral_invariants.reasoning.mean_entropy == pytest.approx(
            snap.behavioral_invariants.reasoning.mean_entropy
        )


# ─────────────────────────────────────────────────────────────────────────────
# DeltaCalculator — GO verdict
# ─────────────────────────────────────────────────────────────────────────────

class TestDeltaCalculatorGoVerdict:
    def test_clean_run_produces_go(self):
        baseline_path = _baseline_file()
        calc = DeltaCalculator(baseline_path)
        report = calc.compute_drift(_clean_current())
        assert report.verdict == "GO"
        assert len([r for r in report.regressions if r.tier == "FATAL"]) == 0

    def test_go_allows_small_latency_increase(self):
        baseline_path = _baseline_file()
        calc = DeltaCalculator(baseline_path)
        current = _clean_current()
        # 10% latency increase — under 25% threshold
        current.system_perf.avg_latency_ms = 1850.0 * 1.10
        report = calc.compute_drift(current)
        assert report.verdict == "GO"


# ─────────────────────────────────────────────────────────────────────────────
# DeltaCalculator — FATAL regressions
# ─────────────────────────────────────────────────────────────────────────────

class TestDeltaCalculatorFatal:
    def test_escalation_rate_fatal_breach(self):
        baseline_path = _baseline_file()
        calc = DeltaCalculator(baseline_path)
        current = _clean_current()
        # Baseline 0.04, push it 0.06 above threshold (0.05) → 0.11
        current.behavioral_invariants.safety.escalation_rate = 0.04 + 0.06
        report = calc.compute_drift(current)
        assert report.verdict == "NO_GO"
        assert any(r.metric == "safety.escalation_rate" and r.tier == "FATAL"
                   for r in report.regressions)

    def test_guard_trigger_rate_fatal_breach(self):
        baseline_path = _baseline_file()
        calc = DeltaCalculator(baseline_path)
        current = _clean_current()
        # Baseline 0.06, push it 0.09 above threshold (0.08) → 0.15
        current.behavioral_invariants.safety.guard_trigger_rate = 0.06 + 0.09
        report = calc.compute_drift(current)
        assert report.verdict == "NO_GO"
        assert any(r.metric == "safety.guard_trigger_rate" and r.tier == "FATAL"
                   for r in report.regressions)

    def test_multiple_fatals_all_appear_in_report(self):
        baseline_path = _baseline_file()
        calc = DeltaCalculator(baseline_path)
        current = _clean_current()
        current.behavioral_invariants.safety.escalation_rate = 0.04 + 0.10
        current.behavioral_invariants.safety.guard_trigger_rate = 0.06 + 0.15
        report = calc.compute_drift(current)
        assert report.verdict == "NO_GO"
        fatals = [r for r in report.regressions if r.tier == "FATAL"]
        assert len(fatals) == 2


# ─────────────────────────────────────────────────────────────────────────────
# DeltaCalculator — WARNING regressions
# ─────────────────────────────────────────────────────────────────────────────

class TestDeltaCalculatorWarning:
    def test_entropy_std_dev_warning(self):
        """The 'hallucinating with confidence' detector."""
        baseline_path = _baseline_file()
        calc = DeltaCalculator(baseline_path)
        current = _clean_current()
        # Baseline std-dev = 0.10; 31% jump → 0.131
        current.behavioral_invariants.reasoning.entropy_std_dev = 0.10 * 1.31
        report = calc.compute_drift(current)
        assert any(r.metric == "reasoning.entropy_std_dev" and r.tier == "WARNING"
                   for r in report.regressions)
        # Should still be GO unless there's a FATAL
        assert report.verdict == "GO"

    def test_mean_entropy_warning(self):
        baseline_path = _baseline_file()
        calc = DeltaCalculator(baseline_path)
        current = _clean_current()
        # 16% relative increase over 0.28 → 0.325
        current.behavioral_invariants.reasoning.mean_entropy = 0.28 * 1.16
        report = calc.compute_drift(current)
        assert any(r.metric == "reasoning.mean_entropy" and r.tier == "WARNING"
                   for r in report.regressions)

    def test_mean_instability_warning(self):
        baseline_path = _baseline_file()
        calc = DeltaCalculator(baseline_path)
        current = _clean_current()
        # 21% jump over 0.18 → 0.218
        current.behavioral_invariants.reasoning.mean_instability = 0.18 * 1.21
        report = calc.compute_drift(current)
        assert any(r.metric == "reasoning.mean_instability" and r.tier == "WARNING"
                   for r in report.regressions)


# ─────────────────────────────────────────────────────────────────────────────
# DeltaCalculator — INFO regressions
# ─────────────────────────────────────────────────────────────────────────────

class TestDeltaCalculatorInfo:
    def test_latency_info_fires_on_25pct_increase(self):
        baseline_path = _baseline_file()
        calc = DeltaCalculator(baseline_path)
        current = _clean_current()
        current.system_perf.avg_latency_ms = 1850.0 * 1.30  # 30% up
        report = calc.compute_drift(current)
        assert any(r.metric == "system_perf.avg_latency_ms" and r.tier == "INFO"
                   for r in report.regressions)
        # INFO alone → GO
        assert report.verdict == "GO"

    def test_confidence_drop_info_fires(self):
        baseline_path = _baseline_file()
        calc = DeltaCalculator(baseline_path)
        current = _clean_current()
        current.behavioral_invariants.reliability.mean_confidence = 0.82 - 0.06
        report = calc.compute_drift(current)
        assert any(r.metric == "reliability.mean_confidence" and r.tier == "INFO"
                   for r in report.regressions)


# ─────────────────────────────────────────────────────────────────────────────
# RegressionReport.render()
# ─────────────────────────────────────────────────────────────────────────────

class TestRegressionReportRender:
    def test_render_contains_verdict_line(self):
        baseline_path = _baseline_file()
        calc = DeltaCalculator(baseline_path)
        report = calc.compute_drift(_clean_current())
        rendered = report.render()
        assert "GO" in rendered or "NO_GO" in rendered

    def test_render_no_go_contains_metric_name(self):
        baseline_path = _baseline_file()
        calc = DeltaCalculator(baseline_path)
        current = _clean_current()
        current.behavioral_invariants.safety.escalation_rate = 0.25
        report = calc.compute_drift(current)
        rendered = report.render()
        assert "escalation_rate" in rendered
        assert "FATAL" in rendered

    def test_render_clean_run_no_regression_text(self):
        baseline_path = _baseline_file()
        calc = DeltaCalculator(baseline_path)
        report = calc.compute_drift(_clean_current())
        rendered = report.render()
        assert "within tolerance" in rendered

    def test_to_json_is_valid_json_with_verdict(self):
        baseline_path = _baseline_file()
        calc = DeltaCalculator(baseline_path)
        report = calc.compute_drift(_clean_current())
        parsed = json.loads(report.to_json())
        assert "verdict" in parsed
        assert "regressions" in parsed
        assert "regression_count" in parsed
