"""
Behavioral Snapshot — schema, capture, and (de)serialization.

A BehavioralSnapshot is the "frozen Behavioral DNA" of a model run.
It is produced by SnapshotCapturer.from_benchmark_summary() and consumed
by DeltaCalculator to drive Go/No-Go release decisions.

Schema version: 1.0.0
"""

from __future__ import annotations

import datetime
import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.voice.contract_loader import get_loader
from scripts.artifact_digest import get_deterministic_json, get_sha256_digest

SCHEMA_VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Schema dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SnapshotMetadata:
    schema_version: str
    model_id: str
    timestamp: str
    eval_prompt_count: int
    contract_fingerprint: str = ""
    quantization: str = "none"
    deployment_env: str = "local"


@dataclass
class SafetyInvariants:
    escalation_rate: float        # fraction of prompts that triggered escalation
    guard_trigger_rate: float     # fraction where reliability guard resampled


@dataclass
class ReasoningInvariants:
    mean_entropy: float           # mean Monte Carlo entropy across prompts
    entropy_std_dev: float        # std-dev of entropy — high = "confident hallucinator"
    mean_instability: float       # mean MC instability score
    raw_entropy_trace: list[float] = field(default_factory=list)
    raw_instability_trace: list[float] = field(default_factory=list)


@dataclass
class ReliabilityInvariants:
    mean_confidence: float        # mean deterministic-pass confidence
    confidence_p90: float         # 90th-percentile confidence
    hallucination_proxy_rate: float  # fraction with stochastic_instability or semantic_divergence


@dataclass
class BehavioralInvariants:
    safety: SafetyInvariants
    reasoning: ReasoningInvariants
    reliability: ReliabilityInvariants


@dataclass
class SystemPerf:
    avg_latency_ms: float
    avg_output_tokens: float


@dataclass
class TelemetryDNA:
    raw_entropy_trace: list[float] = field(default_factory=list)
    raw_instability_trace: list[float] = field(default_factory=list)
    raw_confidence_trace: list[float] = field(default_factory=list)


@dataclass
class BehavioralSnapshot:
    snapshot_metadata: SnapshotMetadata
    behavioral_invariants: BehavioralInvariants
    system_perf: SystemPerf
    telemetry_dna: TelemetryDNA = field(default_factory=TelemetryDNA)

    # ── Serialization ──────────────────────────────────────────────────────

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    # ── Deserialization ────────────────────────────────────────────────────

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BehavioralSnapshot":
        m = dict(d["snapshot_metadata"])
        inv = d["behavioral_invariants"]
        reasoning = dict(inv["reasoning"])
        perf = d["system_perf"]
        telemetry = d.get("telemetry_dna") or {}
        if not m.get("contract_fingerprint"):
            m["contract_fingerprint"] = _compute_contract_fingerprint()

        eval_prompt_count = int(m.get("eval_prompt_count", 0) or 0)
        entropy_trace = _coerce_trace(
            reasoning.get("raw_entropy_trace", telemetry.get("raw_entropy_trace")),
            reasoning.get("mean_entropy", 0.0),
            eval_prompt_count,
        )
        instability_trace = _coerce_trace(
            reasoning.get("raw_instability_trace", telemetry.get("raw_instability_trace")),
            reasoning.get("mean_instability", 0.0),
            eval_prompt_count,
        )
        confidence_trace = _coerce_trace(
            telemetry.get("raw_confidence_trace"),
            inv["reliability"].get("mean_confidence", 0.0),
            eval_prompt_count,
        )
        reasoning["raw_entropy_trace"] = entropy_trace
        reasoning["raw_instability_trace"] = instability_trace

        return cls(
            snapshot_metadata=SnapshotMetadata(**m),
            behavioral_invariants=BehavioralInvariants(
                safety=SafetyInvariants(**inv["safety"]),
                reasoning=ReasoningInvariants(**reasoning),
                reliability=ReliabilityInvariants(**inv["reliability"]),
            ),
            system_perf=SystemPerf(**perf),
            telemetry_dna=TelemetryDNA(
                raw_entropy_trace=entropy_trace,
                raw_instability_trace=instability_trace,
                raw_confidence_trace=confidence_trace,
            ),
        )

    @classmethod
    def load(cls, path: str | Path) -> "BehavioralSnapshot":
        text = Path(path).read_text(encoding="utf-8")
        return cls.from_dict(json.loads(text))


# ─────────────────────────────────────────────────────────────────────────────
# Capturer — converts a benchmark summary dict → BehavioralSnapshot
# ─────────────────────────────────────────────────────────────────────────────

class SnapshotCapturer:
    """
    Converts the dict returned by app.eval.benchmark_runner.summarize_benchmark()
    into a fully-typed BehavioralSnapshot.

    The capturer reads the distribution arrays already present in summary["distributions"]
    to compute any derived stats (p90, std-dev) that are not pre-aggregated.
    """

    @staticmethod
    def from_benchmark_summary(
        summary: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> BehavioralSnapshot:
        """
        Args:
            summary:  Dict from summarize_benchmark().
            metadata: Optional overrides for snapshot_metadata fields.
                      Defaults: schema_version=SCHEMA_VERSION, model_id from summary["model"],
                                timestamp=now, eval_prompt_count=summary["total"].
        """
        meta_defaults: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "model_id": summary.get("model", "unknown"),
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            "eval_prompt_count": summary.get("total", 0),
            "contract_fingerprint": _compute_contract_fingerprint(),
            "quantization": "none",
            "deployment_env": "local",
        }
        if metadata:
            meta_defaults.update(metadata)

        snap_meta = SnapshotMetadata(**meta_defaults)

        # ── Distribution-derived stats ─────────────────────────────────────
        distributions = summary.get("distributions", {})
        entropy_dist: list[float] = distributions.get(
            "entropy", summary.get("_entropy_raw", [])
        )
        instability_dist: list[float] = distributions.get("instability", [])
        confidence_dist: list[float] = distributions.get("confidence", [])
        eval_prompt_count = int(summary.get("total", 0) or 0)

        entropy_std_dev = (
            statistics.stdev(entropy_dist) if len(entropy_dist) >= 2 else 0.0
        )
        confidence_p90 = (
            _percentile(confidence_dist, 90) if confidence_dist else 0.0
        )

        # ── Hallucination proxy ────────────────────────────────────────────
        hallucination_proxy_rate = summary.get("hallucination_proxy_rate", 0.0)

        return BehavioralSnapshot(
            snapshot_metadata=snap_meta,
            behavioral_invariants=BehavioralInvariants(
                safety=SafetyInvariants(
                    escalation_rate=summary.get("escalation_rate", 0.0),
                    guard_trigger_rate=summary.get("guard_trigger_rate", 0.0),
                ),
                reasoning=ReasoningInvariants(
                    mean_entropy=summary.get("mean_entropy", 0.0),
                    entropy_std_dev=entropy_std_dev,
                    mean_instability=summary.get("mean_instability", 0.0),
                    raw_entropy_trace=_coerce_trace(
                        entropy_dist,
                        summary.get("mean_entropy", 0.0),
                        eval_prompt_count,
                    ),
                    raw_instability_trace=_coerce_trace(
                        instability_dist,
                        summary.get("mean_instability", 0.0),
                        eval_prompt_count,
                    ),
                ),
                reliability=ReliabilityInvariants(
                    mean_confidence=summary.get("mean_confidence", 0.0),
                    confidence_p90=confidence_p90,
                    hallucination_proxy_rate=hallucination_proxy_rate,
                ),
            ),
            system_perf=SystemPerf(
                avg_latency_ms=summary.get("avg_latency", 0.0),
                avg_output_tokens=summary.get("avg_output_tokens", 0.0),
            ),
            telemetry_dna=TelemetryDNA(
                raw_entropy_trace=_coerce_trace(
                    entropy_dist,
                    summary.get("mean_entropy", 0.0),
                    eval_prompt_count,
                ),
                raw_instability_trace=_coerce_trace(
                    instability_dist,
                    summary.get("mean_instability", 0.0),
                    eval_prompt_count,
                ),
                raw_confidence_trace=_coerce_trace(
                    confidence_dist,
                    summary.get("mean_confidence", 0.0),
                    eval_prompt_count,
                ),
            ),
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _percentile(data: list[float], pct: float) -> float:
    """Nearest-rank percentile (no external deps)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = int(len(sorted_data) * pct / 100)
    k = max(0, min(k, len(sorted_data) - 1))
    return sorted_data[k]


def _coerce_trace(
    values: Any,
    fallback_value: float,
    fallback_count: int,
) -> list[float]:
    if isinstance(values, list) and values:
        return [float(value) for value in values]
    if fallback_count <= 0:
        return []
    return [float(fallback_value)] * int(fallback_count)


def _compute_contract_fingerprint() -> str:
    contract_data = get_loader()
    canonical_contract = get_deterministic_json(contract_data)
    return get_sha256_digest(canonical_contract)
