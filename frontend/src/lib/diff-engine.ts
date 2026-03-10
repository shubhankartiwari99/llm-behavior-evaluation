import { BehavioralSnapshot } from "@/types/snapshot"

export interface MetricDelta {
  baseValue: number
  currentValue: number
  percentChange: number
  isRegression: boolean
}

export interface SnapshotDiff {
  entropyShift: MetricDelta
  safetyShift: MetricDelta
  latencyShift: MetricDelta
  instabilityShift: MetricDelta
}

function getDelta(baseValue: number, currentValue: number, lowerIsBetter = true): MetricDelta {
  const denominator = Math.abs(baseValue) > 1e-9 ? Math.abs(baseValue) : 1
  const percentChange = ((currentValue - baseValue) / denominator) * 100

  return {
    baseValue,
    currentValue,
    percentChange,
    isRegression: lowerIsBetter ? currentValue > baseValue : currentValue < baseValue,
  }
}

export function calculateSnapshotDiff(
  base: BehavioralSnapshot,
  current: BehavioralSnapshot,
): SnapshotDiff {
  return {
    entropyShift: getDelta(
      base.behavioral_invariants.reasoning.mean_entropy,
      current.behavioral_invariants.reasoning.mean_entropy,
    ),
    safetyShift: getDelta(
      base.behavioral_invariants.safety.escalation_rate,
      current.behavioral_invariants.safety.escalation_rate,
    ),
    latencyShift: getDelta(
      base.system_perf.avg_latency_ms,
      current.system_perf.avg_latency_ms,
    ),
    instabilityShift: getDelta(
      base.behavioral_invariants.reasoning.mean_instability,
      current.behavioral_invariants.reasoning.mean_instability,
    ),
  }
}
