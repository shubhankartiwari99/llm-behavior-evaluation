export interface SnapshotMetadata {
  schema_version: string
  model_id: string
  timestamp: string
  eval_prompt_count: number
  contract_fingerprint?: string
  quantization?: string
  deployment_env?: string
}

export interface SafetyInvariants {
  escalation_rate: number
  guard_trigger_rate: number
}

export interface ReasoningInvariants {
  mean_entropy: number
  entropy_std_dev: number
  mean_instability: number
}

export interface ReliabilityInvariants {
  mean_confidence: number
  confidence_p90: number
  hallucination_proxy_rate: number
}

export interface BehavioralInvariants {
  safety: SafetyInvariants
  reasoning: ReasoningInvariants
  reliability: ReliabilityInvariants
}

export interface SystemPerf {
  avg_latency_ms: number
  avg_output_tokens: number
}

export interface BehavioralSnapshot {
  snapshot_metadata: SnapshotMetadata
  behavioral_invariants: BehavioralInvariants
  system_perf: SystemPerf
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

function asString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined
}

function asNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined
}

export function parseBehavioralSnapshot(payload: unknown): BehavioralSnapshot {
  if (!isRecord(payload)) {
    throw new Error("Invalid snapshot payload.")
  }

  const metadata = payload.snapshot_metadata
  const invariants = payload.behavioral_invariants
  const perf = payload.system_perf

  if (!isRecord(metadata) || !isRecord(invariants) || !isRecord(perf)) {
    throw new Error("Incomplete snapshot payload.")
  }

  const safety = invariants.safety
  const reasoning = invariants.reasoning
  const reliability = invariants.reliability

  if (!isRecord(safety) || !isRecord(reasoning) || !isRecord(reliability)) {
    throw new Error("Incomplete behavioral invariants.")
  }

  const snapshot: BehavioralSnapshot = {
    snapshot_metadata: {
      schema_version: asString(metadata.schema_version) ?? "unknown",
      model_id: asString(metadata.model_id) ?? "unknown-model",
      timestamp: asString(metadata.timestamp) ?? "",
      eval_prompt_count: asNumber(metadata.eval_prompt_count) ?? 0,
      contract_fingerprint: asString(metadata.contract_fingerprint),
      quantization: asString(metadata.quantization),
      deployment_env: asString(metadata.deployment_env),
    },
    behavioral_invariants: {
      safety: {
        escalation_rate: asNumber(safety.escalation_rate) ?? 0,
        guard_trigger_rate: asNumber(safety.guard_trigger_rate) ?? 0,
      },
      reasoning: {
        mean_entropy: asNumber(reasoning.mean_entropy) ?? 0,
        entropy_std_dev: asNumber(reasoning.entropy_std_dev) ?? 0,
        mean_instability: asNumber(reasoning.mean_instability) ?? 0,
      },
      reliability: {
        mean_confidence: asNumber(reliability.mean_confidence) ?? 0,
        confidence_p90: asNumber(reliability.confidence_p90) ?? 0,
        hallucination_proxy_rate: asNumber(reliability.hallucination_proxy_rate) ?? 0,
      },
    },
    system_perf: {
      avg_latency_ms: asNumber(perf.avg_latency_ms) ?? 0,
      avg_output_tokens: asNumber(perf.avg_output_tokens) ?? 0,
    },
  }

  return snapshot
}
