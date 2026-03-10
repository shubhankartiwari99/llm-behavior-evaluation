export enum ReleaseStatus {
  STAGING = "staging",
  PRODUCTION = "production",
  ROLLBACK = "rollback",
  RETIRED = "retired",
}

export enum ReleaseVerdict {
  GO = "GO",
  NO_GO = "NO_GO",
}

export interface RegressionSummary {
  fatal_count: number
  warning_count: number
  info_count: number
  triggered_checks: string[]
}

export interface EvaluationEvidence {
  evaluation_run_id: string
  snapshot_id: string
  verdict: ReleaseVerdict
  regression_summary: RegressionSummary
  policy_version: string
  harness_version: string
  dataset_id: string
  snapshot_uri?: string | null
  snapshot_digest?: string | null
  report_uri?: string | null
  baseline_release_id?: string | null
}

export interface DeploymentEntry {
  release_id: string
  model_family: string
  model_id: string
  git_commit_hash: string
  weights_fingerprint: string
  inference_config_fingerprint: string
  evaluation: EvaluationEvidence
  fragility_score: number
  status: ReleaseStatus
  environment: string
  deployed_at: string
  updated_at: string
  deployed_by: string
  approved_by?: string | null
  previous_release_id?: string | null
  rollback_of_release_id?: string | null
  is_adversarially_verified: boolean
  runtime_metadata: Record<string, unknown>
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

function asBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined
}

function parseReleaseStatus(value: unknown): ReleaseStatus | null {
  return Object.values(ReleaseStatus).includes(value as ReleaseStatus)
    ? (value as ReleaseStatus)
    : null
}

function parseReleaseVerdict(value: unknown): ReleaseVerdict | null {
  return Object.values(ReleaseVerdict).includes(value as ReleaseVerdict)
    ? (value as ReleaseVerdict)
    : null
}

function parseRegressionSummary(value: unknown): RegressionSummary | null {
  if (!isRecord(value)) {
    return null
  }

  const fatal = asNumber(value.fatal_count) ?? asNumber(value.fatal) ?? 0
  const warning = asNumber(value.warning_count) ?? asNumber(value.warning) ?? 0
  const info = asNumber(value.info_count) ?? asNumber(value.info) ?? 0
  const triggeredChecks = Array.isArray(value.triggered_checks)
    ? value.triggered_checks.filter((item): item is string => typeof item === "string")
    : []

  return {
    fatal_count: fatal,
    warning_count: warning,
    info_count: info,
    triggered_checks: triggeredChecks,
  }
}

function parseEvaluationEvidence(value: unknown, host: Record<string, unknown>): EvaluationEvidence | null {
  if (isRecord(value)) {
    const verdict = parseReleaseVerdict(value.verdict)
    const regressionSummary = parseRegressionSummary(value.regression_summary)
    const evaluationRunId = asString(value.evaluation_run_id)
    const snapshotId = asString(value.snapshot_id)
    const policyVersion = asString(value.policy_version)
    const harnessVersion = asString(value.harness_version)
    const datasetId = asString(value.dataset_id)

    if (
      !verdict ||
      !regressionSummary ||
      !evaluationRunId ||
      !snapshotId ||
      !policyVersion ||
      !harnessVersion ||
      !datasetId
    ) {
      return null
    }

    return {
      evaluation_run_id: evaluationRunId,
      snapshot_id: snapshotId,
      verdict,
      regression_summary: regressionSummary,
      policy_version: policyVersion,
      harness_version: harnessVersion,
      dataset_id: datasetId,
      snapshot_uri: asString(value.snapshot_uri) ?? null,
      snapshot_digest: asString(value.snapshot_digest) ?? null,
      report_uri: asString(value.report_uri) ?? null,
      baseline_release_id: asString(value.baseline_release_id) ?? null,
    }
  }

  const legacyVerdict = parseReleaseVerdict(host.verdict)
  const legacySnapshotId = asString(host.snapshot_id)
  const legacyRegressionSummary = parseRegressionSummary(host.regression_summary)

  if (!legacyVerdict || !legacySnapshotId || !legacyRegressionSummary) {
    return null
  }

  return {
    evaluation_run_id: asString(host.evaluation_run_id) ?? `legacy-${asString(host.release_id) ?? "release"}`,
    snapshot_id: legacySnapshotId,
    verdict: legacyVerdict,
    regression_summary: legacyRegressionSummary,
    policy_version: asString(host.policy_version) ?? "unknown-policy",
    harness_version: asString(host.harness_version) ?? "unknown-harness",
    dataset_id: asString(host.dataset_id) ?? "unknown-dataset",
    snapshot_uri: asString(host.snapshot_uri) ?? null,
    snapshot_digest: asString(host.snapshot_digest) ?? null,
    report_uri: asString(host.report_uri) ?? null,
    baseline_release_id: asString(host.baseline_release_id) ?? null,
  }
}

function parseDeploymentEntry(value: unknown): DeploymentEntry | null {
  if (!isRecord(value)) {
    return null
  }

  const status = parseReleaseStatus(value.status)
  const evaluation = parseEvaluationEvidence(value.evaluation, value)
  const releaseId = asString(value.release_id)
  const modelId = asString(value.model_id)
  const gitCommitHash = asString(value.git_commit_hash)
  const weightsFingerprint = asString(value.weights_fingerprint)
  const fragilityScore = asNumber(value.fragility_score)
  const deployedAt = asString(value.deployed_at)

  if (
    !status ||
    !evaluation ||
    !releaseId ||
    !modelId ||
    !gitCommitHash ||
    !weightsFingerprint ||
    fragilityScore === undefined ||
    !deployedAt
  ) {
    return null
  }

  return {
    release_id: releaseId,
    model_family: asString(value.model_family) ?? modelId,
    model_id: modelId,
    git_commit_hash: gitCommitHash,
    weights_fingerprint: weightsFingerprint,
    inference_config_fingerprint: asString(value.inference_config_fingerprint) ?? "unknown-config",
    evaluation,
    fragility_score: fragilityScore,
    status,
    environment: asString(value.environment) ?? "unknown-env",
    deployed_at: deployedAt,
    updated_at: asString(value.updated_at) ?? deployedAt,
    deployed_by: asString(value.deployed_by) ?? "unknown",
    approved_by: asString(value.approved_by) ?? null,
    previous_release_id: asString(value.previous_release_id) ?? null,
    rollback_of_release_id: asString(value.rollback_of_release_id) ?? null,
    is_adversarially_verified: asBoolean(value.is_adversarially_verified) ?? false,
    runtime_metadata: isRecord(value.runtime_metadata) ? value.runtime_metadata : {},
  }
}

export function parseDeploymentEntries(payload: unknown): DeploymentEntry[] {
  if (!Array.isArray(payload)) {
    throw new Error("Invalid registry response.")
  }

  return payload
    .map(parseDeploymentEntry)
    .filter((entry): entry is DeploymentEntry => entry !== null)
}
