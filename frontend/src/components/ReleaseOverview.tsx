import { useEffect, useMemo, useState } from "react"

import SnapshotDiffView from "@/components/SnapshotDiffView"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent } from "@/components/ui/card"
import { DeploymentEntry, ReleaseStatus } from "@/types/release"
import { BehavioralSnapshot, parseBehavioralSnapshot } from "@/types/snapshot"

function getVerdictClasses(verdict: DeploymentEntry["evaluation"]["verdict"]): string {
  return verdict === "GO"
    ? "bg-emerald-500/15 text-emerald-300 border border-emerald-500/30"
    : "bg-red-500/15 text-red-300 border border-red-500/30"
}

function getStatusClasses(status: ReleaseStatus): string {
  if (status === ReleaseStatus.PRODUCTION) return "bg-emerald-500/10 text-emerald-300 border border-emerald-500/20"
  if (status === ReleaseStatus.ROLLBACK) return "bg-amber-500/10 text-amber-300 border border-amber-500/20"
  if (status === ReleaseStatus.RETIRED) return "bg-slate-700/40 text-slate-300 border border-slate-600/50"
  return "bg-sky-500/10 text-sky-300 border border-sky-500/20"
}

function getFragilityClasses(score: number): string {
  if (score >= 20) return "text-orange-400"
  if (score >= 10) return "text-amber-300"
  return "text-emerald-400"
}

function shortHash(value: string, size = 8): string {
  return value.slice(0, size)
}

export default function ReleaseOverview({
  deployments,
  loading = false,
  error = null,
  apiBase,
}: {
  deployments: DeploymentEntry[]
  loading?: boolean
  error?: string | null
  apiBase?: string
}) {
  const comparableDeployments = useMemo(
    () => deployments.filter((deployment) => Boolean(deployment.evaluation.snapshot_uri)),
    [deployments],
  )
  const [baseReleaseId, setBaseReleaseId] = useState<string>("")
  const [currentReleaseId, setCurrentReleaseId] = useState<string>("")
  const [baseSnapshot, setBaseSnapshot] = useState<BehavioralSnapshot | null>(null)
  const [currentSnapshot, setCurrentSnapshot] = useState<BehavioralSnapshot | null>(null)
  const [snapshotLoading, setSnapshotLoading] = useState(false)
  const [snapshotError, setSnapshotError] = useState<string | null>(null)

  useEffect(() => {
    if (comparableDeployments.length >= 2) {
      setCurrentReleaseId((existing) => existing || comparableDeployments[0].release_id)
      setBaseReleaseId((existing) => existing || comparableDeployments[1].release_id)
      return
    }
    setCurrentReleaseId("")
    setBaseReleaseId("")
  }, [comparableDeployments])

  useEffect(() => {
    const baseRelease = comparableDeployments.find((deployment) => deployment.release_id === baseReleaseId)
    const currentRelease = comparableDeployments.find((deployment) => deployment.release_id === currentReleaseId)

    if (!baseRelease || !currentRelease || !baseRelease.evaluation.snapshot_uri || !currentRelease.evaluation.snapshot_uri) {
      setBaseSnapshot(null)
      setCurrentSnapshot(null)
      setSnapshotError(null)
      return
    }

    const resolveSnapshotUrl = (uri: string): string => {
      if (/^https?:\/\//i.test(uri)) {
        return uri
      }

      if (!apiBase) {
        return `/${uri.replace(/^\/+/, "")}`
      }

      return new URL(uri.replace(/^\/+/, ""), apiBase.endsWith("/") ? apiBase : `${apiBase}/`).toString()
    }

    const fetchSnapshots = async () => {
      setSnapshotLoading(true)
      setSnapshotError(null)

      try {
        const [baseResponse, currentResponse] = await Promise.all([
          fetch(resolveSnapshotUrl(baseRelease.evaluation.snapshot_uri!), {
            cache: "no-store",
            headers: { "ngrok-skip-browser-warning": "true" },
          }),
          fetch(resolveSnapshotUrl(currentRelease.evaluation.snapshot_uri!), {
            cache: "no-store",
            headers: { "ngrok-skip-browser-warning": "true" },
          }),
        ])

        if (!baseResponse.ok || !currentResponse.ok) {
          throw new Error("Snapshot artifacts are not reachable from the dashboard.")
        }

        const [basePayload, currentPayload] = await Promise.all([
          baseResponse.json(),
          currentResponse.json(),
        ])

        setBaseSnapshot(parseBehavioralSnapshot(basePayload))
        setCurrentSnapshot(parseBehavioralSnapshot(currentPayload))
      } catch (err) {
        setBaseSnapshot(null)
        setCurrentSnapshot(null)
        setSnapshotError(err instanceof Error ? err.message : "Failed to load snapshot artifacts.")
      } finally {
        setSnapshotLoading(false)
      }
    }

    void fetchSnapshots()
  }, [apiBase, baseReleaseId, comparableDeployments, currentReleaseId])

  if (loading) {
    return (
      <div className="rounded-2xl border border-slate-800 bg-slate-950 p-8 text-center text-sm text-slate-500">
        Loading release registry...
      </div>
    )
  }

  if (error) {
    return (
      <div className="rounded-2xl border border-red-500/20 bg-red-500/5 p-8 text-center text-sm text-red-300">
        {error}
      </div>
    )
  }

  if (deployments.length === 0) {
    return (
      <div className="rounded-2xl border border-slate-800 bg-slate-950 p-8 text-center text-sm text-slate-500">
        No deployment history available yet.
      </div>
    )
  }

  return (
    <div className="space-y-6 rounded-3xl border border-slate-800 bg-slate-950/90 p-6 text-slate-50">
      <header className="flex flex-col gap-4 border-b border-slate-800 pb-5 md:flex-row md:items-center md:justify-between">
        <div>
          <h2 className="text-2xl font-black tracking-tight text-slate-100">Model Release Governance</h2>
          <p className="mt-1 text-sm text-slate-400">
            Behavioral provenance for production sign-off, rollback, and audit.
          </p>
        </div>
        <Badge variant="outline" className="border-emerald-500/30 bg-emerald-500/10 text-emerald-300">
          Evaluator System Active
        </Badge>
      </header>

      <div className="grid grid-cols-1 gap-4">
        {deployments.map((release) => {
          const regressions = release.evaluation.regression_summary
          return (
            <Card key={release.release_id} className="border-slate-800 bg-slate-900/70 py-0">
              <CardContent className="p-6">
                <div className="flex flex-col gap-6 lg:flex-row lg:items-start lg:justify-between">
                  <div className="space-y-3">
                    <div className="flex flex-wrap items-center gap-3">
                      <h3 className="text-xl font-black text-sky-300">{release.model_id}</h3>
                      <Badge className={getVerdictClasses(release.evaluation.verdict)}>
                        {release.evaluation.verdict} VERDICT
                      </Badge>
                      <Badge className={getStatusClasses(release.status)}>
                        {release.status}
                      </Badge>
                    </div>
                    <div className="space-y-1 text-sm text-slate-400">
                      <p className="font-mono">
                        Release: {release.release_id} · Commit: {shortHash(release.git_commit_hash, 7)}
                      </p>
                      <p className="font-mono">
                        Weights: {shortHash(release.weights_fingerprint, 15)} · Snapshot: {release.evaluation.snapshot_id}
                      </p>
                      <p>
                        Env: {release.environment} · Approved by: {release.approved_by ?? "pending"}
                      </p>
                    </div>
                  </div>

                  <div className="rounded-2xl border border-slate-800 bg-slate-950/80 px-5 py-4 text-right">
                    <div className="text-[11px] font-black uppercase tracking-[0.28em] text-slate-500">
                      Fragility (Stress Delta)
                    </div>
                    <div className={`mt-2 text-3xl font-black ${getFragilityClasses(release.fragility_score)}`}>
                      {release.fragility_score.toFixed(1)}%
                    </div>
                    <div className="mt-2 text-[11px] uppercase tracking-widest text-slate-500">
                      {release.is_adversarially_verified ? "Adversarially verified" : "Adversarial verification pending"}
                    </div>
                  </div>
                </div>

                <div className="mt-6 grid grid-cols-1 gap-4 border-t border-slate-800 pt-4 md:grid-cols-5">
                  <div>
                    <span className="block text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">
                      Regressions
                    </span>
                    <div className="mt-2 flex gap-3 font-mono text-sm font-bold">
                      <span className="text-red-400">{regressions.fatal_count}F</span>
                      <span className="text-amber-300">{regressions.warning_count}W</span>
                      <span className="text-slate-400">{regressions.info_count}I</span>
                    </div>
                  </div>
                  <div>
                    <span className="block text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">
                      Dataset
                    </span>
                    <span className="mt-2 block text-sm text-slate-300">{release.evaluation.dataset_id}</span>
                  </div>
                  <div>
                    <span className="block text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">
                      Policy
                    </span>
                    <span className="mt-2 block text-sm text-slate-300">{release.evaluation.policy_version}</span>
                  </div>
                  <div>
                    <span className="block text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">
                      Deployed At
                    </span>
                    <span className="mt-2 block text-sm text-slate-300">
                      {new Date(release.deployed_at).toLocaleString()}
                    </span>
                  </div>
                  <div className="text-left md:text-right">
                    <span className="block text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">
                      Decision Trace
                    </span>
                    <span className="mt-2 inline-flex rounded-full border border-sky-500/20 bg-sky-500/10 px-3 py-1 text-xs font-bold text-sky-300">
                      {release.evaluation.evaluation_run_id}
                    </span>
                  </div>
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      <div className="space-y-4 border-t border-slate-800 pt-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <h3 className="text-lg font-black text-slate-100">Snapshot Diff Engine</h3>
            <p className="mt-1 text-sm text-slate-400">
              Compare two behavioral snapshots to investigate uncertainty, safety, and latency drift.
            </p>
          </div>
          <Badge variant="outline" className="border-sky-500/30 bg-sky-500/10 text-sky-300">
            Behavioral Forensics
          </Badge>
        </div>

        {comparableDeployments.length < 2 ? (
          <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-5 text-sm text-slate-500">
            Snapshot comparison becomes available when at least two releases expose browser-fetchable `snapshot_uri` artifacts.
          </div>
        ) : (
          <div className="space-y-4">
            <div className="grid grid-cols-1 gap-4 rounded-2xl border border-slate-800 bg-slate-950/70 p-5 md:grid-cols-2">
              <label className="space-y-2">
                <span className="block text-[10px] font-black uppercase tracking-[0.24em] text-slate-500">
                  Baseline Release
                </span>
                <select
                  value={baseReleaseId}
                  onChange={(event) => setBaseReleaseId(event.target.value)}
                  className="w-full rounded-xl border border-slate-800 bg-slate-900 px-3 py-3 text-sm text-slate-200 outline-none transition focus:border-sky-500/40"
                >
                  {comparableDeployments.map((release) => (
                    <option key={release.release_id} value={release.release_id}>
                      {release.model_id} · {release.release_id}
                    </option>
                  ))}
                </select>
              </label>

              <label className="space-y-2">
                <span className="block text-[10px] font-black uppercase tracking-[0.24em] text-slate-500">
                  Current Release
                </span>
                <select
                  value={currentReleaseId}
                  onChange={(event) => setCurrentReleaseId(event.target.value)}
                  className="w-full rounded-xl border border-slate-800 bg-slate-900 px-3 py-3 text-sm text-slate-200 outline-none transition focus:border-sky-500/40"
                >
                  {comparableDeployments.map((release) => (
                    <option key={release.release_id} value={release.release_id}>
                      {release.model_id} · {release.release_id}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            {snapshotLoading ? (
              <div className="rounded-2xl border border-slate-800 bg-slate-950/70 p-8 text-center text-sm text-slate-500">
                Loading behavioral snapshots...
              </div>
            ) : snapshotError ? (
              <div className="rounded-2xl border border-red-500/20 bg-red-500/5 p-8 text-center text-sm text-red-300">
                {snapshotError}
              </div>
            ) : baseSnapshot && currentSnapshot ? (
              <SnapshotDiffView
                base={baseSnapshot}
                current={currentSnapshot}
                baseLabel={baseReleaseId}
                currentLabel={currentReleaseId}
              />
            ) : null}
          </div>
        )}
      </div>
    </div>
  )
}
