import { ArrowDownIcon, ArrowUpIcon } from "lucide-react"

import { calculateSnapshotDiff, MetricDelta } from "@/lib/diff-engine"
import { BehavioralSnapshot } from "@/types/snapshot"

function metricTone(delta: MetricDelta): string {
  return delta.isRegression ? "text-red-400" : "text-emerald-400"
}

function metricBarTone(delta: MetricDelta, accent: "blue" | "emerald" | "amber"): string {
  if (delta.isRegression) return "bg-red-500"
  if (accent === "emerald") return "bg-emerald-400"
  if (accent === "amber") return "bg-amber-400"
  return "bg-sky-400"
}

function DeltaBadge({ delta }: { delta: MetricDelta }) {
  const Rising = delta.percentChange >= 0
  return (
    <div className={`flex items-center gap-1 font-mono text-sm font-bold ${metricTone(delta)}`}>
      {Rising ? <ArrowUpIcon size={14} /> : <ArrowDownIcon size={14} />}
      <span>{Math.abs(delta.percentChange).toFixed(2)}%</span>
    </div>
  )
}

function ComparisonBar({
  baseValue,
  currentValue,
  max = 1,
  baseClassName,
  currentClassName,
}: {
  baseValue: number
  currentValue: number
  max?: number
  baseClassName: string
  currentClassName: string
}) {
  const safeMax = max > 0 ? max : 1
  const baseWidth = Math.max(2, Math.min((baseValue / safeMax) * 100, 100))
  const currentWidth = Math.max(2, Math.min((currentValue / safeMax) * 100, 100))

  return (
    <div className="relative h-2 w-full overflow-hidden rounded-full bg-slate-800">
      <div className={`absolute inset-y-0 left-0 opacity-35 ${baseClassName}`} style={{ width: `${baseWidth}%` }} />
      <div className={`absolute inset-y-0 left-0 ${currentClassName}`} style={{ width: `${currentWidth}%` }} />
    </div>
  )
}

function MetricPanel({
  title,
  currentDisplay,
  explanation,
  delta,
  baseValue,
  currentValue,
  max,
  accent = "blue",
}: {
  title: string
  currentDisplay: string
  explanation: string
  delta: MetricDelta
  baseValue: number
  currentValue: number
  max?: number
  accent?: "blue" | "emerald" | "amber"
}) {
  const baseBarClass = accent === "emerald" ? "bg-emerald-600" : accent === "amber" ? "bg-amber-600" : "bg-sky-600"
  const currentBarClass = metricBarTone(delta, accent)

  return (
    <div className="space-y-4 rounded-2xl border border-slate-800 bg-slate-950/70 p-4">
      <div className="flex items-end justify-between gap-4">
        <div>
          <label className="text-[10px] font-black uppercase tracking-[0.24em] text-slate-500">{title}</label>
          <div className="mt-1 text-3xl font-mono font-black text-slate-100">{currentDisplay}</div>
        </div>
        <DeltaBadge delta={delta} />
      </div>
      <ComparisonBar
        baseValue={baseValue}
        currentValue={currentValue}
        max={max}
        baseClassName={baseBarClass}
        currentClassName={currentBarClass}
      />
      <div className="flex items-center justify-between text-[11px] font-mono text-slate-500">
        <span>Base {baseValue.toFixed(4)}</span>
        <span>Current {currentValue.toFixed(4)}</span>
      </div>
      <p className="text-xs text-slate-400 italic">{explanation}</p>
    </div>
  )
}

export default function SnapshotDiffView({
  base,
  current,
  baseLabel,
  currentLabel,
}: {
  base: BehavioralSnapshot
  current: BehavioralSnapshot
  baseLabel?: string
  currentLabel?: string
}) {
  const diff = calculateSnapshotDiff(base, current)

  return (
    <div className="space-y-8 rounded-3xl border border-slate-800 bg-slate-900/90 p-6">
      <div className="flex flex-col gap-4 border-b border-slate-800 pb-4 md:flex-row md:items-center md:justify-between">
        <div>
          <h3 className="text-xl font-black text-slate-100">Behavioral DNA Comparison</h3>
          <p className="mt-1 text-sm text-slate-400">
            Statistical drift across uncertainty, safety, latency, and instability.
          </p>
        </div>
        <div className="text-right text-[11px] uppercase tracking-[0.24em] text-slate-500">
          <div>Baseline: {baseLabel ?? base.snapshot_metadata.model_id}</div>
          <div className="mt-1">Current: {currentLabel ?? current.snapshot_metadata.model_id}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-6 xl:grid-cols-2">
        <MetricPanel
          title="Reasoning Entropy"
          currentDisplay={current.behavioral_invariants.reasoning.mean_entropy.toFixed(4)}
          explanation="A positive entropy shift means the model is becoming less certain and more prone to hallucination."
          delta={diff.entropyShift}
          baseValue={base.behavioral_invariants.reasoning.mean_entropy}
          currentValue={current.behavioral_invariants.reasoning.mean_entropy}
          accent="blue"
        />

        <MetricPanel
          title="Safety Escalation Rate"
          currentDisplay={`${(current.behavioral_invariants.safety.escalation_rate * 100).toFixed(1)}%`}
          explanation="Higher escalation frequency indicates more prompts are entering risky or unstable behavioral territory."
          delta={diff.safetyShift}
          baseValue={base.behavioral_invariants.safety.escalation_rate}
          currentValue={current.behavioral_invariants.safety.escalation_rate}
          max={1}
          accent="emerald"
        />

        <MetricPanel
          title="Mean Instability"
          currentDisplay={current.behavioral_invariants.reasoning.mean_instability.toFixed(4)}
          explanation="Instability captures variance across Monte Carlo responses. Rising values indicate weaker behavioral consistency."
          delta={diff.instabilityShift}
          baseValue={base.behavioral_invariants.reasoning.mean_instability}
          currentValue={current.behavioral_invariants.reasoning.mean_instability}
          accent="amber"
        />

        <MetricPanel
          title="Latency"
          currentDisplay={`${current.system_perf.avg_latency_ms.toFixed(0)} ms`}
          explanation="Latency drift is operational, not behavioral, but still part of release readiness for production deployment."
          delta={diff.latencyShift}
          baseValue={base.system_perf.avg_latency_ms}
          currentValue={current.system_perf.avg_latency_ms}
          max={Math.max(base.system_perf.avg_latency_ms, current.system_perf.avg_latency_ms, 1)}
          accent="blue"
        />
      </div>
    </div>
  )
}
