"use client"

import { ChangeEvent, useEffect, useMemo, useState } from "react"
import {
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Download,
  FlaskConical,
  Play,
  Server,
  Trash2,
  Upload,
} from "lucide-react"

type InferenceMode = "factual" | "mixed" | "emotional"

type InferenceConfig = {
  mode: InferenceMode
  temperature: number
  top_p: number
  max_new_tokens: number
}

type ReviewPacket = {
  entropy_samples?: string[]
  embedding_similarity?: number
  ambiguity?: number
}

type CoreComparison = {
  core_a_output: string
  core_b_output: string
  embedding_similarity: number
  token_delta: number
  length_delta: number
}

type InferenceResult = {
  response_text: string
  latency_ms: number
  input_tokens: number
  output_tokens: number
  confidence: number
  instability: number
  escalate: boolean
}

type InferenceApiResponse = InferenceResult & {
  core_comparison?: CoreComparison
  trace?: Record<string, unknown>
  review_packet?: ReviewPacket
}

type ExperimentItem = {
  prompt: string
  category?: string
}

type ExperimentResult = {
  prompt: string
  category: string
  confidence: number
  instability: number
  escalate: boolean
  latency_ms: number
  input_tokens: number
  output_tokens: number
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0
  if (value < 0) return 0
  if (value > 1) return 1
  return value
}

function toPretty(value: unknown): string {
  if (typeof value === "string") return value
  if (typeof value === "number" || typeof value === "boolean" || value === null) {
    return String(value)
  }
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function parseInferenceResponse(payload: unknown): InferenceApiResponse {
  if (!isRecord(payload)) {
    throw new Error("Invalid inference response payload.")
  }

  if (typeof payload.response_text !== "string") {
    throw new Error("Missing response_text in inference response.")
  }

  const numericFields: (keyof InferenceResult)[] = [
    "latency_ms",
    "input_tokens",
    "output_tokens",
    "confidence",
    "instability",
  ]

  for (const field of numericFields) {
    if (typeof payload[field] !== "number" || Number.isNaN(payload[field])) {
      throw new Error(`Missing ${field} in inference response.`)
    }
  }

  if (typeof payload.escalate !== "boolean") {
    throw new Error("Missing escalate in inference response.")
  }

  if ("core_comparison" in payload && payload.core_comparison !== undefined) {
    if (!isRecord(payload.core_comparison)) {
      throw new Error("Invalid core_comparison in inference response.")
    }

    const cc = payload.core_comparison
    if (
      typeof cc.core_a_output !== "string" ||
      typeof cc.core_b_output !== "string" ||
      typeof cc.embedding_similarity !== "number" ||
      typeof cc.token_delta !== "number" ||
      typeof cc.length_delta !== "number"
    ) {
      throw new Error("Incomplete core_comparison in inference response.")
    }
  }

  if ("trace" in payload && payload.trace !== undefined && !isRecord(payload.trace)) {
    throw new Error("Invalid trace in inference response.")
  }

  if (
    "review_packet" in payload &&
    payload.review_packet !== undefined &&
    !isRecord(payload.review_packet)
  ) {
    throw new Error("Invalid review_packet in inference response.")
  }

  return payload as InferenceApiResponse
}

function parseExperimentDataset(payload: unknown): ExperimentItem[] {
  const normalizeArray = (items: unknown[]): ExperimentItem[] => {
    const out: ExperimentItem[] = []
    for (const item of items) {
      if (typeof item === "string" && item.trim()) {
        out.push({ prompt: item.trim(), category: "uncategorized" })
        continue
      }

      if (isRecord(item) && typeof item.prompt === "string" && item.prompt.trim()) {
        out.push({
          prompt: item.prompt.trim(),
          category:
            typeof item.category === "string" && item.category.trim()
              ? item.category.trim()
              : "uncategorized",
        })
      }
    }
    return out
  }

  if (Array.isArray(payload)) {
    return normalizeArray(payload)
  }

  if (isRecord(payload) && Array.isArray(payload.dataset)) {
    return normalizeArray(payload.dataset)
  }

  if (isRecord(payload) && Array.isArray(payload.prompts)) {
    return normalizeArray(payload.prompts)
  }

  throw new Error("Dataset must be an array or object with dataset/prompts array.")
}

function Panel({
  title,
  subtitle,
  children,
  className = "",
}: {
  title: string
  subtitle?: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <section
      className={`rounded-xl border border-[#0dccf2]/20 bg-[#101f22]/75 p-4 md:p-5 shadow-[0_0_20px_rgba(13,204,242,0.06)] ${className}`}
    >
      <header className="mb-3 border-b border-[#0dccf2]/10 pb-2">
        <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-[#0dccf2]">{title}</h2>
        {subtitle ? <p className="mt-1 text-xs text-slate-400">{subtitle}</p> : null}
      </header>
      {children}
    </section>
  )
}

function MetricCard({ label, value, tone = "text-[#0dccf2]" }: { label: string; value: string; tone?: string }) {
  return (
    <div className="rounded-lg border border-[#0dccf2]/15 bg-black/25 p-3">
      <p className="text-[11px] uppercase tracking-[0.14em] text-slate-400">{label}</p>
      <p className={`font-mono text-base ${tone}`}>{value}</p>
    </div>
  )
}

function RibbonMetric({
  label,
  value,
  tone = "text-[#0dccf2]",
}: {
  label: string
  value: string
  tone?: string
}) {
  return (
    <div className="min-w-[148px] rounded-lg border border-[#0dccf2]/20 bg-black/25 px-3 py-2">
      <p className="text-[10px] uppercase tracking-[0.14em] text-slate-400">{label}</p>
      <p className={`font-mono text-sm ${tone}`}>{value}</p>
    </div>
  )
}

export default function Home() {
  const [prompt, setPrompt] = useState("")
  const [config, setConfig] = useState<InferenceConfig>({
    mode: "factual",
    temperature: 0.7,
    top_p: 0.9,
    max_new_tokens: 512,
  })

  const [systemStatus, setSystemStatus] = useState<"Connected" | "Disconnected">("Disconnected")
  const [loading, setLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const [result, setResult] = useState<InferenceResult | null>(null)
  const [coreComparison, setCoreComparison] = useState<CoreComparison | null>(null)
  const [trace, setTrace] = useState<Record<string, unknown> | null>(null)
  const [reviewPacket, setReviewPacket] = useState<ReviewPacket | null>(null)
  const [traceExpanded, setTraceExpanded] = useState(false)
  const [showCoreComparison, setShowCoreComparison] = useState(false)

  const [datasetName, setDatasetName] = useState("No dataset loaded")
  const [datasetItems, setDatasetItems] = useState<ExperimentItem[]>([])
  const [experimentRunning, setExperimentRunning] = useState(false)
  const [experimentProgress, setExperimentProgress] = useState({ done: 0, total: 0 })
  const [experimentError, setExperimentError] = useState<string | null>(null)
  const [experimentResults, setExperimentResults] = useState<ExperimentResult[]>([])
  const [clockText, setClockText] = useState("--:--:--")

  const monteCarlo = useMemo(() => {
    const mc = trace?.monte_carlo_analysis
    return isRecord(mc) ? mc : null
  }, [trace])

  const comparisonVisible = showCoreComparison || Boolean(result?.escalate)

  const experimentSummary = useMemo(() => {
    if (experimentResults.length === 0) return null

    const total = experimentResults.length
    const meanConfidence =
      experimentResults.reduce((sum, item) => sum + item.confidence, 0) / total
    const meanInstability =
      experimentResults.reduce((sum, item) => sum + item.instability, 0) / total
    const escalationRate = experimentResults.filter((item) => item.escalate).length / total

    return { total, meanConfidence, meanInstability, escalationRate }
  }, [experimentResults])

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch("/api/health")
        setSystemStatus(response.ok ? "Connected" : "Disconnected")
      } catch {
        setSystemStatus("Disconnected")
      }
    }

    checkHealth()
    const id = setInterval(checkHealth, 10000)
    return () => clearInterval(id)
  }, [])

  useEffect(() => {
    const tick = () => {
      setClockText(
        new Date().toLocaleTimeString("en-GB", {
          hour12: false,
        }),
      )
    }
    tick()
    const id = setInterval(tick, 1000)
    return () => clearInterval(id)
  }, [])

  const requestInference = async (promptText: string): Promise<InferenceApiResponse> => {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: promptText,
        mode: config.mode,
        temperature: config.temperature,
        top_p: config.top_p,
        max_new_tokens: config.max_new_tokens,
      }),
    })

    const payload = await response.json().catch(() => null)

    if (!response.ok) {
      const apiError =
        isRecord(payload) && typeof payload.error === "string"
          ? payload.error
          : `Inference failed (HTTP ${response.status}).`
      throw new Error(apiError)
    }

    return parseInferenceResponse(payload)
  }

  const runPrompt = async () => {
    if (!prompt.trim()) {
      setErrorMessage("Prompt cannot be empty.")
      return
    }

    setLoading(true)
    setErrorMessage(null)
    setReviewPacket(null)

    try {
      const data = await requestInference(prompt)
      setResult({
        response_text: data.response_text,
        latency_ms: data.latency_ms,
        input_tokens: data.input_tokens,
        output_tokens: data.output_tokens,
        confidence: data.confidence,
        instability: data.instability,
        escalate: data.escalate,
      })
      setCoreComparison(data.core_comparison ?? null)
      setTrace(data.trace ?? null)
      setReviewPacket(data.review_packet ?? null)
    } catch (error) {
      const message = error instanceof Error ? error.message : "Inference server unavailable."
      setErrorMessage(message)
    } finally {
      setLoading(false)
    }
  }

  const handleDatasetUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    setExperimentError(null)

    try {
      const text = await file.text()
      const parsed = JSON.parse(text)
      const items = parseExperimentDataset(parsed)
      if (items.length === 0) {
        throw new Error("Dataset has no valid prompts.")
      }

      setDatasetName(file.name)
      setDatasetItems(items)
    } catch (error) {
      const message = error instanceof Error ? error.message : "Invalid dataset file."
      setExperimentError(message)
      setDatasetName("No dataset loaded")
      setDatasetItems([])
    }
  }

  const runExperiment = async () => {
    if (datasetItems.length === 0) {
      setExperimentError("Load a dataset first.")
      return
    }

    setExperimentRunning(true)
    setExperimentError(null)
    setExperimentResults([])
    setExperimentProgress({ done: 0, total: datasetItems.length })

    const rows: ExperimentResult[] = []
    let firstError: string | null = null

    for (let i = 0; i < datasetItems.length; i += 1) {
      const item = datasetItems[i]
      try {
        const data = await requestInference(item.prompt)
        rows.push({
          prompt: item.prompt,
          category: item.category ?? "uncategorized",
          confidence: data.confidence,
          instability: data.instability,
          escalate: data.escalate,
          latency_ms: data.latency_ms,
          input_tokens: data.input_tokens,
          output_tokens: data.output_tokens,
        })
      } catch (error) {
        const message = error instanceof Error ? error.message : "Experiment request failed."
        if (!firstError) firstError = message

        rows.push({
          prompt: item.prompt,
          category: item.category ?? "uncategorized",
          confidence: 0,
          instability: 1,
          escalate: true,
          latency_ms: 0,
          input_tokens: 0,
          output_tokens: 0,
        })
      } finally {
        setExperimentProgress({ done: i + 1, total: datasetItems.length })
        setExperimentResults([...rows])
      }
    }

    setExperimentError(firstError)
    setExperimentRunning(false)
  }

  const instabilityPercent = result ? clamp01(result.instability) : 0
  const confidenceTone = !result
    ? "text-slate-400"
    : result.confidence >= 0.75
      ? "text-emerald-400"
      : result.confidence >= 0.5
        ? "text-amber-300"
        : "text-red-400"
  const instabilityTone = !result
    ? "text-slate-400"
    : result.instability <= 0.25
      ? "text-emerald-400"
      : result.instability <= 0.4
        ? "text-amber-300"
        : "text-red-400"
  const escalationTone = !result
    ? "text-slate-400"
    : result.escalate
      ? "text-red-400"
      : "text-emerald-400"

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#080e10] text-slate-100">
      <header className="sticky top-0 z-30 border-b border-[#0dccf2]/20 bg-[#101f22]/95 px-4 py-3 backdrop-blur md:px-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-sm font-bold uppercase tracking-[0.18em] text-[#0dccf2] md:text-base">
              AI Research Command Center
            </h1>
            <p className="mt-1 text-xs text-slate-400">
              Model: Qwen2.5-7B | Backend: Kaggle | Runtime: Connected Flow
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono">
            <div className="flex items-center gap-2">
              <span
                className={`h-2 w-2 rounded-full ${systemStatus === "Connected" ? "bg-emerald-500" : "bg-red-500"
                  }`}
              />
              <span className={systemStatus === "Connected" ? "text-emerald-400" : "text-red-400"}>
                {systemStatus}
              </span>
            </div>
            <div className="text-slate-500">{clockText}</div>
          </div>
        </div>
      </header>

      <section className="border-b border-[#0dccf2]/15 bg-[#0f181c]/95 px-4 py-2 md:px-6">
        <div className="flex gap-3 overflow-x-auto pb-1">
          <RibbonMetric
            label="Confidence"
            value={result ? result.confidence.toFixed(3) : "--"}
            tone={confidenceTone}
          />
          <RibbonMetric
            label="Instability"
            value={result ? result.instability.toFixed(3) : "--"}
            tone={instabilityTone}
          />
          <RibbonMetric
            label="Latency"
            value={result ? `${result.latency_ms} ms` : "--"}
          />
          <RibbonMetric
            label="Tokens"
            value={result ? `${result.input_tokens} -> ${result.output_tokens}` : "--"}
          />
          <RibbonMetric
            label="Escalation"
            value={result ? (result.escalate ? "TRUE" : "FALSE") : "--"}
            tone={escalationTone}
          />
        </div>
      </section>

      <main className="min-h-0 flex-1 overflow-hidden p-4 md:p-6">
        <div className="grid min-h-0 gap-4 lg:h-full lg:grid-rows-[minmax(0,1fr)_minmax(240px,36%)]">
          <div className="grid min-h-0 grid-cols-1 gap-4 xl:grid-cols-[minmax(280px,30%)_minmax(0,45%)_minmax(240px,25%)]">
            <Panel title="Prompt Lab" subtitle="Prompt + controls + run" className="min-h-0 flex flex-col">
              <div className="min-h-0 flex-1 space-y-4 overflow-y-auto pr-1">
                <div className="space-y-2">
                  <label className="text-xs uppercase tracking-[0.14em] text-slate-400">Prompt</label>
                  <textarea
                    value={prompt}
                    onChange={(event) => setPrompt(event.target.value)}
                    placeholder="Enter prompt..."
                    className="min-h-[140px] w-full resize-y rounded-lg border border-[#0dccf2]/20 bg-black/30 p-3 text-sm text-slate-100 outline-none focus:border-[#0dccf2]/50"
                  />
                </div>

                <div className="space-y-2">
                  <label className="text-xs uppercase tracking-[0.14em] text-slate-400">Mode</label>
                  <select
                    value={config.mode}
                    onChange={(event) =>
                      setConfig((prev) => ({ ...prev, mode: event.target.value as InferenceMode }))
                    }
                    className="w-full rounded-lg border border-[#0dccf2]/20 bg-black/30 px-3 py-2 text-sm outline-none focus:border-[#0dccf2]/50"
                  >
                    <option value="factual">factual</option>
                    <option value="mixed">mixed</option>
                    <option value="emotional">emotional</option>
                  </select>
                </div>

                {[{
                  label: "Temperature",
                  key: "temperature",
                  value: config.temperature,
                  min: 0,
                  max: 2,
                  step: 0.1,
                }, {
                  label: "Top-P",
                  key: "top_p",
                  value: config.top_p,
                  min: 0,
                  max: 1,
                  step: 0.05,
                }, {
                  label: "Max Tokens",
                  key: "max_new_tokens",
                  value: config.max_new_tokens,
                  min: 32,
                  max: 8192,
                  step: 32,
                }].map((control) => (
                  <div key={control.key} className="space-y-1">
                    <div className="flex items-center justify-between text-xs">
                      <label className="uppercase tracking-[0.14em] text-slate-400">{control.label}</label>
                      <span className="font-mono text-[#0dccf2]">{control.value}</span>
                    </div>
                    <input
                      type="range"
                      min={control.min}
                      max={control.max}
                      step={control.step}
                      value={control.value}
                      onChange={(event) =>
                        setConfig((prev) => ({ ...prev, [control.key]: Number(event.target.value) }))
                      }
                      className="w-full accent-[#0dccf2]"
                    />
                  </div>
                ))}

                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={runPrompt}
                    disabled={loading}
                    className="inline-flex flex-1 items-center justify-center gap-2 rounded-lg border border-[#0dccf2]/30 bg-[#0dccf2]/90 px-4 py-2 text-sm font-semibold text-[#081014] transition hover:bg-[#33d5f3] disabled:opacity-60"
                  >
                    <Play className="h-4 w-4" />
                    {loading ? "Running..." : "Run Prompt"}
                  </button>
                  <button
                    type="button"
                    onClick={() => {
                      setPrompt("")
                      setResult(null)
                      setCoreComparison(null)
                      setTrace(null)
                      setReviewPacket(null)
                      setErrorMessage(null)
                    }}
                    className="inline-flex items-center justify-center gap-1 rounded-lg border border-[#0dccf2]/20 bg-black/25 px-3 py-2 text-xs uppercase tracking-[0.12em] text-slate-300 transition hover:border-red-500/40 hover:text-red-300"
                    title="Clear prompt and results"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </Panel>

            <Panel title="Core A / Core B" subtitle="Final output plus deterministic/entropy comparison" className="min-h-0 flex flex-col">
              <div className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
                <div className="rounded-lg border border-[#0dccf2]/20 bg-black/25 p-3">
                  <p className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">Final Output</p>
                  <div className="max-h-[350px] overflow-y-auto pr-1">
                    {result ? (
                      <p className="whitespace-pre-wrap text-sm leading-6 text-slate-100">{result.response_text}</p>
                    ) : (
                      <p className="text-sm text-slate-500">Run a prompt to view output.</p>
                    )}
                  </div>
                </div>

                {errorMessage ? (
                  <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-200">
                    {errorMessage}
                  </div>
                ) : null}

                <button
                  type="button"
                  onClick={() => setShowCoreComparison((prev) => !prev)}
                  className="inline-flex items-center gap-2 rounded-lg border border-[#0dccf2]/20 bg-black/25 px-3 py-2 text-xs uppercase tracking-[0.12em] text-slate-200"
                >
                  {comparisonVisible ? <ChevronDown className="h-4 w-4 text-[#0dccf2]" /> : <ChevronRight className="h-4 w-4 text-[#0dccf2]" />}
                  Compare Cores
                  {result?.escalate ? <span className="ml-1 text-red-400">(auto: escalation)</span> : null}
                </button>

                {comparisonVisible ? (
                  <div className="space-y-3 rounded-lg border border-[#0dccf2]/20 bg-black/20 p-3">
                    {coreComparison ? (
                      <>
                        <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
                          <div className="rounded-lg border border-[#0dccf2]/15 bg-black/25 p-3">
                            <p className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">Core A (Deterministic)</p>
                            <div className="max-h-36 overflow-y-auto text-sm text-slate-200">
                              {coreComparison.core_a_output || "-"}
                            </div>
                          </div>
                          <div className="rounded-lg border border-[#0dccf2]/15 bg-black/25 p-3">
                            <p className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">Core B (Entropy)</p>
                            <div className="max-h-36 overflow-y-auto text-sm text-slate-200">
                              {coreComparison.core_b_output || "-"}
                            </div>
                          </div>
                        </div>
                        <div className="grid grid-cols-1 gap-2 md:grid-cols-3">
                          <MetricCard label="Similarity" value={coreComparison.embedding_similarity.toFixed(3)} />
                          <MetricCard label="Token Delta" value={coreComparison.token_delta.toFixed(0)} />
                          <MetricCard label="Length Delta" value={coreComparison.length_delta.toFixed(0)} />
                        </div>
                      </>
                    ) : (
                      <p className="text-sm text-slate-500">Core comparison data unavailable for this run.</p>
                    )}
                  </div>
                ) : null}
              </div>
            </Panel>

            <div className="min-h-0 space-y-4 overflow-y-auto pr-1">
              <Panel title="Reliability" subtitle="Confidence and instability view">
                {result ? (
                  <div className="space-y-3">
                    <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
                      <MetricCard label="Confidence" value={result.confidence.toFixed(3)} tone="text-emerald-400" />
                      <MetricCard label="Instability" value={result.instability.toFixed(3)} tone="text-amber-300" />
                      <MetricCard
                        label="Escalation"
                        value={result.escalate ? "TRUE" : "FALSE"}
                        tone={result.escalate ? "text-red-400" : "text-emerald-400"}
                      />
                    </div>

                    <div className="rounded-lg border border-[#0dccf2]/15 bg-black/25 p-3">
                      <p className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">Instability Meter</p>
                      <div className="h-3 overflow-hidden rounded bg-[#0dccf2]/15">
                        <div
                          className={`h-full ${result.escalate ? "bg-red-500" : "bg-emerald-500"}`}
                          style={{ width: `${(instabilityPercent * 100).toFixed(2)}%` }}
                        />
                      </div>
                      <p className="mt-2 font-mono text-xs text-slate-300">{result.instability.toFixed(3)}</p>
                    </div>

                    <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                      <MetricCard
                        label="Det vs Ent Similarity"
                        value={
                          isRecord(monteCarlo) && typeof monteCarlo.det_entropy_similarity === "number"
                            ? monteCarlo.det_entropy_similarity.toFixed(3)
                            : "n/a"
                        }
                      />
                      <MetricCard
                        label="Entropy Consistency"
                        value={
                          isRecord(monteCarlo) && typeof monteCarlo.entropy_consistency === "number"
                            ? monteCarlo.entropy_consistency.toFixed(3)
                            : "n/a"
                        }
                      />
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-slate-500">No reliability data yet.</p>
                )}
              </Panel>

              <Panel title="Telemetry" subtitle="Latency and token accounting">
                {result ? (
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
                    <MetricCard label="Latency" value={`${result.latency_ms} ms`} />
                    <MetricCard label="Input Tokens" value={result.input_tokens.toString()} />
                    <MetricCard label="Output Tokens" value={result.output_tokens.toString()} />
                  </div>
                ) : (
                  <p className="text-sm text-slate-500">Telemetry appears after running inference.</p>
                )}
              </Panel>

              {result?.escalate ? (
                <Panel title="Escalation" subtitle="Uncertainty diagnostics">
                  <div className="space-y-2 text-sm text-red-200">
                    <p className="rounded-lg border border-red-500/30 bg-red-500/10 p-2 uppercase tracking-[0.12em]">
                      Model uncertainty detected
                    </p>
                    <p className="font-mono">
                      Embedding Similarity: {typeof reviewPacket?.embedding_similarity === "number"
                        ? reviewPacket.embedding_similarity.toFixed(3)
                        : "n/a"}
                    </p>
                    <p className="font-mono">
                      Ambiguity: {typeof reviewPacket?.ambiguity === "number"
                        ? reviewPacket.ambiguity.toFixed(3)
                        : "n/a"}
                    </p>
                    <p className="font-mono">Entropy Samples: {reviewPacket?.entropy_samples?.length ?? 0}</p>
                  </div>
                </Panel>
              ) : null}

              <Panel title="Trace" subtitle="Collapsible decision trace">
                <button
                  type="button"
                  onClick={() => setTraceExpanded((prev) => !prev)}
                  className="flex w-full items-center justify-between rounded-lg border border-[#0dccf2]/20 bg-black/25 px-3 py-2 text-left text-sm"
                >
                  <span className="uppercase tracking-[0.12em] text-slate-300">Trace Details</span>
                  {traceExpanded ? <ChevronDown className="h-4 w-4 text-[#0dccf2]" /> : <ChevronRight className="h-4 w-4 text-[#0dccf2]" />}
                </button>

                {traceExpanded ? (
                  trace ? (
                    <div className="mt-3 space-y-2">
                      {Object.entries(trace).map(([key, value]) => (
                        <div key={key} className="rounded-lg border border-[#0dccf2]/15 bg-black/25 p-2">
                          <p className="mb-1 text-[11px] uppercase tracking-[0.14em] text-slate-400">{key}</p>
                          <pre className="max-h-40 overflow-y-auto whitespace-pre-wrap break-words text-xs text-slate-200">
                            {toPretty(value)}
                          </pre>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="mt-3 text-sm text-slate-500">Run an inference to populate trace data.</p>
                  )
                ) : null}
              </Panel>
            </div>
          </div>

          <Panel
            title="Experiment Runner"
            subtitle="Dataset testing with confidence, instability and escalation tracking"
            className="min-h-0 flex flex-col"
          >
            <div className="mb-4 flex flex-wrap items-center gap-3">
              <label className="inline-flex cursor-pointer items-center gap-2 rounded-lg border border-[#0dccf2]/30 bg-black/25 px-3 py-2 text-xs uppercase tracking-[0.12em] text-slate-200 hover:border-[#0dccf2]/60">
                <Upload className="h-4 w-4 text-[#0dccf2]" />
                Upload JSON
                <input type="file" accept="application/json" className="hidden" onChange={handleDatasetUpload} />
              </label>

              <button
                type="button"
                onClick={runExperiment}
                disabled={experimentRunning || datasetItems.length === 0}
                className="inline-flex items-center gap-2 rounded-lg border border-[#0dccf2]/30 bg-[#0dccf2]/90 px-3 py-2 text-xs font-semibold uppercase tracking-[0.12em] text-[#081014] disabled:opacity-60"
              >
                <FlaskConical className="h-4 w-4" />
                {experimentRunning ? "Running" : "Run Experiment"}
              </button>

              {experimentResults.length > 0 ? (
                <button
                  type="button"
                  onClick={() => {
                    const blob = new Blob([JSON.stringify(experimentResults, null, 2)], { type: "application/json" })
                    const url = URL.createObjectURL(blob)
                    const a = document.createElement("a")
                    a.href = url
                    a.download = `experiment_results_${Date.now()}.json`
                    a.click()
                    URL.revokeObjectURL(url)
                  }}
                  className="inline-flex items-center gap-2 rounded-lg border border-[#0dccf2]/30 bg-black/25 px-3 py-2 text-xs uppercase tracking-[0.12em] text-slate-200 hover:border-[#0dccf2]/60"
                >
                  <Download className="h-4 w-4 text-[#0dccf2]" />
                  Export
                </button>
              ) : null}

              <div className="text-xs text-slate-400">
                <span className="font-semibold text-slate-200">Dataset:</span> {datasetName}
                {datasetItems.length > 0 ? ` (${datasetItems.length} prompts)` : ""}
              </div>
            </div>

            {experimentRunning ? (
              <div className="mb-3 rounded-lg border border-[#0dccf2]/20 bg-black/25 p-3 text-xs text-slate-300">
                Running {experimentProgress.done}/{experimentProgress.total}
              </div>
            ) : null}

            {experimentError ? (
              <div className="mb-3 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-200">
                {experimentError}
              </div>
            ) : null}

            {experimentSummary ? (
              <>
                <div className="mb-3 grid grid-cols-1 gap-2 md:grid-cols-4">
                  <MetricCard label="Prompts" value={experimentSummary.total.toString()} />
                  <MetricCard label="Mean Confidence" value={experimentSummary.meanConfidence.toFixed(3)} />
                  <MetricCard label="Avg Instability" value={experimentSummary.meanInstability.toFixed(3)} />
                  <MetricCard label="Escalation Rate" value={`${(experimentSummary.escalationRate * 100).toFixed(1)}%`} />
                </div>

                <div className="mb-4 rounded-lg border border-[#0dccf2]/15 bg-black/25 p-3">
                  <p className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">Stability Timeline</p>
                  <div className="flex h-20 items-end gap-1 overflow-x-auto">
                    {experimentResults.map((row, index) => (
                      <div
                        key={`${row.prompt}-${index}`}
                        title={`${index + 1}: ${row.instability.toFixed(3)}`}
                        className={`w-2 shrink-0 rounded-t ${row.escalate ? "bg-red-500" : "bg-[#0dccf2]"}`}
                        style={{ height: `${Math.max(6, Math.round(clamp01(row.instability) * 100))}%` }}
                      />
                    ))}
                  </div>
                </div>
              </>
            ) : null}

            <div className="min-h-0 flex-1 overflow-y-auto rounded-lg border border-[#0dccf2]/15 bg-black/20">
              {experimentResults.length === 0 ? (
                <div className="flex h-full min-h-[120px] items-center justify-center p-4 text-sm text-slate-500">
                  Upload a JSON dataset and run experiment to populate results.
                </div>
              ) : (
                <table className="w-full min-w-[960px] border-collapse text-xs">
                  <thead className="sticky top-0 bg-[#101f22]">
                    <tr className="text-left uppercase tracking-[0.12em] text-slate-400">
                      <th className="border-b border-[#0dccf2]/15 px-3 py-2">Prompt</th>
                      <th className="border-b border-[#0dccf2]/15 px-3 py-2">Category</th>
                      <th className="border-b border-[#0dccf2]/15 px-3 py-2">Confidence</th>
                      <th className="border-b border-[#0dccf2]/15 px-3 py-2">Instability</th>
                      <th className="border-b border-[#0dccf2]/15 px-3 py-2">Escalate</th>
                      <th className="border-b border-[#0dccf2]/15 px-3 py-2">Latency (ms)</th>
                      <th className="border-b border-[#0dccf2]/15 px-3 py-2">Tokens In</th>
                      <th className="border-b border-[#0dccf2]/15 px-3 py-2">Tokens Out</th>
                    </tr>
                  </thead>
                  <tbody>
                    {experimentResults.map((row, index) => (
                      <tr key={`${row.prompt}-${index}`} className="border-b border-[#0dccf2]/10 text-slate-200">
                        <td className="max-w-[340px] px-3 py-2">
                          <div className="line-clamp-2">{row.prompt}</div>
                        </td>
                        <td className="px-3 py-2">{row.category}</td>
                        <td className="px-3 py-2 font-mono">{row.confidence.toFixed(3)}</td>
                        <td className="px-3 py-2 font-mono">{row.instability.toFixed(3)}</td>
                        <td className="px-3 py-2 font-mono">
                          {row.escalate ? (
                            <span className="inline-flex items-center gap-1 text-red-400">
                              <AlertTriangle className="h-3 w-3" />
                              true
                            </span>
                          ) : (
                            <span className="text-emerald-400">false</span>
                          )}
                        </td>
                        <td className="px-3 py-2 font-mono">{row.latency_ms}</td>
                        <td className="px-3 py-2 font-mono">{row.input_tokens}</td>
                        <td className="px-3 py-2 font-mono">{row.output_tokens}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              )}
            </div>
          </Panel>
        </div>
      </main>

      <footer className="border-t border-[#0dccf2]/15 bg-[#101f22]/90 px-4 py-2 text-xs text-slate-400 md:px-6">
        <div className="flex flex-wrap items-center gap-4 font-mono">
          <span className="inline-flex items-center gap-2">
            <Server className="h-3.5 w-3.5 text-[#0dccf2]" />
            Backend: {systemStatus}
          </span>
          {result ? (
            <span>
              Last run: {result.latency_ms}ms | {result.input_tokens} -&gt; {result.output_tokens} tokens
            </span>
          ) : (
            <span>Run a prompt to populate telemetry.</span>
          )}
        </div>
      </footer>
    </div>
  )
}
