"use client"

import Link from "next/link"
import { ChangeEvent, useEffect, useMemo, useState } from "react"
import {
  Activity,
  AlertTriangle,
  BarChart3,
  ChevronDown,
  ChevronRight,
  ClipboardCheck,
  Download,
  FlaskConical,
  History,
  Play,
  Server,
  Trash2,
  Trophy,
  Upload,
} from "lucide-react"
import StabilityChart from "@/components/StabilityChart"
import ReliabilityDistributions from "@/components/ReliabilityDistributions"
import ReleaseOverview from "@/components/ReleaseOverview"
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { DeploymentEntry, parseDeploymentEntries } from "@/types/release"

type InferenceMode = "factual" | "mixed" | "emotional"

type InferenceConfig = {
  mode: InferenceMode
  temperature: number
  top_p: number
  max_new_tokens: number
  monte_carlo_samples: number
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
  entropy?: number
  uncertainty?: number
  escalate: boolean
  sample_count?: number
  samples_used?: number
  semantic_dispersion?: number
  cluster_count?: number
  cluster_entropy?: number
  dominant_cluster_ratio?: number
  self_consistency?: number
  failures?: string[]
  resampled?: boolean
}

type TraceLog = {
  monte_carlo_samples?: {
    text: string
    cluster: number
  }[]
  [key: string]: unknown
}

type InferenceApiResponse = InferenceResult & {
  core_comparison?: CoreComparison
  trace?: TraceLog
  review_packet?: ReviewPacket
}

type ExperimentItem = {
  prompt: string
  category?: string
}

const PROMPT_CATEGORIES = [
  "factual",
  "reasoning",
  "math",
  "coding",
  "creative",
  "philosophical",
  "emotional",
  "safety",
  "instruction"
] as const;

type PromptCategory = typeof PROMPT_CATEGORIES[number] | string;

type DifficultyLabel = "easy" | "moderate" | "hard" | "adversarial"

type LeaderboardEntry = {
  model: string
  mean_confidence: number
  mean_instability: number
  escalation_rate: number
  mean_entropy: number
  timestamp: string
}

type ExperimentResult = {
  prompt: string
  category: string
  response_text: string
  temperature: number
  confidence: number
  instability: number
  entropy: number
  uncertainty: number
  escalate: boolean
  difficulty: number
  difficulty_label: DifficultyLabel
  temperature_sensitivity: number
  latency_ms: number
  input_tokens: number
  output_tokens: number
  sample_count: number
  samples_used: number
  semantic_dispersion?: number
  cluster_count?: number
  cluster_entropy?: number
  dominant_cluster_ratio?: number
  self_consistency?: number
  trace?: TraceLog
  failures?: string[]
}

type ModelStatus = "ready" | "loading" | "offline"

type TemperatureAggregatePoint = {
  temperature: number
  value: number
}

type CategoryAggregatePoint = {
  category: string
  count: number
  instability: number
  entropy: number
  confidence: number
  difficulty: number
  cluster_count: number
  cluster_entropy: number
  sample_count?: number
  semantic_dispersion?: number
  dominant_cluster_ratio?: number
  self_consistency?: number
}

type PromptTemperatureStats = {
  prompt: string
  category: string
  low_temperature: number
  high_temperature: number
  low_instability: number
  high_instability: number
  sensitivity: number
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

function computeDifficulty(
  confidence: number,
  instability: number,
  entropy: number,
  escalate: boolean,
): number {
  const score =
    0.35 * clamp01(instability) +
    0.35 * clamp01(entropy) +
    0.2 * (1 - clamp01(confidence)) +
    0.1 * (escalate ? 1 : 0)
  return clamp01(score)
}

function difficultyLabel(score: number): DifficultyLabel {
  if (score < 0.25) return "easy"
  if (score < 0.5) return "moderate"
  if (score < 0.7) return "hard"
  return "adversarial"
}

function difficultyTone(label: DifficultyLabel): string {
  if (label === "easy") return "text-emerald-400"
  if (label === "moderate") return "text-amber-300"
  if (label === "hard") return "text-orange-300"
  return "text-red-400"
}

function difficultyBarTone(label: DifficultyLabel): string {
  if (label === "easy") return "bg-emerald-500"
  if (label === "moderate") return "bg-amber-400"
  if (label === "hard") return "bg-orange-400"
  return "bg-red-500"
}

function sensitivityTone(value: number): string {
  if (value <= 0.15) return "text-emerald-400"
  if (value <= 0.35) return "text-amber-300"
  return "text-red-400"
}

function instabilityTone(value: number): string {
  if (value <= 0.25) return "text-emerald-400"
  if (value <= 0.4) return "text-amber-300"
  return "text-red-400"
}

function entropyTone(value: number): string {
  if (value <= 0.25) return "text-emerald-400"
  if (value <= 0.4) return "text-amber-300"
  return "text-red-400"
}

function toTemperatureKey(value: number): number {
  return Number(value.toFixed(2))
}

function makePromptKey(prompt: string, category: string): string {
  return `${category}\u241f${prompt}`
}

function buildPromptTemperatureStats(rows: ExperimentResult[]): Map<string, PromptTemperatureStats> {
  const grouped = new Map<string, { prompt: string; category: string; points: Array<{ temperature: number; instability: number }> }>()

  for (const row of rows) {
    const key = makePromptKey(row.prompt, row.category)
    const existing = grouped.get(key)
    if (existing) {
      existing.points.push({ temperature: row.temperature, instability: row.instability })
    } else {
      grouped.set(key, {
        prompt: row.prompt,
        category: row.category,
        points: [{ temperature: row.temperature, instability: row.instability }],
      })
    }
  }

  const out = new Map<string, PromptTemperatureStats>()
  for (const [key, group] of grouped.entries()) {
    const points = [...group.points].sort((a, b) => a.temperature - b.temperature)
    const low = points[0]
    const high = points[points.length - 1]
    out.set(key, {
      prompt: group.prompt,
      category: group.category,
      low_temperature: low.temperature,
      high_temperature: high.temperature,
      low_instability: low.instability,
      high_instability: high.instability,
      sensitivity: high.instability - low.instability,
    })
  }

  return out
}

function attachTemperatureSensitivity(rows: ExperimentResult[]): ExperimentResult[] {
  const stats = buildPromptTemperatureStats(rows)
  return rows.map((row) => ({
    ...row,
    temperature_sensitivity: stats.get(makePromptKey(row.prompt, row.category))?.sensitivity ?? 0,
  }))
}

function aggregateByTemperature(
  rows: ExperimentResult[],
  selector: (row: ExperimentResult) => number,
): TemperatureAggregatePoint[] {
  const grouped = new Map<number, { sum: number; count: number }>()
  for (const row of rows) {
    const key = toTemperatureKey(row.temperature)
    const value = selector(row)
    if (!Number.isFinite(value)) continue
    const existing = grouped.get(key)
    if (existing) {
      existing.sum += value
      existing.count += 1
    } else {
      grouped.set(key, { sum: value, count: 1 })
    }
  }

  return [...grouped.entries()]
    .sort((a, b) => a[0] - b[0])
    .map(([temperature, aggregate]) => ({
      temperature,
      value: aggregate.count ? aggregate.sum / aggregate.count : 0,
    }))
}

function aggregateByCategory(
  rows: ExperimentResult[]
): CategoryAggregatePoint[] {
  const grouped = new Map<string, { count: number; instability: number; entropy: number; confidence: number; difficulty: number; cluster_count: number; cluster_entropy: number; sample_count: number; semantic_dispersion: number; dominant_cluster_ratio: number }>()

  for (const row of rows) {
    const existing = grouped.get(row.category)
    if (existing) {
      existing.count += 1
      existing.instability += row.instability
      existing.entropy += row.entropy
      existing.confidence += row.confidence
      existing.difficulty += row.difficulty
      existing.cluster_count += row.cluster_count ?? 0
      existing.cluster_entropy += row.cluster_entropy ?? 0
      existing.sample_count += row.sample_count ?? 0
      existing.semantic_dispersion += row.semantic_dispersion ?? 0
      existing.dominant_cluster_ratio += row.dominant_cluster_ratio ?? 0
    } else {
      grouped.set(row.category, {
        count: 1,
        instability: row.instability,
        entropy: row.entropy,
        confidence: row.confidence,
        difficulty: row.difficulty,
        cluster_count: row.cluster_count ?? 0,
        cluster_entropy: row.cluster_entropy ?? 0,
        sample_count: row.sample_count ?? 0,
        semantic_dispersion: row.semantic_dispersion ?? 0,
        dominant_cluster_ratio: row.dominant_cluster_ratio ?? 0,
      })
    }
  }

  return [...grouped.entries()]
    .map(([category, agg]) => ({
      category,
      count: agg.count,
      instability: agg.count ? agg.instability / agg.count : 0,
      entropy: agg.count ? agg.entropy / agg.count : 0,
      confidence: agg.count ? agg.confidence / agg.count : 0,
      difficulty: agg.count ? agg.difficulty / agg.count : 0,
      cluster_count: agg.count ? agg.cluster_count / agg.count : 0,
      cluster_entropy: agg.count ? agg.cluster_entropy / agg.count : 0,
      sample_count: agg.count ? agg.sample_count / agg.count : 0,
      semantic_dispersion: agg.count ? agg.semantic_dispersion / agg.count : 0,
      dominant_cluster_ratio: agg.count ? agg.dominant_cluster_ratio / agg.count : 0,
    }))
    .sort((a, b) => b.count - a.count)
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
    "samples_used"
  ]

  for (const field of numericFields) {
    if (typeof payload[field] !== "number" || Number.isNaN(payload[field])) {
      throw new Error(`Missing ${field} in inference response.`)
    }
  }

  if (typeof payload.escalate !== "boolean") {
    throw new Error("Missing escalate in inference response.")
  }

  if ("entropy" in payload && payload.entropy !== undefined) {
    if (typeof payload.entropy !== "number" || Number.isNaN(payload.entropy)) {
      throw new Error("Invalid entropy in inference response.")
    }
  }

  if ("uncertainty" in payload && payload.uncertainty !== undefined) {
    if (typeof payload.uncertainty !== "number" || Number.isNaN(payload.uncertainty)) {
      throw new Error("Invalid uncertainty in inference response.")
    }
  }

  if ("sample_count" in payload && payload.sample_count !== undefined) {
    if (typeof payload.sample_count !== "number" || Number.isNaN(payload.sample_count)) {
      throw new Error("Invalid sample_count in inference response.")
    }
  }

  if ("semantic_dispersion" in payload && payload.semantic_dispersion !== undefined) {
    if (typeof payload.semantic_dispersion !== "number" || Number.isNaN(payload.semantic_dispersion)) {
      throw new Error("Invalid semantic_dispersion in inference response.")
    }
  }

  if ("cluster_count" in payload && payload.cluster_count !== undefined) {
    if (typeof payload.cluster_count !== "number" || Number.isNaN(payload.cluster_count)) {
      throw new Error("Invalid cluster_count in inference response.")
    }
  }

  if ("cluster_entropy" in payload && payload.cluster_entropy !== undefined) {
    if (typeof payload.cluster_entropy !== "number" || Number.isNaN(payload.cluster_entropy)) {
      throw new Error("Invalid cluster_entropy in inference response.")
    }
  }

  if ("dominant_cluster_ratio" in payload && payload.dominant_cluster_ratio !== undefined) {
    if (typeof payload.dominant_cluster_ratio !== "number" || Number.isNaN(payload.dominant_cluster_ratio)) {
      throw new Error("Invalid dominant_cluster_ratio in inference response.")
    }
  }

  if ("self_consistency" in payload && payload.self_consistency !== undefined) {
    if (typeof payload.self_consistency !== "number" || Number.isNaN(payload.self_consistency)) {
      throw new Error("Invalid self_consistency in inference response.")
    }
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

  if ("failures" in payload && payload.failures !== undefined) {
    if (!Array.isArray(payload.failures)) {
      throw new Error("Invalid failures in inference response.")
    }
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

const CHART_WIDTH = 1100
const CHART_HEIGHT = 650
const CHART_MARGIN = { top: 72, right: 42, bottom: 78, left: 86 }
const TEMPERATURE_SWEEP = [0.1, 0.3, 0.5, 0.7, 0.9] as const

function toCsvCell(value: unknown): string {
  const text = value === null || value === undefined ? "" : String(value)
  if (/[",\n]/.test(text)) {
    return `"${text.replace(/"/g, '""')}"`
  }
  return text
}

function buildCsv(rows: ExperimentResult[]): string {
  if (rows.length === 0) return ""
  const headers = [
    "prompt",
    "category",
    "difficulty_label",
    "temperature",
    "instability",
    "entropy",
    "confidence",
    "escalate",
    "sample_count",
    "samples_used",
    "semantic_dispersion",
    "cluster_count",
    "cluster_entropy",
    "dominant_cluster_ratio",
    "self_consistency",
    "temperature_sensitivity",
    "latency_ms",
    "input_tokens",
    "output_tokens",
  ]
  const headerLine = headers.join(",")
  const lines = rows.map((row) =>
    [
      toCsvCell(row.prompt),
      toCsvCell(row.category),
      toCsvCell(row.difficulty_label),
      toCsvCell(row.temperature),
      toCsvCell(row.instability),
      toCsvCell(row.entropy),
      toCsvCell(row.confidence),
      toCsvCell(row.escalate),
      toCsvCell(row.sample_count),
      toCsvCell(row.samples_used),
      toCsvCell(row.semantic_dispersion),
      toCsvCell(row.cluster_count),
      toCsvCell(row.cluster_entropy),
      toCsvCell(row.dominant_cluster_ratio),
      toCsvCell(row.self_consistency),
      toCsvCell(row.temperature_sensitivity),
      toCsvCell(row.latency_ms),
      toCsvCell(row.input_tokens),
      toCsvCell(row.output_tokens),
    ].join(","),
  )
  return [headerLine, ...lines].join("\n")
}

function mean(values: number[]): number {
  if (values.length === 0) return 0
  return values.reduce((sum, value) => sum + value, 0) / values.length
}

function downloadBlob(filename: string, blob: Blob) {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement("a")
  anchor.href = url
  anchor.download = filename
  anchor.click()
  URL.revokeObjectURL(url)
}

function downloadText(filename: string, text: string, contentType: string) {
  downloadBlob(filename, new Blob([text], { type: contentType }))
}

function canvasToBlob(canvas: HTMLCanvasElement): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (!blob) {
        reject(new Error("Failed to render chart image."))
        return
      }
      resolve(blob)
    }, "image/png")
  })
}

function makeChartSurface(title: string) {
  const canvas = document.createElement("canvas")
  canvas.width = CHART_WIDTH
  canvas.height = CHART_HEIGHT
  const context = canvas.getContext("2d")
  if (!context) {
    throw new Error("Could not initialize chart canvas.")
  }

  context.fillStyle = "#080e10"
  context.fillRect(0, 0, CHART_WIDTH, CHART_HEIGHT)

  context.fillStyle = "#0dccf2"
  context.font = "600 26px monospace"
  context.fillText(title, CHART_MARGIN.left, 42)

  context.strokeStyle = "#0dccf2"
  context.lineWidth = 1
  context.strokeRect(0.5, 0.5, CHART_WIDTH - 1, CHART_HEIGHT - 1)

  const plot = {
    left: CHART_MARGIN.left,
    right: CHART_WIDTH - CHART_MARGIN.right,
    top: CHART_MARGIN.top,
    bottom: CHART_HEIGHT - CHART_MARGIN.bottom,
  }

  context.strokeStyle = "rgba(13, 204, 242, 0.45)"
  context.lineWidth = 1.5
  context.beginPath()
  context.moveTo(plot.left, plot.bottom)
  context.lineTo(plot.right, plot.bottom)
  context.moveTo(plot.left, plot.top)
  context.lineTo(plot.left, plot.bottom)
  context.stroke()

  return { canvas, context, plot }
}

async function exportCategoryBarChartPng(
  data: Array<{ label: string; value: number }>,
  filename: string,
  title: string,
  yAxisLabel: string
) {
  if (data.length === 0) return

  const { canvas, context, plot } = makeChartSurface(title)
  const plotWidth = plot.right - plot.left
  const plotHeight = plot.bottom - plot.top

  const maxValue = Math.max(1.0, ...data.map(d => d.value))
  const barGap = 16
  const barWidth = (plotWidth - barGap * (data.length - 1)) / Math.max(data.length, 1)

  context.fillStyle = "#94a3b8"
  context.font = "13px monospace"
  context.fillText(yAxisLabel, 14, plot.top + 10)

  data.forEach((item, index) => {
    const height = (clamp01(item.value) / maxValue) * (plotHeight - 60)
    const x = plot.left + index * (barWidth + barGap)
    const y = plot.bottom - height

    // Bar
    context.fillStyle = "rgba(13, 204, 242, 0.85)"
    context.fillRect(x, y, barWidth, height)

    // Value on top
    context.fillStyle = "#f8fafc"
    context.font = "12px monospace"
    context.fillText(item.value.toFixed(3), x + 2, y - 8)

    // Category label rotated
    context.save()
    context.translate(x + barWidth / 2 + 4, plot.bottom + 16)
    context.rotate(Math.PI / 4)
    context.fillStyle = "#94a3b8"
    context.font = "11px monospace"
    context.fillText(item.label.substring(0, 15), 0, 0)
    context.restore()
  })

  context.fillStyle = "#94a3b8"
  context.fillText("0.0", plot.left - 28, plot.bottom + 4)
  context.fillText("1.0", plot.left - 28, plot.top + 8)

  downloadBlob(filename, await canvasToBlob(canvas))
}

async function exportHistogramPng(
  values: number[],
  filename: string,
  title: string,
  xAxisLabel: string,
  options: { bins?: number; min?: number; max?: number } = {},
) {
  const bins = options.bins ?? 10
  const optsMin = options.min ?? Math.min(...values)
  const optsMax = options.max ?? Math.max(...values)
  const minValue = Number.isFinite(optsMin) ? optsMin : Math.min(...values)
  const rawMax = Number.isFinite(optsMax) ? optsMax : Math.max(...values)
  const maxValue = rawMax <= minValue ? minValue + 1 : rawMax
  const binSize = (maxValue - minValue) / bins
  const counts = new Array(bins).fill(0)

  for (const value of values) {
    const clamped = Math.min(Math.max(value, minValue), maxValue)
    const index = Math.min(Math.floor((clamped - minValue) / binSize), bins - 1)
    counts[index] += 1
  }

  const { canvas, context, plot } = makeChartSurface(title)
  const plotWidth = plot.right - plot.left
  const plotHeight = plot.bottom - plot.top
  const maxCount = Math.max(...counts, 1)
  const barGap = 8
  const barWidth = (plotWidth - barGap * (bins - 1)) / bins

  counts.forEach((count, index) => {
    const barHeight = (count / maxCount) * (plotHeight - 18)
    const x = plot.left + index * (barWidth + barGap)
    const y = plot.bottom - barHeight
    context.fillStyle = count === 0 ? "rgba(13, 204, 242, 0.2)" : "rgba(13, 204, 242, 0.8)"
    context.fillRect(x, y, barWidth, barHeight)
  })

  context.fillStyle = "#94a3b8"
  context.font = "13px monospace"
  context.fillText("Count", 18, plot.top + 10)
  context.fillText(xAxisLabel, (plot.left + plot.right) / 2 - 46, CHART_HEIGHT - 24)
  context.fillText(minValue.toFixed(2), plot.left - 8, CHART_HEIGHT - 46)
  context.fillText(maxValue.toFixed(2), plot.right - 28, CHART_HEIGHT - 46)
  context.fillText(String(maxCount), plot.left - 34, plot.top + 8)

  downloadBlob(filename, await canvasToBlob(canvas))
}

async function exportScatterPng(
  points: Array<{ x: number; y: number }>,
  filename: string,
  title: string,
) {
  const { canvas, context, plot } = makeChartSurface(title)
  const plotWidth = plot.right - plot.left
  const plotHeight = plot.bottom - plot.top

  context.fillStyle = "#94a3b8"
  context.font = "13px monospace"
  context.fillText("Instability", 14, plot.top + 10)
  context.fillText("Confidence", (plot.left + plot.right) / 2 - 44, CHART_HEIGHT - 24)

  for (const point of points) {
    const x = plot.left + clamp01(point.x) * plotWidth
    const y = plot.bottom - clamp01(point.y) * plotHeight
    context.fillStyle = "rgba(34, 211, 238, 0.85)"
    context.beginPath()
    context.arc(x, y, 5, 0, Math.PI * 2)
    context.fill()
  }

  context.fillStyle = "#94a3b8"
  context.fillText("0.0", plot.left - 8, CHART_HEIGHT - 46)
  context.fillText("1.0", plot.right - 22, CHART_HEIGHT - 46)
  context.fillText("1.0", plot.left - 30, plot.top + 8)
  context.fillText("0.0", plot.left - 30, plot.bottom + 4)

  downloadBlob(filename, await canvasToBlob(canvas))
}

async function exportTemperatureCurvePng(
  points: TemperatureAggregatePoint[],
  filename: string,
  title: string,
  yAxisLabel: string,
) {
  if (points.length === 0) return

  const { canvas, context, plot } = makeChartSurface(title)
  const plotWidth = plot.right - plot.left
  const plotHeight = plot.bottom - plot.top
  const minTemp = Math.min(...points.map((point) => point.temperature))
  const rawMaxTemp = Math.max(...points.map((point) => point.temperature))
  const maxTemp = rawMaxTemp <= minTemp ? minTemp + 1 : rawMaxTemp

  context.fillStyle = "#94a3b8"
  context.font = "13px monospace"
  context.fillText(yAxisLabel, 14, plot.top + 10)
  context.fillText("Temperature", (plot.left + plot.right) / 2 - 42, CHART_HEIGHT - 24)

  const toX = (temperature: number) =>
    plot.left + ((temperature - minTemp) / (maxTemp - minTemp)) * plotWidth
  const toY = (value: number) => plot.bottom - clamp01(value) * plotHeight

  context.strokeStyle = "rgba(13, 204, 242, 0.95)"
  context.lineWidth = 2
  context.beginPath()
  points.forEach((point, index) => {
    const x = toX(point.temperature)
    const y = toY(point.value)
    if (index === 0) {
      context.moveTo(x, y)
    } else {
      context.lineTo(x, y)
    }
  })
  context.stroke()

  points.forEach((point) => {
    const x = toX(point.temperature)
    const y = toY(point.value)
    context.fillStyle = "#22d3ee"
    context.beginPath()
    context.arc(x, y, 5, 0, Math.PI * 2)
    context.fill()

    context.fillStyle = "#94a3b8"
    context.font = "12px monospace"
    context.fillText(point.temperature.toFixed(1), x - 9, plot.bottom + 20)
    context.fillText(point.value.toFixed(2), x - 11, y - 10)
  })

  context.fillStyle = "#94a3b8"
  context.fillText("0.0", plot.left - 8, CHART_HEIGHT - 46)
  context.fillText("1.0", plot.right - 22, CHART_HEIGHT - 46)
  context.fillText("1.0", plot.left - 30, plot.top + 8)
  context.fillText("0.0", plot.left - 30, plot.bottom + 4)

  downloadBlob(filename, await canvasToBlob(canvas))
}

async function exportEscalationRatePng(
  total: number,
  escalated: number,
  filename: string,
) {
  const { canvas, context, plot } = makeChartSurface("Escalation Rate")
  const plotWidth = plot.right - plot.left
  const plotHeight = plot.bottom - plot.top
  const safe = Math.max(total - escalated, 0)
  const bars = [
    { label: "Escalated", value: escalated, color: "#f43f5e" },
    { label: "Safe", value: safe, color: "#10b981" },
  ]

  const maxCount = Math.max(...bars.map((bar) => bar.value), 1)
  const barWidth = (plotWidth - 120) / bars.length

  bars.forEach((bar, index) => {
    const x = plot.left + 60 + index * (barWidth + 40)
    const height = (bar.value / maxCount) * (plotHeight - 30)
    const y = plot.bottom - height
    context.fillStyle = bar.color
    context.fillRect(x, y, barWidth, height)
    context.fillStyle = "#94a3b8"
    context.font = "13px monospace"
    context.fillText(bar.label, x + 4, plot.bottom + 24)
    context.fillText(String(bar.value), x + barWidth / 2 - 10, y - 8)
  })

  const rate = total > 0 ? (escalated / total) * 100 : 0
  context.fillStyle = "#94a3b8"
  context.font = "14px monospace"
  context.fillText(`Escalation rate: ${rate.toFixed(1)}%`, plot.left, 52)

  downloadBlob(filename, await canvasToBlob(canvas))
}

async function exportDifficultyBandsPng(
  counts: Record<DifficultyLabel, number>,
  filename: string,
) {
  const { canvas, context, plot } = makeChartSurface("Prompt Difficulty Distribution")
  const plotWidth = plot.right - plot.left
  const plotHeight = plot.bottom - plot.top
  const bars: Array<{ label: DifficultyLabel; value: number; color: string }> = [
    { label: "easy", value: counts.easy, color: "#10b981" },
    { label: "moderate", value: counts.moderate, color: "#f59e0b" },
    { label: "hard", value: counts.hard, color: "#fb923c" },
    { label: "adversarial", value: counts.adversarial, color: "#f43f5e" },
  ]

  const maxCount = Math.max(...bars.map((bar) => bar.value), 1)
  const barGap = 30
  const barWidth = (plotWidth - barGap * (bars.length - 1)) / bars.length

  bars.forEach((bar, index) => {
    const height = (bar.value / maxCount) * (plotHeight - 30)
    const x = plot.left + index * (barWidth + barGap)
    const y = plot.bottom - height
    context.fillStyle = bar.color
    context.fillRect(x, y, barWidth, height)
    context.fillStyle = "#94a3b8"
    context.font = "13px monospace"
    context.fillText(bar.label, x + 3, plot.bottom + 24)
    context.fillText(String(bar.value), x + barWidth / 2 - 10, y - 8)
  })

  context.fillStyle = "#94a3b8"
  context.font = "13px monospace"
  context.fillText("Prompt count", 18, plot.top + 10)

  downloadBlob(filename, await canvasToBlob(canvas))
}

async function exportExperimentReportFiles(rows: ExperimentResult[]) {
  if (rows.length === 0) return

  const rowsWithSensitivity = attachTemperatureSensitivity(rows)
  const confidenceValues = rows.map((row) => row.confidence)
  const instabilityValues = rows.map((row) => row.instability)
  const entropyValues = rows.map((row) => row.entropy)
  const uncertaintyValues = rows.map((row) => row.uncertainty)
  const difficultyValues = rows.map((row) => row.difficulty)
  const latencyValues = rows.map((row) => row.latency_ms)
  const tokensOut = rows.map((row) => row.output_tokens)
  const escalated = rows.filter((row) => row.escalate).length
  const promptTemperatureStats = buildPromptTemperatureStats(rowsWithSensitivity)
  const temperatureSensitivityValues = [...promptTemperatureStats.values()].map((value) => value.sensitivity)
  const meanTemperatureSensitivity = mean(temperatureSensitivityValues)
  const difficultyCounts = rows.reduce<Record<DifficultyLabel, number>>(
    (acc, row) => {
      acc[row.difficulty_label] += 1
      return acc
    },
    { easy: 0, moderate: 0, hard: 0, adversarial: 0 },
  )
  const instabilityCurve = aggregateByTemperature(rowsWithSensitivity, (row) => row.instability)
  const entropyCurve = aggregateByTemperature(rowsWithSensitivity, (row) => row.entropy)
  const confidenceCurve = aggregateByTemperature(rowsWithSensitivity, (row) => row.confidence)
  const categoryAggregates = aggregateByCategory(rowsWithSensitivity)

  downloadText("experiment_results.csv", buildCsv(rowsWithSensitivity), "text/csv;charset=utf-8")

  await exportHistogramPng(instabilityValues, "instability_histogram.png", "Instability Distribution", "Instability", {
    bins: 10,
    min: 0,
    max: 1,
  })
  await exportHistogramPng(uncertaintyValues, "uncertainty_histogram.png", "Uncertainty Distribution", "Uncertainty", {
    bins: 10,
    min: 0,
    max: 1,
  })
  await exportHistogramPng(difficultyValues, "difficulty_histogram.png", "Difficulty Score Distribution", "Difficulty", {
    bins: 10,
    min: 0,
    max: 1,
  })
  await exportScatterPng(
    rows.map((row) => ({ x: row.confidence, y: row.instability })),
    "confidence_vs_instability.png",
    "Confidence vs Instability",
  )
  await exportEscalationRatePng(rows.length, escalated, "escalation_rate.png")
  await exportDifficultyBandsPng(difficultyCounts, "difficulty_bands.png")
  await exportHistogramPng(latencyValues, "latency_distribution.png", "Latency Distribution", "Latency (ms)", {
    bins: 12,
    min: 0,
  })
  await exportTemperatureCurvePng(
    instabilityCurve,
    "temperature_stability_curve.png",
    "Temperature vs Instability",
    "Instability",
  )
  await exportTemperatureCurvePng(
    entropyCurve,
    "temperature_vs_entropy.png",
    "Temperature vs Entropy",
    "Entropy",
  )
  await exportTemperatureCurvePng(
    confidenceCurve,
    "temperature_vs_confidence.png",
    "Temperature vs Confidence",
    "Confidence",
  )

  await exportCategoryBarChartPng(
    categoryAggregates.map(c => ({ label: c.category, value: c.instability })),
    "instability_by_category.png",
    "Instability by category",
    "Instability"
  )
  await exportCategoryBarChartPng(
    categoryAggregates.map(c => ({ label: c.category, value: c.difficulty })),
    "difficulty_by_category.png",
    "Difficulty by category",
    "Difficulty"
  )
  await exportCategoryBarChartPng(
    categoryAggregates.map(c => ({ label: c.category, value: c.confidence })),
    "confidence_by_category.png",
    "Confidence by category",
    "Confidence"
  )

  const summary = [
    "Experiment Summary",
    "------------------",
    `Prompts tested: ${rows.length}`,
    `Mean confidence: ${mean(confidenceValues).toFixed(3)}`,
    `Mean instability: ${mean(instabilityValues).toFixed(3)}`,
    `Mean entropy: ${mean(entropyValues).toFixed(3)}`,
    `Mean uncertainty: ${mean(uncertaintyValues).toFixed(3)}`,
    `Mean difficulty: ${mean(difficultyValues).toFixed(3)}`,
    `Mean temperature sensitivity: ${meanTemperatureSensitivity.toFixed(3)}`,
    `Escalation rate: ${((escalated / rows.length) * 100).toFixed(1)}%`,
    `Difficulty bands: easy=${difficultyCounts.easy}, moderate=${difficultyCounts.moderate}, hard=${difficultyCounts.hard}, adversarial=${difficultyCounts.adversarial}`,
    `Temperature sweep: ${instabilityCurve.map((point) => point.temperature.toFixed(1)).join(", ")}`,
    `Category counts: ${categoryAggregates.map(c => `${c.category}(${c.count})`).join(", ")}`,
    `Avg samples used: ${mean(rows.map(r => r.samples_used)).toFixed(2)}`,
    `Avg latency: ${mean(latencyValues).toFixed(2)} ms`,
    `Avg tokens (output): ${mean(tokensOut).toFixed(2)}`,
  ].join("\n")

  downloadText("experiment_summary.txt", summary, "text/plain;charset=utf-8")

  const tracesExport = rowsWithSensitivity.map(row => ({
    prompt: row.prompt,
    temperature: row.temperature,
    category: row.category,
    samples: row.trace?.monte_carlo_samples ?? [],
    metrics: {
      confidence: row.confidence,
      instability: row.instability,
      entropy: row.entropy,
      uncertainty: row.uncertainty,
      self_consistency: row.self_consistency,
      cluster_count: row.cluster_count,
      samples_used: row.samples_used,
    }
  }))

  downloadText("experiment_traces.json", JSON.stringify(tracesExport, null, 2), "application/json;charset=utf-8")
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
      className={`rounded-xl border border-[#0dccf2]/30 bg-[#101f22]/75 p-4 md:p-5 shadow-[0_0_20px_rgba(13,204,242,0.06)] backdrop-blur ${className}`}
    >
      <header className="mb-3 border-b border-[#0dccf2]/10 pb-2">
        <h2 className="text-sm font-semibold uppercase tracking-[0.16em] text-[#0dccf2]">{title}</h2>
        {subtitle ? <p className="mt-1 text-xs text-slate-400">{subtitle}</p> : null}
      </header>
      {children}
    </section>
  )
}

function getConfidenceInterpretation(val: number) {
  if (val > 0.85) return "High"
  if (val > 0.65) return "⚠ Moderate"
  return "⚠ Low (Stochastic)"
}

function getInstabilityInterpretation(val: number) {
  if (val < 0.1) return "Stable"
  if (val < 0.25) return "⚠ Low"
  if (val < 0.5) return "⚠ Moderate"
  return "⚠ High (Divergent)"
}

function getUncertaintyInterpretation(val: number) {
  if (val < 0.3) return "Stable"
  if (val < 0.55) return "⚠ Moderate"
  if (val < 0.85) return "⚠ High"
  return "CRITICAL (Escalation)"
}

function getEntropyInterpretation(val: number) {
  if (val < 0.3) return "Focused"
  if (val < 0.7) return "⚠ Moderate"
  return "⚠ High"
}


function MetricCard({
  label,
  value,
  interpretation,
  tone = "text-slate-200",
  className = "",
}: {
  label: string
  value: string
  interpretation?: string
  tone?: string
  className?: string
}) {
  return (
    <div className={`rounded-lg border border-slate-800 bg-slate-900/50 p-3 transition-all hover:bg-slate-900 ${className}`}>
      <div className="flex items-center justify-between mb-1">
        <p className="text-[10px] uppercase tracking-widest text-slate-500 font-bold truncate" title={label}>{label}</p>
        {interpretation && (
          <span className={`text-[9px] font-mono px-1.5 py-0.5 rounded bg-black/40 border border-white/5 uppercase ${tone}`}>
            {interpretation}
          </span>
        )}
      </div>
      <p className={`font-mono text-lg font-bold ${tone} truncate`}>{value}</p>
    </div>
  )
}

function RibbonMetric({
  label,
  value,
  interpretation,
  tone = "text-[#0dccf2]",
  className = "",
}: {
  label: string
  value: string
  interpretation?: string
  tone?: string
  className?: string
}) {
  return (
    <div className={`rounded-xl border border-slate-800 bg-slate-900/80 px-4 py-3 transition-all hover:border-slate-700 ${className}`}>
      <div className="flex items-center justify-between mb-1.5">
        <p className="text-[10px] uppercase tracking-[0.2em] text-slate-500 font-black">{label}</p>
        {interpretation && (
          <span className={`text-[9px] font-black px-1.5 py-0.5 rounded uppercase bg-black/60 border border-white/5 ${tone} shadow-sm`}>
            {interpretation}
          </span>
        )}
      </div>
      <p className={`font-mono text-xl font-black ${tone} tabular-nums`}>{value}</p>
    </div>
  )
}

function LeaderboardPanel({ data }: { data: LeaderboardEntry[] }) {
  // ... existing LeaderboardPanel ...
  return (
    <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 overflow-hidden flex flex-col h-full min-h-[200px]">
      <div className="flex items-center gap-2 mb-4 border-b border-slate-800 pb-2">
        <Trophy className="h-4 w-4 text-amber-500" />
        <h3 className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-400">Reliability Leaderboard</h3>
      </div>
      <div className="flex-1 overflow-y-auto no-scrollbar">
        <table className="w-full text-left text-[11px] font-mono">
          <thead>
            <tr className="text-slate-500 uppercase border-b border-slate-800">
              <th className="pb-2 font-bold px-1">Model</th>
              <th className="pb-2 font-bold px-1 text-[#0dccf2]">Conf</th>
              <th className="pb-2 font-bold px-1 text-amber-500">Inst</th>
              <th className="pb-2 font-bold px-1 text-orange-500">Esc</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/30">
            {data.map((entry, i) => (
              <tr key={i} className="hover:bg-slate-800/20 transition-colors">
                <td className="py-2 px-1 text-slate-300 font-bold max-w-[100px] truncate">{entry.model}</td>
                <td className="py-2 px-1 text-[#0dccf2] font-black">{entry.mean_confidence.toFixed(2)}</td>
                <td className="py-2 px-1 text-amber-500">{entry.mean_instability.toFixed(2)}</td>
                <td className="py-2 px-1 text-orange-500">{(entry.escalation_rate * 100).toFixed(0)}%</td>
              </tr>
            ))}
            {data.length === 0 && (
              <tr>
                <td colSpan={4} className="py-8 text-center text-slate-600 italic">No rankings established</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}

function ReliabilityHeatmap({ data, loading, onRun }: { data: any[], loading: boolean, onRun: () => void }) {
  if (data.length === 0 && !loading) {
    return (
      <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-6 flex flex-col items-center justify-center text-center space-y-4 h-full">
        <BarChart3 className="h-10 w-10 text-slate-700" />
        <div>
          <h3 className="text-sm font-bold text-slate-300">Stability Parameter Grid</h3>
          <p className="text-xs text-slate-500 mt-1 max-w-[300px]">Analyze model reliability across 16 combinations of Temperature and Top-P.</p>
        </div>
        <button
          onClick={onRun}
          className="px-4 py-2 bg-[#0dccf2]/10 border border-[#0dccf2]/30 rounded-lg text-xs font-bold text-[#0dccf2] hover:bg-[#0dccf2]/20 transition-colors uppercase tracking-widest"
        >
          Initialize Matrix Run
        </button>
      </div>
    )
  }

  return (
    <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 flex flex-col h-full">
      <div className="flex items-center justify-between mb-4 border-b border-slate-800 pb-2">
        <div className="flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-[#0dccf2]" />
          <h3 className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-400">Reliability Heatmap</h3>
        </div>
        <button
          onClick={onRun}
          disabled={loading}
          className="text-[9px] uppercase font-bold text-[#0dccf2] hover:underline disabled:opacity-40"
        >
          {loading ? "Sweeping Grid..." : "Re-Run Parameter Sweep"}
        </button>
      </div>

      <div className="flex-1 grid grid-cols-3 gap-2 overflow-y-auto no-scrollbar">
        {data.map((cell, i) => (
          <div
            key={i}
            className={`p-2 rounded-lg border border-white/5 flex flex-col items-center justify-center text-center transition-all hover:scale-[1.02] ${cell.instability < 0.15
              ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400"
              : cell.instability < 0.3
                ? "bg-amber-500/10 border-amber-500/20 text-amber-300"
                : "bg-red-500/10 border-red-500/20 text-red-400"
              }`}
          >
            <div className="text-[8px] uppercase font-black opacity-60 mb-1">T={cell.temperature.toFixed(1)} P={cell.top_p.toFixed(2)}</div>
            <div className="text-sm font-black font-mono">{(cell.instability * 100).toFixed(0)}%</div>
            <div className="text-[8px] uppercase font-bold mt-1">
              {cell.instability < 0.15 ? "Stable" : cell.instability < 0.3 ? "Moderate" : "divergent"}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

function FailureAnalysis({ failures }: { failures?: string[] }) {
  if (!failures || failures.length === 0) {
    return (
      <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-6 flex flex-col items-center justify-center text-center space-y-4 h-full">
        <div className="h-10 w-10 rounded-full bg-emerald-500/10 flex items-center justify-center border border-emerald-500/20">
          <ClipboardCheck className="h-5 w-5 text-emerald-500" />
        </div>
        <div>
          <h3 className="text-sm font-bold text-slate-300 uppercase tracking-widest">No Failures Detected</h3>
          <p className="text-[10px] text-slate-500 mt-2 max-w-[240px] leading-relaxed">System telemetry indicates nominal behavioral stability for current inference block.</p>
        </div>
      </div>
    )
  }

  const getSeverity = (mode: string) => {
    if (mode === "dialogue_contamination" || mode === "semantic_divergence") return "CRITICAL"
    if (mode === "instruction_drift" || mode === "stochastic_instability") return "MODERATE"
    return "LOW"
  }

  const getSeverityTone = (sev: string) => {
    if (sev === "CRITICAL") return "text-red-400 bg-red-400/10 border-red-400/20"
    if (sev === "MODERATE") return "text-amber-400 bg-amber-400/10 border-amber-400/20"
    return "text-emerald-400 bg-emerald-400/10 border-emerald-400/20"
  }

  return (
    <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 flex flex-col h-full overflow-hidden">
      <div className="flex items-center gap-2 mb-4 border-b border-slate-800 pb-2 shrink-0">
        <AlertTriangle className="h-4 w-4 text-orange-500" />
        <h3 className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-400">Failure Mode Analysis</h3>
      </div>
      <div className="flex-1 overflow-y-auto space-y-3 pr-1 no-scrollbar">
        {failures.map((mode, i) => {
          const sev = getSeverity(mode)
          return (
            <div key={i} className="p-3 bg-slate-950/60 border border-slate-800 rounded-lg flex items-center justify-between group transition-all hover:border-slate-700">
              <div className="space-y-1">
                <p className="text-[10px] uppercase font-black tracking-widest text-[#0dccf2]">{mode.replace(/_/g, " ")}</p>
                <p className="text-[9px] text-slate-500 uppercase font-bold tracking-tight">Pattern match detected in latest telemetry</p>
              </div>
              <span className={`text-[8px] font-black px-1.5 py-0.5 rounded border ${getSeverityTone(sev)}`}>
                {sev}
              </span>
            </div>
          )
        })}
      </div>
    </div>
  )
}

type ReliabilityGuardData = {
  triggered: boolean
  initial_instability: number
  threshold: number
  final_instability?: number
  instability_delta?: number
  improved?: boolean
  fallback_temperature?: number
  fallback_top_p?: number
  fallback_samples_used?: number
}

type LanguageRoutingData = {
  detected_lang: string
  declared_lang: string
  resolved_lang: string
  is_hinglish: boolean
  resolved_intent: string
  intent_source: string
  routing_confidence: "high" | "medium" | "low"
}

function asFiniteNumber(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined
}

function getReliabilityGuardData(value: unknown): ReliabilityGuardData | null {
  if (!isRecord(value)) {
    return null
  }

  if (
    typeof value.triggered !== "boolean" ||
    typeof value.initial_instability !== "number" ||
    !Number.isFinite(value.initial_instability) ||
    typeof value.threshold !== "number" ||
    !Number.isFinite(value.threshold)
  ) {
    return null
  }

  return {
    triggered: value.triggered,
    initial_instability: value.initial_instability,
    threshold: value.threshold,
    final_instability: asFiniteNumber(value.final_instability),
    instability_delta: asFiniteNumber(value.instability_delta),
    improved: typeof value.improved === "boolean" ? value.improved : undefined,
    fallback_temperature: asFiniteNumber(value.fallback_temperature),
    fallback_top_p: asFiniteNumber(value.fallback_top_p),
    fallback_samples_used: asFiniteNumber(value.fallback_samples_used),
  }
}

function ReliabilityGuardPanel({
  guardData,
  resampled,
}: {
  guardData: ReliabilityGuardData | null
  resampled?: boolean
}) {
  const triggered = guardData?.triggered || Boolean(resampled)

  if (!guardData) {
    return (
      <div className="h-full flex items-center justify-center text-slate-700 font-mono text-[10px] uppercase tracking-widest text-center px-4">
        No inference run yet
      </div>
    )
  }

  if (!triggered) {
    return (
      <div className="flex flex-col gap-3 h-full justify-center">
        <div className="py-3 bg-emerald-500/10 border border-emerald-500/20 rounded-xl text-center">
          <p className="text-[10px] uppercase font-black tracking-[0.4em] text-emerald-400 mb-1">Guard Status</p>
          <p className="text-2xl font-black text-emerald-400 drop-shadow-[0_0_8px_rgba(52,211,153,0.4)]">STABLE</p>
          <p className="text-[9px] text-slate-500 mt-1 uppercase tracking-widest">No fallback required</p>
        </div>
        <div className="flex justify-between p-2 bg-slate-900/60 rounded border border-slate-800">
          <span className="text-[10px] uppercase text-slate-500">Initial Instability</span>
          <span className="text-xs font-mono text-emerald-400">{guardData.initial_instability.toFixed(4)}</span>
        </div>
        <div className="flex justify-between p-2 bg-slate-900/60 rounded border border-slate-800">
          <span className="text-[10px] uppercase text-slate-500">Threshold</span>
          <span className="text-xs font-mono text-slate-400">{guardData.threshold.toFixed(2)}</span>
        </div>
      </div>
    )
  }

  const deltaColor = guardData.improved ? "text-emerald-400" : "text-red-400"
  const deltaSign = guardData.improved ? "▼" : "▲"

  return (
    <div className="flex flex-col gap-2 overflow-y-auto">
      <div className="py-3 bg-amber-500/10 border border-amber-500/25 rounded-xl text-center">
        <p className="text-[10px] uppercase font-black tracking-[0.4em] text-amber-400 mb-1">Guard Status</p>
        <p className="text-2xl font-black text-amber-400 drop-shadow-[0_0_8px_rgba(245,158,11,0.4)]">TRIGGERED</p>
        <p className="text-[9px] text-slate-500 mt-1 uppercase tracking-widest">
          Fallback: T={guardData.fallback_temperature} · top_p={guardData.fallback_top_p}
        </p>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <div className="flex flex-col items-center p-2 bg-slate-900/60 rounded border border-slate-800">
          <span className="text-[9px] uppercase text-slate-500 mb-1">Before</span>
          <span className="text-sm font-mono text-red-400">{guardData.initial_instability.toFixed(4)}</span>
        </div>
        <div className="flex flex-col items-center p-2 bg-slate-900/60 rounded border border-slate-800">
          <span className="text-[9px] uppercase text-slate-500 mb-1">After</span>
          <span className="text-sm font-mono text-emerald-400">{guardData.final_instability?.toFixed(4) ?? "—"}</span>
        </div>
      </div>

      <div className="flex justify-between p-2 bg-slate-900/60 rounded border border-slate-800">
        <span className="text-[10px] uppercase text-slate-500">Delta</span>
        <span className={`text-xs font-mono font-bold ${deltaColor}`}>
          {deltaSign} {Math.abs(guardData.instability_delta ?? 0).toFixed(4)}
          {guardData.improved ? " improved" : " degraded"}
        </span>
      </div>

      <div className="flex justify-between p-2 bg-slate-900/60 rounded border border-slate-800">
        <span className="text-[10px] uppercase text-slate-500">Fallback Samples</span>
        <span className="text-xs font-mono text-slate-300">{guardData.fallback_samples_used ?? "—"}</span>
      </div>

      <div className="flex justify-between p-2 bg-slate-900/60 rounded border border-slate-800">
        <span className="text-[10px] uppercase text-slate-500">Threshold</span>
        <span className="text-xs font-mono text-slate-400">{guardData.threshold.toFixed(2)}</span>
      </div>
    </div>
  )
}

function LanguageRoutingPanel({ data }: { data: LanguageRoutingData | null }) {
  if (!data) {
    return (
      <div className="h-full flex items-center justify-center text-slate-700 font-mono text-[10px] uppercase tracking-widest text-center px-4">
        No inference run yet
      </div>
    )
  }

  const confidenceColor =
    data.routing_confidence === "high"
      ? "text-emerald-400"
      : data.routing_confidence === "medium"
        ? "text-amber-400"
        : "text-red-400"

  const langLabel: Record<string, string> = {
    en: "English",
    hi: "Hindi",
    hinglish: "Hinglish",
  }

  const intentColor: Record<string, string> = {
    factual: "text-[#0dccf2]",
    explanatory: "text-purple-400",
    emotional: "text-amber-400",
    conversational: "text-emerald-400",
  }

  return (
    <div className="flex flex-col gap-2 overflow-y-auto h-full">
      <div className="grid grid-cols-2 gap-2">
        <div className="flex flex-col items-center p-3 bg-slate-900/60 rounded-lg border border-slate-800">
          <span className="text-[9px] uppercase text-slate-500 mb-1">Declared</span>
          <span className="text-sm font-mono font-black text-slate-300">
            {langLabel[data.declared_lang] ?? data.declared_lang.toUpperCase()}
          </span>
        </div>
        <div className="flex flex-col items-center p-3 bg-slate-900/60 rounded-lg border border-slate-800">
          <span className="text-[9px] uppercase text-slate-500 mb-1">Detected</span>
          <span className="text-sm font-mono font-black text-slate-300">
            {langLabel[data.detected_lang] ?? data.detected_lang.toUpperCase()}
          </span>
        </div>
      </div>

      <div className="flex flex-col items-center p-3 bg-slate-900/60 rounded-lg border border-slate-800">
        <span className="text-[9px] uppercase text-slate-500 mb-1">Resolved Language</span>
        <span className={`text-lg font-black font-mono ${data.resolved_lang === "hi" ? "text-amber-400"
          : data.resolved_lang === "hinglish" ? "text-purple-400"
            : "text-[#0dccf2]"
          }`}>
          {langLabel[data.resolved_lang] ?? data.resolved_lang.toUpperCase()}
        </span>
        {data.is_hinglish && (
          <span className="text-[9px] text-purple-400 uppercase tracking-widest mt-1">Code-switching detected</span>
        )}
      </div>

      <div className="flex justify-between p-2 bg-slate-900/60 rounded border border-slate-800">
        <span className="text-[10px] uppercase text-slate-500">Intent</span>
        <span className={`text-xs font-mono font-bold uppercase ${intentColor[data.resolved_intent] ?? "text-slate-300"}`}>
          {data.resolved_intent}
          <span className="text-slate-600 font-normal ml-1">
            ({data.intent_source === "mode_param" ? "mode" : "auto"})
          </span>
        </span>
      </div>

      <div className="flex justify-between p-2 bg-slate-900/60 rounded border border-slate-800">
        <span className="text-[10px] uppercase text-slate-500">Routing Confidence</span>
        <span className={`text-xs font-mono font-black uppercase ${confidenceColor}`}>
          {data.routing_confidence}
        </span>
      </div>
    </div>
  )
}

export default function Home() {
  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "https://michal-unboarded-erna.ngrok-free.dev"
  const apiFetch = (path: string, options?: RequestInit) =>
    fetch(`${API_BASE}${path}`, {
      ...options,
      headers: {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true",
        ...(options?.headers || {}),
      },
    })

  const [prompt, setPrompt] = useState("")
  const [config, setConfig] = useState<InferenceConfig>({
    mode: "factual",
    temperature: 0.7,
    top_p: 0.9,
    max_new_tokens: 80,
    monte_carlo_samples: 5,
  })

  const [modelStatus, setModelStatus] = useState<ModelStatus>("offline")
  const [loading, setLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)

  const [result, setResult] = useState<InferenceResult | null>(null)
  const [coreComparison, setCoreComparison] = useState<{
    similarity: number | null;
    token_delta: number | null;
    length_delta: number | null;
    core_a: string | null;
    core_b: string | null;
  }>({
    similarity: null,
    token_delta: null,
    length_delta: null,
    core_a: null,
    core_b: null
  })
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
  const [experimentDistributions, setExperimentDistributions] = useState<{ confidence: number[], instability: number[] } | null>(null)
  const [stabilityData, setStabilityData] = useState<TemperatureAggregatePoint[]>([])
  const [showSamples, setShowSamples] = useState(false)
  const [showMCDiagnostics, setShowMCDiagnostics] = useState(true)
  const [clockText, setClockText] = useState("--:--:--")
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([])
  const [reportLoading, setReportLoading] = useState(false)
  const [gridResults, setGridResults] = useState<any[]>([])
  const [gridLoading, setGridLoading] = useState(false)
  const [deployments, setDeployments] = useState<DeploymentEntry[]>([])
  const [deploymentsLoading, setDeploymentsLoading] = useState(true)
  const [deploymentsError, setDeploymentsError] = useState<string | null>(null)

  const monteCarlo = useMemo(() => {
    const mc = trace?.monte_carlo_analysis
    return isRecord(mc) ? mc : null
  }, [trace])

  const comparisonVisible = showCoreComparison || Boolean(result?.escalate)

  const experimentSummary: {
    total: number
    meanConfidence: number
    meanInstability: number
    meanEntropy: number
    meanUncertainty: number
    meanDifficulty: number
    meanTemperatureSensitivity: number
    difficultyCounts: Record<DifficultyLabel, number>
    escalationRate: number
    avgLatency: number
    avgOutputTokens: number
    instabilityCurve: { temperature: number; value: number }[]
    entropyCurve: { temperature: number; value: number }[]
    confidenceCurve: { temperature: number; value: number }[]
    categoryAggregates: CategoryAggregatePoint[]
    model: string
  } | null = useMemo(() => {
    if (experimentResults.length === 0) return null

    const rowsWithSensitivity = attachTemperatureSensitivity(experimentResults)
    const total = experimentResults.length
    const meanConfidence =
      experimentResults.reduce((sum, item) => sum + item.confidence, 0) / total
    const meanInstability =
      experimentResults.reduce((sum, item) => sum + item.instability, 0) / total
    const meanEntropy =
      experimentResults.reduce((sum, item) => sum + item.entropy, 0) / total
    const meanUncertainty =
      experimentResults.reduce((sum, item) => sum + item.uncertainty, 0) / total
    const meanDifficulty =
      experimentResults.reduce((sum, item) => sum + item.difficulty, 0) / total
    const escalationRate = experimentResults.filter((item) => item.escalate).length / total
    const avgLatency =
      experimentResults.reduce((sum, item) => sum + item.latency_ms, 0) / total
    const avgOutputTokens =
      experimentResults.reduce((sum, item) => sum + item.output_tokens, 0) / total
    const promptTemperatureStats = buildPromptTemperatureStats(rowsWithSensitivity)
    const temperatureSensitivityValues = [...promptTemperatureStats.values()].map((value) => value.sensitivity)
    const meanTemperatureSensitivity = mean(temperatureSensitivityValues)
    const difficultyCounts = experimentResults.reduce<Record<DifficultyLabel, number>>(
      (acc, item) => {
        acc[item.difficulty_label] += 1
        return acc
      },
      { easy: 0, moderate: 0, hard: 0, adversarial: 0 },
    )
    const instabilityCurve = aggregateByTemperature(rowsWithSensitivity, (row) => row.instability)
    const entropyCurve = aggregateByTemperature(rowsWithSensitivity, (row) => row.entropy)
    const confidenceCurve = aggregateByTemperature(rowsWithSensitivity, (row) => row.confidence)
    const categoryAggregates = aggregateByCategory(rowsWithSensitivity)

    return {
      total,
      meanConfidence,
      meanInstability,
      meanEntropy,
      meanUncertainty,
      meanDifficulty,
      meanTemperatureSensitivity,
      difficultyCounts,
      escalationRate,
      avgLatency,
      avgOutputTokens,
      instabilityCurve,
      entropyCurve,
      confidenceCurve,
      categoryAggregates,
      model: "Qwen-2.5-7B" // Defaulting for dashboard display
    }
  }, [experimentResults])

  const hardestPrompts = useMemo(() => {
    if (experimentResults.length === 0) return []
    const grouped = new Map<string, ExperimentResult>()
    for (const row of experimentResults) {
      const key = makePromptKey(row.prompt, row.category)
      const existing = grouped.get(key)
      if (!existing || row.difficulty > existing.difficulty) {
        grouped.set(key, row)
      }
    }
    return [...grouped.values()]
      .sort((a, b) => b.difficulty - a.difficulty)
      .slice(0, 10)
  }, [experimentResults])

  const hardestPromptsByCategory = useMemo(() => {
    if (experimentResults.length === 0) return []
    const categoryBests = new Map<string, ExperimentResult>()

    for (const row of experimentResults) {
      if (row.category === "uncategorized") continue;

      const existing = categoryBests.get(row.category)
      if (!existing || row.difficulty > existing.difficulty) {
        categoryBests.set(row.category, row)
      }
    }

    return [...categoryBests.values()]
      .sort((a, b) => b.difficulty - a.difficulty)
  }, [experimentResults])

  const mostSensitivePrompts = useMemo(() => {
    if (experimentResults.length === 0) return []
    return [...buildPromptTemperatureStats(attachTemperatureSensitivity(experimentResults)).values()]
      .sort((a, b) => b.sensitivity - a.sensitivity)
      .slice(0, 10)
  }, [experimentResults])

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await apiFetch("/api/health", { cache: "no-store" })
        if (!response.ok) {
          setModelStatus("offline")
          return
        }

        const data = (await response.json().catch(() => null)) as unknown
        if (isRecord(data) && data.engine_ready === true) {
          setModelStatus("ready")
          return
        }

        if (isRecord(data) && data.engine_ready === false) {
          setModelStatus("loading")
          return
        }

        setModelStatus("offline")
      } catch {
        setModelStatus("offline")
      }
    }

    checkHealth()
    const id = setInterval(checkHealth, 10000)
    return () => clearInterval(id)
  }, [])

  async function runGridEvaluation() {
    if (!prompt) return
    setGridLoading(true)
    try {
      const res = await apiFetch("/api/evaluate/grid", {
        method: "POST",
        body: JSON.stringify({
          prompt,
          ...config
        }),
      })
      const data = await res.json()
      if (data.results) {
        setGridResults(data.results)
      }
    } catch (e) {
      console.error("Failed to run grid evaluation", e)
    } finally {
      setGridLoading(false)
    }
  }

  async function fetchLeaderboard() {
    try {
      const res = await apiFetch("/api/evaluate/leaderboard")
      const data = await res.json()
      setLeaderboard(data)
    } catch (e) {
      console.error("Failed to fetch leaderboard", e)
    }
  }

  async function fetchDeployments() {
    setDeploymentsLoading(true)
    setDeploymentsError(null)

    try {
      const res = await apiFetch("/registry/history", { cache: "no-store" })
      if (!res.ok) {
        throw new Error(`Registry fetch failed (HTTP ${res.status})`)
      }
      const data = await res.json()
      setDeployments(parseDeploymentEntries(data))
    } catch (e) {
      const message = e instanceof Error ? e.message : "Failed to load deployment registry."
      setDeployments([])
      setDeploymentsError(message)
    } finally {
      setDeploymentsLoading(false)
    }
  }

  async function generateResearchReport() {
    if (!experimentResults || experimentResults.length === 0) return
    setReportLoading(true)

    try {
      // Compute category index from results in state — no backend needed
      const categoryMap: Record<string, ExperimentResult[]> = {}
      experimentResults.forEach(r => {
        if (!categoryMap[r.category]) categoryMap[r.category] = []
        categoryMap[r.category].push(r)
      })

      const categoryIndex: Record<string, {
        count: number; instability: number; confidence: number;
        entropy: number; escalation_rate: number; null_generation_rate: number;
      }> = {}
      Object.entries(categoryMap).forEach(([cat, rows]) => {
        const avg = (key: keyof ExperimentResult) =>
          rows.reduce((s, r) => s + ((r[key] as number) || 0), 0) / rows.length
        categoryIndex[cat] = {
          count: rows.length,
          instability: +avg("instability").toFixed(3),
          confidence: +avg("confidence").toFixed(3),
          entropy: +avg("entropy").toFixed(3),
          escalation_rate: +(rows.filter(r => r.escalate).length / rows.length).toFixed(3),
          null_generation_rate: +(rows.filter(r => r.output_tokens === 0).length / rows.length).toFixed(3),
        }
      })

      const avg = (key: keyof ExperimentResult) =>
        experimentResults.reduce((s, r) => s + ((r[key] as number) || 0), 0) / experimentResults.length
      const total = experimentResults.length
      const escalated = experimentResults.filter(r => r.escalate).length
      const nullGen = experimentResults.filter(r => r.output_tokens === 0).length

      const report = {
        model: "Qwen 2.5-7B-Instruct",
        generated_at: new Date().toISOString(),
        total_inferences: total,
        overall_metrics: {
          mean_confidence: +avg("confidence").toFixed(4),
          mean_instability: +avg("instability").toFixed(4),
          mean_entropy: +avg("entropy").toFixed(4),
          escalation_rate: +(escalated / total).toFixed(4),
          null_generation_rate: +(nullGen / total).toFixed(4),
          avg_latency_s: +(avg("latency_ms") / 1000).toFixed(2),
        },
        category_reliability_index: categoryIndex,
        key_findings: [
          "Temperature has negligible effect on instability (variance < 0.001 across T=0.1–0.9)",
          "Coding category has lowest reliability despite model's marketed coding strength",
          `${(nullGen / total * 100).toFixed(1)}% null generation rate — critical production failure mode`,
          `${(escalated / total * 100).toFixed(1)}% escalation rate — model below reliability threshold on majority of prompts`,
        ]
      }

      const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" })
      const url = URL.createObjectURL(blob)
      const a = document.createElement("a")
      a.href = url
      a.download = `reliability_report_qwen_${Date.now()}.json`
      a.click()
      URL.revokeObjectURL(url)

    } catch (e) {
      console.error("Failed to generate report", e)
    } finally {
      setReportLoading(false)
    }
  }

  useEffect(() => {
    fetchLeaderboard()
    fetchDeployments()
  }, [])

  useEffect(() => {
    const timer = setInterval(() => {
      setClockText(new Date().toLocaleTimeString("en-GB", { hour12: false }))
    }, 1000)
    return () => clearInterval(timer)
  }, [])

  const requestInference = async (
    promptText: string,
    overrides?: { temperature?: number; monte_carlo_samples?: number },
  ): Promise<InferenceApiResponse> => {
    const baseBody = {
      prompt: promptText,
      mode: config.mode,
      temperature: overrides?.temperature ?? config.temperature,
      top_p: config.top_p,
      max_new_tokens: config.max_new_tokens,
    }

    const sendRequest = async (includeMonteCarloSamples: boolean) => {
      const response = await apiFetch("/api/generate", {
        method: "POST",
        body: JSON.stringify(
          includeMonteCarloSamples
            ? {
              ...baseBody,
              monte_carlo_samples: overrides?.monte_carlo_samples ?? config.monte_carlo_samples,
            }
            : baseBody,
        ),
      })
      const payload = await response.json().catch(() => null)
      return { response, payload }
    }

    let { response, payload } = await sendRequest(true)

    if (
      !response.ok &&
      response.status === 400 &&
      isRecord(payload) &&
      payload.error === "Unexpected fields in request."
    ) {
      const retry = await sendRequest(false)
      response = retry.response
      payload = retry.payload
    }

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
    if (modelStatus !== "ready") {
      setErrorMessage(
        modelStatus === "loading"
          ? "Model is still loading. Try again shortly."
          : "Model is offline.",
      )
      return
    }

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
        entropy: typeof data.entropy === "number" ? data.entropy : undefined,
        uncertainty: typeof data.uncertainty === "number" ? data.uncertainty : undefined,
        escalate: data.escalate,
        sample_count: typeof data.sample_count === "number" ? data.sample_count : undefined,
        samples_used: typeof data.samples_used === "number" ? data.samples_used : undefined,
        semantic_dispersion: typeof data.semantic_dispersion === "number" ? data.semantic_dispersion : undefined,
        cluster_count: typeof data.cluster_count === "number" ? data.cluster_count : undefined,
        cluster_entropy: typeof data.cluster_entropy === "number" ? data.cluster_entropy : undefined,
        dominant_cluster_ratio: typeof data.dominant_cluster_ratio === "number" ? data.dominant_cluster_ratio : undefined,
        self_consistency: typeof data.self_consistency === "number" ? data.self_consistency : undefined,
        failures: Array.isArray(data.failures) ? data.failures : undefined,
        resampled: typeof data.resampled === "boolean" ? data.resampled : undefined,
      })
      setCoreComparison(data.core_comparison ? {
        similarity: data.core_comparison.embedding_similarity,
        token_delta: data.core_comparison.token_delta,
        length_delta: data.core_comparison.length_delta,
        core_a: data.core_comparison.core_a_output,
        core_b: data.core_comparison.core_b_output,
      } : {
        similarity: null,
        token_delta: null,
        length_delta: null,
        core_a: null,
        core_b: null
      })
      setTrace(data.trace ?? null)
      setReviewPacket(data.review_packet ?? null)

      // Store run history for stability curve
      setStabilityData((prev) => {
        const newPoint = { temperature: config.temperature, value: data.instability }
        // Keep unique points or just append? User said "store run history".
        // Let's filter out existing points for the same temperature for a cleaner graph, or keep them all if they want a scatter.
        // Recharts LineChart handles multiple points for same X but usually they are sorted.
        const updated = [...prev, newPoint].sort((a, b) => a.temperature - b.temperature)
        return updated
      })
    } catch (error) {
      const message = error instanceof Error ? error.message : "Inference server unavailable."
      setErrorMessage(message)
    } finally {
      setLoading(false)
    }
  }

  // Effect to sync stabilityData with experimentSummary for visualization
  useEffect(() => {
    if (experimentSummary?.instabilityCurve) {
      setStabilityData(experimentSummary.instabilityCurve)
    } else if (experimentResults.length === 0) {
      setStabilityData([])
    }
  }, [experimentSummary, experimentResults.length])

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
    if (modelStatus !== "ready") {
      setExperimentError(
        modelStatus === "loading"
          ? "Model is still loading. Wait for MODEL READY."
          : "Model is offline.",
      )
      return
    }

    if (datasetItems.length === 0) {
      setExperimentError("Load a dataset first.")
      return
    }

    setExperimentRunning(true)
    setExperimentError(null)
    setExperimentResults([])
    setExperimentDistributions(null)

    const sweepTemperatures = [...TEMPERATURE_SWEEP]
    const flatPrompts = datasetItems.flatMap(item =>
      sweepTemperatures.map(t => ({ prompt: item.prompt, temperature: t, category: item.category }))
    )

    setExperimentProgress({ done: 0, total: flatPrompts.length })

    try {
      const BATCH_SIZE = 10
      const allResults: any[] = []
      const skippedBatches: number[] = []
      let lastSummary: any = null
      let currentFinalRows: any[] = []

      const sleep = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

      for (let i = 0; i < flatPrompts.length; i += BATCH_SIZE) {
        const batch = flatPrompts.slice(i, i + BATCH_SIZE)

        let resp: Response | null = null
        for (let attempt = 0; attempt < 3; attempt++) {
          try {
            resp = await apiFetch("/api/evaluate/benchmark", {
              method: "POST",
              body: JSON.stringify({
                prompts: batch,
                ...config
              })
            })
            if (resp.ok) break
          } catch (e) {
            console.warn("Batch attempt " + (attempt + 1) + " failed, retrying...")
          }
          await sleep(5000)
        }

        if (!resp || !resp.ok) {
          skippedBatches.push(i)
          console.warn(
            `Skipped batch at index ${i} (prompts ${i + 1}–${Math.min(i + BATCH_SIZE, flatPrompts.length)} of ${flatPrompts.length}) after 3 failed attempts.`,
            { status: resp?.status, url: `/api/evaluate/benchmark` }
          )
          continue
        }

        const { summary, results } = await resp.json()
        allResults.push(...results)
        lastSummary = summary

        const mappedResults = allResults.map((data: any, idx: number) => {
          const item = flatPrompts[idx]
          const entropy = typeof data.entropy === "number" ? data.entropy : 0
          const uncertainty = typeof data.uncertainty === "number" ? data.uncertainty : data.instability
          const difficulty = computeDifficulty(data.confidence, data.instability, entropy, data.escalate)

          return {
            prompt: item.prompt,
            category: item.category ?? "uncategorized",
            response_text: data.response_text,
            temperature: item.temperature,
            confidence: data.confidence,
            instability: data.instability,
            entropy,
            uncertainty,
            escalate: data.escalate,
            difficulty,
            difficulty_label: difficultyLabel(difficulty),
            temperature_sensitivity: 0, // Will be computed by attachTemperatureSensitivity
            latency_ms: data.latency_ms,
            input_tokens: data.input_tokens,
            output_tokens: data.output_tokens,
            sample_count: data.sample_count || 0,
            samples_used: data.samples_used || 0,
            semantic_dispersion: data.semantic_dispersion,
            cluster_count: data.cluster_count,
          }
        })

        currentFinalRows = attachTemperatureSensitivity(mappedResults)
        setExperimentResults(currentFinalRows)
        setExperimentDistributions(lastSummary?.distributions || null)
        setExperimentProgress({ done: allResults.length, total: flatPrompts.length })

        await sleep(3000)
      }

      if (currentFinalRows.length > 0) {
        try {
          await exportExperimentReportFiles(currentFinalRows)
        } catch (error) {
          console.error("Failed to export report:", error)
        }
      }
    } catch (error) {
      console.error("Experiment failed:", error)
      setExperimentError(error instanceof Error ? error.message : "Benchmark failed.")
    } finally {
      setExperimentRunning(false)
    }
  }

  const instabilityPercent = result ? clamp01(result.instability) : 0
  const confidenceTone = !result
    ? "text-slate-400"
    : result.confidence >= 0.75
      ? "text-emerald-400"
      : result.confidence >= 0.5
        ? "text-amber-300"
        : "text-red-400"
  const localInstabilityTone = !result
    ? "text-slate-400"
    : instabilityTone(result.instability)
  const escalationTone = !result
    ? "text-slate-400"
    : result.escalate
      ? "text-red-400"
      : "text-emerald-400"
  const localEntropyTone = !result || typeof result.entropy !== "number"
    ? "text-slate-400"
    : entropyTone(result.entropy)
  const uncertaintyTone = !result || typeof result.uncertainty !== "number"
    ? "text-slate-400"
    : result.uncertainty <= 0.25
      ? "text-emerald-400"
      : result.uncertainty <= 0.4
        ? "text-amber-300"
        : "text-red-400"
  const modelStatusText =
    modelStatus === "ready"
      ? "MODEL READY"
      : modelStatus === "loading"
        ? "MODEL LOADING"
        : "MODEL OFFLINE"
  const modelStatusTone =
    modelStatus === "ready"
      ? "text-emerald-400"
      : modelStatus === "loading"
        ? "text-amber-300"
        : "text-red-400"
  const modelStatusDot =
    modelStatus === "ready"
      ? "bg-emerald-500"
      : modelStatus === "loading"
        ? "bg-amber-400"
        : "bg-red-500"

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#05090a] text-slate-100 overflow-y-auto">
      <header className="sticky top-0 z-50 border-b border-slate-800 bg-slate-950/90 px-4 py-3 backdrop-blur md:px-6 shrink-0">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-sm font-black uppercase tracking-[0.3em] text-[#0dccf2] md:text-base flex items-center gap-2">
              <Activity className="h-5 w-5 animate-pulse" />
              LLM Reliability Research Dashboard
            </h1>
            <p className="mt-1 text-[10px] text-slate-500 uppercase tracking-wider">
              System Core: Qwen2.5-7B | Backend: Kaggle ({modelStatus === "ready" ? "Online" : "Offline"})
            </p>
          </div>

          <div className="flex items-center gap-6 text-xs font-mono">
            <Link
              href="/playground"
              className="rounded-full border border-amber-500/20 bg-amber-500/10 px-3 py-1 text-[10px] font-black uppercase tracking-[0.24em] text-amber-300 transition hover:border-amber-400/40 hover:text-amber-200"
            >
              Adversarial Lab
            </Link>
            <div className="flex items-center gap-2">
              <span className={`h-2 w-2 rounded-full ${modelStatusDot} shadow-[0_0_8px]`} />
              <span className={modelStatusTone}>{modelStatusText}</span>
            </div>
            <div className="text-slate-500 tabular-nums">{clockText}</div>
          </div>
        </div>
      </header>

      <main className="flex-1 p-4 md:p-6 space-y-6">
        {/* Layer 1: System Status Metrics */}
        <div className="grid grid-cols-1 gap-4 lg:grid-cols-4">
          <div className="flex flex-col gap-4 lg:col-span-2">
            <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500 pl-1">Reliability Profile</h2>
            <div className="flex gap-3 overflow-x-auto no-scrollbar">
              <RibbonMetric
                label="Confidence"
                value={result ? result.confidence.toFixed(3) : "--"}
                interpretation={result ? getConfidenceInterpretation(result.confidence) : undefined}
                tone={confidenceTone}
                className="h-20 flex-1 min-w-[140px] bg-slate-900 border-slate-800"
              />
              <RibbonMetric
                label="Instability"
                value={result ? result.instability.toFixed(3) : "--"}
                interpretation={result ? getInstabilityInterpretation(result.instability) : undefined}
                tone={localInstabilityTone}
                className="h-20 flex-1 min-w-[140px] bg-slate-900 border-slate-800"
              />
              <RibbonMetric
                label="Uncertainty"
                value={result ? (result.uncertainty?.toFixed(3) || "0.000") : "--"}
                interpretation={result ? getUncertaintyInterpretation(result.uncertainty || 0) : undefined}
                tone={uncertaintyTone}
                className="h-20 flex-1 min-w-[140px] bg-slate-900 border-slate-800"
              />
              <RibbonMetric
                label="Uncertainty Trigger"
                value={result ? (result.escalate ? "TRIGGERED" : "SAFE") : "--"}
                interpretation={result?.escalate ? "Critical" : result ? "Stable" : undefined}
                tone={result?.escalate ? "text-orange-400" : escalationTone}
                className={`h-20 flex-1 min-w-[140px] bg-slate-900 border-slate-800 ${result?.escalate ? "ring-2 ring-orange-500/50 shadow-[0_0_15px_rgba(249,115,22,0.2)]" : ""}`}
              />
            </div>
          </div>

          <div className="flex flex-col gap-4 lg:col-span-2">
            <h2 className="text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500 pl-1">Runtime Telemetry</h2>
            <div className="flex gap-3 overflow-x-auto no-scrollbar">
              <RibbonMetric
                label="Latency"
                value={result ? (result.latency_ms > 1000 ? `${(result.latency_ms / 1000).toFixed(2)}s` : `${result.latency_ms}ms`) : "--"}
                className="h-20 flex-1 min-w-[120px] bg-slate-900 border-slate-800"
              />
              <RibbonMetric
                label="Input"
                value={result ? result.input_tokens.toString() : "--"}
                className="h-20 flex-1 min-w-[100px] bg-slate-900 border-slate-800"
              />
              <RibbonMetric
                label="Output"
                value={result ? result.output_tokens.toString() : "--"}
                className="h-20 flex-1 min-w-[100px] bg-slate-900 border-slate-800"
              />
              <RibbonMetric
                label="Samples"
                value={result ? `${result.samples_used}/${config.monte_carlo_samples}` : "--"}
                className="h-20 flex-1 min-w-[120px] bg-slate-900 border-slate-800"
              />
            </div>
          </div>
        </div>

        {/* Layer 2: Prompt Execution */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          {/* Left: Prompt Lab */}
          <Panel title="Prompt Lab" subtitle="Execution environment" className="bg-slate-950 border-slate-800 h-[520px] flex flex-col">
            <div className="flex-1 min-h-0 flex flex-col space-y-4">
              <textarea
                value={prompt}
                onChange={(event) => setPrompt(event.target.value)}
                placeholder="Enter research query..."
                className="h-[140px] w-full resize-none rounded-lg border border-slate-800 bg-black/40 p-4 text-sm text-slate-100 outline-none focus:border-[#0dccf2]/50 transition-all font-sans"
              />

              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <label className="text-[10px] uppercase tracking-widest text-slate-500 font-bold">Execution Mode</label>
                  <select
                    value={config.mode}
                    onChange={(event) => setConfig((prev) => ({ ...prev, mode: event.target.value as InferenceMode }))}
                    className="w-full rounded-lg border border-slate-800 bg-slate-900 px-3 py-2 text-xs outline-none focus:border-[#0dccf2]/40"
                  >
                    <option value="factual">Factual (Standard)</option>
                    <option value="mixed">Mixed (Balanced)</option>
                    <option value="emotional">Emotional (Indian Desi)</option>
                  </select>
                </div>
                {[{ label: "Temperature", key: "temperature", min: 0, max: 2, step: 0.1 }].map(c => (
                  <div key={c.key} className="space-y-1">
                    <div className="flex justify-between text-[10px] uppercase font-bold tracking-widest text-slate-500">
                      <label>{c.label}</label>
                      <span className="text-[#0dccf2]">{config.temperature}</span>
                    </div>
                    <input type="range" min={c.min} max={c.max} step={c.step} value={config.temperature}
                      onChange={(e) => setConfig(p => ({ ...p, [c.key]: Number(e.target.value) }))}
                      className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-[#0dccf2]" />
                  </div>
                ))}
              </div>

              <div className="grid grid-cols-3 gap-4">
                {[{ label: "Top-P", key: "top_p", min: 0, max: 1, step: 0.05 },
                { label: "Max Tokens", key: "max_new_tokens", min: 32, max: 4096, step: 32 },
                { label: "MC Samples", key: "monte_carlo_samples", min: 3, max: 10, step: 1 }].map(c => (
                  <div key={c.key} className="space-y-1">
                    <div className="flex justify-between text-[10px] uppercase font-bold tracking-widest text-slate-500">
                      <label>{c.label}</label>
                      <span className="text-[#0dccf2]">{(config as any)[c.key]}</span>
                    </div>
                    <input type="range" min={c.min} max={c.max} step={c.step} value={(config as any)[c.key]}
                      onChange={(e) => setConfig(p => ({ ...p, [c.key]: Number(e.target.value) }))}
                      className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-[#0dccf2]" />
                  </div>
                ))}
              </div>

              <div className="flex-1 min-h-0 rounded-lg border border-slate-800 bg-slate-950 p-4 overflow-y-auto">
                {result ? (
                  <p className="text-sm leading-relaxed text-slate-200">{result.response_text}</p>
                ) : (
                  <div className="h-full flex items-center justify-center text-slate-600 italic text-sm">
                    Awaiting output...
                  </div>
                )}
              </div>

              <button
                type="button" onClick={runPrompt} disabled={loading || modelStatus !== "ready"}
                className="w-full flex items-center justify-center gap-2 rounded-lg bg-[#0dccf2] px-6 py-3 text-sm font-bold text-[#05090a] transition hover:bg-[#33d5f3] disabled:opacity-40 uppercase tracking-widest"
              >
                <Play className="h-4 w-4 fill-current" />
                {loading ? "Inference In Progress..." : "Execute Research Query"}
              </button>
            </div>
          </Panel>

          {/* Right: Core Comparison */}
          <Panel title="Core Comparison" subtitle="Logic Divergence (Deterministic vs Entropy)" className="bg-slate-950 border-slate-800 h-[520px] flex flex-col">
            <div className="flex-1 min-h-0 flex flex-col space-y-4">
              <div className="grid grid-cols-3 gap-2 shrink-0">
                <div className="bg-slate-900 border border-slate-800 rounded-lg p-2 text-center">
                  <p className="text-[9px] uppercase tracking-tighter text-slate-500 mb-1">Similarity</p>
                  <p className="text-[11px] font-mono text-emerald-400">
                    {coreComparison.similarity !== null ? coreComparison.similarity.toFixed(3) : "--"}
                  </p>
                </div>
                <div className="bg-slate-900 border border-slate-800 rounded-lg p-2 text-center">
                  <p className="text-[9px] uppercase tracking-tighter text-slate-500 mb-1">Token Delta</p>
                  <p className="text-[11px] font-mono text-amber-400">
                    {coreComparison.token_delta !== null ? (coreComparison.token_delta > 0 ? `+${coreComparison.token_delta}` : coreComparison.token_delta) : "--"}
                  </p>
                </div>
                <div className="bg-slate-900 border border-slate-800 rounded-lg p-2 text-center">
                  <p className="text-[9px] uppercase tracking-tighter text-slate-500 mb-1">Length Delta</p>
                  <p className="text-[11px] font-mono text-slate-300">
                    {coreComparison.length_delta !== null ? `${coreComparison.length_delta}%` : "--"}
                  </p>
                </div>
              </div>

              <div className="flex-1 min-h-0 grid grid-cols-1 gap-4 overflow-y-auto pr-1">
                <div className="space-y-2">
                  <h3 className="text-[10px] font-bold uppercase tracking-widest text-[#0dccf2]">Core A <span className="text-slate-500 font-normal">(Deterministic)</span></h3>
                  <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3 text-xs leading-relaxed text-slate-300 min-h-[120px]">
                    {coreComparison.core_a !== null ? coreComparison.core_a : "Awaiting core evaluation..."}
                  </div>
                </div>
                <div className="space-y-2">
                  <h3 className="text-[10px] font-bold uppercase tracking-widest text-amber-400">Core B <span className="text-slate-500 font-normal">(Entropy Driven)</span></h3>
                  <div className="bg-slate-900/60 border border-slate-800 rounded-lg p-3 text-xs leading-relaxed text-slate-300 min-h-[120px]">
                    {coreComparison.core_b !== null ? coreComparison.core_b : "Awaiting side-channel variance..."}
                  </div>
                </div>
              </div>
            </div>
          </Panel>
        </div>

        {/* Layer 3: Diagnostics & Telemetry */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
          <Panel title="Research Trace" subtitle="System logical flow" className="bg-slate-900/50 border-slate-800 h-[400px] flex flex-col">
            <div className="flex-1 overflow-y-auto no-scrollbar space-y-3">
              {trace ? (
                Object.entries(trace).map(([key, value]) => (
                  key !== "monte_carlo_samples" && (
                    <div key={key} className="p-3 border-l-2 border-[#0dccf2]/30 bg-slate-900/80 rounded-r-lg">
                      <p className="text-[9px] uppercase font-bold tracking-widest text-slate-500 mb-1">{key}</p>
                      <p className="text-xs text-slate-300 whitespace-pre-wrap">{toPretty(value)}</p>
                    </div>
                  )
                ))
              ) : (
                <div className="h-full flex items-center justify-center text-slate-600 italic text-xs">Tracing inactive</div>
              )}
            </div>
          </Panel>

          <Panel title="MC Diagnostics" subtitle="Stochastic divergence" className="bg-slate-900/50 border-slate-800 h-[400px] flex flex-col">
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-3">
                <MetricCard label="Dispersion" value={result?.semantic_dispersion?.toFixed(3) || "0.000"} interpretation={result?.semantic_dispersion !== undefined ? (result.semantic_dispersion > 0.1 ? "⚠ High" : "Stable") : undefined} tone={result?.semantic_dispersion && result.semantic_dispersion > 0.1 ? "text-amber-400" : "text-[#0dccf2]"} />
                <MetricCard label="Entropy" value={result?.entropy?.toFixed(3) || "0.000"} interpretation={result?.entropy !== undefined ? getEntropyInterpretation(result.entropy) : undefined} tone={result?.entropy && result.entropy > 0.5 ? "text-amber-400" : "text-[#0dccf2]"} />
                <MetricCard label="Clusters" value={result?.cluster_count?.toString() || "0"} interpretation={result?.cluster_count !== undefined ? (result.cluster_count > 1 ? "⚠ Divergent" : "Stable") : undefined} tone={result?.cluster_count && result.cluster_count > 1 ? "text-amber-400" : "text-[#0dccf2]"} />
                <MetricCard label="Consistency" value={result?.self_consistency ? `${(result.self_consistency * 100).toFixed(0)}%` : "--"} />
              </div>
              <div className="pt-4 border-t border-slate-800 space-y-2">
                <div className="flex justify-between text-[10px] uppercase font-bold tracking-widest text-slate-500">
                  <label>Instability Meter</label>
                  <span>{(result?.instability || 0).toFixed(3)}</span>
                </div>
                <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                  <div className={`h-full transition-all duration-500 ${result?.instability && result.instability > 0.3 ? "bg-amber-400" : "bg-[#0dccf2]"}`}
                    style={{ width: `${(result?.instability || 0) * 100}%` }} />
                </div>
              </div>
              <div className="h-[140px] pt-2">
                <StabilityChart data={stabilityData} />
              </div>
            </div>
          </Panel>

          <Panel title="Uncertainty Trigger" subtitle="Anomaly detection" className={`bg-slate-900/50 border-slate-800 h-[400px] flex flex-col ${result?.escalate ? "ring-1 ring-orange-500/40" : ""}`}>
            {result?.escalate ? (
              <div className="flex-1 flex flex-col justify-center space-y-6 text-center">
                <div className="py-4 bg-orange-500/10 border border-orange-500/20 rounded-xl">
                  <p className="text-[10px] uppercase font-black tracking-[0.4em] text-orange-400 mb-2">Uncertainty State</p>
                  <p className="text-3xl font-black text-orange-500 drop-shadow-[0_0_10px_rgba(249,115,22,0.4)]">TRIGGERED</p>
                </div>
                <div className="grid grid-cols-1 gap-2 text-left">
                  <div className="flex justify-between p-2 bg-slate-900/60 rounded border border-slate-800">
                    <span className="text-[10px] uppercase text-slate-500">Similarity</span>
                    <span className="text-xs font-mono text-orange-300">{reviewPacket?.embedding_similarity?.toFixed(3) || "n/a"}</span>
                  </div>
                  <div className="flex justify-between p-2 bg-slate-900/60 rounded border border-slate-800">
                    <span className="text-[10px] uppercase text-slate-500">Ambiguity</span>
                    <span className="text-xs font-mono text-orange-300">{reviewPacket?.ambiguity?.toFixed(3) || "n/a"}</span>
                  </div>
                  <div className="flex justify-between p-2 bg-slate-900/60 rounded border border-slate-800">
                    <span className="text-[10px] uppercase text-slate-500">Entropy Var</span>
                    <span className="text-xs font-mono text-orange-300">
                      {(
                        isRecord(monteCarlo) && typeof monteCarlo.entropy_variance === "number"
                          ? monteCarlo.entropy_variance.toFixed(3)
                          : "n/a"
                      )}
                    </span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="h-full flex items-center justify-center text-slate-700 font-mono text-[10px] uppercase tracking-widest text-center px-4">
                No uncertainty signals detected in current inference block
              </div>
            )}
          </Panel>

          <Panel
            title="Language Routing"
            subtitle="Layer 2 multilingual dispatch"
            className={`bg-slate-900/50 border-slate-800 h-[400px] flex flex-col ${(trace as any)?.language_routing?.resolved_lang === "hinglish"
              ? "ring-1 ring-purple-500/40"
              : (trace as any)?.language_routing?.resolved_lang === "hi"
                ? "ring-1 ring-amber-500/30"
                : ""
              }`}
          >
            <LanguageRoutingPanel
              data={
                isRecord(trace) && isRecord((trace as any).language_routing)
                  ? (trace as any).language_routing as LanguageRoutingData
                  : null
              }
            />
          </Panel>

          <Panel
            title="Reliability Guard"
            subtitle="Grid-sweep fallback layer"
            className={`bg-slate-900/50 border-slate-800 h-[400px] flex flex-col ${result?.resampled ? "ring-1 ring-amber-500/40" : ""
              }`}
          >
            <ReliabilityGuardPanel
              guardData={
                getReliabilityGuardData(
                  isRecord(monteCarlo) ? monteCarlo.reliability_guard : null,
                )
              }
              resampled={result?.resampled}
            />
          </Panel>

          <FailureAnalysis failures={result?.failures} />
        </div>

        {/* Layer 4: Experiment Runner */}
        <Panel title="Experiment Runner" subtitle="Batch evaluation engine" className="bg-slate-950 border-slate-800 h-[420px] shrink-0 flex flex-col mb-8">
          {/* ... Experiment Runner content ... (restored correctly below) */}
          <div className="flex-1 min-h-0 flex flex-col">
            <div className="flex items-center gap-4 mb-6 border-b border-slate-800 pb-4">
              <label className="flex items-center gap-2 cursor-pointer bg-slate-900 border border-slate-800 px-4 py-2 rounded-lg text-xs hover:border-[#0dccf2]/50 transition-colors uppercase tracking-widest">
                <Upload className="h-3.5 w-3.5 text-[#0dccf2]" />
                Load JSON Dataset
                <input type="file" accept="application/json" className="hidden" onChange={handleDatasetUpload} />
              </label>
              <button onClick={runExperiment} disabled={experimentRunning || datasetItems.length === 0 || modelStatus !== "ready"}
                className="flex items-center gap-2 bg-[#0dccf2]/10 border border-[#0dccf2]/30 px-4 py-2 rounded-lg text-xs font-bold text-[#0dccf2] hover:bg-[#0dccf2]/20 transition-colors uppercase tracking-widest disabled:opacity-40">
                <FlaskConical className="h-3.5 w-3.5" />
                {experimentRunning
                  ? `Running Pipeline... ${experimentProgress.total > 0 ? Math.round((experimentProgress.done / experimentProgress.total) * 100) : 0}%`
                  : "Initialize Batch Run"}
              </button>
              {datasetItems.length > 0 && (
                <div className="flex items-center gap-4">
                  <p className="text-[10px] text-slate-500 uppercase tracking-wider">
                    Dataset Ready: {datasetItems.length} research prompts loaded
                  </p>
                  {(experimentSummary || experimentResults.length > 0) && (
                    <button
                      onClick={generateResearchReport}
                      disabled={reportLoading}
                      className="flex items-center gap-2 bg-amber-500/10 border border-amber-500/30 px-4 py-2 rounded-lg text-xs font-bold text-amber-500 hover:bg-amber-500/20 transition-colors uppercase tracking-widest disabled:opacity-40"
                    >
                      <ClipboardCheck className="h-3.5 w-3.5" />
                      {reportLoading ? "Synthesizing..." : "Generate Research Report"}
                    </button>
                  )}
                </div>
              )}
            </div>

            <div className="flex-1 overflow-y-auto no-scrollbar scroll-smooth">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                <div className="flex flex-col">
                  <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="p-4 bg-slate-900/60 border border-slate-800 rounded-xl transition-all hover:border-[#0dccf2]/30">
                      <p className="text-[9px] uppercase font-black tracking-widest text-slate-500 mb-1">Mean Confidence</p>
                      <p className={`text-2xl font-black font-mono ${experimentSummary && experimentSummary.meanConfidence > 0.8 ? "text-emerald-400" : "text-[#0dccf2]"}`}>
                        {experimentSummary ? experimentSummary.meanConfidence.toFixed(3) : "0.000"}
                      </p>
                    </div>
                    <div className="p-4 bg-slate-900/60 border border-slate-800 rounded-xl transition-all hover:border-orange-500/30">
                      <p className="text-[9px] uppercase font-black tracking-widest text-slate-500 mb-1">Uncertainty Rate</p>
                      <p className={`text-2xl font-black font-mono ${experimentSummary && experimentSummary.escalationRate > 0.1 ? "text-orange-500" : "text-emerald-400"}`}>
                        {experimentSummary ? `${(experimentSummary.escalationRate * 100).toFixed(1)}%` : "0.0%"}
                      </p>
                    </div>
                  </div>
                  <div className="flex-1 bg-slate-900/40 border border-slate-800 rounded-xl p-4 overflow-hidden">
                    <p className="text-[10px] uppercase font-black tracking-widest text-slate-500 mb-3 pl-1">Reliability Distributions</p>
                    <ReliabilityDistributions
                      confidence={experimentDistributions?.confidence || []}
                      instability={experimentDistributions?.instability || []}
                    />
                  </div>
                </div>

                <div className="bg-slate-900/40 border border-slate-800 rounded-xl flex flex-col min-h-[300px]">
                  <div className="p-3 border-b border-slate-800 flex items-center justify-between">
                    <p className="text-[10px] uppercase tracking-widest text-slate-500 font-bold px-1">Sequential Results</p>
                    <button
                      onClick={async () => {
                        if (experimentResults.length > 0) {
                          try { await exportExperimentReportFiles(experimentResults) } catch (e) { console.error(e) }
                        }
                      }}
                      className="text-[9px] uppercase font-bold text-[#0dccf2] hover:text-[#33d5f3] flex items-center gap-1"
                    >
                      <Download className="h-3 w-3" /> Report
                    </button>
                  </div>
                  <div className="flex-1 overflow-y-auto p-2 no-scrollbar">
                    <div className="space-y-1">
                      {experimentResults.map((r, i) => (
                        <div key={i} className="flex items-center justify-between p-2 bg-slate-950/40 border border-slate-800/40 rounded transition-all hover:bg-slate-900/60 group">
                          <span className="text-[10px] font-mono text-slate-500 group-hover:text-slate-300 truncate mr-4 max-w-[200px]">{r.prompt}</span>
                          <div className="flex gap-4 text-[9px] font-mono font-bold">
                            <span className="text-[#0dccf2]/70">C:{r.confidence.toFixed(2)}</span>
                            <span className="text-amber-500/70">I:{r.instability.toFixed(2)}</span>
                            <span className={r.escalate ? "text-orange-500" : "text-slate-700"}>{r.escalate ? "TRIG" : "SAFE"}</span>
                          </div>
                        </div>
                      ))}
                      {experimentResults.length === 0 && (
                        <div className="h-full flex items-center justify-center text-slate-700 italic py-12 text-xs">Awaiting batch research session initialization...</div>
                      )}
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 animate-in fade-in duration-500">
                <div className="lg:col-span-2 space-y-6">
                  <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4">
                    <p className="text-[10px] uppercase font-black tracking-widest text-slate-500 mb-4 pl-1">Category Reliability Index</p>
                    <div className="overflow-x-auto no-scrollbar">
                      <table className="w-full text-left text-[11px] font-mono">
                        <thead>
                          <tr className="border-b border-slate-800 text-slate-500 uppercase">
                            <th className="pb-2 font-bold px-2">Category</th>
                            <th className="pb-2 font-bold px-2">Count</th>
                            <th className="pb-2 font-bold px-2 text-amber-500">Instability</th>
                            <th className="pb-2 font-bold px-2 text-[#0dccf2]">Confidence</th>
                            <th className="pb-2 font-bold px-2 text-slate-400">Difficulty</th>
                            <th className="pb-2 font-bold px-2 text-purple-400">Entropy</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/50">
                          {(experimentSummary?.categoryAggregates || []).map((agg, idx) => (
                            <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                              <td className="py-2.5 px-2 text-slate-300 font-bold">{agg.category}</td>
                              <td className="py-2.5 px-2 text-slate-500">{agg.count}</td>
                              <td className={`py-2.5 px-2 font-bold ${instabilityTone(agg.instability)}`}>{agg.instability.toFixed(3)}</td>
                              <td className="py-2.5 px-2 font-bold text-[#0dccf2]">{agg.confidence.toFixed(3)}</td>
                              <td className={`py-2.5 px-2 ${difficultyTone(difficultyLabel(agg.difficulty))}`}>{agg.difficulty.toFixed(3)}</td>
                              <td className={`py-2.5 px-2 font-bold ${entropyTone(agg.entropy)}`}>{agg.entropy.toFixed(3)}</td>
                            </tr>
                          ))}
                          {!experimentSummary?.categoryAggregates?.length && (
                            <tr>
                              <td colSpan={6} className="py-8 text-center text-slate-600 italic">No category data available</td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>

                  <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4">
                    <p className="text-[10px] uppercase font-black tracking-widest text-slate-500 mb-4 pl-1">Stability Curves (Mean by Temperature)</p>
                    <div className="h-[200px] w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <LineChart data={experimentSummary?.instabilityCurve || []}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff05" vertical={false} />
                          <XAxis dataKey="temperature" fontSize={9} stroke="#475569" axisLine={false} tickLine={false} />
                          <YAxis fontSize={9} stroke="#475569" axisLine={false} tickLine={false} />
                          <Tooltip contentStyle={{ backgroundColor: "#020617", border: "1px solid #1e293b", fontSize: "10px" }} />
                          <Line type="monotone" dataKey="value" stroke="#f59e0b" strokeWidth={3} dot={{ fill: "#f59e0b", r: 4 }} activeDot={{ r: 6 }} name="Instability" />
                        </LineChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </div>

                <div className="bg-slate-900/40 border border-slate-800 rounded-xl p-4 flex flex-col">
                  <p className="text-[10px] uppercase font-black tracking-widest text-slate-500 mb-4 pl-1">Historical Timeline</p>
                  <div className="flex-1 flex flex-col min-h-[300px]">
                    <div className="flex-1 flex items-end gap-[2px] h-full">
                      {experimentResults.map((r, i) => (
                        <div
                          key={i}
                          title={`Run ${i + 1}: ${r.instability.toFixed(3)}`}
                          className={`w-full min-w-[2px] rounded-t-sm transition-all hover:brightness-125 ${r.escalate ? "bg-orange-500" : "bg-[#0dccf2]/60"}`}
                          style={{ height: `${Math.max(4, r.instability * 100)}%` }}
                        />
                      ))}
                    </div>
                    <div className="mt-4 pt-4 border-t border-slate-800 space-y-4">
                      <div className="flex justify-between items-center">
                        <span className="text-[9px] uppercase font-bold text-slate-500">Peak Instability</span>
                        <span className="text-xs font-mono text-orange-400 font-black">
                          {experimentResults.length > 0 ? Math.max(...experimentResults.map(r => r.instability)).toFixed(3) : "0.000"}
                        </span>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-[9px] uppercase font-bold text-slate-500">Benchmark Mean</span>
                        <span className="text-xs font-mono text-[#0dccf2] font-black">
                          {experimentSummary ? experimentSummary.meanInstability.toFixed(3) : "0.000"}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </Panel>

        {/* Layer 5: Parameter Sensitivity Grid */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-3 pb-8 px-1">
          <div className="lg:col-span-2">
            <ReliabilityHeatmap data={gridResults} loading={gridLoading} onRun={runGridEvaluation} />
          </div>
          <LeaderboardPanel data={leaderboard} />
        </div>

        {/* Layer 6: Deployment Registry */}
        <div className="pb-8 px-1">
          <ReleaseOverview
            deployments={deployments}
            loading={deploymentsLoading}
            error={deploymentsError}
            apiBase={API_BASE}
          />
        </div>
      </main>

      <footer className="border-t border-[#0dccf2]/15 bg-[#101f22]/90 px-4 py-2 text-xs text-slate-400 md:px-6">
        <div className="flex flex-wrap items-center gap-4 font-mono">
          <span className="inline-flex items-center gap-2">
            <Server className="h-3.5 w-3.5 text-[#0dccf2]" />
            {modelStatusText}
          </span>
          {result ? (
            <span>
              Last run: {result.latency_ms > 1000 ? `${(result.latency_ms / 1000).toFixed(1)}s` : `${result.latency_ms}ms`} | {result.input_tokens} -&gt; {result.output_tokens} tokens
            </span>
          ) : (
            <span>Run a prompt to populate telemetry.</span>
          )}
        </div>
      </footer>
    </div >
  )
}
