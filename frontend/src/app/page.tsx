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
import StabilityChart from "@/components/StabilityChart"

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

function MetricCard({ label, value, tone = "text-[#0dccf2]" }: { label: string; value: string; tone?: string }) {
  return (
    <div className="rounded-lg border border-[#0dccf2]/15 bg-black/25 p-3 min-w-0">
      <p className="text-[10px] uppercase tracking-[0.14em] text-slate-400 truncate" title={label}>{label}</p>
      <p className={`font-mono text-base ${tone} truncate`}>{value}</p>
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
    monte_carlo_samples: 5,
  })

  const [modelStatus, setModelStatus] = useState<ModelStatus>("offline")
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
  const [stabilityData, setStabilityData] = useState<TemperatureAggregatePoint[]>([])
  const [showSamples, setShowSamples] = useState(false)
  const [showMCDiagnostics, setShowMCDiagnostics] = useState(true)
  const [clockText, setClockText] = useState("--:--:--")

  const monteCarlo = useMemo(() => {
    const mc = trace?.monte_carlo_analysis
    return isRecord(mc) ? mc : null
  }, [trace])

  const comparisonVisible = showCoreComparison || Boolean(result?.escalate)

  const experimentSummary = useMemo(() => {
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
        const response = await fetch("/api/health", { cache: "no-store" })
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
      const response = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
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
      })
      setCoreComparison(data.core_comparison ?? null)
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
    const sweepTemperatures = [...TEMPERATURE_SWEEP]
    const totalRuns = datasetItems.length * sweepTemperatures.length
    setExperimentProgress({ done: 0, total: totalRuns })

    const rows: ExperimentResult[] = []
    let firstError: string | null = null
    let completedRuns = 0

    for (let i = 0; i < datasetItems.length; i += 1) {
      const item = datasetItems[i]
      for (const temperature of sweepTemperatures) {
        try {
          const data = await requestInference(item.prompt, { temperature })
          const monteCarloTrace =
            isRecord(data.trace) && isRecord(data.trace.monte_carlo_analysis)
              ? data.trace.monte_carlo_analysis
              : null
          const entropy =
            typeof data.entropy === "number"
              ? data.entropy
              : monteCarloTrace && typeof monteCarloTrace.entropy === "number"
                ? monteCarloTrace.entropy
                : 0
          const uncertainty =
            typeof data.uncertainty === "number"
              ? data.uncertainty
              : monteCarloTrace && typeof monteCarloTrace.uncertainty === "number"
                ? monteCarloTrace.uncertainty
                : data.instability
          const difficulty = computeDifficulty(data.confidence, data.instability, entropy, data.escalate)

          rows.push({
            prompt: item.prompt,
            category: item.category ?? "uncategorized",
            response_text: data.response_text,
            temperature,
            confidence: data.confidence,
            instability: data.instability,
            entropy,
            uncertainty,
            escalate: data.escalate,
            difficulty,
            difficulty_label: difficultyLabel(difficulty),
            temperature_sensitivity: 0,
            latency_ms: data.latency_ms,
            input_tokens: data.input_tokens,
            output_tokens: data.output_tokens,
            sample_count: typeof data.sample_count === "number" ? data.sample_count : 0,
            samples_used: typeof data.samples_used === "number" ? data.samples_used : 0,
            semantic_dispersion: typeof data.semantic_dispersion === "number" ? data.semantic_dispersion : undefined,
            cluster_count: typeof data.cluster_count === "number" ? data.cluster_count : undefined,
            cluster_entropy: typeof data.cluster_entropy === "number" ? data.cluster_entropy : undefined,
            dominant_cluster_ratio: typeof data.dominant_cluster_ratio === "number" ? data.dominant_cluster_ratio : undefined,
            self_consistency: typeof data.self_consistency === "number" ? data.self_consistency : undefined,
            trace: data.trace as TraceLog | undefined,
          })
        } catch (error) {
          const message = error instanceof Error ? error.message : "Experiment request failed."
          if (!firstError) firstError = message

          rows.push({
            prompt: item.prompt,
            category: item.category ?? "uncategorized",
            response_text: "",
            temperature,
            confidence: 0,
            instability: 1,
            entropy: 1,
            uncertainty: 1,
            escalate: true,
            difficulty: 1,
            difficulty_label: "adversarial",
            temperature_sensitivity: 0,
            latency_ms: 0,
            input_tokens: 0,
            output_tokens: 0,
            sample_count: 0,
            samples_used: 0,
            semantic_dispersion: 1,
            cluster_count: 1,
            cluster_entropy: 1,
            dominant_cluster_ratio: 0,
            self_consistency: 0,
          })
        } finally {
          completedRuns += 1
          setExperimentProgress({ done: completedRuns, total: totalRuns })
          setExperimentResults(attachTemperatureSensitivity([...rows]))
        }
      }
    }

    setExperimentError(firstError)
    setExperimentRunning(false)

    const finalRows = attachTemperatureSensitivity(rows)
    setExperimentResults(finalRows)

    if (finalRows.length > 0) {
      try {
        await exportExperimentReportFiles(finalRows)
      } catch (error) {
        const message = error instanceof Error ? error.message : "Failed to generate report files."
        setExperimentError(firstError ? `${firstError} | ${message}` : message)
      }
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
    <div className="flex h-full min-h-0 flex-col bg-[#080e10] text-slate-100">
      <header className="sticky top-0 z-30 border-b border-[#0dccf2]/20 bg-[#101f22]/95 px-4 py-3 backdrop-blur md:px-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-sm font-bold uppercase tracking-[0.18em] text-[#0dccf2] md:text-base">
              AI Research Command Center
            </h1>
            <p className="mt-1 text-xs text-slate-400">
              Model: Qwen2.5-7B | Backend: Kaggle ({modelStatus === "ready" ? "Online" : modelStatus === "loading" ? "Loading" : "Offline"})
            </p>
          </div>

          <div className="flex items-center gap-4 text-xs font-mono">
            <div className="flex items-center gap-2">
              <span className={`h-2 w-2 rounded-full ${modelStatusDot}`} />
              <span className={modelStatusTone}>{modelStatusText}</span>
            </div>
            <div className="text-slate-500">{clockText}</div>
          </div>
        </div>
      </header>

      <section className="border-b border-[#0dccf2]/15 bg-[#0f181c]/95 px-4 py-3 md:px-6">
        <div className="flex flex-col gap-4">
          {/* Row 1 - Reliability */}
          <div className="flex gap-4 overflow-x-auto pb-1">
            <RibbonMetric
              label="Confidence"
              value={result ? result.confidence.toFixed(3) : "--"}
              tone={confidenceTone}
            />
            <RibbonMetric
              label="Instability"
              value={result ? result.instability.toFixed(3) : "--"}
              tone={localInstabilityTone}
            />
            <RibbonMetric
              label="Uncertainty"
              value={result ? (result.uncertainty?.toFixed(3) || "0.000") : "--"}
              tone={uncertaintyTone}
            />
            <RibbonMetric
              label="Escalation"
              value={result ? (result.escalate ? "TRUE" : "FALSE") : "--"}
              tone={escalationTone}
            />
          </div>
        </div>
      </section>

      <main className="min-h-0 flex-1 overflow-hidden p-4 md:p-6">
        <div className="grid h-full grid-cols-1 gap-6 lg:grid-cols-2 overflow-y-auto pr-1">
          {/* Top Left: Prompt Lab */}
          <div className="flex flex-col gap-6">
            <Panel title="Prompt Lab" subtitle="Prompt + controls + run" className="min-h-0 flex flex-col">
              <div className="min-h-0 flex-1 space-y-4 pr-1">
                <div className="space-y-2">
                  <label className="text-xs uppercase tracking-[0.14em] text-slate-400">Prompt</label>
                  <textarea
                    value={prompt}
                    onChange={(event) => setPrompt(event.target.value)}
                    placeholder="Enter prompt..."
                    className="min-h-[120px] w-full resize-y rounded-lg border border-[#0dccf2]/20 bg-black/30 p-3 text-sm text-slate-100 outline-none focus:border-[#0dccf2]/50"
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
                }, {
                  label: "MC Samples",
                  key: "monte_carlo_samples",
                  value: config.monte_carlo_samples,
                  min: 3,
                  max: 7,
                  step: 1,
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
                    disabled={loading || modelStatus !== "ready"}
                    className="inline-flex flex-1 items-center justify-center gap-2 rounded-lg border border-[#0dccf2]/30 bg-[#0dccf2]/90 px-4 py-2 text-sm font-semibold text-[#081014] transition hover:bg-[#33d5f3] disabled:opacity-60"
                  >
                    <Play className="h-4 w-4" />
                    {loading
                      ? "Running..."
                      : modelStatus === "loading"
                        ? "Model Loading"
                        : modelStatus === "offline"
                          ? "Run Prompt (backend offline)"
                          : "Run Prompt"}
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

            <Panel title="Trace" subtitle="Collapsible decision trace" className="flex-1 min-h-0 flex flex-col">
              <div className="min-h-0 flex-1 overflow-y-auto pr-1">
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
                    <div className="mt-3 space-y-4">
                      {Object.entries(trace).map(([key, value]) => {
                        if (key === "monte_carlo_samples" && Array.isArray(value)) {
                          const groups: Record<number, string[]> = {}
                          value.forEach((sample: any) => {
                            const c = sample.cluster ?? 0
                            if (!groups[c]) groups[c] = []
                            groups[c].push(sample.text)
                          })
                          return (
                            <div key={key} className="rounded-lg border border-[#0dccf2]/15 bg-black/25 p-3">
                              <div className="flex items-center justify-between mb-3 border-b border-[#0dccf2]/10 pb-2">
                                <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-[#0dccf2]">
                                  Monte Carlo Reasoning Trace
                                </p>
                                <button
                                  onClick={() => setShowSamples(!showSamples)}
                                  className="text-[10px] uppercase tracking-wider text-slate-400 hover:text-[#0dccf2]"
                                >
                                  {showSamples ? "Hide Samples ▲" : "Show Samples ▼"}
                                </button>
                              </div>
                              {showSamples && (
                                <div className="space-y-4">
                                  {Object.entries(groups).map(([clusterId, texts]) => (
                                    <div key={clusterId} className="space-y-2 border-l-2 border-[#0dccf2]/30 pl-3">
                                      <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-amber-300">
                                        Cluster {clusterId}
                                      </p>
                                      <ul className="ml-4 list-disc space-y-1 text-xs text-slate-200">
                                        {texts.map((t, idx) => (
                                          <li key={idx} className="leading-relaxed">
                                            {t}
                                          </li>
                                        ))}
                                      </ul>
                                    </div>
                                  ))}
                                </div>
                              )}
                            </div>
                          )
                        }

                        return (
                          <div key={key} className="rounded-lg border border-[#0dccf2]/15 bg-black/25 p-2">
                            <p className="mb-1 text-[11px] uppercase tracking-[0.14em] text-slate-400">{key}</p>
                            <pre className="max-h-40 overflow-y-auto whitespace-pre-wrap break-words text-xs text-slate-200">
                              {toPretty(value)}
                            </pre>
                          </div>
                        )
                      })}
                    </div>
                  ) : (
                    <p className="mt-3 text-sm text-slate-500">Run an inference to populate trace data.</p>
                  )
                ) : null}
              </div>
            </Panel>
          </div>

          {/* Top Right: Core A / Core B */}
          <div className="flex flex-col gap-6 overflow-y-auto pr-1">
            <Panel title="Core Comparison" subtitle="Core A (Deterministic) vs Core B (Entropy)" className="flex-none">
              <div className="space-y-4">
                <div className="rounded-lg border border-[#0dccf2]/20 bg-black/25 p-3">
                  <p className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">Final Output</p>
                  <div className="max-h-[320px] overflow-y-auto pr-1">
                    {result ? (
                      <p className="whitespace-pre-wrap text-sm leading-6 text-slate-100">{result.response_text}</p>
                    ) : (
                      <p className="text-sm text-slate-500">Run a prompt to view output.</p>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                  <div className="rounded-lg border border-[#0dccf2]/15 bg-black/25 p-3">
                    <p className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">Core A (Deterministic)</p>
                    <div className="max-h-[240px] overflow-y-auto text-sm text-slate-200 pr-1">
                      {coreComparison?.core_a_output || "-"}
                    </div>
                  </div>
                  <div className="rounded-lg border border-[#0dccf2]/15 bg-black/25 p-3">
                    <p className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">Core B (Entropy)</p>
                    <div className="max-h-[240px] overflow-y-auto text-sm text-slate-200 pr-1">
                      {coreComparison?.core_b_output || "-"}
                    </div>
                  </div>
                </div>
              </div>
            </Panel>

            <div className="grid grid-cols-1 gap-6">
              <Panel title="Monte Carlo Diagnostics" subtitle="Semantic dispersion and cluster analysis">
                {result?.cluster_count !== undefined ? (
                  <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                    <MetricCard
                      label="Cluster Count"
                      value={result.cluster_count.toString()}
                      tone={result.cluster_count > 1 ? "text-amber-300" : "text-emerald-400"}
                    />
                    <MetricCard
                      label="Cluster Entropy"
                      value={result.cluster_entropy?.toFixed(3) ?? "n/a"}
                      tone={result.cluster_entropy && result.cluster_entropy > 0.4 ? "text-amber-300" : "text-emerald-400"}
                    />
                    <MetricCard
                      label="Dominant Cluster %"
                      value={result.dominant_cluster_ratio ? `${(result.dominant_cluster_ratio * 100).toFixed(1)}%` : "n/a"}
                      tone={result.dominant_cluster_ratio && result.dominant_cluster_ratio < 0.6 ? "text-amber-300" : "text-emerald-400"}
                    />
                    <MetricCard
                      label="Semantic Dispersion"
                      value={result.semantic_dispersion?.toFixed(3) ?? "n/a"}
                    />
                  </div>
                ) : (
                  <p className="text-sm text-slate-500">No Monte Carlo data yet.</p>
                )}
              </Panel>

              <Panel title="Telemetry" subtitle="Runtime and token accounting">
                {result ? (
                  <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                    <MetricCard label="Latency" value={`${result.latency_ms} ms`} />
                    <MetricCard label="Input Tokens" value={result.input_tokens.toString()} />
                    <MetricCard label="Output Tokens" value={result.output_tokens.toString()} />
                    <MetricCard
                      label="Samples Used"
                      value={`${result.samples_used} / ${config.monte_carlo_samples}`}
                    />
                  </div>
                ) : (
                  <p className="text-sm text-slate-500">Telemetry appears after running inference.</p>
                )}
              </Panel>

              {result?.escalate ? (
                <Panel title="Uncertainty Trigger" subtitle="Orange diagnostic signal" className="border-orange-500/50">
                  <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <div className="space-y-2 rounded-lg border border-orange-500/30 bg-orange-500/10 p-4">
                      <p className="text-[10px] uppercase tracking-[0.12em] text-orange-300">Uncertainty Status</p>
                      <p className="font-mono text-lg font-bold text-orange-400">TRIGGERED</p>
                    </div>
                    <div className="space-y-2 rounded-lg border border-[#0dccf2]/15 bg-black/25 p-4">
                      <p className="font-mono text-xs text-slate-300">
                        Embedding Similarity: {typeof reviewPacket?.embedding_similarity === "number"
                          ? reviewPacket.embedding_similarity.toFixed(3)
                          : "n/a"}
                      </p>
                      <p className="font-mono text-xs text-slate-300">
                        Ambiguity Score: {typeof reviewPacket?.ambiguity === "number"
                          ? reviewPacket.ambiguity.toFixed(3)
                          : "n/a"}
                      </p>
                      <p className="font-mono text-xs text-slate-300">
                        Entropy Variance: {isRecord(monteCarlo) && typeof monteCarlo.entropy_variance === "number"
                          ? monteCarlo.entropy_variance.toFixed(3)
                          : "n/a"}
                      </p>
                    </div>
                  </div>
                </Panel>
              ) : null}

              <Panel title="Model Stability Curve" subtitle="Temperature vs Instability history">
                <div className="h-[240px]">
                  <StabilityChart data={stabilityData} />
                </div>
              </Panel>
            </div>
          </div>
        </div>
      </main>

      <section className="border-t border-[#0dccf2]/15 bg-[#101f22]/95 h-[400px] shrink-0 overflow-hidden flex flex-col">
        <Panel
          title="Experiment Runner"
          subtitle="Dataset testing with confidence, instability and escalation tracking"
          className="flex-1 min-h-0 flex flex-col m-4"
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
              disabled={experimentRunning || datasetItems.length === 0 || modelStatus !== "ready"}
              className="inline-flex items-center gap-2 rounded-lg border border-[#0dccf2]/30 bg-[#0dccf2]/90 px-3 py-2 text-xs font-semibold uppercase tracking-[0.12em] text-[#081014] disabled:opacity-60"
            >
              <FlaskConical className="h-4 w-4" />
              {experimentRunning ? "Running" : "Run Experiment"}
            </button>

            {experimentResults.length > 0 ? (
              <>
                <button
                  type="button"
                  onClick={async () => {
                    try {
                      await exportExperimentReportFiles(experimentResults)
                    } catch (error) {
                      const message =
                        error instanceof Error ? error.message : "Failed to generate report files."
                      setExperimentError(message)
                    }
                  }}
                  className="inline-flex items-center gap-2 rounded-lg border border-[#0dccf2]/30 bg-black/25 px-3 py-2 text-xs uppercase tracking-[0.12em] text-slate-200 hover:border-[#0dccf2]/60"
                >
                  <Download className="h-4 w-4 text-[#0dccf2]" />
                  Export Report
                </button>

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
                  Export JSON
                </button>
              </>
            ) : null}

            <div className="text-xs text-slate-400">
              <span className="font-semibold text-slate-200">Dataset:</span> {datasetName}
              {datasetItems.length > 0 ? ` (${datasetItems.length} prompts)` : ""}
              {datasetItems.length > 0
                ? ` | Runs: ${datasetItems.length * TEMPERATURE_SWEEP.length} (${TEMPERATURE_SWEEP.join(", ")})`
                : ""}
            </div>
          </div>

          {
            experimentRunning ? (
              <div className="mb-3 rounded-lg border border-[#0dccf2]/20 bg-black/25 p-3 text-xs text-slate-300">
                Running {experimentProgress.done}/{experimentProgress.total}
              </div>
            ) : null
          }

          {
            experimentError ? (
              <div className="mb-3 rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-xs text-red-200">
                {experimentError}
              </div>
            ) : null
          }

          {
            experimentSummary ? (
              <>
                <div className="mb-3 grid grid-cols-1 gap-2 md:grid-cols-4 xl:grid-cols-10">
                  <MetricCard label="Prompts" value={experimentSummary?.total.toString() || "0"} />
                  <MetricCard label="Mean Confidence" value={experimentSummary?.meanConfidence.toFixed(3) || "0.000"} />
                  <MetricCard label="Avg Instability" value={experimentSummary?.meanInstability.toFixed(3) || "0.000"} />
                  <MetricCard label="Mean Entropy" value={experimentSummary?.meanEntropy.toFixed(3) || "0.000"} />
                  <MetricCard label="Mean Uncertainty" value={experimentSummary?.meanUncertainty.toFixed(3) || "0.000"} />
                  <MetricCard label="Mean Difficulty" value={experimentSummary?.meanDifficulty.toFixed(3) || "0.000"} />
                  <MetricCard
                    label="Temp Sensitivity"
                    value={experimentSummary?.meanTemperatureSensitivity.toFixed(3) || "0.000"}
                    tone={sensitivityTone(experimentSummary?.meanTemperatureSensitivity || 0)}
                  />
                  <MetricCard label="Escalation Rate" value={`${((experimentSummary?.escalationRate || 0) * 100).toFixed(1)}%`} />
                  <MetricCard label="Avg Latency" value={`${(experimentSummary?.avgLatency || 0).toFixed(1)} ms`} />
                  <MetricCard label="Avg Tokens Out" value={(experimentSummary?.avgOutputTokens || 0).toFixed(1)} />
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

                <div className="mb-4 rounded-lg border border-[#0dccf2]/15 bg-black/25 p-3">
                  <p className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">
                    Temperature Sensitivity Curves (mean by temperature)
                  </p>
                  <div className="overflow-x-auto">
                    <table className="min-w-[640px] border-collapse text-xs">
                      <thead>
                        <tr className="text-left uppercase tracking-[0.1em] text-slate-400">
                          <th className="border-b border-[#0dccf2]/10 px-2 py-1">Temperature</th>
                          <th className="border-b border-[#0dccf2]/10 px-2 py-1">Instability</th>
                          <th className="border-b border-[#0dccf2]/10 px-2 py-1">Entropy</th>
                          <th className="border-b border-[#0dccf2]/10 px-2 py-1">Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {experimentSummary.instabilityCurve.map((point, index) => (
                          <tr key={`${point.temperature}-${index}`} className="border-b border-[#0dccf2]/10 text-slate-200">
                            <td className="px-2 py-1 font-mono">{point.temperature.toFixed(1)}</td>
                            <td className="px-2 py-1 font-mono">
                              {point.value.toFixed(3)}
                            </td>
                            <td className="px-2 py-1 font-mono">
                              {experimentSummary.entropyCurve[index]?.value.toFixed(3) ?? "--"}
                            </td>
                            <td className="px-2 py-1 font-mono">
                              {experimentSummary.confidenceCurve[index]?.value.toFixed(3) ?? "--"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="mb-4 rounded-lg border border-[#0dccf2]/15 bg-black/25 p-3">
                  <p className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">
                    Category Stability Panel
                  </p>
                  <div className="overflow-x-auto">
                    <table className="min-w-[640px] border-collapse text-xs">
                      <thead>
                        <tr className="text-left uppercase tracking-[0.1em] text-slate-400">
                          <th className="border-b border-[#0dccf2]/10 px-2 py-1">Category</th>
                          <th className="border-b border-[#0dccf2]/10 px-2 py-1">Count</th>
                          <th className="border-b border-[#0dccf2]/10 px-2 py-1">Instability</th>
                          <th className="border-b border-[#0dccf2]/10 px-2 py-1">Difficulty</th>
                          <th className="border-b border-[#0dccf2]/10 px-2 py-1">Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {experimentSummary.categoryAggregates.map((agg, index) => (
                          <tr key={`${agg.category}-${index}`} className="border-b border-[#0dccf2]/10 text-slate-200">
                            <td className="px-2 py-1 font-mono text-[#0dccf2]">{agg.category}</td>
                            <td className="px-2 py-1 font-mono">{agg.count}</td>
                            <td className={`px-2 py-1 font-mono ${instabilityTone(agg.instability)}`}>
                              {agg.instability.toFixed(3)}
                            </td>
                            <td className={`px-2 py-1 font-mono ${difficultyTone(difficultyLabel(agg.difficulty))}`}>
                              {agg.difficulty.toFixed(3)}
                            </td>
                            <td className="px-2 py-1 font-mono">
                              {agg.confidence.toFixed(3)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="mb-4 grid grid-cols-1 gap-3 xl:grid-cols-[minmax(260px,34%)_minmax(0,1fr)]">
                  <div className="rounded-lg border border-[#0dccf2]/15 bg-black/25 p-3">
                    <p className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">
                      Prompt Difficulty Distribution
                    </p>
                    <div className="space-y-2">
                      {(["easy", "moderate", "hard", "adversarial"] as DifficultyLabel[]).map((label) => {
                        const count = experimentSummary.difficultyCounts[label]
                        const ratio = experimentSummary.total > 0 ? count / experimentSummary.total : 0
                        return (
                          <div key={label}>
                            <div className="mb-1 flex items-center justify-between text-[11px] uppercase tracking-[0.1em]">
                              <span className={difficultyTone(label)}>{label}</span>
                              <span className="font-mono text-slate-300">
                                {count} ({(ratio * 100).toFixed(1)}%)
                              </span>
                            </div>
                            <div className="h-2 overflow-hidden rounded bg-[#0dccf2]/15">
                              <div
                                className={`h-full ${difficultyBarTone(label)}`}
                                style={{ width: `${(ratio * 100).toFixed(2)}%` }}
                              />
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>

                  <div className="rounded-lg border border-[#0dccf2]/15 bg-black/25 p-3">
                    <p className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">
                      Hardest Prompt Per Category
                    </p>
                    <div className="mb-4 max-h-40 space-y-2 overflow-y-auto pr-1">
                      {hardestPromptsByCategory.map((row, index) => (
                        <div key={`${row.category}-${index}`} className="rounded border border-[#0dccf2]/10 bg-black/30 p-2">
                          <div className="mb-1 flex items-center justify-between gap-2">
                            <span className="font-mono text-[11px] text-[#0dccf2] uppercase tracking-[0.08em]">{row.category}</span>
                            <span className={`font-mono text-[11px] uppercase tracking-[0.08em] ${difficultyTone(row.difficulty_label)}`}>
                              {row.difficulty.toFixed(3)}
                            </span>
                          </div>
                          <p className="line-clamp-2 text-xs text-slate-200">{row.prompt}</p>
                        </div>
                      ))}
                    </div>

                    <p className="mb-2 text-[11px] uppercase tracking-[0.14em] text-slate-400">
                      Top 10 Hardest Prompts (Overall)
                    </p>
                    <div className="max-h-40 space-y-2 overflow-y-auto pr-1">
                      {hardestPrompts.map((row, index) => (
                        <div key={`${row.prompt}-${index}`} className="rounded border border-[#0dccf2]/10 bg-black/30 p-2">
                          <div className="mb-1 flex items-center justify-between gap-2">
                            <span className="font-mono text-[11px] text-slate-400">#{index + 1}</span>
                            <span className={`font-mono text-[11px] uppercase tracking-[0.08em] ${difficultyTone(row.difficulty_label)}`}>
                              {row.difficulty_label} {row.difficulty.toFixed(3)}
                            </span>
                          </div>
                          <p className="line-clamp-2 text-xs text-slate-200">{row.prompt}</p>
                        </div>
                      ))}
                    </div>

                    <p className="mb-2 mt-4 text-[11px] uppercase tracking-[0.14em] text-slate-400">
                      Top 10 Temperature-Sensitive Prompts
                    </p>
                    <div className="max-h-40 space-y-2 overflow-y-auto pr-1">
                      {mostSensitivePrompts.map((item, index) => (
                        <div key={`${item.prompt}-${index}`} className="rounded border border-[#0dccf2]/10 bg-black/30 p-2">
                          <div className="mb-1 flex items-center justify-between gap-2">
                            <span className="font-mono text-[11px] text-slate-400">#{index + 1}</span>
                            <span className={`font-mono text-[11px] ${sensitivityTone(item.sensitivity)}`}>
                              sens {item.sensitivity.toFixed(3)}
                            </span>
                          </div>
                          <p className="line-clamp-2 text-xs text-slate-200">{item.prompt}</p>
                          <p className="mt-1 font-mono text-[10px] text-slate-400">
                            {item.low_temperature.toFixed(1)}:{item.low_instability.toFixed(3)} -&gt;{" "}
                            {item.high_temperature.toFixed(1)}:{item.high_instability.toFixed(3)}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </>
            ) : null
          }

          <div className="min-h-0 flex-1 overflow-y-auto rounded-lg border border-[#0dccf2]/15 bg-black/20">
            {experimentResults.length === 0 ? (
              <div className="flex h-full min-h-[120px] items-center justify-center p-4 text-sm text-slate-500">
                Upload a JSON dataset and run experiment to populate results.
              </div>
            ) : (
              <table className="w-full min-w-[1660px] border-collapse text-xs">
                <thead className="sticky top-0 bg-[#101f22]">
                  <tr className="text-left uppercase tracking-[0.12em] text-slate-400">
                    <th className="border-b border-[#0dccf2]/15 px-3 py-2">Prompt</th>
                    <th className="border-b border-[#0dccf2]/15 px-3 py-2">Response</th>
                    <th className="border-b border-[#0dccf2]/15 px-3 py-2">Category</th>
                    <th className="border-b border-[#0dccf2]/15 px-3 py-2">Temperature</th>
                    <th className="border-b border-[#0dccf2]/15 px-3 py-2">Confidence</th>
                    <th className="border-b border-[#0dccf2]/15 px-3 py-2">Instability</th>
                    <th className="border-b border-[#0dccf2]/15 px-3 py-2">Entropy</th>
                    <th className="border-b border-[#0dccf2]/15 px-3 py-2">Uncertainty</th>
                    <th className="border-b border-[#0dccf2]/15 px-3 py-2">Temp Sensitivity</th>
                    <th className="border-b border-[#0dccf2]/15 px-3 py-2">Difficulty</th>
                    <th className="border-b border-[#0dccf2]/15 px-3 py-2">Difficulty Label</th>
                    <th className="border-b border-[#0dccf2]/15 px-3 py-2">Escalate</th>
                    <th className="border-b border-[#0dccf2]/15 px-3 py-2">Latency (ms)</th>
                    <th className="border-b border-[#0dccf2]/15 px-3 py-2">Tokens In</th>
                    <th className="border-b border-[#0dccf2]/15 px-3 py-2">Samples Used</th>
                    <th className="border-b border-[#0dccf2]/15 px-3 py-2">Tokens Out</th>
                  </tr>
                </thead>
                <tbody>
                  {experimentResults.map((row, index) => (
                    <tr key={`${row.prompt}-${index}`} className="border-b border-[#0dccf2]/10 text-slate-200">
                      <td className="max-w-[340px] px-3 py-2">
                        <div className="line-clamp-2">{row.prompt}</div>
                      </td>
                      <td className="max-w-[340px] px-3 py-2">
                        <div className="line-clamp-2">{row.response_text || "-"}</div>
                      </td>
                      <td className="px-3 py-2">{row.category}</td>
                      <td className="px-3 py-2 font-mono">{row.temperature.toFixed(1)}</td>
                      <td className="px-3 py-2 font-mono">{row.confidence.toFixed(3)}</td>
                      <td className="px-3 py-2 font-mono">{row.instability.toFixed(3)}</td>
                      <td className="px-3 py-2 font-mono">{row.entropy.toFixed(3)}</td>
                      <td className="px-3 py-2 font-mono">{row.uncertainty.toFixed(3)}</td>
                      <td className={`px-3 py-2 font-mono ${sensitivityTone(row.temperature_sensitivity)}`}>
                        {row.temperature_sensitivity.toFixed(3)}
                      </td>
                      <td className={`px-3 py-2 font-mono ${difficultyTone(row.difficulty_label)}`}>
                        {row.difficulty.toFixed(3)}
                      </td>
                      <td className="px-3 py-2">
                        <span className={`inline-flex rounded border border-current/30 px-2 py-0.5 font-mono text-[10px] uppercase tracking-[0.08em] ${difficultyTone(row.difficulty_label)}`}>
                          {row.difficulty_label}
                        </span>
                      </td>
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
                      <td className="px-3 py-2 font-mono">
                        <span className={`${row.samples_used >= config.monte_carlo_samples ? "text-red-400 font-bold" : "text-emerald-400"}`}>
                          {row.samples_used} / {config.monte_carlo_samples}
                        </span>
                        {row.samples_used >= config.monte_carlo_samples && (
                          <span className="ml-2 inline-flex rounded bg-red-500/20 px-1 py-0.5 text-[9px] uppercase tracking-wider text-red-500">
                            High Complexity
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-2 font-mono">{row.output_tokens}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </Panel>
      </section>

      <footer className="border-t border-[#0dccf2]/15 bg-[#101f22]/90 px-4 py-2 text-xs text-slate-400 md:px-6">
        <div className="flex flex-wrap items-center gap-4 font-mono">
          <span className="inline-flex items-center gap-2">
            <Server className="h-3.5 w-3.5 text-[#0dccf2]" />
            {modelStatusText}
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
