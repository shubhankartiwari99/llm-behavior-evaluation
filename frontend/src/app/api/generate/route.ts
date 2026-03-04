import { NextResponse } from "next/server"

import { getInferenceEndpoint } from "@/lib/inference-endpoint"

const GENERATE_TIMEOUT_MS = 180000

type GenerateContract = {
    response_text: string
    latency_ms: number
    input_tokens: number
    output_tokens: number
    confidence: number
    instability: number
    escalate: boolean
    core_comparison?: Record<string, unknown>
    trace?: Record<string, unknown>
    review_packet?: Record<string, unknown>
}

class ContractError extends Error { }

function isObject(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null
}

function isFiniteNumber(value: unknown): value is number {
    return typeof value === "number" && Number.isFinite(value)
}

function assertGenerateContract(data: unknown): asserts data is GenerateContract {
    if (!isObject(data)) {
        throw new ContractError("Invalid response payload.")
    }

    if (typeof data.response_text !== "string") {
        throw new ContractError("Missing response_text in inference response.")
    }

    if (!isFiniteNumber(data.latency_ms)) {
        throw new ContractError("Missing latency_ms in inference response.")
    }

    if (!isFiniteNumber(data.input_tokens)) {
        throw new ContractError("Missing input_tokens in inference response.")
    }

    if (!isFiniteNumber(data.output_tokens)) {
        throw new ContractError("Missing output_tokens in inference response.")
    }

    if (!isFiniteNumber(data.confidence)) {
        throw new ContractError("Missing confidence in inference response.")
    }

    if (!isFiniteNumber(data.instability)) {
        throw new ContractError("Missing instability in inference response.")
    }

    if (typeof data.escalate !== "boolean") {
        throw new ContractError("Missing escalate in inference response.")
    }

    if ("trace" in data && data.trace !== undefined && !isObject(data.trace)) {
        throw new ContractError("Invalid trace in inference response.")
    }

    if ("core_comparison" in data && data.core_comparison !== undefined && !isObject(data.core_comparison)) {
        throw new ContractError("Invalid core_comparison in inference response.")
    }

    if ("review_packet" in data && data.review_packet !== undefined && !isObject(data.review_packet)) {
        throw new ContractError("Invalid review_packet in inference response.")
    }
}

async function safeErrorText(response: Response): Promise<string> {
    const text = await response.text()
    if (!text) {
        return `Inference server unavailable (HTTP ${response.status}).`
    }

    try {
        const parsed = JSON.parse(text) as { error?: unknown }
        if (typeof parsed.error === "string" && parsed.error.trim()) {
            return parsed.error
        }
    } catch {
        // keep raw text fallback
    }

    return text
}

export async function POST(req: Request) {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), GENERATE_TIMEOUT_MS)

    try {
        const body = await req.json()
        const endpoint = getInferenceEndpoint()
        const response = await fetch(endpoint, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                "ngrok-skip-browser-warning": "true",
            },
            body: JSON.stringify(body),
            cache: "no-store",
            signal: controller.signal,
        })

        if (!response.ok) {
            const message = await safeErrorText(response)
            return NextResponse.json({ error: message }, { status: response.status })
        }

        const data = await response.json()
        assertGenerateContract(data)
        return NextResponse.json(data)
    } catch (error: unknown) {
        if (error instanceof Error && error.name === "AbortError") {
            return NextResponse.json(
                { error: "Inference server timeout." },
                { status: 504 },
            )
        }

        if (error instanceof ContractError) {
            return NextResponse.json({ error: error.message }, { status: 502 })
        }

        console.error("Inference Error:", error)
        return NextResponse.json({ error: "Inference server unavailable." }, { status: 500 })
    } finally {
        clearTimeout(timeoutId)
    }
}
