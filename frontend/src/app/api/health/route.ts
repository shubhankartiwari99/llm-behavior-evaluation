import { NextResponse } from "next/server"

import { getInferenceEndpoint } from "@/lib/inference-endpoint"

function getHealthEndpoint() {
  const endpoint = getInferenceEndpoint()
  return endpoint.replace(/\/generate\/?$/, "/health")
}

export async function GET() {
  try {
    const res = await fetch(getHealthEndpoint(), {
      method: "GET",
      headers: { "ngrok-skip-browser-warning": "true" },
      signal: AbortSignal.timeout(3000),
    })

    if (!res.ok) {
      return NextResponse.json({ status: "offline" }, { status: 502 })
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch (error) {
    return NextResponse.json({ status: "offline", error: String(error) }, { status: 503 })
  }
}
