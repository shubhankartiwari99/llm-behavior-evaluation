export function getInferenceEndpoint() {
  const endpoint = process.env.NEXT_PUBLIC_INFERENCE_URL

  if (!endpoint) {
    throw new Error("NEXT_PUBLIC_INFERENCE_URL is not defined")
  }

  return endpoint
}
