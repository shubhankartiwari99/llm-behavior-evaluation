Indian Desi LLM Inference Engine

Public API Contract - Version 1.2.0 (B22)

---

CHANGELOG from v1.1.0 (B21)
  - Added `GET /api/registry/history` for deployment provenance and release governance.
  - Added token-level forensics fields: `token_entropy_available`, `token_entropy[*].entropy`,
    and `token_entropy[*].instability`.
  - Added `trace.monte_carlo_analysis.token_instability_trace` for per-position stochastic drift.
  - API contract version advanced to 1.2.0 for the expanded response and registry surface.

---

CHANGELOG from v1.0.0 (B20)
  - Response shape extended with reliability metrics (confidence, instability,
    entropy, uncertainty, escalate, cluster metrics, failures, samples).
  - resampled and samples_used added to expose reliability guard state.
  - trace.monte_carlo_analysis added with full MC diagnostics and guard telemetry.
  - Determinism guarantee updated: applies to guardrail-overridden responses only.
    Probabilistic inference paths are intentionally non-deterministic.

---

1. Overview

This document defines the public API surface for contract version 1.2.0.

The engine guarantees:
  - Multilingual guardrail parity (English and Hindi).
  - Contract-backed guardrail override enforcement.
  - Strict input validation.
  - Sealed response shape.
  - No internal state leakage.
  - Reliability guard: automatic fallback resampling when instability > 0.25.

---

2. Endpoint: /generate

Method

POST

---

Request Body

{
  "prompt": "string (1-10000 characters)",
  "emotional_lang": "en" | "hi",
  "mode": "factual" | "emotional" | "mixed" | "",
  "temperature": number (0.0-2.0, default 0.7),
  "top_p": number (0.0-1.0, default 0.9),
  "max_new_tokens": integer (1-8192, default 128),
  "do_sample": boolean (default true),
  "monte_carlo_samples": integer (3-10, default 5)
}

Required fields: prompt
All other fields are optional with defaults as shown.

Invalid Input Behavior

{
  "error": "Error message",
  "code": "INVALID_INPUT"
}

HTTP Status: 400

---

3. Response Body

On success (HTTP 200):

{
  "response_text": "string",

  "confidence": number,
  "instability": number,
  "entropy": number,
  "uncertainty": number,
  "escalate": boolean,
  "sample_count": integer,
  "resampled": boolean,
  "samples_used": integer,
  "token_entropy_available": boolean,
  "token_entropy": [
    {
      "text": "string",
      "entropy": number,
      "instability": number | null
    }
  ],
  "semantic_dispersion": number,
  "cluster_count": integer,
  "cluster_entropy": number,
  "dominant_cluster_ratio": number,
  "self_consistency": number,
  "failures": string[],
  "samples": string[],

  "latency_ms": number,
  "input_tokens": integer,
  "output_tokens": integer,

  "trace": {
    "turn": { ... },
    "guardrail": { ... },
    "skeleton": { ... },
    "tone_profile": "string (optional)",
    "selection": { ... },
    "replay_hash": "sha256:...",
    "monte_carlo_analysis": {
      "sample_count": integer,
      "entropy_consistency": number,
      "entropy_variance": number,
      "semantic_dispersion": number,
      "pairwise_disagreement_entropy": number,
      "det_entropy_similarity": number,
      "entropy": number,
      "uncertainty": number,
      "uncertainty_level": "low" | "moderate" | "high" | "critical",
      "token_instability_trace": number[],
      "resampled": boolean,
      "samples_used": integer,
      "reliability_guard": {
        "triggered": boolean,
        "initial_instability": number,
        "threshold": number,
        "final_instability": number (only if triggered),
        "instability_delta": number (only if triggered),
        "improved": boolean (only if triggered),
        "fallback_temperature": number (only if triggered),
        "fallback_top_p": number (only if triggered),
        "fallback_samples_used": integer (only if triggered)
      }
    }
  },

  "core_comparison": {
    "core_a_output": "string",
    "core_b_output": "string",
    "embedding_similarity": number,
    "token_delta": integer,
    "length_delta": integer
  },

  "review_packet": {
    "entropy_samples": string[],
    "embedding_similarity": number,
    "ambiguity": number
  }
}

Note: core_comparison is always present. review_packet is only present when
escalate is true.

---

4. Reliability Guard

The engine runs a grid-sweep-informed fallback when instability exceeds 0.25
(empirically derived from a 400-inference parameter sweep over Qwen 2.5-7B).

When triggered:
  - Resamples at T=0.1, top_p=0.5 (optimal parameters from the grid sweep).
  - resampled: true in the response.
  - Full before/after telemetry in trace.monte_carlo_analysis.reliability_guard.

When not triggered:
  - resampled: false.
  - reliability_guard.triggered: false.

---

5. Determinism

Guardrail-overridden responses are deterministic for identical inputs.
Probabilistic inference paths (resampled or not) are intentionally
non-deterministic due to Monte Carlo sampling.

replay_hash is deterministic and stable across identical guardrail inputs.

---

6. Multilingual Support

Supported languages:
  - English ("en")
  - Hindi ("hi")

Guardrail resolution occurs in the requested language.
English fallback is used if the requested language block is missing.
If both are missing, a hard runtime error is raised.

---

7. Endpoint: /health

Method

GET

Response

{
  "status": "ok",
  "version": "string",
  "engine_ready": boolean,
  "queue_depth": integer,
  "queue_worker_running": boolean
}

---

8. Endpoint: /version

Method

GET

Response

{
  "engine_name": "indian-desi-llm-inference-core",
  "engine_version": "1.0.2",
  "release_stage": "B22"
}

---

9. Endpoint: /api/registry/history

Method

GET

Response

[
  {
    "release_id": "string",
    "model_id": "string",
    "git_commit_hash": "string",
    "weights_fingerprint": "string",
    "fragility_score": number,
    "status": "staging" | "production" | "rollback" | "retired",
    "environment": "string",
    "deployed_at": "ISO-8601 timestamp",
    "evaluation": {
      "evaluation_run_id": "string",
      "snapshot_id": "string",
      "verdict": "GO" | "NO_GO",
      "regression_summary": {
        "fatal_count": integer,
        "warning_count": integer,
        "info_count": integer,
        "triggered_checks": string[]
      }
    }
  }
]

---

10. Error Handling

400 - INVALID_INPUT

{
  "error": "Message",
  "code": "INVALID_INPUT"
}

500 - INFERENCE_FAILED

{
  "error": "Inference failed.",
  "code": "INFERENCE_FAILED"
}

503 - ENGINE_NOT_READY

{
  "error": "Message",
  "code": "ENGINE_NOT_READY"
}

No internal stack traces are exposed.

---

11. Engine Version

This document reflects API contract version 1.2.0 for engine release 1.0.2 (B22).
Changes to request or response shape constitute a breaking change
and require a version increment.
