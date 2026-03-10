"""
CI Behavioral Gate — regression referee for model releases.

Runs the inference benchmark harness against a set of prompts, captures the
behavioural snapshot, and compares it against a frozen baseline.

Exit codes:
  0  → verdict is GO
  1  → verdict is NO_GO (fatal regressions detected) or runtime error
  2  → argument error

Usage:
  python scripts/ci_behavioral_gate.py \\
      --baseline  artifacts/behavioral_baseline.json \\
      --prompts   eval/prompts_adversarial.json \\
      [--output   artifacts/regression_report.json] \\
      [--model-id "Qwen/Qwen2.5-7B-Instruct"] \\
      [--dry-run]           # parse + validate baseline only, no inference
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# ── Make sure the project root is importable ──────────────────────────────────
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ci_behavioral_gate")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Behavioral regression gate for model releases.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (requires inference engine)
  python scripts/ci_behavioral_gate.py \\
      --baseline artifacts/behavioral_baseline.json \\
      --prompts eval/prompts_adversarial.json

  # Dry-run: validate baseline schema only (no inference, CI-safe)
  python scripts/ci_behavioral_gate.py \\
      --baseline artifacts/behavioral_baseline.json \\
      --dry-run
        """,
    )
    parser.add_argument(
        "--baseline",
        default="artifacts/behavioral_baseline.json",
        help="Path to frozen baseline snapshot JSON.",
    )
    parser.add_argument(
        "--prompts",
        default=None,
        help="Path to prompts JSON (array of {prompt, category} objects).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to write regression report JSON.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override model ID in snapshot metadata (defaults to ENGINE_NAME).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate baseline schema only; skip inference and exit 0.",
    )
    return parser.parse_args()


def _load_prompts(path: str) -> list[dict]:
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    if isinstance(data, list):
        return data
    # Support {dataset: [...]} or {prompts: [...]} wrappers
    return data.get("dataset", data.get("prompts", []))


def _run_inline_benchmark(
    prompts: list[dict],
    model_id: str,
) -> dict:
    """
    Runs the inference harness inline (no HTTP server needed).
    Requires MODEL_DIR or the remote backend to be configured.
    """
    import asyncio
    from app.api import run_inference_pipeline
    from app.inference import InferenceEngine
    from app.engine_config import MODEL_BACKEND
    from app.eval.benchmark_runner import summarize_benchmark

    # Resolve engine
    model_dir = os.environ.get("MODEL_DIR", "model_gguf" if MODEL_BACKEND == "gguf" else ".")
    log.info("Loading inference engine from %s …", model_dir)
    engine = InferenceEngine(model_dir)

    base_params = {
        "emotional_lang": "en",
        "mode": "",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 128,
        "do_sample": True,
        "monte_carlo_samples": 5,
    }

    results = []
    total = len(prompts)
    for i, item in enumerate(prompts, 1):
        prompt_text = item.get("prompt", "") if isinstance(item, dict) else str(item)
        if not prompt_text.strip():
            continue
        payload = {**base_params, "prompt": prompt_text}
        log.info("  [%d/%d] %s", i, total, prompt_text[:60])
        try:
            result = asyncio.get_event_loop().run_until_complete(
                asyncio.to_thread(run_inference_pipeline, engine, payload)
            )
            results.append(result)
        except Exception as exc:
            log.warning("  Prompt %d failed (%s) — skipping.", i, exc)

    summary = summarize_benchmark(results)
    summary["model"] = model_id
    return summary


def main() -> int:
    args = _parse_args()

    # ── Load and validate baseline ────────────────────────────────────────
    from app.eval.behavioral_snapshot import BehavioralSnapshot, SnapshotCapturer
    from app.eval.delta_calculator import DeltaCalculator

    baseline_path = Path(args.baseline)
    if not baseline_path.exists():
        log.error("Baseline not found: %s", baseline_path)
        return 1

    log.info("Loading baseline: %s", baseline_path)
    try:
        baseline = BehavioralSnapshot.load(baseline_path)
    except Exception as exc:
        log.error("Failed to parse baseline snapshot: %s", exc)
        return 1

    log.info(
        "  model_id=%s  schema=%s  eval_prompts=%d",
        baseline.snapshot_metadata.model_id,
        baseline.snapshot_metadata.schema_version,
        baseline.snapshot_metadata.eval_prompt_count,
    )

    if args.dry_run:
        log.info("Dry-run complete — baseline schema is valid. ✓")
        return 0

    # ── Run benchmark ─────────────────────────────────────────────────────
    if not args.prompts:
        log.error("--prompts is required unless --dry-run is set.")
        return 2

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        log.error("Prompts file not found: %s", prompts_path)
        return 1

    prompts = _load_prompts(str(prompts_path))
    log.info("Loaded %d prompts from %s", len(prompts), prompts_path)

    try:
        from app.engine_identity import ENGINE_NAME
        model_id = args.model_id or ENGINE_NAME
        summary = _run_inline_benchmark(prompts, model_id)
    except Exception as exc:
        log.error("Benchmark run failed: %s", exc)
        return 1

    # ── Capture current snapshot ──────────────────────────────────────────
    current_snapshot = SnapshotCapturer.from_benchmark_summary(
        summary,
        metadata={"model_id": model_id},
    )
    log.info(
        "Current snapshot: entropy=%.4f  instability=%.4f  escalation_rate=%.4f",
        current_snapshot.behavioral_invariants.reasoning.mean_entropy,
        current_snapshot.behavioral_invariants.reasoning.mean_instability,
        current_snapshot.behavioral_invariants.safety.escalation_rate,
    )

    # ── Compute drift ─────────────────────────────────────────────────────
    calculator = DeltaCalculator(baseline_path)
    report = calculator.compute_drift(current_snapshot)

    # ── Print report ──────────────────────────────────────────────────────
    print("\n" + report.render() + "\n")

    # ── Persist JSON report (optional) ────────────────────────────────────
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report.to_json(), encoding="utf-8")
        log.info("Report written to %s", output_path)

    return 0 if report.verdict == "GO" else 1


if __name__ == "__main__":
    sys.exit(main())
