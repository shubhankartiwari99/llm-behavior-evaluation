from __future__ import annotations

import asyncio
import hashlib
import logging
import json
import os
import datetime
import threading
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

from app.guardrails.guardrail_classifier import classify_user_input
from app.guardrails.guardrail_escalation import compute_guardrail_escalation
from app.guardrails.guardrail_strategy import apply_guardrail_strategy
from app.engine_config import MODEL_BACKEND
from app.engine_identity import ENGINE_NAME, ENGINE_RELEASE_STAGE, ENGINE_VERSION
from app.inference import InferenceEngine
from app.tone.tone_calibration import calibrate_tone
from app.intelligence.dual_plane import evaluate_dual_plane

app = FastAPI(
    title="Indian Desi Multilingual LLM",
    description="Inference API for the Indian Desi Multilingual LLM",
    version="0.1.0",
)
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_PROMPT_LENGTH = 10000
ALLOWED_LANGS = {"en", "hi"}
_REQUEST_KEYS = {
    "prompt",
    "emotional_lang",
    "mode",
    "temperature",
    "top_p",
    "max_new_tokens",
    "do_sample",
    "monte_carlo_samples",
}
_TRACE_KEYS = ("turn", "guardrail", "skeleton", "tone_profile", "selection", "replay_hash")
_ALLOWED_MODES = {"", "factual", "emotional", "mixed"}
_MODE_ALIASES = {"explanatory": "factual"}

def build_prompt(mode: str, user_prompt: str):
    if mode == "factual":
        system = (
            "You are a formal educational assistant. "
            "Use professional tone. Do not use slang. "
            "Avoid casual expressions like 'buddy' or 'yaar'."
        )
    elif mode == "emotional":
        system = (
            "You are a compassionate and emotionally supportive assistant."
        )
    elif mode == "mixed":
        system = (
            "You may use a casual conversational tone."
        )
    else:
        system = ""

    if system:
        return system + "\n\nUser: " + user_prompt
    return user_prompt

_engine: InferenceEngine | None = None
_engine_lock = threading.Lock()
_engine_initialized = False
_request_queue: asyncio.Queue[dict[str, Any]] | None = None
_request_worker: asyncio.Task[None] | None = None
_request_worker_loop: asyncio.AbstractEventLoop | None = None


def _resolve_model_dir() -> str:
    model_dir = os.environ.get("MODEL_DIR")
    if not model_dir and MODEL_BACKEND == "gguf":
        model_dir = "model_gguf"
    if not model_dir and MODEL_BACKEND == "remote":
        model_dir = "."  # unused for remote backend
    if not model_dir:
        raise RuntimeError("MODEL_DIR environment variable must be set.")
    return model_dir


def _initialize_engine_once() -> InferenceEngine:
    global _engine
    global _engine_initialized

    if _engine is not None:
        return _engine

    with _engine_lock:
        if _engine is not None:
            return _engine

        model_dir = _resolve_model_dir()
        logging.info("Loading inference engine...")
        _engine = InferenceEngine(model_dir)
        _engine_initialized = True
        logging.info("Inference engine ready.")

    return _engine


def _get_engine() -> InferenceEngine:
    if _engine is None:
        raise RuntimeError("Inference engine not initialized")
    return _engine


def _validate_generate_request(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Invalid request body.")

    if set(payload.keys()) - _REQUEST_KEYS:
        raise ValueError("Unexpected fields in request.")

    prompt = payload.get("prompt")
    if not isinstance(prompt, str):
        raise ValueError("Prompt must be a string.")

    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    if len(prompt) > MAX_PROMPT_LENGTH:
        raise ValueError("Prompt exceeds maximum length.")

    lang = payload.get("emotional_lang", "en")
    if lang is None:
        lang = "en"

    if not isinstance(lang, str):
        raise ValueError("Invalid emotional_lang.")

    if lang not in ALLOWED_LANGS:
        raise ValueError("Unsupported emotional_lang.")

    mode = payload.get("mode", "")
    if mode is None:
        mode = ""
    if not isinstance(mode, str):
        raise ValueError("Invalid mode.")
    mode = _MODE_ALIASES.get(mode.strip().lower(), mode.strip().lower())
    if mode not in _ALLOWED_MODES:
        raise ValueError("Unsupported mode.")

    t_val = payload.get("temperature", 0.7)
    if t_val is None:
        t_val = 0.7
    if not isinstance(t_val, (int, float)) or isinstance(t_val, bool):
        raise ValueError("Invalid temperature.")
    t_val = float(t_val)
    if t_val < 0.0 or t_val > 2.0:
        raise ValueError("Invalid temperature.")

    top_p = payload.get("top_p", 0.9)
    if top_p is None:
        top_p = 0.9
    if not isinstance(top_p, (int, float)) or isinstance(top_p, bool):
        raise ValueError("Invalid top_p.")
    top_p = float(top_p)
    if top_p <= 0.0 or top_p > 1.0:
        raise ValueError("Invalid top_p.")

    max_new_tokens = payload.get("max_new_tokens", 128)
    if max_new_tokens is None:
        max_new_tokens = 80
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise ValueError("Invalid max_new_tokens.")
    if max_new_tokens <= 0 or max_new_tokens > 8192:
        raise ValueError("Invalid max_new_tokens.")

    ds_val = payload.get("do_sample", True)
    if ds_val is None:
        ds_val = True
    if not isinstance(ds_val, bool):
        raise ValueError("Invalid do_sample.")

    mc_samples = payload.get("monte_carlo_samples", 5)
    if mc_samples is None:
        mc_samples = 5
    if not isinstance(mc_samples, int) or isinstance(mc_samples, bool):
        raise ValueError("Invalid monte_carlo_samples.")
    if mc_samples < 3 or mc_samples > 10:
        raise ValueError("Invalid monte_carlo_samples.")

    return {
        "prompt": prompt,
        "emotional_lang": lang,
        "mode": mode,
        "temperature": t_val,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "do_sample": ds_val,
        "monte_carlo_samples": mc_samples,
    }


def _build_api_trace(prompt: str, emotional_lang: str) -> dict[str, Any]:
    guardrail_result = classify_user_input(prompt)
    guardrail_action = apply_guardrail_strategy(guardrail_result)
    base_skeleton = "A"
    skeleton_after_guardrail = compute_guardrail_escalation(guardrail_result, base_skeleton)
    if guardrail_result.risk_category == "SELF_HARM_RISK":
        skeleton_after_guardrail = "C"

    trace: dict[str, Any] = {
        "turn": {
            "emotional_turn_index": 1,
            "intent": "emotional",
            "emotional_lang": emotional_lang,
            "previous_skeleton": base_skeleton,
            "resolved_skeleton": skeleton_after_guardrail,
            "skeleton_transition": f"{base_skeleton}->{skeleton_after_guardrail}",
            "transition_legal": True,
            "escalation_state": "none",
            "latched_theme": None,
            "signals": {
                "overwhelm": False,
                "resignation": False,
                "guilt": False,
                "wants_action": False,
            },
        },
        "guardrail": {
            "classifier_version": guardrail_result.guardrail_schema_version,
            "strategy_version": guardrail_action.guardrail_strategy_version,
            "risk_category": guardrail_result.risk_category,
            "severity": guardrail_result.severity,
            "override": bool(guardrail_action.override),
        },
        "skeleton": {
            "base": base_skeleton,
            "after_guardrail": skeleton_after_guardrail,
        },
        "selection": {
            "eligible_count": 0,
            "selected_variant_indices": {},
        },
    }

    if not guardrail_action.override:
        try:
            trace["tone_profile"] = calibrate_tone(
                skeleton_after_guardrail,
                guardrail_result.severity,
                guardrail_result.risk_category,
            )
        except ValueError:
            pass

    replay_subset: dict[str, Any] = {
        "emotional_lang": emotional_lang,
        "skeleton_after_guardrail": skeleton_after_guardrail,
        "guardrail": trace["guardrail"],
        "selection": trace["selection"],
    }
    if "tone_profile" in trace:
        replay_subset["tone_profile"] = trace["tone_profile"]
    canonical = json.dumps(replay_subset, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    trace["replay_hash"] = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    sealed_trace: dict[str, Any] = {}
    for key in _TRACE_KEYS:
        if key in trace:
            sealed_trace[key] = trace[key]
    return sealed_trace


def _run_inference_pipeline(runtime_engine: InferenceEngine, validated: dict[str, Any]) -> dict[str, Any]:
    structured_prompt = build_prompt(validated.get("mode", ""), validated["prompt"])
    max_samples = validated["monte_carlo_samples"]
    min_samples = 4
    stability_threshold = 0.03

    _t0 = datetime.datetime.now()
    det_response_text, det_meta = runtime_engine.generate(
        structured_prompt,
        return_meta=True,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
        max_new_tokens=validated["max_new_tokens"],
    )

    det_token_count = det_meta.get("output_tokens", len(det_response_text.split()))

    entropy_kwargs = {
        "temperature": validated["temperature"],
        "top_p": validated["top_p"],
        "do_sample": validated["do_sample"],
        "max_new_tokens": validated["max_new_tokens"],
    }

    ent_outputs = []
    ent_metas = []
    instability_history = []
    final_analysis = None

    while len(ent_outputs) < max_samples:
        res_text, res_meta = runtime_engine.generate(
            structured_prompt,
            return_meta=True,
            **entropy_kwargs,
        )
        ent_outputs.append(res_text)
        ent_metas.append(res_meta)

        if len(ent_outputs) >= 2:
            current_ent_token_counts = [m.get("output_tokens", len(o.split())) for o, m in zip(ent_outputs, ent_metas)]
            analysis = evaluate_dual_plane(
                det_response_text,
                ent_outputs,
                det_token_count,
                current_ent_token_counts,
            )
            instability_history.append(analysis["instability"])

            if len(ent_outputs) >= min_samples and len(instability_history) >= 2:
                delta = abs(instability_history[-1] - instability_history[-2])
                if delta < stability_threshold:
                    final_analysis = analysis
                    break
            final_analysis = analysis

    core_b_output = ent_outputs[0] if ent_outputs else det_response_text

    ent_token_counts = [meta.get("output_tokens", len(out.split())) for out, meta in zip(ent_outputs, ent_metas)]
    first_entropy_token_count = ent_token_counts[0] if ent_token_counts else 0

    if final_analysis is None:
        analysis = evaluate_dual_plane(
            det_response_text,
            ent_outputs,
            det_token_count,
            ent_token_counts,
        )
    else:
        analysis = final_analysis

    trace_data = _build_api_trace(structured_prompt, validated["emotional_lang"])
    trace_data["monte_carlo_analysis"] = {
        "sample_count": analysis["sample_count"],
        "entropy_consistency": analysis["entropy_consistency"],
        "entropy_variance": analysis["entropy_variance"],
        "semantic_dispersion": analysis["semantic_dispersion"],
        "pairwise_disagreement_entropy": analysis["pairwise_disagreement_entropy"],
        "det_entropy_similarity": analysis["det_entropy_similarity"],
        "entropy": analysis["entropy"],
        "uncertainty": analysis["uncertainty"],
        "uncertainty_level": analysis["uncertainty_level"],
    }
    
    trace_log = []
    labels = analysis.get("cluster_labels", [])
    for idx, sample in enumerate(ent_outputs):
        label = labels[idx] if idx < len(labels) else 0
        trace_log.append({
            "text": sample,
            "cluster": int(label)
        })
        
    trace_data["monte_carlo_samples"] = trace_log

    _latency_ms = round((datetime.datetime.now() - _t0).total_seconds() * 1000, 2)

    response_payload = {
        "response_text": det_response_text,
        "latency_ms": _latency_ms,
        "input_tokens": det_meta.get("input_tokens", 0),
        "output_tokens": det_token_count,
        "confidence": analysis["confidence"],
        "instability": analysis["instability"],
        "entropy": analysis["entropy"],
        "uncertainty": analysis["uncertainty"],
        "escalate": analysis["escalate"],
        "sample_count": analysis["sample_count"],
        "samples_used": len(ent_outputs),
        "semantic_dispersion": analysis["semantic_dispersion"],
        "cluster_count": analysis["cluster_count"],
        "cluster_entropy": analysis["cluster_entropy"],
        "dominant_cluster_ratio": analysis["dominant_cluster_ratio"],
        "self_consistency": analysis["self_consistency"],
        "samples": ent_outputs,
        "core_comparison": {
            "core_a_output": det_response_text,
            "core_b_output": core_b_output,
            "embedding_similarity": analysis["det_entropy_similarity"],
            "token_delta": abs(det_token_count - first_entropy_token_count),
            "length_delta": abs(len(det_response_text) - len(core_b_output)),
        },
        "trace": trace_data,
    }

    if analysis["escalate"]:
        response_payload["review_packet"] = {
            "entropy_samples": ent_outputs,
            "embedding_similarity": analysis["embedding_similarity"],
            "ambiguity": analysis["ambiguity"],
        }

    return response_payload


async def _inference_worker(queue: asyncio.Queue[dict[str, Any]]) -> None:
    while True:
        job = await queue.get()
        future: asyncio.Future[dict[str, Any]] = job["future"]
        try:
            result = await asyncio.to_thread(
                _run_inference_pipeline,
                job["engine"],
                job["validated"],
            )
            if not future.cancelled():
                future.set_result(result)
        except Exception as exc:
            if not future.cancelled():
                future.set_exception(exc)
        finally:
            queue.task_done()


async def _ensure_inference_worker() -> asyncio.Queue[dict[str, Any]]:
    global _request_queue
    global _request_worker
    global _request_worker_loop

    loop = asyncio.get_running_loop()
    needs_new_worker = (
        _request_queue is None
        or _request_worker is None
        or _request_worker.done()
        or _request_worker_loop is not loop
    )

    if needs_new_worker:
        _request_queue = asyncio.Queue()
        _request_worker = loop.create_task(_inference_worker(_request_queue))
        _request_worker_loop = loop
        logging.info("Inference queue worker started.")

    return _request_queue


async def _enqueue_inference(runtime_engine: InferenceEngine, validated: dict[str, Any]) -> dict[str, Any]:
    queue = await _ensure_inference_worker()
    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, Any]] = loop.create_future()

    await queue.put(
        {
            "engine": runtime_engine,
            "validated": validated,
            "future": future,
        }
    )
    return await future


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "engine_version": ENGINE_VERSION,
        },
    )


@app.on_event("startup")
async def load_engine_on_startup():
    # Tests can disable eager load and inject their own stub engine.
    if os.environ.get("SKIP_ENGINE_STARTUP") == "1" or os.environ.get("PYTEST_CURRENT_TEST"):
        logging.info("Skipping startup engine initialization.")
        return

    _initialize_engine_once()
    await _ensure_inference_worker()


@app.on_event("shutdown")
async def shutdown_inference_queue_worker():
    global _request_queue
    global _request_worker
    global _request_worker_loop

    if _request_worker is not None and not _request_worker.done():
        _request_worker.cancel()
        try:
            await _request_worker
        except asyncio.CancelledError:
            pass

    _request_queue = None
    _request_worker = None
    _request_worker_loop = None


@app.get("/health")
def health_check():
    queue_depth = _request_queue.qsize() if _request_queue is not None else 0
    worker_running = bool(_request_worker is not None and not _request_worker.done())
    return JSONResponse(
        status_code=200,
        content={
            "status": "ok",
            "version": ENGINE_VERSION,
            "engine_ready": _engine_initialized,
            "queue_depth": queue_depth,
            "queue_worker_running": worker_running,
        },
    )


@app.post("/generate")
async def generate_text(request: Request):
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "Invalid request body.", "code": "INVALID_INPUT"})

    try:
        validated = _validate_generate_request(payload)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc), "code": "INVALID_INPUT"})

    try:
        runtime_engine = _get_engine()
    except RuntimeError as exc:
        logging.exception("Engine not ready: %s", exc)
        return JSONResponse(status_code=503, content={"error": str(exc), "code": "ENGINE_NOT_READY"})

    try:
        response_payload = await _enqueue_inference(runtime_engine, validated)
        return JSONResponse(status_code=200, content=response_payload)
    except Exception as exc:
        logging.exception("Generate failed: %s", exc)
        return JSONResponse(status_code=500, content={"error": "Inference failed.", "code": "INFERENCE_FAILED"})


@app.get("/version")
def version():
    return {
        "engine_name": ENGINE_NAME,
        "engine_version": ENGINE_VERSION,
        "release_stage": ENGINE_RELEASE_STAGE,
    }
