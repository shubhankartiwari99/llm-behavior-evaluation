from __future__ import annotations

import hashlib
import logging
import json
import os
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
_REQUEST_KEYS = {"prompt", "emotional_lang", "mode", "temperature", "top_p", "max_new_tokens", "do_sample"}
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

engine: InferenceEngine | None = None


def _get_engine() -> InferenceEngine:
    global engine
    if engine is None:
        model_dir = os.environ.get("MODEL_DIR")
        if not model_dir and MODEL_BACKEND == "gguf":
            model_dir = "model_gguf"
        if not model_dir and MODEL_BACKEND == "remote":
            model_dir = "."  # unused for remote backend
        if not model_dir:
            raise RuntimeError("MODEL_DIR environment variable must be set.")
        engine = InferenceEngine(model_dir)
    return engine


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
        max_new_tokens = 128
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool):
        raise ValueError("Invalid max_new_tokens.")
    if max_new_tokens <= 0 or max_new_tokens > 8192:
        raise ValueError("Invalid max_new_tokens.")

    ds_val = payload.get("do_sample", True)
    if ds_val is None:
        ds_val = True
    if not isinstance(ds_val, bool):
        raise ValueError("Invalid do_sample.")

    return {
        "prompt": prompt,
        "emotional_lang": lang,
        "mode": mode,
        "temperature": t_val,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "do_sample": ds_val,
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


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "engine_version": ENGINE_VERSION,
        },
    )


@app.get("/health")
def health_check():
    return JSONResponse(status_code=200, content={"status": "ok", "version": ENGINE_VERSION})


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
        structured_prompt = build_prompt(validated.get("mode", ""), validated["prompt"])
        
        # Deterministic
        det_response_text, det_meta = runtime_engine.generate(
            structured_prompt,
            return_meta=True,
            temperature=0.0,
            top_p=1.0,
            do_sample=False,
            max_new_tokens=validated["max_new_tokens"],
        )
        
        # 5x Entropy runs (Monte Carlo execution for robust uncertainty estimation)
        from concurrent.futures import ThreadPoolExecutor
        
        def run_entropy():
            kwargs = {
                "temperature": validated["temperature"],
                "top_p": validated["top_p"],
                "do_sample": validated["do_sample"],
                "max_new_tokens": validated["max_new_tokens"],
            }
            return runtime_engine.generate(
                structured_prompt,
                return_meta=True,
                **kwargs
            )

        with ThreadPoolExecutor(max_workers=5) as executor:
            entropy_results = list(executor.map(lambda _: run_entropy(), range(5)))
            
        ent_outputs = [res[0] for res in entropy_results]
        ent_metas = [res[1] for res in entropy_results]
        core_b_output = ent_outputs[0] if ent_outputs else ""
        
        # We can use the first entropy run's token count for backwards compatibility,
        # or we could pass the entire list. `evaluate_dual_plane` now expects a list.
        det_token_count = det_meta.get("output_tokens", len(det_response_text.split()))
        ent_token_counts = [meta.get("output_tokens", len(out.split())) for out, meta in zip(ent_outputs, ent_metas)]
        first_entropy_token_count = ent_token_counts[0] if ent_token_counts else 0
        
        analysis = evaluate_dual_plane(
            det_response_text,
            ent_outputs,
            det_token_count,
            ent_token_counts
        )

        trace_data = _build_api_trace(structured_prompt, validated["emotional_lang"])
        trace_data["monte_carlo_analysis"] = {
            "entropy_consistency": analysis["entropy_consistency"],
            "entropy_variance": analysis["entropy_variance"],
            "det_entropy_similarity": analysis["det_entropy_similarity"],
        }

        response_payload = {
            "response_text": det_response_text,
            "latency_ms": det_meta.get("latency_ms", 0),
            "input_tokens": det_meta.get("input_tokens", 0),
            "output_tokens": det_token_count,
            "confidence": analysis["confidence"],
            "instability": analysis["instability"],
            "escalate": analysis["escalate"],
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
