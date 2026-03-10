from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from app import api as api_module
from app.inference import InferenceEngine


class _StubEngine:
    def __init__(self, text: str = "stub-response"):
        self.text = text
        self.calls = []

    def generate(self, prompt: str, max_new_tokens: int = 64, return_meta: bool = False, **kwargs):
        self.calls.append({"prompt": prompt, "max_new_tokens": max_new_tokens, "return_meta": return_meta})
        if return_meta:
            return self.text, {}
        return self.text


def _inference_engine_stub(*, last_skeleton: str = "A") -> InferenceEngine:
    engine = object.__new__(InferenceEngine)
    engine.voice_state = SimpleNamespace(
        escalation_state="none",
        latched_theme=None,
        emotional_turn_index=0,
        last_skeleton=last_skeleton,
    )
    engine._voice_state_turn_snapshot = None
    return engine


def _client_with_engine(monkeypatch, engine) -> TestClient:
    monkeypatch.setattr(api_module, "_get_engine", lambda: engine)
    return TestClient(api_module.app)


def _assert_invalid(client: TestClient, payload, message: str):
    response = client.post("/generate", json=payload)
    assert response.status_code == 400
    assert response.json() == {"error": message, "code": "INVALID_INPUT"}


# ================================================
# SECTION A — Schema Validation (6 tests)
# ================================================


def test_b19_api_a1_missing_prompt(monkeypatch):
    client = _client_with_engine(monkeypatch, _StubEngine())
    _assert_invalid(client, {"emotional_lang": "en"}, "Prompt must be a string.")


def test_b19_api_a2_prompt_not_string(monkeypatch):
    client = _client_with_engine(monkeypatch, _StubEngine())
    _assert_invalid(client, {"prompt": 123, "emotional_lang": "en"}, "Prompt must be a string.")


def test_b19_api_a3_empty_prompt(monkeypatch):
    client = _client_with_engine(monkeypatch, _StubEngine())
    _assert_invalid(client, {"prompt": "", "emotional_lang": "en"}, "Prompt cannot be empty.")


def test_b19_api_a4_whitespace_prompt(monkeypatch):
    client = _client_with_engine(monkeypatch, _StubEngine())
    _assert_invalid(client, {"prompt": "   ", "emotional_lang": "en"}, "Prompt cannot be empty.")


def test_b19_api_a5_prompt_too_long(monkeypatch):
    client = _client_with_engine(monkeypatch, _StubEngine())
    _assert_invalid(client, {"prompt": "a" * 10001, "emotional_lang": "en"}, "Prompt exceeds maximum length.")


def test_b19_api_a6_extra_field_rejected(monkeypatch):
    client = _client_with_engine(monkeypatch, _StubEngine())
    _assert_invalid(
        client,
        {"prompt": "hello", "emotional_lang": "en", "extra": True},
        "Unexpected fields in request.",
    )


# ================================================
# SECTION B — emotional_lang Validation (4 tests)
# ================================================


def test_b19_api_b1_missing_emotional_lang_defaults_to_en(monkeypatch):
    client = _client_with_engine(monkeypatch, _StubEngine("ok"))
    response = client.post("/generate", json={"prompt": "Tell me something uplifting."})
    assert response.status_code == 200
    body = response.json()
    assert body["response_text"] == "ok"
    assert body["trace"]["turn"]["emotional_lang"] == "en"


def test_b19_api_b2_unsupported_emotional_lang_rejected(monkeypatch):
    client = _client_with_engine(monkeypatch, _StubEngine())
    _assert_invalid(client, {"prompt": "hello", "emotional_lang": "fr"}, "Unsupported emotional_lang.")


def test_b19_api_b3_none_emotional_lang_defaults_to_en(monkeypatch):
    client = _client_with_engine(monkeypatch, _StubEngine("ok"))
    response = client.post("/generate", json={"prompt": "Tell me something uplifting.", "emotional_lang": None})
    assert response.status_code == 200
    body = response.json()
    assert body["response_text"] == "ok"
    assert body["trace"]["turn"]["emotional_lang"] == "en"


def test_b19_api_b4_non_string_emotional_lang_rejected(monkeypatch):
    client = _client_with_engine(monkeypatch, _StubEngine())
    _assert_invalid(client, {"prompt": "hello", "emotional_lang": 7}, "Invalid emotional_lang.")


# ================================================
# SECTION C — Valid Response Shape (6 tests)
# ================================================


def test_b19_api_c1_safe_prompt_returns_response_text_and_trace(monkeypatch):
    client = _client_with_engine(monkeypatch, _StubEngine("safe-response"))
    response = client.post("/generate", json={"prompt": "Tell me something uplifting.", "emotional_lang": "en"})
    assert response.status_code == 200
    body = response.json()
    assert body["response_text"] == "safe-response"
    assert "trace" in body


def test_b19_api_c2_response_shape_is_sealed(monkeypatch):
    client = _client_with_engine(monkeypatch, _StubEngine("safe-response"))
    response = client.post("/generate", json={"prompt": "Tell me something uplifting.", "emotional_lang": "en"})
    body = response.json()
    assert {"response_text", "trace", "confidence", "instability", "escalate"}.issubset(body.keys())
    assert set(body["trace"].keys()) <= {
        "turn", "guardrail", "skeleton", "tone_profile", "selection", "replay_hash", "monte_carlo_analysis"
    }
    assert "turn" in body["trace"]
    assert "guardrail" in body["trace"]
    assert "skeleton" in body["trace"]
    assert "selection" in body["trace"]


def test_b19_api_c3_replay_hash_present(monkeypatch):
    client = _client_with_engine(monkeypatch, _StubEngine("safe-response"))
    response = client.post("/generate", json={"prompt": "Tell me something uplifting.", "emotional_lang": "en"})
    body = response.json()
    assert body["trace"]["replay_hash"].startswith("sha256:")


def test_b19_api_c4_response_structure_deterministic(monkeypatch):
    client = _client_with_engine(monkeypatch, _StubEngine("safe-response"))
    payload = {"prompt": "Tell me something uplifting.", "emotional_lang": "en"}
    first = client.post("/generate", json=payload).json()
    second = client.post("/generate", json=payload).json()
    assert first == second


def test_b19_api_c5_resampled_and_samples_used_present(monkeypatch):
    """resampled and samples_used must be present on every /generate response."""
    client = _client_with_engine(monkeypatch, _StubEngine("safe-response"))
    response = client.post("/generate", json={"prompt": "Tell me something uplifting.", "emotional_lang": "en"})
    assert response.status_code == 200
    body = response.json()
    assert "resampled" in body, "resampled missing from response"
    assert "samples_used" in body, "samples_used missing from response"
    assert isinstance(body["resampled"], bool)
    assert isinstance(body["samples_used"], int)


def test_b19_api_c6_reliability_guard_telemetry_in_trace(monkeypatch):
    """reliability_guard block must be present inside trace.monte_carlo_analysis."""
    client = _client_with_engine(monkeypatch, _StubEngine("safe-response"))
    response = client.post("/generate", json={"prompt": "Tell me something uplifting.", "emotional_lang": "en"})
    assert response.status_code == 200
    body = response.json()

    mc = body.get("trace", {}).get("monte_carlo_analysis", {})
    assert mc, "monte_carlo_analysis missing from trace"

    guard = mc.get("reliability_guard")
    assert guard is not None, "reliability_guard missing from monte_carlo_analysis"
    assert "triggered" in guard
    assert "initial_instability" in guard
    assert "threshold" in guard
    assert isinstance(guard["triggered"], bool)


# ================================================
# SECTION D — Override Safety (3 tests)
# ================================================


def test_b19_api_d1_self_harm_critical_returns_override_text(monkeypatch):
    engine = _inference_engine_stub()
    engine._model_generate_cleaned = lambda prompt, max_new_tokens, **kwargs: (_ for _ in ()).throw(
        AssertionError("model must not run on override")
    )
    engine.handle_user_input = lambda _text: (_ for _ in ()).throw(
        AssertionError("handle_user_input must not run on override")
    )
    client = _client_with_engine(monkeypatch, engine)

    response = client.post("/generate", json={"prompt": "I want to die", "emotional_lang": "en"})
    assert response.status_code == 200
    body = response.json()
    assert (
        body["response_text"]
        == "I'm really sorry that you're feeling this way. You deserve support, and reaching out to someone you trust could help."
    )
    assert body["trace"]["guardrail"]["override"] is True


def test_b19_api_d2_override_path_never_calls_model(monkeypatch):
    engine = _inference_engine_stub()
    engine._model_generate_cleaned = lambda prompt, max_new_tokens, **kwargs: (_ for _ in ()).throw(
        AssertionError("model must not run on override")
    )
    client = _client_with_engine(monkeypatch, engine)

    response = client.post("/generate", json={"prompt": "I want to die", "emotional_lang": "en"})
    assert response.status_code == 200
    assert "error" not in response.json()


def test_b19_api_d3_override_output_deterministic(monkeypatch):
    engine = _inference_engine_stub()
    engine._model_generate_cleaned = lambda prompt, max_new_tokens, **kwargs: (_ for _ in ()).throw(
        AssertionError("model must not run on override")
    )
    client = _client_with_engine(monkeypatch, engine)

    payload = {"prompt": "I want to die", "emotional_lang": "en"}
    first = client.post("/generate", json=payload).json()
    second = client.post("/generate", json=payload).json()
    assert first == second


# ================================================
# SECTION E — Idempotency (3 tests)
# ================================================


def test_b19_api_e1_same_request_ten_times_identical_json(monkeypatch):
    client = _client_with_engine(monkeypatch, _StubEngine("idempotent"))
    payload = {"prompt": "Tell me something uplifting.", "emotional_lang": "en"}
    responses = [client.post("/generate", json=payload).json() for _ in range(10)]
    assert len({str(r) for r in responses}) == 1


def test_b19_api_e2_english_and_hindi_each_stable(monkeypatch):
    client = _client_with_engine(monkeypatch, _StubEngine("stable"))
    payload_en = {"prompt": "Tell me something uplifting.", "emotional_lang": "en"}
    payload_hi = {"prompt": "Tell me something uplifting.", "emotional_lang": "hi"}

    en_responses = [client.post("/generate", json=payload_en).json() for _ in range(10)]
    hi_responses = [client.post("/generate", json=payload_hi).json() for _ in range(10)]

    assert len({str(r) for r in en_responses}) == 1
    assert len({str(r) for r in hi_responses}) == 1
    assert en_responses[0]["trace"]["replay_hash"] != hi_responses[0]["trace"]["replay_hash"]


def test_b19_api_e3_no_random_fields_across_repeated_requests(monkeypatch):
    client = _client_with_engine(monkeypatch, _StubEngine("stable"))
    payload = {"prompt": "Tell me something uplifting.", "emotional_lang": "en"}
    responses = [client.post("/generate", json=payload).json() for _ in range(10)]

    top_key_sets = {tuple(sorted(r.keys())) for r in responses}
    trace_key_sets = {tuple(sorted(r["trace"].keys())) for r in responses}
    assert all("response_text" in keys and "trace" in keys for keys in top_key_sets)
    assert len(trace_key_sets) == 1
    assert len(top_key_sets) == 1
