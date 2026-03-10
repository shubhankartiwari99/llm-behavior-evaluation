from __future__ import annotations

import importlib
import sys
import types

from fastapi.testclient import TestClient


def _stable_analysis(_det_text: str, ent_outputs: list[str], _det_count: int, _ent_counts: list[int]) -> dict:
    sample_count = len(ent_outputs)
    return {
        "instability": 0.10,
        "confidence": 0.85,
        "entropy": 0.22,
        "uncertainty": 0.18,
        "escalate": False,
        "sample_count": sample_count,
        "semantic_dispersion": 0.05,
        "cluster_count": 1,
        "cluster_entropy": 0.0,
        "dominant_cluster_ratio": 1.0,
        "self_consistency": 0.9,
        "det_entropy_similarity": 0.95,
        "entropy_consistency": 0.9,
        "entropy_variance": 0.01,
        "pairwise_disagreement_entropy": 0.02,
        "uncertainty_level": "low",
        "embedding_similarity": 0.95,
        "ambiguity": 0.05,
        "cluster_labels": [0 for _ in range(sample_count)],
    }


def _load_api_module():
    dual_plane_stub = types.ModuleType("app.intelligence.dual_plane")
    dual_plane_stub.evaluate_dual_plane = _stable_analysis
    sys.modules["app.intelligence.dual_plane"] = dual_plane_stub
    sys.modules.pop("app.api", None)
    return importlib.import_module("app.api")


class _TokenTraceStubEngine:
    def __init__(self):
        self.calls = 0
        self.sample_traces = [
            [
                {"text": "alpha", "entropy": 0.08, "token_id": 11, "logprob": -0.12},
                {"text": " beta", "entropy": 0.20, "token_id": 21, "logprob": -0.20},
            ],
            [
                {"text": "alpha", "entropy": 0.07, "token_id": 11, "logprob": -0.14},
                {"text": " gamma", "entropy": 0.42, "token_id": 22, "logprob": -0.72},
            ],
            [
                {"text": "alpha", "entropy": 0.09, "token_id": 11, "logprob": -0.11},
                {"text": " beta", "entropy": 0.18, "token_id": 21, "logprob": -0.24},
            ],
            [
                {"text": "alpha", "entropy": 0.10, "token_id": 11, "logprob": -0.15},
                {"text": " gamma", "entropy": 0.45, "token_id": 22, "logprob": -0.68},
            ],
        ]

    def generate(self, _prompt: str, max_new_tokens: int = 64, return_meta: bool = False, **_kwargs):
        self.calls += 1
        if self.calls == 1:
            meta = {
                "output_tokens": 2,
                "input_tokens": 5,
                "token_entropy_available": True,
                "token_trace": [
                    {"text": "alpha", "entropy": 0.08, "token_id": 11, "logprob": -0.10},
                    {"text": " beta", "entropy": 0.19, "token_id": 21, "logprob": -0.18},
                ],
                "token_entropy": [
                    {"text": "alpha", "entropy": 0.08},
                    {"text": " beta", "entropy": 0.19},
                ],
            }
            return "alpha beta", meta

        trace = self.sample_traces[(self.calls - 2) % len(self.sample_traces)]
        meta = {
            "output_tokens": len(trace),
            "token_trace": trace,
        }
        return f"sample_{self.calls}", meta


def test_compute_per_token_instability_ranks_divergent_positions_higher():
    api_module = _load_api_module()
    mc_metas = [
        {
            "token_trace": [
                {"token_id": 1, "logprob": -0.11},
                {"token_id": 2, "logprob": -0.20},
            ]
        },
        {
            "token_trace": [
                {"token_id": 1, "logprob": -0.14},
                {"token_id": 3, "logprob": -0.74},
            ]
        },
        {
            "token_trace": [
                {"token_id": 1, "logprob": -0.10},
                {"token_id": 2, "logprob": -0.22},
            ]
        },
        {
            "token_trace": [
                {"token_id": 1, "logprob": -0.15},
                {"token_id": 3, "logprob": -0.70},
            ]
        },
    ]

    trace = api_module.compute_per_token_instability(mc_metas)

    assert len(trace) == 2
    assert trace[0] >= 0.0
    assert trace[1] > trace[0]


def test_generate_enriches_token_entropy_with_per_token_instability(monkeypatch):
    api_module = _load_api_module()
    monkeypatch.setattr(api_module, "_get_engine", lambda: _TokenTraceStubEngine())
    monkeypatch.setattr(api_module, "evaluate_dual_plane", _stable_analysis)
    monkeypatch.setattr(api_module, "detect_failures", lambda *_args, **_kwargs: [])

    client = TestClient(api_module.app)
    response = client.post(
        "/generate",
        json={"prompt": "Explain caching simply.", "monte_carlo_samples": 4},
    )

    assert response.status_code == 200
    body = response.json()
    tokens = body["token_entropy"]

    assert body["token_entropy_available"] is True
    assert len(tokens) == 2
    assert isinstance(tokens[0]["instability"], float)
    assert isinstance(tokens[1]["instability"], float)
    assert tokens[1]["instability"] > tokens[0]["instability"]
    assert body["trace"]["monte_carlo_analysis"]["token_instability_trace"] == [
        tokens[0]["instability"],
        tokens[1]["instability"],
    ]
