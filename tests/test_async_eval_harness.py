from __future__ import annotations

import asyncio
import time

from app.eval.async_runner import AsyncEvalHarness


def test_async_eval_harness_respects_concurrency_limit():
    active = 0
    max_active = 0

    def runner(payload: dict) -> dict:
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        time.sleep(0.02)
        active -= 1
        return {"prompt": payload["prompt"]}

    harness = AsyncEvalHarness(runner=runner, concurrency=2, timeout_seconds=2)
    prompts = [{"prompt": f"prompt-{index}"} for index in range(5)]
    results = asyncio.run(harness.run_batch(prompts))

    assert [result["prompt"] for result in results] == [item["prompt"] for item in prompts]
    assert max_active <= 2


def test_async_eval_harness_returns_timeout_error_payload():
    def slow_runner(_payload: dict) -> dict:
        time.sleep(0.05)
        return {"ok": True}

    harness = AsyncEvalHarness(runner=slow_runner, concurrency=1, timeout_seconds=0.01)
    results = asyncio.run(harness.run_batch([{"prompt": "slow"}]))

    assert len(results) == 1
    assert results[0]["error_type"] == "timeout"
    assert results[0]["prompt"] == "slow"


def test_async_eval_harness_returns_error_payload_on_exception():
    def failing_runner(_payload: dict) -> dict:
        raise RuntimeError("boom")

    harness = AsyncEvalHarness(runner=failing_runner, concurrency=1, timeout_seconds=1)
    results = asyncio.run(harness.run_batch([{"prompt": "fail"}]))

    assert len(results) == 1
    assert results[0]["error"] == "boom"
    assert results[0]["error_type"] == "exception"
    assert results[0]["prompt"] == "fail"
