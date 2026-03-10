from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, List

log = logging.getLogger("async_runner")

class AsyncEvalHarness:
    def __init__(
        self,
        engine: Any | None = None,
        concurrency: int = 8,
        timeout_seconds: float = 180,
        runner: Callable[[Dict[str, Any]], Any] | None = None,
    ):
        if engine is None and runner is None:
            raise ValueError("AsyncEvalHarness requires an engine or runner.")

        self.engine = engine
        self.runner = runner
        self.concurrency = max(1, int(concurrency))
        self.timeout_seconds = float(timeout_seconds)

    def _resolve_runner(self) -> Callable[[Dict[str, Any]], Any]:
        if self.runner is not None:
            return self.runner

        def _engine_runner(payload: Dict[str, Any]) -> Any:
            from app.api import run_inference_pipeline

            return run_inference_pipeline(self.engine, payload)

        return _engine_runner

    async def _run_single_exploration(
        self,
        payload: Dict[str, Any],
        semaphore: asyncio.Semaphore,
    ) -> Dict[str, Any]:
        async with semaphore:
            runner = self._resolve_runner()
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(runner, payload),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                prompt = str(payload.get("prompt", ""))
                log.error(
                    "Inference timed out for prompt: %s...",
                    prompt[:50] or "N/A",
                )
                return {
                    "error": f"Timed out after {self.timeout_seconds:g}s",
                    "error_type": "timeout",
                    "prompt": payload.get("prompt"),
                }
            except Exception as exc:
                prompt = str(payload.get("prompt", ""))
                log.error(
                    "Inference failed for prompt: %s... Error: %s",
                    prompt[:50] or "N/A",
                    exc,
                )
                return {
                    "error": str(exc),
                    "error_type": "exception",
                    "prompt": payload.get("prompt"),
                }

    async def run_batch(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        log.info(
            "Starting async batch execution for %d tasks (concurrency=%d)",
            len(payloads),
            self.concurrency,
        )
        semaphore = asyncio.Semaphore(self.concurrency)
        tasks = [
            asyncio.create_task(self._run_single_exploration(payload, semaphore))
            for payload in payloads
        ]
        results = await asyncio.gather(*tasks)
        log.info("Batch execution complete.")
        return results
