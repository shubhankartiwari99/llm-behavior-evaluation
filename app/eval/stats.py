from __future__ import annotations

from bisect import bisect_right
from typing import Iterable

import numpy as np

try:
    from scipy.stats import wasserstein_distance as _scipy_wasserstein_distance
except Exception:  # pragma: no cover - fallback only used when scipy is absent.
    _scipy_wasserstein_distance = None


def _manual_wasserstein_distance(
    baseline_probs: list[float],
    current_probs: list[float],
) -> float:
    baseline = sorted(float(value) for value in baseline_probs)
    current = sorted(float(value) for value in current_probs)
    support = sorted(set(baseline + current))
    if len(support) <= 1:
        return 0.0

    distance = 0.0
    for left, right in zip(support, support[1:]):
        baseline_cdf = bisect_right(baseline, left) / len(baseline)
        current_cdf = bisect_right(current, left) / len(current)
        distance += abs(baseline_cdf - current_cdf) * (right - left)
    return distance


class StatisticalReferee:
    """Detects subtle shifts in model probability distributions."""

    @staticmethod
    def calculate_distribution_shift(
        baseline_probs: Iterable[float],
        current_probs: Iterable[float],
    ) -> float:
        baseline = [float(value) for value in baseline_probs]
        current = [float(value) for value in current_probs]
        if not baseline or not current:
            return 0.0
        if _scipy_wasserstein_distance is not None:
            return float(_scipy_wasserstein_distance(baseline, current))
        return float(_manual_wasserstein_distance(baseline, current))

    @staticmethod
    def is_statistically_significant(
        distance: float,
        threshold: float = 0.05,
    ) -> bool:
        return float(distance) > float(threshold)

    @staticmethod
    def get_distribution_summary(trace: Iterable[float]) -> dict[str, float | str]:
        values = np.asarray([float(value) for value in trace], dtype=float)
        if values.size == 0:
            return {
                "std_dev": 0.0,
                "variance": 0.0,
                "p50": 0.0,
                "p95": 0.0,
                "modality_check": "unimodal",
            }
        variance = float(np.var(values))
        return {
            "std_dev": float(np.std(values)),
            "variance": variance,
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "modality_check": "bimodal" if values.size > 10 and variance > 0.1 else "unimodal",
        }
