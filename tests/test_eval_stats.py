from __future__ import annotations

from app.eval.stats import StatisticalReferee


def test_wasserstein_distance_is_zero_for_identical_distributions():
    baseline = [0.2, 0.25, 0.3, 0.35]
    current = [0.2, 0.25, 0.3, 0.35]
    assert StatisticalReferee.calculate_distribution_shift(baseline, current) == 0.0


def test_wasserstein_distance_detects_bimodal_shift():
    baseline = [0.25, 0.28, 0.30, 0.27, 0.29, 0.26, 0.28, 0.30, 0.25, 0.32]
    current = [0.12, 0.12, 0.12, 0.12, 0.12, 0.44, 0.44, 0.44, 0.44, 0.44]
    distance = StatisticalReferee.calculate_distribution_shift(baseline, current)
    assert distance > 0.1
    assert StatisticalReferee.is_statistically_significant(distance, threshold=0.1) is True


def test_distribution_summary_reports_shape_statistics():
    summary = StatisticalReferee.get_distribution_summary([0.1, 0.2, 0.3, 0.4, 0.5])
    assert summary["std_dev"] > 0.0
    assert summary["variance"] > 0.0
    assert summary["p50"] == 0.3
    assert summary["p95"] >= 0.4
    assert summary["modality_check"] == "unimodal"
