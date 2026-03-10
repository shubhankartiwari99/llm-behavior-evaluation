from __future__ import annotations

from pathlib import Path

import pytest

from app.registry.manager import RegistryManager
from app.registry.schema import (
    DeploymentEntry,
    EvaluationEvidence,
    RegressionSummary,
    ReleaseStatus,
    ReleaseVerdict,
)


def _evidence(verdict: ReleaseVerdict = ReleaseVerdict.GO) -> EvaluationEvidence:
    return EvaluationEvidence(
        evaluation_run_id="eval-123",
        snapshot_id="snapshot-123",
        verdict=verdict,
        regression_summary=RegressionSummary(
            fatal_count=0,
            warning_count=1 if verdict == ReleaseVerdict.GO else 2,
            info_count=0,
            triggered_checks=["reasoning.mean_entropy"],
        ),
        policy_version="delta-calculator-v1",
        harness_version="harness-v1",
        dataset_id="dataset-v1",
        snapshot_uri="snapshots/snapshot-123.json",
        report_uri="reports/eval-123.json",
    )


def _entry(
    *,
    release_id: str,
    model_id: str,
    status: ReleaseStatus = ReleaseStatus.STAGING,
    verdict: ReleaseVerdict = ReleaseVerdict.GO,
    environment: str = "prod-us",
    approved_by: str | None = None,
    previous_release_id: str | None = None,
    rollback_of_release_id: str | None = None,
) -> DeploymentEntry:
    return DeploymentEntry(
        release_id=release_id,
        model_family="qwen-2.5-7b",
        model_id=model_id,
        git_commit_hash="a" * 40,
        weights_fingerprint="sha256:" + "b" * 64,
        inference_config_fingerprint="sha256:" + "c" * 64,
        evaluation=_evidence(verdict),
        status=status,
        environment=environment,
        deployed_by="ci-bot",
        approved_by=approved_by,
        previous_release_id=previous_release_id,
        rollback_of_release_id=rollback_of_release_id,
        is_adversarially_verified=True,
        fragility_score=0.08,
        runtime_metadata={"temperature": 0.7, "top_p": 0.9},
    )


def _manager(tmp_path: Path) -> RegistryManager:
    return RegistryManager(
        history_path=tmp_path / "history.jsonl",
        active_path=tmp_path / "active.json",
    )


def test_production_requires_go_verdict_and_approver():
    with pytest.raises(ValueError, match="GO evaluation verdict"):
        _entry(
            release_id="rel-001",
            model_id="qwen-2.5-7b-v1",
            status=ReleaseStatus.PRODUCTION,
            verdict=ReleaseVerdict.NO_GO,
            approved_by="review-board",
        )

    with pytest.raises(ValueError, match="approved_by"):
        _entry(
            release_id="rel-002",
            model_id="qwen-2.5-7b-v1",
            status=ReleaseStatus.PRODUCTION,
            verdict=ReleaseVerdict.GO,
        )


def test_rollback_requires_target_release():
    with pytest.raises(ValueError, match="rollback_of_release_id"):
        _entry(
            release_id="rel-003",
            model_id="qwen-2.5-7b-v2",
            status=ReleaseStatus.ROLLBACK,
            verdict=ReleaseVerdict.GO,
            approved_by="review-board",
        )


def test_log_production_sets_active_pointer(tmp_path: Path):
    manager = _manager(tmp_path)
    release = _entry(
        release_id="rel-004",
        model_id="qwen-2.5-7b-v1",
        status=ReleaseStatus.PRODUCTION,
        approved_by="review-board",
    )

    manager.log_deployment(release)

    active = manager.get_active_release("qwen-2.5-7b", "prod-us")
    assert active is not None
    assert active.release_id == "rel-004"
    assert active.model_id == "qwen-2.5-7b-v1"


def test_transition_to_production_records_previous_release(tmp_path: Path):
    manager = _manager(tmp_path)
    current = _entry(
        release_id="rel-005",
        model_id="qwen-2.5-7b-v1",
        status=ReleaseStatus.PRODUCTION,
        approved_by="review-board",
    )
    candidate = _entry(
        release_id="rel-006",
        model_id="qwen-2.5-7b-v2",
        status=ReleaseStatus.STAGING,
    )

    manager.log_deployment(current)
    manager.log_deployment(candidate)

    promoted = manager.transition_release(
        "rel-006",
        ReleaseStatus.PRODUCTION,
        acted_by="release-manager",
        approved_by="review-board",
    )

    assert promoted.status == ReleaseStatus.PRODUCTION
    assert promoted.previous_release_id == "rel-005"
    assert manager.get_active_release("qwen-2.5-7b", "prod-us").release_id == "rel-006"


def test_rollback_restores_prior_release_pointer(tmp_path: Path):
    manager = _manager(tmp_path)
    prior = _entry(
        release_id="rel-007",
        model_id="qwen-2.5-7b-v1",
        status=ReleaseStatus.PRODUCTION,
        approved_by="review-board",
    )
    failing = _entry(
        release_id="rel-008",
        model_id="qwen-2.5-7b-v2",
        status=ReleaseStatus.PRODUCTION,
        approved_by="review-board",
        previous_release_id="rel-007",
    )

    manager.log_deployment(prior)
    manager.transition_release(
        "rel-007",
        ReleaseStatus.RETIRED,
        acted_by="release-manager",
    )
    manager.log_deployment(failing)

    manager.transition_release(
        "rel-008",
        ReleaseStatus.ROLLBACK,
        acted_by="release-manager",
        rollback_of_release_id="rel-007",
    )

    rolled_back = manager.get_release("rel-008")
    restored = manager.get_active_release("qwen-2.5-7b", "prod-us")

    assert rolled_back.status == ReleaseStatus.ROLLBACK
    assert rolled_back.rollback_of_release_id == "rel-007"
    assert restored is not None
    assert restored.release_id == "rel-007"
    assert restored.status == ReleaseStatus.PRODUCTION
