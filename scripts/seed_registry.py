import asyncio
import os
import random
from datetime import datetime, timedelta, timezone
from app.registry.manager import RegistryManager
from app.registry.schema import (
    DeploymentEntry,
    EvaluationEvidence,
    RegressionSummary,
    ReleaseStatus,
    ReleaseVerdict,
)

# Mock data for demonstration purposes
MODELS = ["qwen-2.5-7b", "mistral-v3", "llama-3-8b"]
DEPLOYERS = ["AI_SYSTEM", "CI_PIPELINE", "MANUAL_OVERRIDE"]
VERDICTS = ["GO", "NO_GO"]

def generate_git_hash():
    """Generates a random 7-character hex string representing a Git commit hash."""
    return f"{random.randrange(16**7):07x}"

def _sha256_like() -> str:
    return "sha256:" + "".join(random.choice("0123456789abcdef") for _ in range(64))


def _evidence(
    release_id: str,
    snapshot_id: str,
    verdict: ReleaseVerdict,
    fatal: int,
    warning: int,
    info: int,
) -> EvaluationEvidence:
    return EvaluationEvidence(
        evaluation_run_id=f"eval-{release_id}",
        snapshot_id=snapshot_id,
        verdict=verdict,
        regression_summary=RegressionSummary(
            fatal_count=fatal,
            warning_count=warning,
            info_count=info,
        ),
        policy_version="delta-calculator-v1",
        harness_version="registry-seed-v1",
        dataset_id="seed-demo-dataset",
        snapshot_uri=f"snapshots/{snapshot_id}.json",
        report_uri=f"reports/{release_id}.json",
    )


async def seed_registry(db_path: str = "data/registry/deployment_history.jsonl"):
    """Seeds the deployment registry JSONL file with mock data."""
    print(f"Seeding registry data to {db_path}...")
    
    # Ensure artifacts mapping directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    manager = RegistryManager(history_path=db_path)
    
    # Check if the registry already has data
    if manager.get_all_deployments():
        print("Registry already has data. Skipping seed.")
        return

    # Generate sequential dates for the timeline over the past 30 days
    base_date = datetime.now(timezone.utc) - timedelta(days=30)
    
    entries = []
    
    # 1. Successful production release 30 days ago (Oldest)
    entries.append(DeploymentEntry(
        release_id="rel-old-01",
        model_family="llama-3-8b",
        model_id="llama-3-8b",
        git_commit_hash=generate_git_hash(),
        weights_fingerprint=_sha256_like(),
        inference_config_fingerprint=_sha256_like(),
        evaluation=_evidence("rel-old-01", "snap-v1", ReleaseVerdict.GO, 0, 0, 1),
        status=ReleaseStatus.RETIRED,
        environment="production",
        deployed_at=base_date,
        approved_by="AI_REVIEW_BOARD",
        deployed_by="CI_PIPELINE",
        is_adversarially_verified=True,
        fragility_score=12.5,
    ))
    
    # 2. Failed staging deploy 20 days ago
    entries.append(DeploymentEntry(
        release_id="rel-stage-02",
        model_family="mistral-v3",
        model_id="mistral-v3",
        git_commit_hash=generate_git_hash(),
        weights_fingerprint=_sha256_like(),
        inference_config_fingerprint=_sha256_like(),
        evaluation=_evidence("rel-stage-02", "snap-v2", ReleaseVerdict.NO_GO, 1, 2, 0),
        status=ReleaseStatus.STAGING,
        environment="staging",
        deployed_at=base_date + timedelta(days=10),
        deployed_by="CI_PIPELINE",
        is_adversarially_verified=True,
        fragility_score=25.0,
    ))

    # 3. Successful production release 15 days ago
    entries.append(DeploymentEntry(
        release_id="rel-prod-03",
        model_family="qwen-2.5-7b",
        model_id="qwen-2.5-7b",
        git_commit_hash=generate_git_hash(),
        weights_fingerprint=_sha256_like(),
        inference_config_fingerprint=_sha256_like(),
        evaluation=_evidence("rel-prod-03", "snap-v3", ReleaseVerdict.GO, 0, 1, 0),
        status=ReleaseStatus.RETIRED,
        environment="production",
        deployed_at=base_date + timedelta(days=15),
        approved_by="AI_REVIEW_BOARD",
        deployed_by="AI_SYSTEM",
        is_adversarially_verified=True,
        fragility_score=18.2,
    ))

    # 4. Critical Rollback 5 days ago
    entries.append(DeploymentEntry(
        release_id="rel-fail-04",
        model_family="qwen-2.5-7b",
        model_id="qwen-2.5-7b-v2",
        git_commit_hash=generate_git_hash(),
        weights_fingerprint=_sha256_like(),
        inference_config_fingerprint=_sha256_like(),
        evaluation=_evidence("rel-fail-04", "snap-v4", ReleaseVerdict.GO, 0, 0, 0),
        status=ReleaseStatus.ROLLBACK,
        environment="production",
        deployed_at=base_date + timedelta(days=25),
        approved_by="AI_REVIEW_BOARD",
        deployed_by="CI_PIPELINE",
        is_adversarially_verified=False,
        fragility_score=0.0,
        rollback_of_release_id="rel-prod-03",
    ))

    # 5. Current Production release (Today)
    entries.append(DeploymentEntry(
        release_id="rel-curr-05",
        model_family="qwen-2.5-7b",
        model_id="qwen-2.5-7b-v3",
        git_commit_hash=generate_git_hash(),
        weights_fingerprint=_sha256_like(),
        inference_config_fingerprint=_sha256_like(),
        evaluation=_evidence("rel-curr-05", "snap-v5", ReleaseVerdict.GO, 0, 0, 2),
        status=ReleaseStatus.PRODUCTION,
        environment="production",
        deployed_at=datetime.now(timezone.utc),
        approved_by="AI_REVIEW_BOARD",
        deployed_by="AI_SYSTEM",
        is_adversarially_verified=True,
        fragility_score=8.5,
        previous_release_id="rel-prod-03",
    ))

    for entry in entries:
        manager.log_deployment(entry)
        
    print(f"Successfully seeded {len(entries)} deployment records.")


if __name__ == "__main__":
    asyncio.run(seed_registry())
