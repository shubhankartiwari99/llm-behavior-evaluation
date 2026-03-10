from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ReleaseStatus(str, Enum):
    STAGING = "staging"
    PRODUCTION = "production"
    ROLLBACK = "rollback"
    RETIRED = "retired"


class ReleaseVerdict(str, Enum):
    GO = "GO"
    NO_GO = "NO_GO"


class RegressionSummary(BaseModel):
    fatal_count: int = Field(0, ge=0)
    warning_count: int = Field(0, ge=0)
    info_count: int = Field(0, ge=0)
    triggered_checks: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def coerce_legacy_counts(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        if any(key in values for key in ("FATAL", "WARNING", "INFO")):
            values = {
                "fatal_count": values.get("FATAL", 0),
                "warning_count": values.get("WARNING", 0),
                "info_count": values.get("INFO", 0),
                "triggered_checks": values.get("triggered_checks", []),
            }
        return values


class EvaluationEvidence(BaseModel):
    evaluation_run_id: str = Field(..., min_length=3)
    snapshot_id: str = Field(..., min_length=3)
    verdict: ReleaseVerdict
    regression_summary: RegressionSummary
    policy_version: str = Field(..., min_length=1)
    harness_version: str = Field(..., min_length=1)
    dataset_id: str = Field(..., min_length=1)
    snapshot_uri: Optional[str] = None
    snapshot_digest: Optional[str] = None
    report_uri: Optional[str] = None
    baseline_release_id: Optional[str] = None


class ActiveReleasePointer(BaseModel):
    model_family: str
    environment: str
    release_id: str
    model_id: str
    git_commit_hash: str
    weights_fingerprint: str
    snapshot_id: str
    verdict: ReleaseVerdict
    fragility_score: float = Field(..., ge=0.0)
    updated_at: datetime = Field(default_factory=utc_now)


class DeploymentEntry(BaseModel):
    # Identity and provenance
    release_id: str = Field(..., min_length=3)
    model_family: str = Field(..., min_length=1, description="Stable architecture family, e.g. llama-3-8b-instruct")
    model_id: str = Field(..., min_length=1, description="Versioned release identifier, e.g. llama-3-8b-instruct-v1.2")
    git_commit_hash: str = Field(..., min_length=7, max_length=64)
    weights_fingerprint: str = Field(..., min_length=8)
    inference_config_fingerprint: str = Field(..., min_length=8)

    # Evaluation sign-off
    evaluation: EvaluationEvidence

    # Lifecycle
    status: ReleaseStatus = Field(default=ReleaseStatus.STAGING)
    environment: str = Field(..., min_length=1)
    deployed_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    deployed_by: str = Field(..., min_length=1)
    approved_by: Optional[str] = None
    previous_release_id: Optional[str] = None
    rollback_of_release_id: Optional[str] = None

    # High-stakes audit fields
    is_adversarially_verified: bool = False
    fragility_score: float = Field(..., ge=0.0)
    runtime_metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("git_commit_hash")
    @classmethod
    def normalize_git_hash(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not all(ch in "0123456789abcdef" for ch in normalized):
            raise ValueError("git_commit_hash must be hexadecimal.")
        return normalized

    @field_validator("deployed_at", "updated_at", mode="before")
    @classmethod
    def ensure_timezone_aware(cls, value: Any) -> Any:
        if isinstance(value, datetime) and value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    @model_validator(mode="after")
    def validate_release_contract(self) -> "DeploymentEntry":
        status = self.status
        evaluation = self.evaluation
        approved_by = self.approved_by
        rollback_of_release_id = self.rollback_of_release_id

        if status == ReleaseStatus.PRODUCTION:
            if evaluation is None or evaluation.verdict != ReleaseVerdict.GO:
                raise ValueError("Production releases require a GO evaluation verdict.")
            if not approved_by:
                raise ValueError("Production releases require approved_by for auditability.")

        if status == ReleaseStatus.ROLLBACK and not rollback_of_release_id:
            raise ValueError("Rollback releases must reference rollback_of_release_id.")

        if rollback_of_release_id and status != ReleaseStatus.ROLLBACK:
            raise ValueError("rollback_of_release_id is only valid for rollback releases.")

        return self
