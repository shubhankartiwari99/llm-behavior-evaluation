from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from app.registry.schema import (
    ActiveReleasePointer,
    DeploymentEntry,
    ReleaseStatus,
    utc_now,
)

ALLOWED_TRANSITIONS = {
    ReleaseStatus.STAGING: {ReleaseStatus.PRODUCTION, ReleaseStatus.RETIRED},
    ReleaseStatus.PRODUCTION: {ReleaseStatus.ROLLBACK, ReleaseStatus.RETIRED},
    ReleaseStatus.ROLLBACK: {ReleaseStatus.RETIRED},
    ReleaseStatus.RETIRED: set(),
}


def _default_registry_dir() -> Path:
    base = os.environ.get("DEPLOYMENT_REGISTRY_DIR", "data/registry")
    return Path(base)


class RegistryManager:
    """
    File-backed deployment provenance store.

    History is append-safe JSONL rewritten transactionally for status changes.
    Active production pointers are kept separately so a release lookup does not
    depend on scanning full history during rollback or UI reads.
    """

    def __init__(
        self,
        history_path: str | Path | None = None,
        active_path: str | Path | None = None,
    ):
        root = _default_registry_dir()
        self.history_path = Path(history_path) if history_path else root / "deployment_history.jsonl"
        self.active_path = Path(active_path) if active_path else root / "active_releases.json"
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.active_path.parent.mkdir(parents=True, exist_ok=True)

    def log_deployment(self, entry: DeploymentEntry) -> None:
        if self.get_release(entry.release_id) is not None:
            raise ValueError(f"release_id already exists: {entry.release_id}")

        entries = self._read_entries()
        entries.append(entry)
        self._write_entries(entries)

        if entry.status == ReleaseStatus.PRODUCTION:
            self._set_active_pointer(entry)

    def get_all_deployments(
        self,
        *,
        model_family: Optional[str] = None,
        environment: Optional[str] = None,
        status: Optional[ReleaseStatus] = None,
    ) -> List[DeploymentEntry]:
        entries = self._read_entries()
        if model_family is not None:
            entries = [entry for entry in entries if entry.model_family == model_family]
        if environment is not None:
            entries = [entry for entry in entries if entry.environment == environment]
        if status is not None:
            entries = [entry for entry in entries if entry.status == status]
        entries.sort(key=lambda entry: entry.deployed_at, reverse=True)
        return entries

    def get_release(self, release_id: str) -> Optional[DeploymentEntry]:
        for entry in self._read_entries():
            if entry.release_id == release_id:
                return entry
        return None

    def get_active_release(self, model_family: str, environment: str) -> Optional[DeploymentEntry]:
        pointer = self._read_active_pointers().get(self._pointer_key(model_family, environment))
        if pointer is None:
            return None
        return self.get_release(pointer.release_id)

    def get_latest_production(
        self,
        model_id: str,
        environment: Optional[str] = None,
    ) -> Optional[DeploymentEntry]:
        if environment is not None:
            active = self.get_active_release(model_id, environment)
            if active is not None:
                return active

        for entry in self.get_all_deployments(environment=environment, status=ReleaseStatus.PRODUCTION):
            if entry.model_family == model_id or entry.model_id == model_id:
                return entry
        return None

    def transition_release(
        self,
        release_id: str,
        new_status: ReleaseStatus,
        *,
        acted_by: str,
        approved_by: Optional[str] = None,
        rollback_of_release_id: Optional[str] = None,
    ) -> DeploymentEntry:
        current = self.get_release(release_id)
        if current is None:
            raise ValueError(f"Unknown release_id: {release_id}")

        if new_status == current.status:
            return current

        allowed = ALLOWED_TRANSITIONS[current.status]
        if new_status not in allowed:
            raise ValueError(f"Illegal transition: {current.status} -> {new_status}")

        entries = self._read_entries()
        now = utc_now()
        updated = current

        if new_status == ReleaseStatus.PRODUCTION:
            active = self.get_active_release(current.model_family, current.environment)
            updated = self._validated_update(
                current,
                status=ReleaseStatus.PRODUCTION,
                approved_by=approved_by or current.approved_by or acted_by,
                previous_release_id=(
                    active.release_id
                    if active and active.release_id != current.release_id
                    else current.previous_release_id
                ),
                updated_at=now,
            )
            self._replace_entry(entries, updated)
            self._write_entries(entries)
            self._set_active_pointer(updated)
            return updated

        if new_status == ReleaseStatus.ROLLBACK:
            if not rollback_of_release_id:
                raise ValueError("rollback_of_release_id is required for rollback transitions.")

            rollback_target = self.get_release(rollback_of_release_id)
            if rollback_target is None:
                raise ValueError(f"Unknown rollback target: {rollback_of_release_id}")
            if rollback_target.environment != current.environment:
                raise ValueError("Rollback target must be in the same environment.")
            if rollback_target.model_family != current.model_family:
                raise ValueError("Rollback target must be in the same model_family.")

            updated = self._validated_update(
                current,
                status=ReleaseStatus.ROLLBACK,
                rollback_of_release_id=rollback_of_release_id,
                updated_at=now,
            )
            restored = self._validated_update(
                rollback_target,
                status=ReleaseStatus.PRODUCTION,
                approved_by=rollback_target.approved_by or acted_by,
                updated_at=now,
            )
            self._replace_entry(entries, updated)
            self._replace_entry(entries, restored)
            self._write_entries(entries)
            self._set_active_pointer(restored)
            return updated

        updated = self._validated_update(current, status=new_status, updated_at=now)
        self._replace_entry(entries, updated)
        self._write_entries(entries)

        active = self.get_active_release(current.model_family, current.environment)
        if active is not None and active.release_id == current.release_id and new_status != ReleaseStatus.PRODUCTION:
            self._clear_active_pointer(current.model_family, current.environment)

        return updated

    def _replace_entry(self, entries: List[DeploymentEntry], replacement: DeploymentEntry) -> None:
        for index, entry in enumerate(entries):
            if entry.release_id == replacement.release_id:
                entries[index] = replacement
                return
        raise ValueError(f"Release not found during replacement: {replacement.release_id}")

    def _validated_update(self, entry: DeploymentEntry, **updates) -> DeploymentEntry:
        payload = entry.model_dump()
        payload.update(updates)
        return DeploymentEntry.model_validate(payload)

    def _read_entries(self) -> List[DeploymentEntry]:
        if not self.history_path.exists():
            return []

        entries: List[DeploymentEntry] = []
        with self.history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                entries.append(DeploymentEntry.model_validate_json(line))
        return entries

    def _write_entries(self, entries: List[DeploymentEntry]) -> None:
        with self.history_path.open("w", encoding="utf-8") as handle:
            for entry in sorted(entries, key=lambda item: item.deployed_at):
                handle.write(entry.model_dump_json())
                handle.write("\n")

    def _pointer_key(self, model_family: str, environment: str) -> str:
        return f"{model_family}::{environment}"

    def _read_active_pointers(self) -> Dict[str, ActiveReleasePointer]:
        if not self.active_path.exists():
            return {}

        payload = json.loads(self.active_path.read_text(encoding="utf-8"))
        return {
            key: ActiveReleasePointer.model_validate(value)
            for key, value in payload.items()
        }

    def _write_active_pointers(self, pointers: Dict[str, ActiveReleasePointer]) -> None:
        serializable = {key: pointer.model_dump(mode="json") for key, pointer in pointers.items()}
        self.active_path.write_text(json.dumps(serializable, indent=2, default=str), encoding="utf-8")

    def _set_active_pointer(self, entry: DeploymentEntry) -> None:
        pointers = self._read_active_pointers()
        pointers[self._pointer_key(entry.model_family, entry.environment)] = ActiveReleasePointer(
            model_family=entry.model_family,
            environment=entry.environment,
            release_id=entry.release_id,
            model_id=entry.model_id,
            git_commit_hash=entry.git_commit_hash,
            weights_fingerprint=entry.weights_fingerprint,
            snapshot_id=entry.evaluation.snapshot_id,
            verdict=entry.evaluation.verdict,
            fragility_score=entry.fragility_score,
            updated_at=utc_now(),
        )
        self._write_active_pointers(pointers)

    def _clear_active_pointer(self, model_family: str, environment: str) -> None:
        pointers = self._read_active_pointers()
        pointers.pop(self._pointer_key(model_family, environment), None)
        self._write_active_pointers(pointers)
