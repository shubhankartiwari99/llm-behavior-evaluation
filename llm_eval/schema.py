from __future__ import annotations

from enum import Enum
from typing import Any, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ToneLabel(str, Enum):
    FORMAL = "formal"
    INFORMAL = "informal"
    NEUTRAL = "neutral"


class CulturalSignal(str, Enum):
    STRONG_INDIAN_CONTEXT = "strong_indian_context"
    WEAK_INDIAN_CONTEXT = "weak_indian_context"
    NONE = "none"


class ResponseType(str, Enum):
    GENERIC = "generic"
    SPECIFIC = "specific"
    EXAMPLE_DRIVEN = "example_driven"


class LabeledResponseRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    record_id: Optional[str] = None
    experiment_id: Optional[str] = None
    prompt_id: str
    run_id: Optional[str] = None
    prompt: Optional[str] = None
    prompt_type: Optional[str] = None
    condition: Optional[str] = None
    response: str
    tone: Optional[ToneLabel] = None
    cultural: Optional[CulturalSignal] = None
    response_type: Optional[ResponseType] = Field(
        default=None,
        alias="type",
        serialization_alias="type",
    )
    source_category: Optional[str] = None
    source_file: Optional[str] = None
    confidence: Optional[float] = None
    instability: Optional[float] = None
    latency_ms: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator(
        "record_id",
        "experiment_id",
        "prompt_id",
        "run_id",
        "prompt",
        "prompt_type",
        "condition",
        "response",
        "source_category",
        "source_file",
        mode="before",
    )
    @classmethod
    def strip_string_fields(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
        return value

    @field_validator("prompt_id", "response")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value:
            raise ValueError("Required text fields must be non-empty.")
        return value

    @field_validator("confidence", "instability", "latency_ms", mode="before")
    @classmethod
    def coerce_numeric_metrics(cls, value: Any) -> Any:
        if value is None or value == "":
            return None
        if isinstance(value, bool):
            raise ValueError("Metrics must be numeric.")
        if isinstance(value, (int, float)):
            return float(value)
        raise ValueError("Metrics must be numeric.")

    def is_fully_labeled(self) -> bool:
        return (
            self.tone is not None
            and self.cultural is not None
            and self.response_type is not None
        )

    def label_pattern(self) -> Optional[Tuple[str, str, str]]:
        if not self.is_fully_labeled():
            return None
        assert self.tone is not None
        assert self.cultural is not None
        assert self.response_type is not None
        return (
            self.tone.value,
            self.cultural.value,
            self.response_type.value,
        )

    def to_dict(self, include_none: bool = True) -> dict[str, Any]:
        return self.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=not include_none,
        )


class ExperimentPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    prompt: str
    prompt_type: str = "default"
    condition: Optional[str] = None
    notes: Optional[str] = None

    @field_validator(
        "prompt_id",
        "prompt",
        "prompt_type",
        "condition",
        "notes",
        mode="before",
    )
    @classmethod
    def strip_prompt_fields(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
        return value

    @field_validator("prompt_id", "prompt", "prompt_type")
    @classmethod
    def validate_prompt_fields(cls, value: str) -> str:
        if not value:
            raise ValueError("Prompt fields must be non-empty.")
        return value


class ExperimentSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    title: str
    objective: str
    hypothesis: str
    description: Optional[str] = None
    runs_per_prompt: int = 30
    inference_params: dict[str, Any] = Field(default_factory=dict)
    prompts: list[ExperimentPrompt]
    metrics: list[str] = Field(default_factory=list)

    @field_validator(
        "experiment_id",
        "title",
        "objective",
        "hypothesis",
        "description",
        mode="before",
    )
    @classmethod
    def strip_experiment_fields(cls, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
        return value

    @field_validator("experiment_id", "title", "objective", "hypothesis")
    @classmethod
    def validate_required_strings(cls, value: str) -> str:
        if not value:
            raise ValueError("Experiment strings must be non-empty.")
        return value

    @field_validator("runs_per_prompt")
    @classmethod
    def validate_runs_per_prompt(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("runs_per_prompt must be positive.")
        return value

    @model_validator(mode="after")
    def validate_prompts(self) -> "ExperimentSpec":
        if not self.prompts:
            raise ValueError("At least one prompt is required.")
        return self
