from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Union

from llm_eval.schema import LabeledResponseRecord


def _normalize_record_payloads(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        if isinstance(payload.get("records"), list):
            return [item for item in payload["records"] if isinstance(item, dict)]
        if isinstance(payload.get("results"), list):
            return [item for item in payload["results"] if isinstance(item, dict)]
    raise ValueError(
        "Unsupported dataset format. Expected a JSON array, a 'records' object, "
        "or a 'results' object."
    )


def load_records(path: Union[str, Path]) -> list[LabeledResponseRecord]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if dataset_path.suffix.lower() == ".jsonl":
        payloads: list[dict[str, Any]] = []
        for raw_line in dataset_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            item = json.loads(line)
            if not isinstance(item, dict):
                raise ValueError("JSONL datasets must contain one object per line.")
            payloads.append(item)
    else:
        payloads = _normalize_record_payloads(json.loads(dataset_path.read_text(encoding="utf-8")))

    return [LabeledResponseRecord.model_validate(item) for item in payloads]


def write_records(path: Union[str, Path], records: Iterable[LabeledResponseRecord]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = [record.to_dict(include_none=True) for record in records]

    if output_path.suffix.lower() == ".json":
        output_path.write_text(
            json.dumps(serialized, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    lines = [json.dumps(item, ensure_ascii=False) for item in serialized]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def bootstrap_records_from_eval_results(
    results: Sequence[dict[str, Any]],
    *,
    experiment_id: Optional[str] = None,
    source_file: Optional[str] = None,
    default_prompt_type: Optional[str] = None,
) -> list[LabeledResponseRecord]:
    prompt_ids: OrderedDict[str, str] = OrderedDict()
    prompt_run_counts: dict[str, int] = {}
    bootstrapped: list[LabeledResponseRecord] = []

    for item in results:
        prompt = str(item.get("prompt") or "").strip()
        response = str(item.get("response") or item.get("response_text") or "").strip()
        if not prompt or not response:
            continue

        existing_prompt_id = item.get("prompt_id")
        if isinstance(existing_prompt_id, str) and existing_prompt_id.strip():
            prompt_id = existing_prompt_id.strip()
        else:
            prompt_id = prompt_ids.setdefault(prompt, f"p{len(prompt_ids) + 1}")

        run_index = prompt_run_counts.get(prompt_id, 0) + 1
        prompt_run_counts[prompt_id] = run_index

        record = LabeledResponseRecord(
            record_id=f"{prompt_id}_r{run_index}",
            experiment_id=experiment_id,
            prompt_id=prompt_id,
            run_id=f"{prompt_id}_r{run_index}",
            prompt=prompt,
            prompt_type=str(
                item.get("prompt_type")
                or item.get("category")
                or default_prompt_type
                or "unknown"
            ),
            condition=str(item.get("condition")).strip() if item.get("condition") else None,
            response=response,
            tone=item.get("tone"),
            cultural=item.get("cultural"),
            type=item.get("type"),
            source_category=str(item.get("category")).strip() if item.get("category") else None,
            source_file=source_file,
            confidence=item.get("confidence"),
            instability=item.get("instability"),
            latency_ms=item.get("latency_ms"),
            metadata=item.get("meta") if isinstance(item.get("meta"), dict) else {},
        )
        bootstrapped.append(record)

    if not bootstrapped:
        raise ValueError("No valid prompt/response pairs found in the input results.")
    return bootstrapped


def bootstrap_records_from_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    experiment_id: Optional[str] = None,
    default_prompt_type: Optional[str] = None,
) -> Path:
    source_path = Path(input_path)
    if source_path.suffix.lower() == ".jsonl":
        payloads = [
            json.loads(line)
            for line in source_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        payloads = _normalize_record_payloads(json.loads(source_path.read_text(encoding="utf-8")))

    records = bootstrap_records_from_eval_results(
        payloads,
        experiment_id=experiment_id or source_path.stem,
        source_file=str(source_path),
        default_prompt_type=default_prompt_type,
    )
    return write_records(output_path, records)
