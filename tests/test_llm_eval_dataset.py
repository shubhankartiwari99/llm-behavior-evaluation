from __future__ import annotations

import json

from llm_eval.dataset import bootstrap_records_from_eval_results, load_records, write_records


def test_bootstrap_records_groups_repeated_prompts(tmp_path):
    results = [
        {
            "category": "baseline",
            "prompt": "Explain the importance of discipline.",
            "response": "Discipline helps people stay consistent.",
        },
        {
            "category": "baseline",
            "prompt": "Explain the importance of discipline.",
            "response": "Discipline supports long-term progress.",
        },
        {
            "category": "variation",
            "prompt": "Explain why discipline matters.",
            "response": "It creates routine and steady improvement.",
        },
    ]

    records = bootstrap_records_from_eval_results(
        results,
        experiment_id="exp_bootstrap",
        source_file="eval/results_test.json",
    )

    assert [record.prompt_id for record in records] == ["p1", "p1", "p2"]
    assert [record.run_id for record in records] == ["p1_r1", "p1_r2", "p2_r1"]
    assert records[0].prompt_type == "baseline"
    assert records[0].tone is None
    assert records[0].to_dict(include_none=True)["type"] is None

    output_path = tmp_path / "bootstrapped.jsonl"
    write_records(output_path, records)
    restored = load_records(output_path)

    assert len(restored) == 3
    assert restored[1].response == "Discipline supports long-term progress."
    assert restored[2].source_file == "eval/results_test.json"


def test_load_records_accepts_json_array(tmp_path):
    payload = [
        {
            "prompt_id": "p1",
            "prompt": "What is inflation?",
            "response": "Inflation means prices rise over time.",
            "tone": "neutral",
            "cultural": "none",
            "type": "specific",
        }
    ]
    input_path = tmp_path / "labels.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    records = load_records(input_path)

    assert len(records) == 1
    assert records[0].is_fully_labeled()
    assert records[0].response_type.value == "specific"
