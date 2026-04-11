# Probabilistic LLM Behavior Evaluation

This subsystem turns the repo into a behavior-evaluation workflow instead of a loose collection of model runs.

It adds four things:

1. A manual labeling schema for `tone`, `cultural`, and `type`.
2. Reusable experiment specs for stability, prompt sensitivity, and cultural triggering.
3. CLI tools to bootstrap datasets, run sampling, and compute empirical probabilities.
4. A probability-oriented analysis layer that maps directly to Stat 110 concepts.

## Folder Map

```text
llm_eval/
  ├── data/
  ├── experiments/
  ├── analysis/
  ├── notes/
  ├── scripts/
  ├── dataset.py
  ├── experiment_templates.py
  ├── metrics.py
  ├── sampling.py
  └── schema.py
```

## Label Schema

Each manually labeled response should end up in this shape:

```json
{
  "prompt_id": "p1",
  "response": "In India, especially in tier-2 cities...",
  "tone": "neutral",
  "cultural": "strong_indian_context",
  "type": "specific"
}
```

The CLI bootstrap step leaves labels empty on purpose. Labeling is manual in v1.

## Quickstart

Bootstrap an existing eval file into a labeling dataset:

```bash
python3 -m llm_eval.scripts.bootstrap_dataset \
  --input eval/results_manual_test.json \
  --output llm_eval/data/manual_test_labels.jsonl
```

Run one of the experiment specs and write unlabeled samples:

```bash
python3 -m llm_eval.scripts.run_experiment \
  --spec llm_eval/experiments/exp_01_single_prompt_stability.json \
  --output llm_eval/data/exp_01_single_prompt_stability.jsonl
```

Analyze a labeled dataset:

```bash
python3 -m llm_eval.scripts.analyze_dataset \
  --input llm_eval/data/manual_test_labels.jsonl \
  --require-complete-labels
```

## What The Analysis Computes

- `P(category)` for every label dimension
- `P(cultural | prompt_type)` for conditional comparisons
- consistency by `prompt_id`
- diversity over joint label patterns
- Indian-context rate as a bias-strength proxy
- response-length expectation and variance

Use the files in [llm_eval/notes/labeling_guide.md](/Users/shubhankartiwari/indian-desi-llm-inference/llm_eval/notes/labeling_guide.md) and [llm_eval/analysis/README.md](/Users/shubhankartiwari/indian-desi-llm-inference/llm_eval/analysis/README.md) as the operating docs.
