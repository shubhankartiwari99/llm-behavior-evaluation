# Probabilistic LLM Behavior Evaluation System

This repo now has two connected evaluation layers:

1. runtime reliability and drift analysis under `app/eval`
2. manual behavior labeling and probability analysis under `llm_eval`

The result is a cleaner research project: not just model experiments, but a repeatable evaluation system for studying stability, prompt sensitivity, and cultural conditioning in LLM outputs.

## Core Capabilities

- probabilistic vs deterministic inference tracing
- behavioral snapshot and drift regression tracking
- manual response labeling for `tone`, `cultural`, and `type`
- empirical probability analysis over labeled outputs
- experiment scaffolding for stability, prompt variation, and cultural triggering

## Research Framing

Primary project question:

`How does prompt wording and cultural context change the probability distribution of LLM behaviors?`

This supports a portfolio-grade framing:

`Probabilistic Analysis of Cultural Conditioning in LLM Outputs`

## Repo Map

- [app/eval](/Users/shubhankartiwari/indian-desi-llm-inference/app/eval): runtime reliability, benchmark summaries, drift snapshots
- [eval](/Users/shubhankartiwari/indian-desi-llm-inference/eval): existing prompt/result artifacts
- [llm_eval](/Users/shubhankartiwari/indian-desi-llm-inference/llm_eval): manual labeling schema, experiment specs, and probabilistic analysis
- [tests](/Users/shubhankartiwari/indian-desi-llm-inference/tests): unit coverage

## Quickstart

Bootstrap an old result file into the new schema:

```bash
python3 -m llm_eval.scripts.bootstrap_dataset \
  --input eval/results_manual_test.json \
  --output llm_eval/data/manual_test_labels.jsonl
```

Run one of the experiment specs:

```bash
python3 -m llm_eval.scripts.run_experiment \
  --spec llm_eval/experiments/exp_01_single_prompt_stability.json \
  --output llm_eval/data/exp_01_single_prompt_stability.jsonl
```

Analyze labeled outputs:

```bash
python3 -m llm_eval.scripts.analyze_dataset \
  --input llm_eval/data/exp_01_single_prompt_stability_labeled.jsonl \
  --require-complete-labels
```

See [llm_eval/README.md](/Users/shubhankartiwari/indian-desi-llm-inference/llm_eval/README.md) for the workflow and [llm_eval/notes/labeling_guide.md](/Users/shubhankartiwari/indian-desi-llm-inference/llm_eval/notes/labeling_guide.md) for the manual rubric.
