# Experiment Specs

This directory contains the first three portfolio-grade experiments:

1. `exp_01_single_prompt_stability.json`
2. `exp_02_prompt_variation.json`
3. `exp_03_cultural_triggering.json`

Each spec defines:

- the hypothesis
- the prompts
- the number of runs per prompt
- the inference defaults
- the metrics to compare

Run a spec with:

```bash
python3 -m llm_eval.scripts.run_experiment --spec llm_eval/experiments/<file>.json --output llm_eval/data/<dataset>.jsonl
```
