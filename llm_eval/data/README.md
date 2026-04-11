# Data Directory

Store experiment outputs and manual labels here.

Suggested pattern:

- raw sampled outputs: `exp_01_single_prompt_stability.jsonl`
- manually labeled copy: `exp_01_single_prompt_stability_labeled.jsonl`
- exported summaries: `exp_01_single_prompt_stability_summary.json`

The bootstrap and sampling commands both write records in the same schema so the analysis step stays simple.
