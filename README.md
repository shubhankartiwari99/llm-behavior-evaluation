# Probabilistic Evaluation of LLM Inference-Time Behavior

> **A system for analyzing how inference-time policies reshape LLM output distributions.**

> **Note:** This is an earlier iteration / research exploration.
> This work evolved into llm-generation-control : https://github.com/shubhankartiwari99/llm-generation-control

This project is a **systems-level evaluation framework** for analyzing how inference-time runtime policies reshape Large Language Model (LLM) output distributions.

Instead of treating model outputs as final, this work separates:

- **raw model generation (pre-rescue)**
- **post-processed runtime outputs (post-rescue)**

and measures how runtime interventions affect the distribution. These metrics capture different aspects:

- **Entropy** → uncertainty within a distribution
- **Collapse Ratio** → relative compression
- **KL Divergence** → distance between distributions

> Collapse ratio captures entropy reduction but not direction of change.
> KL divergence captures how far the distribution moved.
> Both are required for complete analysis.

---

## LLM Regression Detection System
This repository now includes a lightweight governance layer for model version promotion.

### What it solves
Prevents silent behavioral regressions when a new model version is compared against a frozen evaluation set.

### Why it matters
A model can improve on one metric while worsening more important safety, format, or consistency behavior.

### What’s new
Unlike traditional offline evaluation, this system enforces behavioral regression constraints at deployment time through CI-gated promotion policies.

### Dataset versioning
- `data/eval_dataset_v1.json` — earlier frozen evaluation set
- `data/eval_dataset_v2.json` — newer, richer evaluation set
- `data/eval_dataset.json` — active default dataset used by the runner

### System components
- `scripts/regression_engine.py` — structured behavioral metrics + decision policy
- `scripts/run_regression_check.py` — runner with explicit promote/reject output
- `artifacts/decision_history.json` — audit trail for every check
- `scripts/ci_regression_gate.py` — CI-friendly promotion gate

### CI gate usage
```bash
python3 scripts/ci_regression_gate.py \
  --prod models/model_v1.json \
  --cand models/model_v2.json
```

This command exits with `0` when the candidate passes promotion policy, and `1` when it is rejected.

---

## 🧭 Research Question

How do inference-time interventions reshape the probability distribution of LLM behaviors under different prompt conditions?

---

## 🎯 Use Cases

- ML Engineers → understand inference-time behavior
- AI Reliability → evaluate system-level bias and stability
- Researchers → study distribution shaping effects

---

## 🚀 Positioning

This project sits at the intersection of:
- ML systems
- LLM evaluation
- AI reliability
- inference-time control

---

## 📌 Status

Superseded / Earlier Iteration.
This work evolved into llm-generation-control : https://github.com/shubhankartiwari99/llm-generation-control

---

## ⚠️ Failure Modes & Limitations

- Some prompts may bypass runtime shaping
- KL divergence may remain low despite structural changes in very sparse probability spaces

---

## ⚡ Performance

Typical latency:
- Mock mode: ~50–200ms
- Real inference: model-dependent

---

## 🔧 Extensibility

The evaluation layer can be extended to:
- new behavioral labels
- alternative metrics (JS divergence, Wasserstein)
- different model backends
