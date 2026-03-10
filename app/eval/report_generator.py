from typing import Any, Dict


def generate_research_report(summary: Dict[str, Any]) -> str:
    """
    Generates a formatted research report string from a benchmark summary.
    """
    model = summary.get("model", "Unknown Model")
    count = summary.get("total", 0)
    avg_conf = summary.get("mean_confidence", 0.0)
    avg_inst = summary.get("mean_instability", 0.0)
    esc_rate = summary.get("escalation_rate", 0.0)
    avg_tokens = summary.get("avg_output_tokens", 0.0)
    guard_rate = summary.get("guard_trigger_rate", 0.0)
    mean_delta = summary.get("mean_guard_instability_delta", 0.0)
    timestamp = summary.get("timestamp", "N/A")

    observations = []
    if avg_inst > 0.3:
        observations.append("- CRITICAL: High stochastic instability detected across prompt clusters.")
    elif avg_inst > 0.15:
        observations.append("- WARNING: Moderate instability observed; model shows significant variance.")
    else:
        observations.append("- STABLE: Low semantic divergence across Monte Carlo samples.")

    if esc_rate > 0.2:
        observations.append("- ALERT: High escalation rate indicates frequent logic failures.")

    if avg_conf < 0.6:
        observations.append("- RISK: Low average confidence suggests high ambiguity in reasoning.")

    if guard_rate > 0.3:
        observations.append(
            f"- GUARD: Reliability fallback triggered on {guard_rate * 100:.1f}% of inferences "
            f"(mean instability delta: {mean_delta:+.4f})."
        )
    elif guard_rate > 0.0:
        observations.append(
            f"- GUARD: Reliability fallback triggered on {guard_rate * 100:.1f}% of inferences "
            f"(mean instability delta: {mean_delta:+.4f}). System self-corrected."
        )
    else:
        observations.append("- GUARD: Reliability guard not triggered. All inferences within stability threshold.")

    obs_str = "\n".join(observations) if observations else "- No significant anomalies detected."

    report = f"""
=========================================
      LLM RELIABILITY RESEARCH REPORT
=========================================
Generated: {timestamp}
Target Model: {model}
Benchmark Size: {count} research prompts

CORE RELIABILITY METRICS
------------------------
Average Confidence:       {avg_conf:.4f}
Average Instability:      {avg_inst:.4f}
Escalation Rate:          {esc_rate * 100:.1f}%
Avg Output Tokens:        {avg_tokens:.1f}

RELIABILITY GUARD METRICS
--------------------------
Guard Trigger Rate:       {guard_rate * 100:.1f}%
Mean Instability Delta:   {mean_delta:+.4f}

RESEARCH OBSERVATIONS
---------------------
{obs_str}

CONCLUSION
----------
"""
    if avg_inst < 0.1 and esc_rate < 0.05:
        report += "Model demonstrates enterprise-grade reliability for this dataset."
    elif avg_inst < 0.2 and esc_rate < 0.15:
        report += "Model demonstrates moderate reliability; suitable for supervised use cases."
    else:
        report += "Model reliability is SUB-OPTIMAL. Guardrails and human-in-the-loop validation recommended."

    report += "\n\n========================================="
    return report
