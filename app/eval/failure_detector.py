import re

def detect_failures(output: str, metrics: dict) -> list[str]:
    """
    Analyzes model output and telemetry to identify specific failure modes.
    
    Args:
        output: The primary text output from the model.
        metrics: Dictionary containing reliability metrics (entropy, instability, etc.).
        
    Returns:
        A list of detected failure mode identifiers.
    """
    failures = []

    # 1. Dialogue Contamination
    # Patterns indicating the model is simulating a multi-turn conversation incorrectly.
    if re.search(r"(User:|Assistant:|Human:|System:)", output, re.IGNORECASE):
        failures.append("dialogue_contamination")

    # 2. Stochastic Instability
    # High randomness/disagreement between Monte Carlo samples.
    if metrics.get("instability", 0) > 0.3 or metrics.get("entropy", 0) > 0.8:
        failures.append("stochastic_instability")

    # 3. Semantic Divergence
    # Significant meaning shifts across samples (indicated by multiple clusters).
    if metrics.get("cluster_count", 0) > 1 and metrics.get("semantic_dispersion", 0) > 0.15:
        failures.append("semantic_divergence")

    # 4. Instruction Drift
    # Typically indicated by excessive verbosity for a prompt asking for brevity.
    # Note: This is an approximation based on token count.
    if metrics.get("output_tokens", 0) > 350:
        failures.append("instruction_drift")

    # 5. Format Corruption
    # Detecting broken number formatting or nonsensical spacing (e.g., "1, 00 0").
    if re.search(r"\d,\s?\d\d\s\d", output) or re.search(r"[a-zA-Z]\s[a-zA-Z]\s[a-zA-Z]\s[a-zA-Z]", output):
        # The latter regex is a simple heuristic for exploded text (e.g. "H e l l o")
        failures.append("format_corruption")

    return failures
