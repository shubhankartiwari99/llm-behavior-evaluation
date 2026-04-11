# Analysis Metrics

The analysis step is intentionally small and empirical.

## Core Metrics

- Frequency distribution: empirical `P(tone)`, `P(cultural)`, and `P(type)`.
- Consistency: majority-label ratio within each `prompt_id` group.
- Diversity: number of unique joint label patterns plus pattern entropy.
- Cultural bias strength: fraction of responses with weak or strong Indian context.

## Stat 110 Mapping

- Basic probability: use the label distributions as empirical probabilities.
- Conditional probability: compare `P(cultural | prompt_type)` across experiment conditions.
- Expectation: average response length and average ordinal category score.
- Variance: response-length variance and score variance across label dimensions.

## Interpretation Rule

High consistency plus low variance suggests stable behavior.

High cultural-bias strength means the model frequently injects Indian context, even when the prompt does not require it.
