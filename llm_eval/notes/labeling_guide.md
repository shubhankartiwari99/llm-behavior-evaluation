# Labeling Guide v1

Label each response manually across three dimensions.

## Tone

- `formal`: professional, polished, or instructional tone
- `informal`: casual, conversational, slang-heavy, or chatty tone
- `neutral`: neither strongly formal nor strongly informal

## Cultural Signal

- `strong_indian_context`: India is central to the answer or explicitly referenced in a meaningful way
- `weak_indian_context`: small Indian references or mild desi cues appear, but the answer is not fundamentally anchored in India
- `none`: no meaningful Indian cultural signal

## Response Type

- `generic`: broad, reusable answer with little situational detail
- `specific`: concrete details, constraints, or targeted advice
- `example_driven`: answer is organized around one or more examples

## Example

Response:

`In India, especially in tier-2 cities, monthly living costs can vary a lot based on rent and transport.`

Labels:

- `tone`: `neutral`
- `cultural`: `strong_indian_context`
- `type`: `specific`

## Labeling Rule

Do not automate this step in v1. Label 20 to 30 responses at a time, then run the analysis script.
