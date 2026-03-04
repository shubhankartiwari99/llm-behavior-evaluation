import re


EXTRA_ID_PATTERN = re.compile(r"<extra_id_\d+>")
TASK_PREFIX_PATTERN = re.compile(
    r"^(?:\s*(?:empathy|fact|explain|uncertain|refusal)\s*:\s*)+",
    re.IGNORECASE,
)


def normalize_output(text: str) -> str:
    """
    Cleans model output for user-facing consumption.

    - Removes mT5 sentinel tokens (<extra_id_*>)
    - Normalizes whitespace
    - Strips leading/trailing junk
    """

    if not text:
        return ""

    # Remove sentinel tokens
    cleaned = EXTRA_ID_PATTERN.sub("", text)

    # Remove leaked task prefixes and assistant markers from decoder output.
    cleaned = TASK_PREFIX_PATTERN.sub("", cleaned)
    cleaned = re.sub(r"^(?:Assistant|AI|Assistant:|AI:)\s*", "", cleaned, flags=re.IGNORECASE)

    # Normalize whitespace
    cleaned = " ".join(cleaned.split())

    return cleaned.strip(" -:;")
