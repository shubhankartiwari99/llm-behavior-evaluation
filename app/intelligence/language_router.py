"""
Layer 2 — Multilingual Routing

Takes a raw prompt + declared language + mode and returns a RoutingDecision
that drives language-aware prompt construction downstream.

Resolution logic:
  1. Script detection  — Unicode ranges determine actual written language
  2. Hinglish detection — Latin script but with Hinglish lexical markers
  3. Language reconciliation — declared vs detected, with deterministic rules
  4. Intent resolution — mode param when set, auto-detected otherwise

Routing confidence:
  high   — declared and detected agree, or script is unambiguous
  medium — Hinglish detected (ambiguous by nature), or auto intent used
  low    — declared and detected conflict, fallback applied
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# Supported resolved languages
ResolvedLang = Literal["en", "hi", "hinglish"]
ResolvedIntent = Literal["factual", "explanatory", "emotional", "conversational"]
IntentSource = Literal["mode_param", "auto_detected"]
RoutingConfidence = Literal["high", "medium", "low"]

# Mode → intent mapping (when mode is explicitly set by the caller)
_MODE_TO_INTENT: dict[str, ResolvedIntent] = {
    "factual": "factual",
    "emotional": "emotional",
    "mixed": "conversational",
    "explanatory": "explanatory",
}

# Hinglish lexical markers — Latin-script Hindi/Urdu loanwords commonly
# code-switched into English text by Indian users.
_HINGLISH_MARKERS: frozenset[str] = frozenset({
    "yaar", "bhai", "bhaiya", "didi", "kya", "hai", "hain", "nahi", "nhi",
    "kar", "karo", "karna", "hoga", "tha", "thi", "the", "mein", "main",
    "aur", "lekin", "par", "pe", "se", "ko", "ka", "ki", "ke", "ne",
    "ho", "hona", "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
    "kuch", "bahut", "bohot", "thoda", "zyada", "accha", "acha",
    "theek", "thik", "bilkul", "zaroor", "zaruri", "samajh", "pata",
    "lagta", "lagti", "lagta hai", "pata nahi", "koi", "sab", "apna",
    "apne", "iska", "uska", "mera", "tera", "unka", "hamara",
    "abhi", "phir", "fir", "baad", "pehle", "kal", "aaj", "agar",
    "toh", "to", "woh", "wo", "yeh", "ye", "bas", "sirf", "matlab",
    "dono", "log", "wala", "wali", "wale",
})


@dataclass(frozen=True)
class RoutingDecision:
    """Immutable routing result. All downstream prompt construction uses this."""
    detected_lang: str           # from Unicode script detection
    declared_lang: str           # from request emotional_lang param
    resolved_lang: ResolvedLang  # final decision
    is_hinglish: bool            # code-switching detected in Latin-script text
    resolved_intent: ResolvedIntent
    intent_source: IntentSource
    routing_confidence: RoutingConfidence

    def to_trace_dict(self) -> dict:
        return {
            "detected_lang": self.detected_lang,
            "declared_lang": self.declared_lang,
            "resolved_lang": self.resolved_lang,
            "is_hinglish": self.is_hinglish,
            "resolved_intent": self.resolved_intent,
            "intent_source": self.intent_source,
            "routing_confidence": self.routing_confidence,
        }


def _detect_script_language(text: str) -> str:
    """
    Detects written script from Unicode character ranges.
    Mirrors app/language.py but inlined here to keep the router self-contained
    and patchable in tests without the language module as a dependency.
    """
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            return "hi"    # Devanagari
        if 0x0A00 <= cp <= 0x0A7F:
            return "pa"    # Gurmukhi
        if 0x0980 <= cp <= 0x09FF:
            return "bn"    # Bengali
        if 0x0B80 <= cp <= 0x0BFF:
            return "ta"    # Tamil
        if 0x0C00 <= cp <= 0x0C7F:
            return "te"    # Telugu
        if 0x0C80 <= cp <= 0x0CFF:
            return "kn"    # Kannada
    return "en"


def _detect_hinglish(text: str) -> bool:
    """
    Returns True if the text is Latin-script but contains Hinglish lexical
    markers — i.e. code-switching by an Indian user writing in English
    but mixing Hindi/Urdu words.
    """
    words = set(text.lower().split())
    matches = words & _HINGLISH_MARKERS
    # Require at least 2 distinct markers to avoid false positives on
    # single common borrowings like "yaar" in otherwise pure English text.
    return len(matches) >= 2


def _resolve_language(
    script_lang: str,
    declared_lang: str,
    is_hinglish: bool,
) -> tuple[ResolvedLang, RoutingConfidence]:
    """
    Reconciles script detection with the caller's declared language.

    Rules (in priority order):
    1. Non-Latin Indic script → use that script's language, high confidence
    2. Devanagari → "hi", regardless of declaration
    3. Caller declared "hi" but script is Latin + Hinglish markers → "hinglish"
    4. Caller declared "hi" but script is Latin, no Hinglish → "hi" (trust declaration)
    5. Both agree on "en" and no Hinglish → "en", high confidence
    6. Conflict (declared "en", script "hi") → script wins, low confidence
    """
    if script_lang == "hi":
        confidence: RoutingConfidence = "high" if declared_lang == "hi" else "low"
        return "hi", confidence

    # Other Indic scripts — currently route as "en" for prompt construction
    # (they will get English instructions but the model will respond in-language
    # since Qwen 2.5 is multilingual). Tagged for future per-script routing.
    if script_lang in {"pa", "bn", "ta", "te", "kn"}:
        logger.info(
            "Routing: non-Hindi Indic script '%s' detected, routing as 'en'.",
            script_lang,
        )
        return "en", "medium"

    # Latin script path
    if declared_lang == "hi":
        if is_hinglish:
            return "hinglish", "medium"
        # Trust the declaration even without Hinglish markers
        return "hi", "medium"

    # Both are "en"
    if is_hinglish:
        return "hinglish", "medium"

    return "en", "high"


def _resolve_intent(
    mode: str,
    prompt: str,
) -> tuple[ResolvedIntent, IntentSource]:
    """
    Returns (intent, source).

    If mode is set and recognized, it takes priority (source = "mode_param").
    Otherwise, auto-detects from prompt text (source = "auto_detected").
    """
    if mode and mode in _MODE_TO_INTENT:
        return _MODE_TO_INTENT[mode], "mode_param"

    # Lazy import — keeps the router patchable without the intent module loaded
    from app.intent import detect_intent

    raw = detect_intent(prompt)
    # Map intent.py's extra categories to the router's 4-way schema
    intent_map: dict[str, ResolvedIntent] = {
        "factual": "factual",
        "explanatory": "explanatory",
        "emotional": "emotional",
        "conversational": "conversational",
        # Uncertain/refusal prompts are edge cases — route as factual so the
        # model responds rather than silently failing; guardrails handle safety.
        "uncertain": "factual",
        "refusal": "factual",
    }
    mapped: ResolvedIntent = intent_map.get(raw, "conversational")
    return mapped, "auto_detected"


def route_prompt(
    prompt: str,
    declared_lang: str,
    mode: str,
) -> RoutingDecision:
    """
    Main entry point for Layer 2 routing.

    Args:
        prompt:        Raw user prompt text.
        declared_lang: The emotional_lang value from the API request ("en"/"hi").
        mode:          The mode value from the API request ("factual"/"emotional"/
                       "mixed"/"explanatory"/"").

    Returns:
        A frozen RoutingDecision used downstream for prompt construction.
    """
    script_lang = _detect_script_language(prompt)
    # Only check Hinglish for Latin-script text
    is_hinglish = False
    if script_lang == "en":
        is_hinglish = _detect_hinglish(prompt)

    resolved_lang, confidence = _resolve_language(script_lang, declared_lang, is_hinglish)
    resolved_intent, intent_source = _resolve_intent(mode, prompt)

    # Downgrade confidence to medium if intent was auto-detected
    if intent_source == "auto_detected" and confidence == "high":
        confidence = "medium"

    decision = RoutingDecision(
        detected_lang=script_lang,
        declared_lang=declared_lang,
        resolved_lang=resolved_lang,
        is_hinglish=is_hinglish,
        resolved_intent=resolved_intent,
        intent_source=intent_source,
        routing_confidence=confidence,
    )

    logger.debug(
        "LanguageRouter: declared=%s detected=%s resolved=%s hinglish=%s "
        "intent=%s(%s) confidence=%s",
        declared_lang, script_lang, resolved_lang, is_hinglish,
        resolved_intent, intent_source, confidence,
    )

    return decision
