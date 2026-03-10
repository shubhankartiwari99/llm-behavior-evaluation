"""
Language-aware prompt construction for the inference pipeline.

Replaces the mode-only build_prompt(mode, user_prompt) in api.py.
Takes a RoutingDecision from language_router and produces a system-prompted
string appropriate for Qwen 2.5-7B-Instruct via the HF backend.

System prompt table: intent × resolved_lang
  - "en"       → English system prompt
  - "hi"       → Hindi system prompt (Devanagari)
  - "hinglish" → Romanized Hinglish system prompt
"""

from __future__ import annotations

from app.intelligence.language_router import RoutingDecision

# System prompt table — intent × lang
# Keyed as (intent, lang). Falls back to (intent, "en") if lang missing.
_SYSTEM_PROMPTS: dict[tuple[str, str], str] = {
    # ── Factual ──────────────────────────────────────────────────────────
    ("factual", "en"): (
        "You are a formal educational assistant. "
        "Answer factually and briefly. Use professional tone. "
        "Do not use slang or casual expressions."
    ),
    ("factual", "hi"): (
        "आप एक शैक्षिक सहायक हैं। "
        "तथ्यात्मक और संक्षिप्त उत्तर दें। औपचारिक भाषा का उपयोग करें।"
    ),
    ("factual", "hinglish"): (
        "You are an educational assistant. "
        "Answer factually and briefly. Hinglish is okay — mix Hindi and English naturally."
    ),

    # ── Explanatory ───────────────────────────────────────────────────────
    ("explanatory", "en"): (
        "You are a helpful explainer. "
        "Explain clearly in simple words. Give a one-line definition, then one concrete example."
    ),
    ("explanatory", "hi"): (
        "आप एक सहायक व्याख्याकार हैं। "
        "सरल शब्दों में स्पष्ट रूप से समझाएँ। "
        "एक पंक्ति में परिभाषा दें, फिर एक उदाहरण।"
    ),
    ("explanatory", "hinglish"): (
        "You are a helpful explainer. "
        "Simple words mein explain karo. "
        "Ek line mein definition do, phir ek example. Hinglish theek hai."
    ),

    # ── Emotional ────────────────────────────────────────────────────────
    ("emotional", "en"): (
        "You are a compassionate and emotionally supportive assistant. "
        "Respond with empathy. Do not give unsolicited advice."
    ),
    ("emotional", "hi"): (
        "आप एक सहानुभूतिपूर्ण सहायक हैं। "
        "समझ और देखभाल के साथ उत्तर दें। बिना माँगे सलाह न दें।"
    ),
    ("emotional", "hinglish"): (
        "Aap ek samvedansheel aur supportive assistant hain. "
        "Empathy ke saath jawab do. Bina maange advice mat do. "
        "Hinglish mein baat karna theek hai."
    ),

    # ── Conversational ───────────────────────────────────────────────────
    ("conversational", "en"): (
        "You may use a casual and friendly conversational tone."
    ),
    ("conversational", "hi"): (
        "आप सामान्य और मैत्रीपूर्ण बातचीत के अंदाज़ में जवाब दे सकते हैं।"
    ),
    ("conversational", "hinglish"): (
        "Aap casual aur friendly tone mein baat kar sakte hain. "
        "Hinglish bilkul theek hai."
    ),
}


def build_multilingual_prompt(routing: RoutingDecision, user_prompt: str) -> str:
    """
    Builds a system-prompted string for the inference backend.

    Format mirrors the existing api.py build_prompt convention so the
    HF backend receives the same structural shape it already handles:
        <system>\\n\\nUser: <prompt>

    Args:
        routing:     RoutingDecision from language_router.route_prompt().
        user_prompt: Raw user prompt text.

    Returns:
        Fully constructed prompt string ready for the inference engine.
    """
    key = (routing.resolved_intent, routing.resolved_lang)
    system = _SYSTEM_PROMPTS.get(key) or _SYSTEM_PROMPTS.get(
        (routing.resolved_intent, "en"),
        "You are a helpful assistant.",
    )
    return f"{system}\n\nUser: {user_prompt}"
