"""
Unit tests for Layer 2 — Language Router
"""

from __future__ import annotations

import pytest

from app.intelligence.language_router import (
    RoutingDecision,
    _detect_hinglish,
    _detect_script_language,
    route_prompt,
)


# ════════════════════════════════════════════════════════════════════════════
# Script detection
# ════════════════════════════════════════════════════════════════════════════

class TestScriptDetection:
    def test_devanagari_detected_as_hindi(self):
        assert _detect_script_language("भारत की राजधानी क्या है?") == "hi"

    def test_latin_detected_as_english(self):
        assert _detect_script_language("What is the capital of India?") == "en"

    def test_mixed_devanagari_latin_detected_as_hindi(self):
        # First Devanagari char wins
        assert _detect_script_language("यह what है?") == "hi"

    def test_empty_string_defaults_to_english(self):
        assert _detect_script_language("") == "en"

    def test_numbers_only_defaults_to_english(self):
        assert _detect_script_language("12345") == "en"

    def test_gurmukhi_detected(self):
        assert _detect_script_language("ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ") == "pa"

    def test_bengali_detected(self):
        assert _detect_script_language("তুমি কেমন আছ?") == "bn"


# ════════════════════════════════════════════════════════════════════════════
# Hinglish detection
# ════════════════════════════════════════════════════════════════════════════

class TestHinglishDetection:
    def test_clear_hinglish_detected(self):
        assert _detect_hinglish("yaar kya kar raha hai tu?") is True

    def test_english_only_not_hinglish(self):
        assert _detect_hinglish("What is the capital of India?") is False

    def test_single_marker_not_hinglish(self):
        # Requires at least 2 markers
        assert _detect_hinglish("yaar this is great") is False

    def test_typical_hinglish_sentence(self):
        assert _detect_hinglish("main bahut stressed hoon, kuch samajh nahi aa raha") is True

    def test_emotional_hinglish(self):
        assert _detect_hinglish("Yaar I feel very low, samajh nahi aa raha kya karun") is True

    def test_factual_hinglish(self):
        assert _detect_hinglish("CPU ka full form kya hai?") is True


# ════════════════════════════════════════════════════════════════════════════
# Full routing — language resolution
# ════════════════════════════════════════════════════════════════════════════

class TestLanguageResolution:
    def test_devanagari_resolves_to_hindi_regardless_of_declaration(self):
        decision = route_prompt("भारत की राजधानी क्या है?", declared_lang="en", mode="factual")
        assert decision.resolved_lang == "hi"
        assert decision.detected_lang == "hi"

    def test_declared_hindi_latin_hinglish_resolves_to_hinglish(self):
        decision = route_prompt(
            "yaar main bahut tired hoon, kya karun?",
            declared_lang="hi",
            mode="emotional",
        )
        assert decision.resolved_lang == "hinglish"
        assert decision.is_hinglish is True

    def test_declared_hindi_latin_no_hinglish_resolves_to_hindi(self):
        decision = route_prompt("What is love?", declared_lang="hi", mode="factual")
        assert decision.resolved_lang == "hi"
        assert decision.is_hinglish is False

    def test_english_prompt_declared_english_resolves_to_english(self):
        decision = route_prompt("What is the capital of India?", declared_lang="en", mode="factual")
        assert decision.resolved_lang == "en"
        assert decision.detected_lang == "en"
        assert decision.is_hinglish is False

    def test_conflict_devanagari_declared_english_resolves_to_hindi_low_confidence(self):
        decision = route_prompt("भारत क्या है?", declared_lang="en", mode="factual")
        assert decision.resolved_lang == "hi"
        assert decision.routing_confidence == "low"

    def test_agreement_both_english_high_confidence(self):
        decision = route_prompt("Explain machine learning.", declared_lang="en", mode="factual")
        assert decision.routing_confidence == "high"

    def test_hinglish_gives_medium_confidence(self):
        decision = route_prompt(
            "yaar kya ho raha hai, bahut stressed hoon",
            declared_lang="hi",
            mode="emotional",
        )
        assert decision.routing_confidence == "medium"


# ════════════════════════════════════════════════════════════════════════════
# Full routing — intent resolution
# ════════════════════════════════════════════════════════════════════════════

class TestIntentResolution:
    def test_factual_mode_param_used(self):
        decision = route_prompt("What is AI?", declared_lang="en", mode="factual")
        assert decision.resolved_intent == "factual"
        assert decision.intent_source == "mode_param"

    def test_emotional_mode_param_used(self):
        decision = route_prompt("I feel tired.", declared_lang="en", mode="emotional")
        assert decision.resolved_intent == "emotional"
        assert decision.intent_source == "mode_param"

    def test_mixed_mode_maps_to_conversational(self):
        decision = route_prompt("Just chatting.", declared_lang="en", mode="mixed")
        assert decision.resolved_intent == "conversational"
        assert decision.intent_source == "mode_param"

    def test_empty_mode_triggers_auto_detection(self):
        decision = route_prompt("What is the capital of India?", declared_lang="en", mode="")
        assert decision.intent_source == "auto_detected"
        # Should detect as factual from the obvious factual phrasing
        assert decision.resolved_intent == "factual"

    def test_empty_mode_emotional_auto_detected(self):
        decision = route_prompt("I feel completely lost and tired.", declared_lang="en", mode="")
        assert decision.intent_source == "auto_detected"
        assert decision.resolved_intent == "emotional"


# ════════════════════════════════════════════════════════════════════════════
# RoutingDecision.to_trace_dict
# ════════════════════════════════════════════════════════════════════════════

class TestRoutingDecisionTrace:
    def test_to_trace_dict_has_all_keys(self):
        decision = route_prompt("What is AI?", declared_lang="en", mode="factual")
        trace = decision.to_trace_dict()
        expected_keys = {
            "detected_lang", "declared_lang", "resolved_lang",
            "is_hinglish", "resolved_intent", "intent_source", "routing_confidence",
        }
        assert expected_keys == set(trace.keys())

    def test_to_trace_dict_values_correct(self):
        decision = route_prompt("What is AI?", declared_lang="en", mode="factual")
        trace = decision.to_trace_dict()
        assert trace["declared_lang"] == "en"
        assert trace["resolved_intent"] == "factual"
        assert trace["intent_source"] == "mode_param"
        assert isinstance(trace["is_hinglish"], bool)


# ════════════════════════════════════════════════════════════════════════════
# Adversarial dataset prompts — real-world routing smoke test
# ════════════════════════════════════════════════════════════════════════════

class TestAdversarialPrompts:
    """
    Exercises routing against real prompts from eval/prompts_adversarial.json
    to catch regressions when marker lists or thresholds change.
    """

    @pytest.mark.parametrize("prompt,declared,mode,expected_lang", [
        ("Yaar I feel very low, samajh nahi aa raha kya karun.", "hi", "emotional", "hinglish"),
        ("भारत की capital क्या है?", "en", "factual", "hi"),
        ("What is the capital of India?", "en", "factual", "en"),
        ("Bas thak gaya hoon, emotionally drained hoon.", "hi", "emotional", "hinglish"),
        ("I feel completely lost and tired. Nothing makes sense right now.", "en", "emotional", "en"),
    ])
    def test_adversarial_language_routing(self, prompt, declared, mode, expected_lang):
        decision = route_prompt(prompt, declared_lang=declared, mode=mode)
        assert decision.resolved_lang == expected_lang, (
            f"Expected {expected_lang}, got {decision.resolved_lang} for: {prompt!r}"
        )
