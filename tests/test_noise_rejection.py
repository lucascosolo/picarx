import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import speech_match  # noqa: E402
import audio_nodes  # noqa: E402
import companion  # noqa: E402


class QualityScoreTest(unittest.TestCase):
    """speech_match.quality_score - the deterministic noise screen."""

    def test_empty_and_lone_fillers_score_zero(self):
        self.assertEqual(speech_match.quality_score(""), 0.0)
        for word in ("the", "a", "and", "or", "um", "uh", "hmm"):
            self.assertEqual(speech_match.quality_score(word), 0.0,
                             f"lone '{word}' should score 0")

    def test_fillers_fine_inside_sentences(self):
        # The blacklist is single-word only.
        self.assertGreater(speech_match.quality_score("play the radio"), 0.6)

    def test_real_requests_score_high(self):
        for text in ("what time is it", "tell me a story",
                      "play the radio", "explore"):
            self.assertGreaterEqual(speech_match.quality_score(text), 0.5, text)

    def test_strong_short_replies_keep_a_floor(self):
        # A mid-conversation "yes" must not be scored like noise.
        self.assertGreaterEqual(speech_match.quality_score("yes", 0.9), 0.5)
        self.assertGreaterEqual(speech_match.quality_score("okay", 0.9), 0.5)

    def test_confidence_scales_score(self):
        high = speech_match.quality_score("play the radio", 0.95)
        low = speech_match.quality_score("play the radio", 0.1)
        self.assertGreater(high, low)

    def test_weak_fragment_scores_low(self):
        # A contentless two-word noise decode lands in the reject bands.
        self.assertLess(speech_match.quality_score("it that", 0.4), 0.2)


class AudioHeardGateTest(unittest.TestCase):
    """_emit_result drops noise before it ever reaches the bus."""

    def _node(self):
        node = audio_nodes.AudioNode.__new__(audio_nodes.AudioNode)
        node.bus = harness.FakeBus()
        node._last_stop_reflex_at = 0.0
        node._utt_peak_rms = 0.0
        node._utt_floor = None
        return node

    def _emit(self, node, text, conf_word=None):
        words = [{"conf": conf_word}] if conf_word is not None else []
        import json
        node._emit_result(json.dumps({"text": text, "result": words}), time.time())

    def test_low_confidence_dropped(self):
        node = self._node()
        self._emit(node, "garbled mush", conf_word=0.2)
        self.assertEqual(node.bus.of("picarx/audio/heard"), [])
        rejected = node.bus.of("picarx/audio/rejected")
        self.assertEqual(len(rejected), 1)
        self.assertIn("confidence", rejected[0]["reason"])

    def test_lone_filler_dropped(self):
        node = self._node()
        self._emit(node, "the", conf_word=0.9)
        self.assertEqual(node.bus.of("picarx/audio/heard"), [])
        self.assertIn("filler", node.bus.of("picarx/audio/rejected")[0]["reason"])

    def test_weak_energy_dropped(self):
        node = self._node()
        node._utt_floor = 100.0
        node._utt_peak_rms = 180.0   # < 2.5x floor - barely over the gate trigger
        self._emit(node, "faint mumble", conf_word=0.9)
        self.assertEqual(node.bus.of("picarx/audio/heard"), [])
        self.assertIn("noise floor", node.bus.of("picarx/audio/rejected")[0]["reason"])

    def test_clean_speech_published(self):
        node = self._node()
        node._utt_floor = 100.0
        node._utt_peak_rms = 900.0
        self._emit(node, "play the radio", conf_word=0.85)
        heard = node.bus.of("picarx/audio/heard")
        self.assertEqual(len(heard), 1)
        self.assertEqual(heard[0]["text"], "play the radio")

    def test_stop_never_filtered(self):
        # Safety words must reach field_agent even on a marginal decode.
        node = self._node()
        node._utt_floor = 100.0
        node._utt_peak_rms = 120.0
        self._emit(node, "stop", conf_word=0.15)   # fails every other check
        self.assertEqual(len(node.bus.of("picarx/audio/heard")), 1)

    def test_peak_resets_between_utterances(self):
        node = self._node()
        node._utt_floor = 100.0
        node._utt_peak_rms = 900.0
        self._emit(node, "hello robot friend", conf_word=0.9)
        self.assertEqual(node._utt_peak_rms, 0.0)


class CompanionChatGateTest(unittest.TestCase):
    """on_unhandled's three tiers: silent drop / soft reply / full pipeline."""

    def _companion(self):
        c = companion.Companion.__new__(companion.Companion)
        import queue
        c.bus = harness.FakeBus()
        c.work_queue = queue.Queue()
        c._last_didnt_catch_at = 0.0
        return c

    def test_noise_dropped_silently(self):
        c = self._companion()
        c.on_unhandled({"text": "it that", "confidence": 0.3})
        self.assertTrue(c.work_queue.empty())              # no LLM work queued
        self.assertEqual(c.bus.of("picarx/audio/speak"), [])   # and no reply
        self.assertEqual(len(c.bus.of("picarx/audio/rejected")), 1)

    def test_midband_gets_soft_reply_no_llm(self):
        c = self._companion()
        c.on_unhandled({"text": "banana curtains", "confidence": 0.5})
        self.assertTrue(c.work_queue.empty())              # still no LLM call
        speaks = c.bus.of("picarx/audio/speak")
        self.assertEqual(len(speaks), 1)
        self.assertIn("didn't catch", speaks[0]["text"])

    def test_soft_reply_throttled(self):
        c = self._companion()
        c.on_unhandled({"text": "banana curtains", "confidence": 0.5})
        c.on_unhandled({"text": "velvet spoon", "confidence": 0.5})
        self.assertEqual(len(c.bus.of("picarx/audio/speak")), 1)   # once, not twice

    def test_real_speech_reaches_pipeline(self):
        c = self._companion()
        c.on_unhandled({"text": "tell me a story about robots", "confidence": 0.85})
        self.assertEqual(c.work_queue.get_nowait(), ("chat", "tell me a story about robots"))

    def test_short_reply_survives_conversation(self):
        # "yes" answering a question mid-conversation must reach the LLM.
        c = self._companion()
        c.on_unhandled({"text": "yes", "confidence": 0.9})
        self.assertEqual(c.work_queue.get_nowait(), ("chat", "yes"))

    def test_no_confidence_still_gated(self):
        # Payloads without confidence (older publishers) still get scored.
        c = self._companion()
        c.on_unhandled({"text": "the"})
        self.assertTrue(c.work_queue.empty())


if __name__ == "__main__":
    unittest.main()
