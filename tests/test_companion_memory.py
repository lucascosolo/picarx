import os
import sys
import tempfile
import threading
import time
import unittest
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import companion  # noqa: E402
from semantic_store import SemanticStore  # noqa: E402


class CompanionMemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "semantic.db")
        # Redirect the on-disk memory file into the tmp dir.
        self._orig_dir = companion.DATA_DIR
        self._orig_mem = companion.COMPANION_MEMORY_PATH
        companion.DATA_DIR = self.tmp
        companion.COMPANION_MEMORY_PATH = os.path.join(self.tmp, "companion_memory.json")

        self.c = companion.Companion.__new__(companion.Companion)
        self.c.lock = threading.Lock()
        self.c.bus = harness.FakeBus()
        self.c.history = deque(maxlen=companion.HISTORY_TURNS)
        self.c.last_turn_at = None
        self.c.semantic = SemanticStore(readonly=True, db_path=self.db)

    def tearDown(self):
        companion.DATA_DIR = self._orig_dir
        companion.COMPANION_MEMORY_PATH = self._orig_mem

    def _writer(self):
        return SemanticStore(readonly=False, db_path=self.db)

    # ---- episode date resolution ----

    def test_episode_query_date_today(self):
        d = self.c._episode_query_date("what did you do today")
        self.assertEqual(d, time.strftime("%Y-%m-%d"))

    def test_episode_query_date_yesterday(self):
        d = self.c._episode_query_date("summarize yesterday")
        expect = time.strftime("%Y-%m-%d", time.localtime(time.time() - 86400))
        self.assertEqual(d, expect)

    def test_non_episode_utterance_returns_none(self):
        self.assertIsNone(self.c._episode_query_date("play some music"))
        self.assertIsNone(self.c._episode_query_date("what happened"))  # no day word

    # ---- episode readback (no LLM) ----

    def test_maybe_answer_episode_reads_from_store(self):
        today = time.strftime("%Y-%m-%d")
        self._writer().upsert_fact(f"episode:{today}",
                                   "Rolled around the living room and met a new chair.", 0.6)
        handled = self.c._maybe_answer_episode("what did you do today")
        self.assertTrue(handled)
        spoken = self.c.bus.of("picarx/audio/speak")
        self.assertEqual(len(spoken), 1)
        self.assertIn("living room", spoken[0]["text"])

    def test_maybe_answer_episode_missing_day(self):
        handled = self.c._maybe_answer_episode("summarize yesterday")
        self.assertTrue(handled)  # still handled - answers "not put together yet"
        spoken = self.c.bus.of("picarx/audio/speak")
        self.assertEqual(len(spoken), 1)
        self.assertIn("yesterday", spoken[0]["text"].lower())

    def test_maybe_answer_episode_declines_non_episode(self):
        self.assertFalse(self.c._maybe_answer_episode("tell me a joke"))
        self.assertEqual(self.c.bus.of("picarx/audio/speak"), [])

    # ---- self-model injected into the personality system prompt ----

    def test_self_model_injected_into_system_prompt(self):
        w = self._writer()
        w.replace_subject("self", [("I back away first when stuck.", 0.75),
                                   ("I have mapped 2 places.", 0.65)],
                          source="self_model")
        prompt = self.c._compose_system_prompt()
        self.assertIn("I back away first when stuck.", prompt)
        self.assertIn(companion.SYSTEM_PROMPT, prompt)

    def test_system_prompt_falls_back_without_self_model(self):
        self.assertEqual(self.c._compose_system_prompt(), companion.SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()
