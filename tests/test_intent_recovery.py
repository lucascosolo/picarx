import json
import os
import queue
import tempfile
import threading
import types
import unittest

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness
import companion
import tool_catalog


class _FakeClient:
    def __init__(self, verdict):
        self.verdict = verdict

    def messages(self):
        raise AssertionError("messages is not callable")

    class _Messages:
        def __init__(self, verdict):
            self.verdict = verdict

        def create(self, **kwargs):
            block = types.SimpleNamespace(
                type="text", text=json.dumps(self.verdict))
            return types.SimpleNamespace(content=[block])

    @property
    def messages(self):
        return self._Messages(self.verdict)


class IntentCatalogTest(unittest.TestCase):
    def test_read_only_is_auto_repairable(self):
        spec = tool_catalog.command_spec("where is bottle")
        self.assertEqual(spec["safety_class"], "read_only")
        self.assertTrue(spec["auto_repair"])
        self.assertEqual(spec["required_fields"], ["object"])

    def test_state_changing_and_remote_are_not_auto_repairable(self):
        for command, safety in (("take a note buy milk", "reversible_write"),
                                ("delete notes old plan", "destructive"),
                                ("ssh into robot.local", "remote")):
            spec = tool_catalog.command_spec(command)
            self.assertEqual(spec["safety_class"], safety)
            self.assertFalse(spec["auto_repair"])


class IntentRepairGateTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.original_paths = companion.DATA_DIR, companion.LEARNED_INTENTS_PATH
        companion.DATA_DIR = self.tmp
        companion.LEARNED_INTENTS_PATH = os.path.join(
            self.tmp, "learned_intents.json")
        self.c = companion.Companion.__new__(companion.Companion)
        self.c.lock = threading.Lock()
        self.c.bus = harness.FakeBus()
        self.c.work_queue = queue.Queue()
        self.c.learned_intents = {}
        self.c._warned_no_key = False
        self.c._client = None

    def tearDown(self):
        companion.DATA_DIR, companion.LEARNED_INTENTS_PATH = self.original_paths

    def _run(self, verdict, text="how is the battery"):
        self.c._client = _FakeClient(verdict)
        self.c._repair_intent(text)
        return self.c.bus.last(companion.INTENT_RECOVERY_STATUS_TOPIC)

    def test_high_confidence_read_only_repair_is_cached_and_dispatched(self):
        status = self._run({
            "command": "battery",
            "confidence": 0.97,
            "rationale": "battery is an explicit local status request",
        })
        self.assertEqual(status["state"], "accepted")
        key = companion.speech_match.canonicalize("how is the battery")
        self.assertEqual(self.c.learned_intents[key]["command"], "battery")
        heard = self.c.bus.last("picarx/audio/heard")
        self.assertEqual(heard, {"text": "battery", "source": "intent_repair"})

    def test_low_confidence_read_only_repair_does_not_execute(self):
        status = self._run({
            "command": "status",
            "confidence": 0.55,
            "rationale": "the transcript is ambiguous",
        })
        self.assertEqual(status["state"], "low_confidence")
        self.assertEqual(self.c.bus.of("picarx/audio/heard"), [])
        self.assertEqual(self.c.learned_intents, {})

    def test_state_changing_repair_requires_explicit_path(self):
        status = self._run({
            "command": "take a note buy milk",
            "confidence": 0.99,
            "rationale": "the words resemble a note request",
        }, text="save this thought buy milk")
        self.assertEqual(status["state"], "confirmation_required")
        self.assertEqual(status["safety_class"], "reversible_write")
        self.assertEqual(self.c.bus.of("picarx/audio/heard"), [])
        self.assertEqual(self.c.learned_intents, {})

    def test_missing_confidence_fails_closed(self):
        status = self._run({"command": "battery"}, text="check my charge")
        self.assertEqual(status["state"], "low_confidence")
        self.assertEqual(self.c.bus.of("picarx/audio/heard"), [])


if __name__ == "__main__":
    unittest.main()
