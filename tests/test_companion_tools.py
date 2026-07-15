import os
import sys
import tempfile
import threading
import unittest
from collections import deque
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import companion  # noqa: E402
from semantic_store import SemanticStore  # noqa: E402


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _tool(tool_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


class _FakeMessages:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=self.responses.pop(0))


class _FakeClient:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


class CompanionToolTest(unittest.TestCase):
    def setUp(self):
        self.c = companion.Companion.__new__(companion.Companion)
        self.c.lock = threading.Lock()
        self.c.bus = harness.FakeBus()
        self.c.semantic = SemanticStore(
            readonly=True, db_path=os.path.join(tempfile.mkdtemp(), "none.db"))

    # ---- direct tool dispatch ----

    def test_schedule_reminder_publishes(self):
        out = self.c._execute_tool("schedule_reminder",
                                   {"message": "call mom", "delay_minutes": 15})
        self.assertIn("scheduled", out.lower())
        msg = self.c.bus.last(companion.REMINDER_SET_TOPIC)
        self.assertEqual(msg["message"], "call mom")
        self.assertEqual(msg["delay_minutes"], 15)

    def test_schedule_reminder_needs_time(self):
        out = self.c._execute_tool("schedule_reminder", {"message": "x"})
        self.assertIn("delay", out.lower())
        self.assertEqual(self.c.bus.of(companion.REMINDER_SET_TOPIC), [])

    def test_start_and_stop_following_publish_mode(self):
        self.c._execute_tool("start_following", {})
        self.c._execute_tool("stop_following", {})
        msgs = self.c.bus.of(companion.FOLLOW_CONTROL_TOPIC)
        self.assertEqual([m["enabled"] for m in msgs], [True, False])
        # Companion must NEVER emit a raw motion primitive.
        self.assertEqual(self.c.bus.of("picarx/intent/move"), [])

    def test_share_connection_publishes(self):
        self.c._execute_tool("share_connection", {"name": "Pixel"})
        msg = self.c.bus.last(companion.NETWORK_CONNECT_TOPIC)
        self.assertEqual(msg["name"], "Pixel")

    def test_unknown_tool(self):
        self.assertIn("Unknown", self.c._execute_tool("frobnicate", {}))

    # ---- full tool loop ----

    def test_tool_loop_executes_then_speaks(self):
        client = _FakeClient([
            [_tool("t1", "start_following", {})],       # round 1: model calls a tool
            [_text("Okay, following you now!")],        # round 2: model speaks
        ])
        messages = [{"role": "user", "content": "follow me"}]
        reply = self.c._chat_with_tools(client, messages)
        self.assertEqual(reply, "Okay, following you now!")
        # The tool actually fired.
        self.assertEqual(self.c.bus.last(companion.FOLLOW_CONTROL_TOPIC)["enabled"], True)
        # Two model round-trips: the tool call and the spoken follow-up.
        self.assertEqual(len(client.messages.calls), 2)
        # Every call advertised the tools.
        self.assertTrue(all("tools" in c for c in client.messages.calls))

    def test_tool_loop_plain_reply_no_tools(self):
        client = _FakeClient([[_text("Just chatting.")]])
        reply = self.c._chat_with_tools(client, [{"role": "user", "content": "hi"}])
        self.assertEqual(reply, "Just chatting.")
        self.assertEqual(len(client.messages.calls), 1)  # no extra round

    def test_tool_loop_bounded(self):
        # A model that keeps calling tools forever must stop at MAX_TOOL_ROUNDS.
        client = _FakeClient([[_tool(f"t{i}", "stop_following", {})]
                              for i in range(companion.MAX_TOOL_ROUNDS + 3)])
        self.c._chat_with_tools(client, [{"role": "user", "content": "x"}])
        self.assertLessEqual(len(client.messages.calls), companion.MAX_TOOL_ROUNDS)


if __name__ == "__main__":
    unittest.main()
