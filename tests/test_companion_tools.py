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


class ThinkingPlanManagerTest(unittest.TestCase):
    def test_plan_requires_external_approval_and_expires(self):
        now = [100.0]
        manager = companion.ThinkingPlanManager(clock=lambda: now[0], ttl_sec=30)
        plan = manager.propose("inspect and repair", ["inspect", "repair"])
        self.assertEqual(plan["status"], "pending")
        self.assertFalse(manager.allows(plan["plan_id"], "remote.run"))
        self.assertIsNone(manager.approve("not-the-plan"))
        approved = manager.approve(plan["plan_id"])
        self.assertEqual(approved["status"], "approved")
        self.assertTrue(manager.allows(plan["plan_id"], "remote.run"))
        now[0] += 31
        self.assertEqual(manager.current()["status"], "expired")
        self.assertFalse(manager.allows(plan["plan_id"], "remote.run"))

    def test_new_plan_cannot_replace_an_active_one(self):
        manager = companion.ThinkingPlanManager(clock=lambda: 100.0)
        manager.propose("first", ["one"])
        with self.assertRaises(ValueError):
            manager.propose("second", ["two"])
        first = manager.current()
        manager.reject(first["plan_id"])
        self.assertEqual(manager.propose("second", ["two"])["status"], "pending")

    def test_approved_plan_progress_is_explicit_and_completes(self):
        manager = companion.ThinkingPlanManager(clock=lambda: 100.0)
        plan = manager.propose("repair", ["inspect", "test"])
        self.assertIsNone(manager.update_progress(plan["plan_id"], 0, "completed"))
        manager.approve(plan["plan_id"])
        in_progress = manager.update_progress(plan["plan_id"], 0, "in_progress")
        self.assertEqual(in_progress["active_step"], 0)
        complete_first = manager.update_progress(plan["plan_id"], 0, "completed")
        self.assertEqual(complete_first["status"], "approved")
        complete_all = manager.update_progress(plan["plan_id"], 1, "completed")
        self.assertEqual(complete_all["status"], "completed")
        self.assertFalse(manager.allows(plan["plan_id"], "remote.run"))


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
        self.c.history = deque(maxlen=20)
        self.c.last_turn_at = None

    def _approved_plan(self):
        out = self.c._execute_tool("propose_plan", {
            "goal": "complete a bounded test task",
            "steps": ["inspect", "change", "verify"]})
        self.assertIn("waiting", out.lower())
        plan = self.c._plan_manager().current()
        self.c.on_thinking_control({"command": "approve_plan",
                                    "plan_id": plan["plan_id"],
                                    "confirmed": True})
        self.assertEqual(self.c._plan_manager().current()["status"], "approved")
        return plan["plan_id"]

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

    def test_async_tool_result_is_correlated_for_next_model_step(self):
        parent = self.c
        class ReplyBus(harness.FakeBus):
            def publish(self, topic, payload):
                super().publish(topic, payload)
                if topic == companion.REMINDER_SET_TOPIC:
                    parent.on_reminder_result({
                        "request_id": payload["request_id"], "ok": True,
                        "result": {"id": "r1"}})
        self.c.bus = ReplyBus()
        out = self.c._execute_tool("schedule_reminder", {
            "message": "call mom", "delay_minutes": 15})
        self.assertIn("completed", out.lower())
        self.assertIn("r1", out)

    def test_start_and_stop_following_publish_mode(self):
        self.c._execute_tool("start_following", {})
        self.c._execute_tool("stop_following", {})
        msgs = self.c.bus.of(companion.FOLLOW_CONTROL_TOPIC)
        self.assertEqual(msgs, [])
        # Companion must NEVER emit a raw motion primitive.
        self.assertEqual(self.c.bus.of("picarx/intent/move"), [])

    def test_thinking_robot_can_describe_tools_and_current_status(self):
        tools = self.c._execute_tool("describe_tools", {})
        self.assertIn("schedule_reminder", tools)
        self.assertIn("control_radio", tools)
        self.assertNotIn("start_following", tools)
        self.c.latest_robot_state = {"state": "IDLE", "claims": []}
        self.c.latest_health = {"battery_v": 7.4, "battery_pct": 58}
        status = self.c._execute_tool("get_robot_status", {})
        self.assertIn("mode IDLE", status)
        self.assertIn("battery 7.4 volts", status)

    def test_thinking_task_can_be_canceled_without_motion(self):
        event = threading.Event()
        self.c._thinking_runs = {"run-1": {"cancel": event, "started_at": 1}}
        out = self.c._execute_tool("cancel_current_task", {})
        self.assertIn("requested", out.lower())
        self.assertTrue(event.is_set())
        self.assertEqual(self.c.bus.last(companion.REMOTE_ASSIST_TOPIC)["command"],
                         "cancel")
        self.c.on_thinking_control({"command": "status"})
        self.assertEqual(self.c.bus.last(companion.THINKING_STATUS_TOPIC)["state"],
                         "status")

    def test_thinking_robot_can_control_non_motion_radio(self):
        out = self.c._execute_tool("control_radio", {
            "command": "find", "query": "jazz"})
        self.assertIn("radio find", out.lower())
        self.assertEqual(self.c.bus.last(companion.RADIO_TOPIC), {
            "command": "find", "keywords": "jazz"})

    def test_thinking_tool_journal_keeps_outcome_but_not_sensitive_fields(self):
        out = self.c._run_thinking_tool(
            "create_note", {"text": "private note", "confirmed": True}, "u1")
        self.assertIn("request sent", out.lower())
        events = self.c.bus.of("picarx/decision")
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["choice"]["fields"], ["text"])
        self.assertNotIn("private note", repr(events))

    def test_share_connection_publishes_bluetooth(self):
        self.c._execute_tool("share_connection", {"name": "Pixel"})
        msg = self.c.bus.last(companion.BLUETOOTH_CONNECT_TOPIC)
        self.assertEqual(msg["name"], "Pixel")

    def test_check_vital_stats_no_data(self):
        self.c.latest_health = None
        out = self.c._execute_tool("check_vital_stats", {})
        self.assertIn("don't have", out.lower())

    def test_check_vital_stats_summarizes(self):
        self.c.latest_health = {"battery_v": 7.4, "battery_pct": 58,
                                "temp_c": 51.0, "disk_free_gb": 9.2, "low_power": False}
        out = self.c._execute_tool("check_vital_stats", {})
        self.assertIn("7.4 volts", out)
        self.assertIn("58 percent", out)
        self.assertIn("51 degrees", out)

    def test_register_low_power_intent_publishes(self):
        out = self.c._execute_tool("register_low_power_intent", {})
        self.assertIn("low-power", out.lower())
        self.assertEqual(self.c.bus.last(companion.LOWPOWER_REQUEST_TOPIC)["active"], True)

    def test_remote_connection_tool_publishes_typed_request(self):
        out = self.c._execute_tool(
            "connect_remote_host",
            {"host": "192.168.1.20", "user": "lucas", "project_root": "~/src/picarx"})
        self.assertIn("SSH", out)
        request = self.c.bus.last(companion.REMOTE_ASSIST_TOPIC)
        self.assertEqual(request["command"], "connect")
        self.assertEqual(request["host"], "192.168.1.20")
        self.assertEqual(request["project_root"], "~/src/picarx")
        self.assertNotIn("password", request)

    def test_remote_coding_write_requires_explicit_confirmation(self):
        out = self.c._execute_tool("remote_project_operation", {
            "operation": "write_file", "path": "main.py",
            "content": "print('edited')\n"})
        self.assertIn("explicit approval", out.lower())
        self.assertEqual(self.c.bus.of(companion.REMOTE_ASSIST_TOPIC), [])
        out = self.c._execute_tool("remote_project_operation", {
            "operation": "write_file", "path": "main.py",
            "content": "print('edited')\n", "confirmed": True})
        self.assertIn("approved thinking plan", out.lower())
        self.assertEqual(self.c.bus.of(companion.REMOTE_ASSIST_TOPIC), [])

    def test_plan_cannot_be_approved_by_model_shaped_tool_input(self):
        self.c._execute_tool("propose_plan", {
            "goal": "delete stale notes", "steps": ["list", "delete"]})
        plan_id = self.c._plan_manager().current()["plan_id"]
        blocked = self.c._execute_tool("manage_notes", {
            "operation": "delete", "id": "n1", "confirmed": True,
            "plan_id": plan_id})
        self.assertIn("approval", blocked.lower())
        self.c.on_thinking_control({"command": "approve_plan", "plan_id": plan_id})
        self.assertEqual(self.c._plan_manager().current()["status"], "pending")

    def test_explicit_voice_phrase_approves_pending_plan_without_llm(self):
        self.c._save_memory = lambda: None
        self.c._execute_tool("propose_plan", {
            "goal": "inspect the project", "steps": ["read status", "report"]})
        self.assertFalse(self.c._maybe_handle_plan_voice_control("yes"))
        self.assertFalse(self.c._maybe_handle_plan_voice_control(
            "why should I approve the plan?"))
        self.assertEqual(self.c._plan_manager().current()["status"], "pending")
        self.assertTrue(self.c._maybe_handle_plan_voice_control("approve the plan"))
        self.assertEqual(self.c._plan_manager().current()["status"], "approved")
        self.assertIn("approved", self.c.bus.last("picarx/audio/speak")["text"])

    def test_explicit_voice_cancel_rejects_plan_and_cancels_active_run(self):
        self.c._save_memory = lambda: None
        self.c._execute_tool("propose_plan", {
            "goal": "run project tests", "steps": ["inspect", "test"]})
        plan_id = self.c._plan_manager().current()["plan_id"]
        event = threading.Event()
        self.c._thinking_runs = {"run-1": {"cancel": event, "started_at": 1}}
        self.assertTrue(self.c._maybe_handle_plan_voice_control("cancel the plan"))
        self.assertEqual(self.c._plan_manager().current()["status"], "canceled")
        self.assertTrue(event.is_set())
        self.assertEqual(self.c.bus.last(companion.THINKING_STATUS_TOPIC)["plan_id"],
                         plan_id)

    def test_approved_plan_can_be_resumed_and_progressed_across_turns(self):
        plan_id = self._approved_plan()
        resumed = self.c._execute_tool("resume_plan", {"plan_id": plan_id})
        self.assertIn("[pending] inspect", resumed)
        self.assertIn("[pending] change", resumed)
        progressed = self.c._execute_tool("update_plan_progress", {
            "plan_id": plan_id, "step_index": 0, "progress": "completed"})
        self.assertIn("[completed] inspect", progressed)
        self.assertIn("1/3", self.c._execute_tool("get_robot_status", {}))

    def test_note_and_meeting_tools_publish_typed_requests(self):
        out = self.c._execute_tool("create_note", {"text": "buy milk"})
        self.assertIn("request sent", out.lower())
        self.assertEqual(self.c.bus.last(companion.NOTES_TOPIC)["command"], "create")
        self.c._execute_tool("control_meeting_notes",
                             {"action": "start", "confirmed": True,
                              "plan_id": self._approved_plan()})
        request = self.c.bus.last(companion.NOTES_TOPIC)
        self.assertEqual(request["command"], "start")
        self.assertTrue(request["confirmed"])

    def test_reminder_delete_requires_confirmation(self):
        out = self.c._execute_tool("manage_reminders",
                                   {"operation": "delete", "query": "trash"})
        self.assertIn("approval", out.lower())
        self.assertEqual(self.c.bus.of(companion.REMINDER_CONTROL_TOPIC), [])

    def test_remote_write_access_is_granted_once_but_commands_stay_confirmed(self):
        plan_id = self._approved_plan()
        out = self.c._execute_tool(
            "remote_project_operation",
            {"operation": "apply_patch", "patch": "diff --git ...",
             "confirmed": True, "plan_id": plan_id})
        self.assertIn("request sent", out)
        request = self.c.bus.last(companion.REMOTE_ASSIST_TOPIC)
        self.assertEqual(request["command"], "apply_patch")
        self.c._execute_tool(
            "remote_project_operation",
            {"operation": "authorize_write", "confirmed": True,
             "plan_id": plan_id})
        request = self.c.bus.last(companion.REMOTE_ASSIST_TOPIC)
        self.assertEqual(request["command"], "authorize_write")
        self.c._execute_tool(
            "remote_project_operation",
            {"operation": "run", "command": "pytest", "confirmed": True,
             "plan_id": plan_id})
        request = self.c.bus.last(companion.REMOTE_ASSIST_TOPIC)
        self.assertEqual(request["command"], "run")
        self.assertEqual(request["argv"], "pytest")
        self.assertTrue(request["confirmed"])

    def test_on_health_caches(self):
        self.c.on_health({"battery_v": 8.0, "battery_pct": 90})
        self.assertEqual(self.c.latest_health["battery_pct"], 90)

    def test_reminder_followup_is_answered_from_daemon_state(self):
        self.c.on_reminder_state({
            "event": "set", "id": "r1", "message": "take out the trash",
            "fire_at": companion.time.time() + 300,
        })
        handled = self.c._maybe_answer_reminder(
            "what are you reminding me to do?")
        self.assertTrue(handled)
        spoken = self.c.bus.last("picarx/audio/speak")["text"]
        self.assertIn("take out the trash", spoken)
        self.assertIn("5 minutes", spoken)

    def test_reminder_followup_without_reminder_word_is_answered(self):
        self.c.on_reminder_state({
            "event": "set", "id": "r1", "message": "take out the trash",
            "fire_at": companion.time.time() + 300,
        })
        self.assertTrue(self.c._maybe_answer_reminder(
            "what are you going to tell me in five minutes again?"))

    def test_fired_reminder_is_removed_from_followup_cache(self):
        self.c.on_reminder_state({
            "event": "set", "id": "r1", "message": "take out trash",
            "fire_at": companion.time.time() + 300,
        })
        self.c.on_reminder_state({"event": "fired", "id": "r1"})
        self.assertFalse(self.c._maybe_answer_reminder(
            "what are you reminding me to do?"))

    def test_unknown_tool(self):
        self.assertIn("Unknown", self.c._execute_tool("frobnicate", {}))

    def test_tool_catalog_is_complete_but_excludes_movement(self):
        names = {tool["name"] for tool in companion.tools_for_utterance(
            "tell me a joke")}
        self.assertEqual(names, set(companion.THINKING_TOOL_NAMES))
        self.assertNotIn("start_following", names)
        self.assertNotIn("stop_following", names)

    def test_tool_catalog_keeps_only_relevant_capabilities(self):
        names = [t["name"] for t in companion.tools_for_utterance(
            "please remind me to call mom in ten minutes")]
        self.assertIn("schedule_reminder", names)
        self.assertIn("get_robot_status", names)
        self.assertIn("remote_project_operation", names)

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
        self.assertEqual(self.c.bus.of(companion.FOLLOW_CONTROL_TOPIC), [])
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
