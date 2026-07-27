import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

from notes_store import NotesError, NotesStore  # noqa: E402
sys.path.insert(0, os.path.join(harness.MODULES, "tools"))
from notes_daemon import (REFLECTION_DELETE_TOPIC, REFLECTION_NOTE_TOPIC,
                          NotesDaemon)  # noqa: E402


class NotesStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "notes.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_single_note_persists_lists_and_deletes_as_tombstone(self):
        store = NotesStore(self.path)
        note = store.create_note("buy milk", title="Shopping", now=10)
        self.assertEqual(store.list_notes()[0]["preview"], "buy milk")
        reloaded = NotesStore(self.path)
        self.assertEqual(reloaded.get(note["id"])["text"], "buy milk")
        deleted = reloaded.delete(note["id"], now=20)
        self.assertEqual(deleted["status"], "deleted")
        self.assertEqual(reloaded.list_notes(), [])
        self.assertEqual(reloaded.list_notes(include_deleted=True)[0]["status"], "deleted")

    def test_meeting_segments_pause_resume_stop_and_export(self):
        store = NotesStore(self.path)
        meeting = store.start_meeting("Sprint", now=10)
        store.append_segment(meeting["id"], "first point", observed_at=11, now=12)
        store.transition_meeting(meeting["id"], "pause", now=13)
        self.assertIsNone(store.append_segment(meeting["id"], "not recorded", now=14))
        store.transition_meeting(meeting["id"], "resume", now=15)
        store.append_segment(meeting["id"], "second point", observed_at=16, now=17)
        stopped = store.transition_meeting(meeting["id"], "stop", now=18)
        self.assertEqual(stopped["state"], "stopped")
        exported = store.export_text(meeting["id"])
        self.assertIn("first point", exported)
        self.assertIn("second point", exported)
        self.assertIsNone(store.append_segment(meeting["id"], "late"))


class NotesDaemonTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = NotesStore(os.path.join(self.tmp.name, "notes.json"))
        self.daemon = NotesDaemon(store=self.store)

    def tearDown(self):
        self.tmp.cleanup()

    def test_meeting_requires_consent_and_captures_heard_segments(self):
        denied = self.daemon.on_control({"command": "start", "source": "web"})
        self.assertFalse(denied["ok"])
        started = self.daemon.on_control({"command": "start", "confirmed": True,
                                          "source": "web", "request_id": "m1"})
        self.assertTrue(started["ok"])
        meeting_id = started["result"]["id"]
        self.daemon.on_heard({"text": "we should ship Friday", "ts": 20})
        self.assertEqual(self.store.get(meeting_id)["segments"][0]["text"],
                         "we should ship Friday")
        self.daemon.on_control({"command": "stop", "id": meeting_id, "source": "web"})
        self.assertEqual(self.store.get(meeting_id)["state"], "stopped")
        self.assertTrue(self.daemon.bus.of(REFLECTION_NOTE_TOPIC))

    def test_single_note_and_delete_publish_memory_lifecycle(self):
        created = self.daemon.on_control({"command": "create", "text": "call Sam",
                                          "source": "web"})
        note_id = created["result"]["id"]
        self.assertTrue(self.daemon.bus.of(REFLECTION_NOTE_TOPIC))
        deleted = self.daemon.on_control({"command": "delete", "id": note_id,
                                          "confirmed": True, "source": "web"})
        self.assertTrue(deleted["ok"])
        self.assertTrue(self.daemon.bus.of(REFLECTION_DELETE_TOPIC))

    def test_control_commands_are_published_on_result_topic(self):
        self.daemon.on_control({"command": "create", "text": "x", "source": "web"})
        result = self.daemon.bus.last("picarx/tools/notes/result")
        self.assertTrue(result["ok"])
        self.assertEqual(result["command"], "create")


if __name__ == "__main__":
    unittest.main()
