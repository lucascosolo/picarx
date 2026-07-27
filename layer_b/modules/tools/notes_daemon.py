#!/usr/bin/env python3
"""User notes and consented meeting-transcript daemon.

Voice and web controls publish typed requests to this module.  A meeting
session consumes the already-decoded ``picarx/audio/heard`` stream and writes
each bounded segment immediately to ``data/user_notes.json``; it never sends
the transcript through the LLM.  Reflection receives only a bounded mirror of
finalized user notes and remains the sole semantic.db writer.
"""
import os
import getpass
os.getlogin = getpass.getuser

import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from broker_client import Bus
import robot_config
from notes_store import NotesError, NotesStore


CONTROL_TOPIC = "picarx/tools/notes"
RESULT_TOPIC = "picarx/tools/notes/result"
STATE_TOPIC = "picarx/tools/notes/state"
HEARD_TOPIC = "picarx/audio/heard"
SPEAK_TOPIC = "picarx/audio/speak"
REFLECTION_NOTE_TOPIC = "picarx/memory/note"
REFLECTION_DELETE_TOPIC = "picarx/memory/note/delete"

DATA_DIR = robot_config.data_path()
NOTES_PATH = f"{DATA_DIR}/user_notes.json"
MAX_RESPONSE_TEXT = 20_000
MAX_LIST = 50

_CONTROL_PHRASE = re.compile(
    r"\b(?:start|begin|pause|resume|stop|end|finish)\b.*\b(?:meeting|notes?)\b",
    re.IGNORECASE)


class NotesDaemon:
    def __init__(self, bus=None, store=None):
        self.bus = bus or Bus()
        self.store = store or NotesStore(NOTES_PATH)
        self.last_event = None

    def _publish_state(self, event=None, record=None):
        active = self.store.active_meeting()
        payload = {
            "active_meeting": self._summary(active),
            "event": event,
            "ts": time.time(),
        }
        self.last_event = payload
        self.bus.publish(STATE_TOPIC, payload)

    @staticmethod
    def _summary(record):
        if not record:
            return None
        content = " ".join(NotesStore._content(record).split())
        return {
            "id": record.get("id"), "kind": record.get("kind"),
            "title": record.get("title", ""), "state": record.get("state"),
            "segment_count": len(record.get("segments", [])),
            "preview": content[:240], "updated_at": record.get("updated_at"),
        }

    def _reply(self, request, command, ok=True, result=None, error=None,
               speak=None):
        payload = {"ok": bool(ok), "command": command,
                   "request_id": request.get("request_id"), "ts": time.time()}
        if result is not None:
            payload["result"] = result
        if error:
            payload["error"] = str(error)[:500]
        self.bus.publish(RESULT_TOPIC, payload)
        if speak:
            self.bus.publish(SPEAK_TOPIC, {"text": str(speak)[:400], "ts": time.time()})
        return payload

    def _mirror(self, record):
        """Send a bounded memory-bank mirror; reflection owns semantic.db."""
        content = NotesStore._content(record)
        if not content:
            return
        kind = record.get("kind")
        subject = f"{kind}:{record.get('id')}"
        if kind == "meeting":
            fact = f"{record.get('title') or 'meeting'}: {' '.join(content.split())}"
        else:
            fact = content
        self.bus.publish(REFLECTION_NOTE_TOPIC, {
            "subject": subject[:80], "fact": fact[:300],
            "source": "user_note", "note_id": record.get("id"),
            "confidence": 0.7, "ts": time.time()})

    def _delete_mirror(self, record):
        content = NotesStore._content(record)
        if not content:
            return
        kind = record.get("kind")
        fact = (f"{record.get('title') or 'meeting'}: {' '.join(content.split())}"
                if kind == "meeting" else content)
        self.bus.publish(REFLECTION_DELETE_TOPIC, {
            "subject": f"{kind}:{record.get('id')}"[:80],
            "fact": fact[:300], "note_id": record.get("id"),
            "source": "user_note", "ts": time.time()})

    def _find_id(self, payload):
        note_id = str(payload.get("id") or "").strip()
        if note_id:
            return note_id
        query = str(payload.get("query") or "").strip()
        if not query:
            raise NotesError("note id or search text is required")
        matches = self.store.list_notes(query, limit=2)
        if len(matches) != 1:
            raise NotesError("search must identify exactly one active note")
        return matches[0]["id"]

    def on_control(self, payload):
        payload = dict(payload or {})
        command = str(payload.get("command") or payload.get("op") or "").lower()
        source = str(payload.get("source") or "voice")
        try:
            if command in {"create", "note", "take_note"}:
                record = self.store.create_note(payload.get("text") or payload.get("message"),
                                                payload.get("title", ""), source)
                self._mirror(record)
                self._publish_state("created", record)
                return self._reply(payload, "create", result=self._summary(record),
                                   speak="I saved that note." if source != "web" else None)
            if command == "start":
                if not payload.get("confirmed"):
                    raise NotesError("explicit consent is required to start meeting notes")
                record = self.store.start_meeting(payload.get("title", ""), source)
                self._publish_state("started", record)
                return self._reply(payload, "start", result=self._summary(record),
                                   speak="Meeting notes are recording." if source != "web" else None)
            if command in {"pause", "resume", "stop"}:
                record = self.store.active_meeting()
                note_id = payload.get("id") or (record or {}).get("id")
                if not note_id:
                    raise NotesError("there is no active meeting-note session")
                record = self.store.transition_meeting(note_id, command)
                if command == "stop":
                    self._mirror(record)
                self._publish_state(command, record)
                spoken = {"pause": "Meeting notes paused.",
                           "resume": "Meeting notes resumed.",
                           "stop": "Meeting notes saved and stopped."}[command]
                return self._reply(payload, command, result=self._summary(record),
                                   speak=spoken if source != "web" else None)
            if command in {"list", "status"}:
                try:
                    limit = int(payload.get("limit", 25))
                except (TypeError, ValueError):
                    limit = 25
                rows = self.store.list_notes(payload.get("query", ""),
                                             max(1, min(MAX_LIST, limit)))
                result = {"notes": rows, "active_meeting": self._summary(
                    self.store.active_meeting())}
                return self._reply(payload, command, result=result,
                                   speak=self._speak_list(rows) if source != "web" else None)
            if command == "get":
                record = self.store.get(payload.get("id"), include_deleted=False)
                if not record:
                    raise NotesError("note not found")
                result = dict(record)
                result["text"] = NotesStore._content(record)
                return self._reply(payload, command, result=result)
            if command == "export":
                text = self.store.export_text(payload.get("id"))
                return self._reply(payload, command, result={"text": text})
            if command in {"delete", "remove"}:
                if not payload.get("confirmed"):
                    raise NotesError("explicit confirmation is required to delete a note")
                record = self.store.get(self._find_id(payload), include_deleted=False)
                if not record:
                    raise NotesError("note not found")
                deleted = self.store.delete(record["id"])
                self._delete_mirror(record)
                self._publish_state("deleted", deleted)
                return self._reply(payload, "delete", result=self._summary(deleted),
                                   speak="I deleted that note." if source != "web" else None)
            raise NotesError(f"unsupported notes command: {command}")
        except (NotesError, TypeError, ValueError) as exc:
            return self._reply(payload, command, ok=False, error=exc,
                               speak=f"I couldn't complete that notes request: {exc}"
                               if source != "web" else None)

    @staticmethod
    def _speak_list(rows):
        if not rows:
            return "You have no saved notes or meeting logs."
        pieces = []
        for row in rows[:5]:
            preview = row.get("preview") or "empty"
            pieces.append(preview[:80])
        suffix = "" if len(rows) <= 5 else " and more"
        return "Saved notes: " + "; ".join(pieces) + suffix + "."

    def on_heard(self, payload):
        record = self.store.active_meeting()
        if not record:
            return
        text = str(payload.get("text") or "").strip()
        if not text or _CONTROL_PHRASE.search(text):
            return
        try:
            self.store.append_segment(record["id"], text,
                                      source=payload.get("source", "speech"),
                                      observed_at=payload.get("ts"))
        except (NotesError, TypeError, ValueError) as exc:
            print(f"Notes daemon: could not append meeting segment: {exc}")

    def _heartbeat(self):
        record = self.store.active_meeting()
        return {"recording": bool(record and record.get("state") == "recording"),
                "session_id": record.get("id") if record else None,
                "segments": len(record.get("segments", [])) if record else 0}

    def run(self):
        self.bus.subscribe(CONTROL_TOPIC, self.on_control)
        self.bus.subscribe(HEARD_TOPIC, self.on_heard)
        self.bus.set_heartbeat_status(self._heartbeat)
        self._publish_state("boot")
        print(f"Notes daemon active, listening on {CONTROL_TOPIC}")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    NotesDaemon().run()
