#!/usr/bin/env python3
"""Durable, bounded user notes and meeting transcripts.

This is a small JSON store rather than another SQLite writer.  It owns only
user-authored notes and meeting sessions; reflection remains the sole writer
of ``semantic.db`` and receives a best-effort mirror of finalized notes over
the bus.  Deleted records become tombstones so the local lifecycle is
auditable, while normal list/search operations hide them.
"""
import json
import os
import threading
import time
import uuid


MAX_NOTES = 500
MAX_NOTE_CHARS = 100_000
MAX_SEGMENTS = 5_000
MAX_SEGMENT_CHARS = 2_000
MAX_TITLE_CHARS = 120
MAX_PREVIEW_CHARS = 240


class NotesError(ValueError):
    """A malformed or unsafe note operation."""


class NotesStore:
    def __init__(self, path, max_notes=MAX_NOTES):
        self.path = os.path.abspath(os.path.expanduser(path))
        self.max_notes = int(max_notes)
        self.lock = threading.RLock()
        self.records = {}
        self._load()

    # ---------- persistence ----------

    def _load(self):
        try:
            with open(self.path, encoding="utf-8") as stream:
                saved = json.load(stream)
        except FileNotFoundError:
            return
        except (OSError, json.JSONDecodeError) as exc:
            # A corrupt optional memory file must not prevent the robot from
            # booting. Preserve the bad file for forensics rather than
            # overwriting it with an empty store.
            print(f"Notes store: could not load {self.path}: {exc}")
            return
        rows = saved.get("notes", saved) if isinstance(saved, dict) else {}
        if not isinstance(rows, dict):
            return
        for note_id, record in rows.items():
            if isinstance(record, dict) and self._valid_record(record):
                self.records[str(note_id)] = dict(record)

    @staticmethod
    def _valid_record(record):
        return (record.get("kind") in ("note", "meeting") and
                record.get("status", "active") in ("active", "deleted") and
                isinstance(record.get("created_at"), (int, float)))

    def _persist_locked(self):
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = self.path + ".tmp"
        payload = {"version": 1, "notes": self.records}
        try:
            with open(tmp, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
            os.replace(tmp, self.path)
        except OSError as exc:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise NotesError(f"could not persist notes: {exc}")

    # ---------- normalization / views ----------

    @staticmethod
    def _text(value, limit):
        text = str(value or "").strip()
        return text[:limit]

    @staticmethod
    def _now(now):
        return time.time() if now is None else float(now)

    def _active_count_locked(self):
        return sum(1 for record in self.records.values()
                   if record.get("status") == "active")

    @staticmethod
    def _content(record):
        if record.get("kind") == "note":
            return record.get("text", "")
        return "\n".join(segment.get("text", "")
                           for segment in record.get("segments", []))

    def _summary_locked(self, record):
        content = " ".join(self._content(record).split())
        return {
            "id": record["id"], "kind": record["kind"],
            "title": record.get("title", ""), "status": record["status"],
            "state": record.get("state"), "created_at": record["created_at"],
            "updated_at": record.get("updated_at", record["created_at"]),
            "deleted_at": record.get("deleted_at"),
            "segment_count": len(record.get("segments", [])),
            "preview": content[:MAX_PREVIEW_CHARS],
        }

    def _copy_locked(self, record):
        # JSON round-tripping is safe here and prevents callers from mutating
        # the in-memory record without going through persistence.
        return json.loads(json.dumps(record))

    # ---------- notes ----------

    def create_note(self, text, title="", source="user", now=None):
        text = self._text(text, MAX_NOTE_CHARS)
        if not text:
            raise NotesError("note text is empty")
        title = self._text(title, MAX_TITLE_CHARS)
        now = self._now(now)
        with self.lock:
            if self._active_count_locked() >= self.max_notes:
                raise NotesError("note capacity is full")
            note_id = uuid.uuid4().hex
            record = {
                "id": note_id, "kind": "note", "title": title,
                "text": text, "source": self._text(source, 40) or "user",
                "status": "active", "created_at": now, "updated_at": now,
            }
            self.records[note_id] = record
            self._persist_locked()
            return self._copy_locked(record)

    # ---------- meeting sessions ----------

    def active_meeting(self):
        with self.lock:
            for record in self.records.values():
                if (record.get("kind") == "meeting" and
                        record.get("status") == "active" and
                        record.get("state") in ("recording", "paused")):
                    return self._copy_locked(record)
        return None

    def start_meeting(self, title="", source="user", now=None):
        title = self._text(title, MAX_TITLE_CHARS)
        now = self._now(now)
        with self.lock:
            if self._active_count_locked() >= self.max_notes:
                raise NotesError("note capacity is full")
            if any(record.get("kind") == "meeting" and
                   record.get("status") == "active" and
                   record.get("state") in ("recording", "paused")
                   for record in self.records.values()):
                raise NotesError("a meeting-note session is already active")
            note_id = uuid.uuid4().hex
            record = {
                "id": note_id, "kind": "meeting", "title": title,
                "segments": [], "source": self._text(source, 40) or "user",
                "status": "active", "state": "recording",
                "created_at": now, "updated_at": now,
            }
            self.records[note_id] = record
            self._persist_locked()
            return self._copy_locked(record)

    def _meeting_locked(self, note_id):
        record = self.records.get(str(note_id or ""))
        if not record or record.get("kind") != "meeting" or \
                record.get("status") != "active":
            raise NotesError("active meeting session not found")
        return record

    def transition_meeting(self, note_id, action, now=None):
        action = str(action or "").lower()
        if action not in {"pause", "resume", "stop"}:
            raise NotesError("meeting action must be pause, resume, or stop")
        now = self._now(now)
        with self.lock:
            record = self._meeting_locked(note_id)
            state = record.get("state")
            if action == "pause":
                if state != "recording":
                    raise NotesError("meeting is not recording")
                record["state"] = "paused"
            elif action == "resume":
                if state != "paused":
                    raise NotesError("meeting is not paused")
                record["state"] = "recording"
            else:
                if state not in ("recording", "paused"):
                    raise NotesError("meeting is already stopped")
                record["state"] = "stopped"
            record["updated_at"] = now
            self._persist_locked()
            return self._copy_locked(record)

    def append_segment(self, note_id, text, source="speech", observed_at=None,
                       now=None):
        text = self._text(text, MAX_SEGMENT_CHARS)
        if not text:
            return None
        now = self._now(now)
        observed_at = now if observed_at is None else float(observed_at)
        with self.lock:
            record = self._meeting_locked(note_id)
            if record.get("state") != "recording":
                return None
            segments = record.setdefault("segments", [])
            if len(segments) >= MAX_SEGMENTS:
                raise NotesError("meeting segment capacity is full")
            if len(self._content(record)) + len(text) > MAX_NOTE_CHARS:
                raise NotesError("meeting note capacity is full")
            segments.append({"text": text, "source": self._text(source, 40) or "speech",
                             "observed_at": observed_at, "recorded_at": now})
            record["updated_at"] = now
            # Persist every segment: a power loss can lose at most the current
            # broker callback, not an entire meeting.
            self._persist_locked()
            return self._copy_locked(record)

    # ---------- query / lifecycle ----------

    def list_notes(self, query="", limit=25, include_deleted=False):
        query = self._text(query, 200).lower()
        try:
            limit = max(1, min(100, int(limit)))
        except (TypeError, ValueError):
            limit = 25
        with self.lock:
            rows = []
            for record in self.records.values():
                if not include_deleted and record.get("status") != "active":
                    continue
                haystack = " ".join((record.get("title", ""),
                                     self._content(record))).lower()
                if query and query not in haystack:
                    continue
                rows.append(self._summary_locked(record))
            rows.sort(key=lambda row: row["updated_at"], reverse=True)
            return rows[:limit]

    def get(self, note_id, include_deleted=True):
        with self.lock:
            record = self.records.get(str(note_id or ""))
            if not record or (not include_deleted and record.get("status") != "active"):
                return None
            return self._copy_locked(record)

    def delete(self, note_id, now=None):
        now = self._now(now)
        with self.lock:
            record = self.records.get(str(note_id or ""))
            if not record or record.get("status") != "active":
                raise NotesError("note not found")
            record["status"] = "deleted"
            record["deleted_at"] = now
            record["updated_at"] = now
            self._persist_locked()
            return self._copy_locked(record)

    def export_text(self, note_id):
        record = self.get(note_id, include_deleted=False)
        if not record:
            raise NotesError("note not found")
        title = record.get("title") or ("Meeting notes" if record["kind"] == "meeting"
                                         else "Note")
        lines = [title, "=" * len(title)]
        if record["kind"] == "note":
            lines.append(record.get("text", ""))
        else:
            for segment in record.get("segments", []):
                stamp = time.strftime("%Y-%m-%d %H:%M:%S",
                                      time.localtime(segment.get("observed_at", 0)))
                lines.append(f"[{stamp}] {segment.get('text', '')}")
        return "\n".join(lines)[:MAX_NOTE_CHARS]
