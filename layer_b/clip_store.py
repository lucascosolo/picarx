"""Atomic, bounded local storage for short robot media clips.

Capture and playback owners use this module rather than accepting filesystem
paths from a model or web request. IDs are generated locally, media stays
under one configured data directory, incomplete captures are disposable, and
metadata never contains media contents.
"""
import json
import math
import os
import re
import threading
import time
import uuid


MAX_CLIP_DURATION_SEC = 15.0
MAX_CLIP_BYTES = 32 * 1024 * 1024
MAX_CLIPS = 30
MAX_TOTAL_BYTES = 256 * 1024 * 1024
_CLIP_ID = re.compile(r"^[0-9a-f]{32}$")
_EXTENSIONS = {"audio": ".wav", "video": ".mjpeg"}


class ClipError(RuntimeError):
    """A user-visible, bounded clip-store failure."""


class ClipStore:
    """Own clip files and metadata under a generated-ID-only directory."""

    def __init__(self, root, max_duration=MAX_CLIP_DURATION_SEC,
                 max_clip_bytes=MAX_CLIP_BYTES, max_clips=MAX_CLIPS,
                 max_total_bytes=MAX_TOTAL_BYTES, clock=None):
        root = os.path.abspath(os.path.expanduser(str(root)))
        if not root:
            raise ClipError("clip storage directory is required")
        self.root = root
        self.max_duration = max(0.1, float(max_duration))
        self.max_clip_bytes = max(1, int(max_clip_bytes))
        self.max_clips = max(1, int(max_clips))
        self.max_total_bytes = max(self.max_clip_bytes, int(max_total_bytes))
        self.clock = clock or time.time
        self._lock = threading.RLock()
        self._reservations = {}
        os.makedirs(self.root, exist_ok=True)
        self.cleanup_incomplete()

    @staticmethod
    def _kind(kind):
        value = str(kind or "").strip().lower()
        if value not in _EXTENSIONS:
            raise ClipError("clip kind must be audio or video")
        return value

    @staticmethod
    def _id(clip_id):
        value = str(clip_id or "").strip().lower()
        if not _CLIP_ID.fullmatch(value):
            raise ClipError("invalid clip id")
        return value

    def _media_path(self, clip_id, kind):
        return os.path.join(self.root, clip_id + _EXTENSIONS[kind])

    def _metadata_path(self, clip_id):
        return os.path.join(self.root, clip_id + ".json")

    def _existing(self):
        rows = []
        for name in os.listdir(self.root):
            if not name.endswith(".json"):
                continue
            clip_id = name[:-5]
            if not _CLIP_ID.fullmatch(clip_id):
                continue
            path = self._metadata_path(clip_id)
            try:
                with open(path, encoding="utf-8") as stream:
                    row = json.load(stream)
                kind = self._kind(row.get("kind"))
                media = self._media_path(clip_id, kind)
                if os.path.islink(media) or not os.path.isfile(media):
                    continue
                size = os.path.getsize(media)
                if size > self.max_clip_bytes:
                    continue
                row = {
                    "id": clip_id,
                    "kind": kind,
                    "bytes": size,
                    "duration_sec": round(max(0.0, float(
                        row.get("duration_sec", 0.0))), 3),
                    "created_at": float(row.get("created_at", 0.0)),
                }
                rows.append(row)
            except (OSError, TypeError, ValueError, ClipError):
                continue
        rows.sort(key=lambda row: (row["created_at"], row["id"]), reverse=True)
        return rows

    def list(self):
        with self._lock:
            return [dict(row) for row in self._existing()]

    def begin(self, kind, duration_sec):
        kind = self._kind(kind)
        try:
            duration = float(duration_sec)
        except (TypeError, ValueError):
            raise ClipError("clip duration must be a number")
        if not math.isfinite(duration) or duration <= 0 or duration > self.max_duration:
            raise ClipError(f"clip duration must be between 0 and {self.max_duration:g} seconds")
        with self._lock:
            rows = self._existing()
            if len(rows) >= self.max_clips:
                raise ClipError("clip storage has reached its count limit")
            if sum(row["bytes"] for row in rows) >= self.max_total_bytes:
                raise ClipError("clip storage has reached its size limit")
            for _ in range(3):
                clip_id = uuid.uuid4().hex
                temporary = os.path.join(self.root, clip_id + ".part")
                if os.path.exists(temporary):
                    continue
                try:
                    with open(temporary, "xb"):
                        pass
                except OSError as exc:
                    raise ClipError(f"could not reserve clip storage: {exc}") from exc
                reservation = {
                    "id": clip_id,
                    "kind": kind,
                    "duration_sec": duration,
                    "started_at": self.clock(),
                    "temporary_path": temporary,
                    "media_path": self._media_path(clip_id, kind),
                }
                self._reservations[clip_id] = reservation
                return dict(reservation)
            raise ClipError("could not allocate a unique clip id")

    def finalize(self, reservation, duration_sec=None):
        if not isinstance(reservation, dict):
            raise ClipError("invalid clip reservation")
        clip_id = self._id(reservation.get("id"))
        with self._lock:
            active = self._reservations.get(clip_id)
            if active is None or active.get("temporary_path") != reservation.get("temporary_path"):
                raise ClipError("clip reservation is no longer active")
            temporary = active["temporary_path"]
            try:
                size = os.path.getsize(temporary)
            except OSError as exc:
                self._reservations.pop(clip_id, None)
                raise ClipError(f"clip capture is unavailable: {exc}") from exc
            if size <= 0:
                self._reservations.pop(clip_id, None)
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
                raise ClipError("clip capture produced no media")
            if size > self.max_clip_bytes:
                self.abort(active)
                raise ClipError("clip exceeds the per-clip size limit")
            total = sum(row["bytes"] for row in self._existing())
            if total + size > self.max_total_bytes:
                self.abort(active)
                raise ClipError("clip would exceed the total storage limit")
            try:
                os.replace(temporary, active["media_path"])
                try:
                    duration = float(duration_sec)
                except (TypeError, ValueError):
                    duration = float(active["duration_sec"])
                if not math.isfinite(duration):
                    duration = float(active["duration_sec"])
                duration = max(0.0, min(self.max_duration, duration))
                metadata = {
                    "id": clip_id,
                    "kind": active["kind"],
                    "duration_sec": round(duration, 3),
                    "created_at": self.clock(),
                }
                self._write_metadata(clip_id, metadata)
            except OSError as exc:
                try:
                    os.unlink(active["media_path"])
                except FileNotFoundError:
                    pass
                raise ClipError(f"could not finalize clip: {exc}") from exc
            finally:
                self._reservations.pop(clip_id, None)
            metadata["bytes"] = size
            return metadata

    def _write_metadata(self, clip_id, metadata):
        path = self._metadata_path(clip_id)
        temporary = path + ".tmp"
        try:
            with open(temporary, "w", encoding="utf-8") as stream:
                json.dump(metadata, stream, separators=(",", ":"))
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def abort(self, reservation):
        if not isinstance(reservation, dict):
            return
        clip_id = str(reservation.get("id") or "")
        with self._lock:
            active = self._reservations.pop(clip_id, None)
            path = (active or reservation).get("temporary_path")
            if path:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass

    def cleanup_incomplete(self):
        with self._lock:
            try:
                names = os.listdir(self.root)
            except OSError:
                return
            for name in names:
                if name.endswith(".part") or name.endswith(".json.tmp"):
                    try:
                        os.unlink(os.path.join(self.root, name))
                    except OSError:
                        pass

    def get(self, clip_id):
        clip_id = self._id(clip_id)
        return next((row for row in self.list() if row["id"] == clip_id), None)

    def path(self, clip_id):
        row = self.get(clip_id)
        if row is None:
            raise ClipError("clip not found")
        path = self._media_path(row["id"], row["kind"])
        if os.path.islink(path) or not os.path.isfile(path):
            raise ClipError("clip media is unavailable")
        return path

    def delete(self, clip_id):
        clip_id = self._id(clip_id)
        with self._lock:
            row = self.get(clip_id)
            if row is None:
                raise ClipError("clip not found")
            media = self._media_path(clip_id, row["kind"])
            metadata = self._metadata_path(clip_id)
            try:
                os.unlink(media)
                os.unlink(metadata)
            except OSError as exc:
                raise ClipError(f"could not delete clip: {exc}") from exc
            return row
