#!/usr/bin/env python3
"""Bounded local media capture and playback coordinator.

Video frames arrive through the central camera controller; this module never
opens Picamera2. Audio capture is delegated to audio_nodes, which already owns
the microphone stream. Both paths share ClipStore, require explicit recording
confirmation, and hold the LOCAL_CAPTURE resource lease while active.
"""
import base64
import os
import signal
import subprocess
import sys
import threading
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
from broker_client import Bus
from camera_client import CameraSubscription
from clip_store import ClipError, ClipStore
import robot_config


CONTROL_TOPIC = "picarx/tools/clip"
RESULT_TOPIC = "picarx/tools/clip/result"
STATE_TOPIC = "picarx/tools/clip/state"
AUDIO_CONTROL_TOPIC = "picarx/tools/clip/audio"
AUDIO_RESULT_TOPIC = "picarx/tools/clip/audio/result"
ROBOT_STATE_TOPIC = "picarx/state/current"
CLAIM_TOPIC = "picarx/state/claim"
RELEASE_TOPIC = "picarx/state/release"
OWNER = "clip_daemon"
CLAIM_TTL_SEC = 3.0
VIDEO_FPS = 8.0
CAPTURE_POLL_SEC = 0.1


class ClipDaemon:
    def __init__(self, store=None, camera=None, process_factory=None,
                 clock=None):
        self.bus = Bus()
        self.store = store or ClipStore(robot_config.data_path("clips"))
        self.clock = clock or time.time
        self.camera = camera or CameraSubscription(
            self.bus, OWNER, VIDEO_FPS, on_frame=self.on_frame, ttl=2.0)
        self.process_factory = process_factory or subprocess.Popen
        self.lock = threading.RLock()
        self._video = None
        self._audio = None
        self._playback = None
        self._robot_state = {}

    def _result(self, payload, command, ok=True, result=None, error=None):
        response = {"ok": bool(ok), "command": command,
                    "request_id": payload.get("request_id"), "ts": self.clock()}
        if result is not None:
            response["result"] = result
        if error:
            response["error"] = str(error)[:400]
        self.bus.publish(RESULT_TOPIC, response)
        return response

    def _claim(self):
        self.bus.publish(CLAIM_TOPIC, {
            "owner": OWNER, "state": "LOCAL_CAPTURE", "ttl": CLAIM_TTL_SEC,
            "reason": "local media capture or playback", "ts": self.clock()})

    def _release(self):
        try:
            self.camera.release()
        except Exception:
            pass
        self.bus.publish(RELEASE_TOPIC, {"owner": OWNER, "ts": self.clock()})

    def _busy(self):
        return bool(self._video or self._audio or self._playback)

    def _resource_available(self):
        state = str((self._robot_state or {}).get("state") or "IDLE")
        owner = str((self._robot_state or {}).get("owner") or "")
        if state in {"RC", "SAFETY_STOP", "SPEAKING"}:
            return False
        return state != "LOCAL_CAPTURE" or owner == OWNER

    def on_robot_state(self, payload):
        if not isinstance(payload, dict):
            return
        with self.lock:
            self._robot_state = dict(payload)
            interrupted = ((self._video or self._audio) and
                           not self._resource_available())
        if interrupted:
            self.on_control({"command": "stop", "reason": "resource preempted"})

    def on_frame(self, payload):
        if not isinstance(payload, dict):
            return
        encoded = payload.get("jpeg")
        if not encoded:
            return
        try:
            frame = base64.b64decode(encoded, validate=True)
        except (ValueError, TypeError):
            return
        with self.lock:
            capture = self._video
            if capture is None:
                return
            if sum(len(item) for item in capture["frames"]) + len(frame) > \
                    self.store.max_clip_bytes:
                request = dict(capture["payload"])
                self._video = None
            else:
                capture["frames"].append(frame)
                return
        self._release()
        self.store.abort(capture["reservation"])
        self._result(request, "capture", ok=False,
                     error="video clip exceeds the per-clip size limit")

    def _finish_video(self, error=None):
        with self.lock:
            capture = self._video
            self._video = None
        if capture is None:
            return self._result({}, "stop", ok=False, error="no active capture")
        self._release()
        try:
            if error:
                self.store.abort(capture["reservation"])
                return self._result(capture["payload"], "capture", ok=False,
                                     error=error)
            frames = capture["frames"]
            if not frames:
                self.store.abort(capture["reservation"])
                return self._result(capture["payload"], "capture", ok=False,
                                     error="camera produced no frames")
            with open(capture["reservation"]["temporary_path"], "ab") as stream:
                for frame in frames:
                    stream.write(frame)
                stream.flush()
                os.fsync(stream.fileno())
            duration = max(0.0, self.clock() - capture["started_at"])
            result = self.store.finalize(capture["reservation"], duration)
            return self._result(capture["payload"], "capture", result=result)
        except (OSError, ClipError) as exc:
            self.store.abort(capture["reservation"])
            return self._result(capture["payload"], "capture", ok=False, error=exc)

    def _video_loop(self, request_id):
        while True:
            with self.lock:
                capture = self._video
                if capture is None or capture["payload"].get("request_id") != request_id:
                    return
                expired = self.clock() - capture["started_at"] >= capture["duration_sec"]
            if expired:
                self._finish_video()
                return
            self._claim()
            self.camera.ensure()
            time.sleep(CAPTURE_POLL_SEC)

    def _audio_watchdog(self, request_id, duration):
        deadline = time.monotonic() + duration + 5.0
        while time.monotonic() < deadline:
            with self.lock:
                if not self._audio or self._audio.get("request_id") != request_id:
                    return
            time.sleep(CAPTURE_POLL_SEC)
        with self.lock:
            active = bool(self._audio and self._audio.get("request_id") == request_id)
        if active:
            self.bus.publish(AUDIO_CONTROL_TOPIC, {
                "command": "stop", "request_id": request_id,
                "reason": "audio capture watchdog timeout"})
            with self.lock:
                if self._audio and self._audio.get("request_id") == request_id:
                    self._audio = None
            self._release()
            self._result({"request_id": request_id}, "capture", ok=False,
                         error="audio capture timed out")

    def on_audio_result(self, payload):
        if not isinstance(payload, dict):
            return
        request_id = payload.get("request_id")
        with self.lock:
            active = self._audio and self._audio.get("request_id") == request_id
            pending = bool((payload.get("result") or {}).get("pending"))
            if active and not pending:
                self._audio = None
        if not active or pending:
            return
        self._release()
        self.bus.publish(STATE_TOPIC, dict(payload, event="audio_complete"))
        self.bus.publish(RESULT_TOPIC, dict(payload))

    def _capture(self, payload):
        if payload.get("confirmed") is not True:
            return self._result(payload, "capture", ok=False,
                                error="explicit recording confirmation is required")
        if not self._resource_available():
            return self._result(payload, "capture", ok=False,
                                error="media resources are currently owned by a higher-priority state")
        if self._busy():
            return self._result(payload, "capture", ok=False,
                                error="another clip operation is active")
        kind = str(payload.get("kind") or "").lower()
        try:
            duration = float(payload.get("duration_sec", 5.0))
            reservation = self.store.begin(kind, duration)
        except (TypeError, ValueError, ClipError) as exc:
            return self._result(payload, "capture", ok=False, error=exc)
        self._claim()
        if kind == "audio":
            with self.lock:
                self._audio = {"request_id": payload.get("request_id"),
                               "reservation_id": reservation["id"],
                               "duration_sec": reservation["duration_sec"]}
            # AudioNode owns the live microphone and must create its own
            # reservation; discard this coordinator reservation immediately.
            self.store.abort(reservation)
            self.bus.publish(AUDIO_CONTROL_TOPIC, dict(payload,
                                                        command="capture"))
            threading.Thread(target=self._audio_watchdog,
                             args=(payload.get("request_id"), duration),
                             daemon=True, name="clip-audio-watchdog").start()
            return self._result(payload, "capture", result={
                "kind": "audio", "pending": True,
                "duration_sec": reservation["duration_sec"]})
        if kind != "video":
            self._release()
            self.store.abort(reservation)
            return self._result(payload, "capture", ok=False,
                                error="clip kind must be audio or video")
        with self.lock:
            self._video = {"payload": dict(payload), "reservation": reservation,
                           "frames": [], "started_at": self.clock(),
                           "duration_sec": reservation["duration_sec"]}
        self.camera.ensure()
        threading.Thread(target=self._video_loop,
                         args=(payload.get("request_id"),), daemon=True,
                         name="clip-video-capture").start()
        return self._result(payload, "capture", result={
            "id": reservation["id"], "kind": "video", "pending": True,
            "duration_sec": reservation["duration_sec"]})

    def _stop_playback(self):
        with self.lock:
            playback = self._playback
            self._playback = None
        if playback is None:
            return False
        process = playback["process"]
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (AttributeError, OSError, ProcessLookupError):
            try:
                process.terminate()
            except Exception:
                pass
        self._release()
        return True

    def _play_loop(self, payload, process):
        try:
            process.wait()
        except Exception:
            pass
        with self.lock:
            if self._playback and self._playback.get("process") is process:
                self._playback = None
                active = True
            else:
                active = False
        if active:
            self._release()
            self._result(payload, "play", result={"completed": True,
                                                  "id": payload.get("id")})

    def _play(self, payload):
        if self._busy():
            return self._result(payload, "play", ok=False,
                                error="another clip operation is active")
        if not self._resource_available():
            return self._result(payload, "play", ok=False,
                                error="media resources are currently unavailable")
        try:
            row = self.store.get(payload.get("id"))
            path = self.store.path(payload.get("id"))
        except ClipError as exc:
            return self._result(payload, "play", ok=False, error=exc)
        argv = (["aplay", "-D", "plug:robot_speaker", "-q", path]
                if row["kind"] == "audio" else
                ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", path])
        try:
            process = self.process_factory(argv, stdout=subprocess.DEVNULL,
                                           stderr=subprocess.DEVNULL,
                                           start_new_session=True)
        except (OSError, TypeError) as exc:
            return self._result(payload, "play", ok=False, error=exc)
        with self.lock:
            self._playback = {"id": payload.get("id"), "process": process}
        self._claim()
        threading.Thread(target=self._play_loop, args=(dict(payload), process),
                         daemon=True, name="clip-playback").start()
        return self._result(payload, "play", result={"started": True,
                                                      "id": payload.get("id")})

    def on_control(self, payload):
        payload = dict(payload or {})
        payload.setdefault("request_id", uuid.uuid4().hex)
        command = str(payload.get("command") or payload.get("op") or "").lower()
        try:
            if command == "capture":
                return self._capture(payload)
            if command == "list":
                return self._result(payload, command, result={"clips": self.store.list()})
            if command == "status":
                with self.lock:
                    active = {"capture": bool(self._video or self._audio),
                              "playback": bool(self._playback)}
                return self._result(payload, command,
                                    result={"active": active,
                                            "clips": self.store.list()})
            if command == "delete":
                if payload.get("confirmed") is not True:
                    return self._result(payload, command, ok=False,
                                        error="explicit confirmation is required to delete a clip")
                row = self.store.delete(payload.get("id"))
                return self._result(payload, command, result={"deleted": row})
            if command == "play":
                return self._play(payload)
            if command == "stop":
                with self.lock:
                    video = bool(self._video)
                    audio = bool(self._audio)
                    playback = bool(self._playback)
                if video:
                    self._finish_video(error="video capture interrupted")
                if audio:
                    self.bus.publish(AUDIO_CONTROL_TOPIC, dict(payload, command="stop"))
                stopped_playback = self._stop_playback() if playback else False
                return self._result(payload, command,
                                    result={"stopped": bool(video or audio or
                                                            stopped_playback)})
            return self._result(payload, command, ok=False,
                                error="unsupported clip command")
        except (ClipError, OSError, ValueError, TypeError) as exc:
            return self._result(payload, command, ok=False, error=exc)

    def run(self):
        self.bus.subscribe(CONTROL_TOPIC, self.on_control)
        self.bus.subscribe(AUDIO_RESULT_TOPIC, self.on_audio_result)
        self.bus.subscribe(ROBOT_STATE_TOPIC, self.on_robot_state)
        print(f"Clip daemon active on {CONTROL_TOPIC}")
        while True:
            time.sleep(1)


if __name__ == "__main__":
    ClipDaemon().run()
