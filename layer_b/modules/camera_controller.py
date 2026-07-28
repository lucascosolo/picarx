#!/usr/bin/env python3
"""The sole owner of the physical Pi camera.

Consumers request a short-lived subscription on ``picarx/camera/subscribe``.
The controller opens Picamera2 once, captures at the maximum requested FPS,
and broadcasts each captured frame to all subscribers.  No detector or tool
ever opens Picamera2 directly, so hand tracking and object recognition cannot
race the libcamera pipeline or repeatedly tear it down during a handoff.
"""
import base64
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus
from camera_client import (CAMERA_FRAME_TOPIC, CAMERA_STATUS_TOPIC,
                           CAMERA_SUBSCRIBE_TOPIC)
from camera_lock import CameraBusy, CameraLease


CAPTURE_SIZE = (640, 480)
JPEG_QUALITY = 75
MAX_FPS = 30.0
SUBSCRIPTION_TTL_SEC = 2.0


class SubscriptionBook:
    """Pure subscription arbitration, separated for hardware-free tests."""

    def __init__(self, ttl=SUBSCRIPTION_TTL_SEC, clock=None):
        self.ttl = max(1.0, float(ttl))
        self.clock = clock or time.time
        self._subscriptions = {}

    def update(self, payload, now=None):
        now = self.clock() if now is None else float(now)
        if not isinstance(payload, dict):
            return
        subscriber = str(payload.get("subscriber") or "").strip()[:80]
        if not subscriber:
            return
        if not bool(payload.get("enabled", True)):
            self._subscriptions.pop(subscriber, None)
            return
        try:
            fps = max(0.1, min(MAX_FPS, float(payload.get("fps", 1.0))))
        except (TypeError, ValueError):
            fps = 1.0
        try:
            ttl = max(1.0, min(10.0, float(payload.get("ttl", self.ttl))))
        except (TypeError, ValueError):
            ttl = self.ttl
        self._subscriptions[subscriber] = {
            "subscriber": subscriber,
            "fps": fps,
            "expires_at": now + ttl,
        }

    def active(self, now=None):
        now = self.clock() if now is None else float(now)
        for subscriber, request in list(self._subscriptions.items()):
            if request["expires_at"] <= now:
                del self._subscriptions[subscriber]
        return [dict(item) for item in self._subscriptions.values()]

    def max_fps(self, now=None):
        active = self.active(now)
        return max((item["fps"] for item in active), default=0.0)


class CameraController:
    def __init__(self):
        self.bus = Bus()
        self.subscriptions = SubscriptionBook()
        self.camera = None
        self.lease = None
        self._sequence = 0
        self._last_status = None
        self._lock = threading.Lock()
        self.bus.subscribe(CAMERA_SUBSCRIBE_TOPIC, self.on_subscription)

    def on_subscription(self, payload):
        with self._lock:
            self.subscriptions.update(payload)

    def _status(self, active, fps, error=None):
        signature = (tuple(item["subscriber"] for item in active), round(fps, 2), error)
        if signature == self._last_status:
            return
        self._last_status = signature
        payload = {
            "active": bool(active),
            "subscribers": [item["subscriber"] for item in active],
            "fps": round(fps, 2),
            "ts": time.time(),
        }
        if error:
            payload["error"] = str(error)[:200]
        self.bus.publish(CAMERA_STATUS_TOPIC, payload)

    def _open(self):
        from picamera2 import Picamera2

        self.lease = CameraLease().acquire()
        try:
            self.camera = Picamera2()
            config = self.camera.create_preview_configuration(
                main={"format": "RGB888", "size": CAPTURE_SIZE})
            self.camera.configure(config)
            self.camera.start()
            time.sleep(0.3)
        except Exception:
            self._close()
            raise

    def _close(self):
        camera, self.camera = self.camera, None
        if camera is not None:
            for method in ("stop", "close"):
                try:
                    getattr(camera, method)()
                except Exception:
                    pass
        if self.lease is not None:
            self.lease.release()
            self.lease = None

    def _publish_frame(self, frame, now):
        import cv2

        ok, buffer = cv2.imencode(
            ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            return
        self._sequence += 1
        height, width = frame.shape[:2]
        self.bus.publish(CAMERA_FRAME_TOPIC, {
            "jpeg": base64.b64encode(buffer.tobytes()).decode("ascii"),
            "w": int(width), "h": int(height), "seq": self._sequence,
            "ts": now,
        })

    def run(self):
        print("Camera controller active (Picamera2 is owned here only)")
        try:
            while True:
                with self._lock:
                    active = self.subscriptions.active()
                    fps = self.subscriptions.max_fps()
                self._status(active, fps)
                if not active:
                    self._close()
                    time.sleep(0.1)
                    continue
                if self.camera is None:
                    try:
                        self._open()
                    except CameraBusy as exc:
                        self._status(active, fps, exc)
                        time.sleep(0.2)
                        continue
                    except Exception as exc:
                        self._status(active, fps, exc)
                        time.sleep(0.5)
                        continue
                started = time.monotonic()
                try:
                    frame = self.camera.capture_array()
                    self._publish_frame(frame, time.time())
                except Exception as exc:
                    self._close()
                    self._status(active, fps, exc)
                    time.sleep(0.5)
                    continue
                interval = 1.0 / max(0.1, fps)
                remaining = interval - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
        finally:
            self._close()


if __name__ == "__main__":
    CameraController().run()
