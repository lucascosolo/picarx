"""Client-side helpers for the single Layer B camera controller.

Camera consumers never import Picamera2.  They publish a short-lived
subscription request and consume the latest JPEG published by
``camera_controller.py``.  The short lease makes a crashed consumer harmless:
the controller drops its subscription automatically after the TTL.
"""
import base64
import threading
import time


CAMERA_SUBSCRIBE_TOPIC = "picarx/camera/subscribe"
CAMERA_FRAME_TOPIC = "picarx/camera/frame"
CAMERA_STATUS_TOPIC = "picarx/camera/status"
DEFAULT_TTL_SEC = 2.0
REFRESH_INTERVAL_SEC = 0.75


class LatestCameraFrame:
    """One-slot payload buffer; slow inference never queues stale frames."""

    def __init__(self):
        self._lock = threading.Lock()
        self._item = None
        self._sequence = 0

    def put(self, payload):
        with self._lock:
            self._sequence += 1
            self._item = (self._sequence, payload, time.monotonic())

    def take(self):
        with self._lock:
            item, self._item = self._item, None
        return item

    def clear(self):
        with self._lock:
            self._item = None


class CameraSubscription:
    """Maintain one expiring camera stream subscription for a module."""

    def __init__(self, bus, subscriber, fps, on_frame=None, ttl=DEFAULT_TTL_SEC):
        self.bus = bus
        self.subscriber = str(subscriber)
        self.fps = max(0.1, min(30.0, float(fps)))
        self.ttl = max(1.0, float(ttl))
        self.on_frame = on_frame
        self._last_request_at = 0.0
        self._active = False
        self.bus.subscribe(CAMERA_FRAME_TOPIC, self._on_frame)

    def _on_frame(self, payload):
        if self.on_frame is not None:
            self.on_frame(payload)

    @property
    def active(self):
        return self._active

    def ensure(self, now=None):
        now = time.monotonic() if now is None else float(now)
        if not self._active or now - self._last_request_at >= REFRESH_INTERVAL_SEC:
            self.bus.publish(CAMERA_SUBSCRIBE_TOPIC, {
                "subscriber": self.subscriber,
                "enabled": True,
                "fps": self.fps,
                "ttl": self.ttl,
                "ts": time.time(),
            })
            self._last_request_at = now
            self._active = True

    def release(self):
        if self._active:
            self.bus.publish(CAMERA_SUBSCRIBE_TOPIC, {
                "subscriber": self.subscriber,
                "enabled": False,
                "ts": time.time(),
            })
        self._active = False
        self._last_request_at = 0.0


def decode_camera_frame(payload):
    """Decode a controller JPEG payload into an RGB numpy array."""
    encoded = payload.get("jpeg") if isinstance(payload, dict) else None
    if not encoded:
        return None
    try:
        import cv2
        import numpy as np
        raw = base64.b64decode(encoded, validate=True)
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return None
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except (ValueError, TypeError, OSError):
        return None
