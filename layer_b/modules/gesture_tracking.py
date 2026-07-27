#!/usr/bin/env python3
"""Low-cost MediaPipe Hands head tracking.

The production loop is deliberately split into a capture thread and a
processing loop.  The capture thread keeps only the newest 320x240 frame, so
slow inference cannot create an ever-growing queue.  The pure controller and
throttle classes are usable off-robot and are kept independent of MediaPipe,
OpenCV, Picamera2, and psutil so the safety behavior can be tested on a CI
machine.

Enable with::

    picarx/gesture/control {"enabled": true}

The module publishes ordinary ``picarx/intent/look`` messages.  It never
publishes drive intents and it relinquishes the head when its lease expires,
when another RobotState wins, or when the camera/model fails.
"""
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus
from camera_lock import CameraBusy, CameraLease

try:
    import robot_config
except Exception:  # pragma: no cover - production import should work
    robot_config = None


PAN_MIN, PAN_MAX = -35.0, 35.0
TILT_MIN, TILT_MAX = -30.0, 30.0
FRAME_SIZE = (320, 240)
CENTER_DEADZONE_PX = 10.0
DEFAULT_SKIP_FACTOR = 3       # process one out of every three captured frames
CPU_HIGH_PERCENT = 90.0
CPU_HIGH_HOLD_SEC = 3.0
THERMAL_LIMIT_C = 80.0
THERMAL_RECOVER_C = 72.0
HAND_LOSS_HOLD_SEC = 1.0
CLAIM_RENEW_INTERVAL_SEC = 0.5
STATE_TOPIC = "picarx/state/current"
CONTROL_TOPIC = "picarx/gesture/control"
STATUS_TOPIC = "picarx/gesture/status"
STATE_CLAIM_TOPIC = "picarx/state/claim"
STATE_RELEASE_TOPIC = "picarx/state/release"
LOOK_TOPIC = "picarx/intent/look"
OWNER = "gesture_tracking"
STATE_NAME = "GESTURE_TRACKING"

# MediaPipe removed the legacy ``mp.solutions`` Python API in 0.10.30.  The
# replacement Tasks API needs a separate .task model asset; keep it beside the
# other ignored, install-time vision models and fetch it lazily on first use.
HAND_MODEL_DIR = robot_config.data_path("models", "mediapipe") \
    if robot_config is not None else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", "mediapipe")
HAND_MODEL_PATH = os.path.join(HAND_MODEL_DIR, "hand_landmarker.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task")


def _clamp(value, low, high):
    return max(low, min(high, float(value)))


class HeadLimits:
    """Single hardware boundary used by every gesture head command."""

    @staticmethod
    def clamp(pan, tilt):
        return round(_clamp(pan, PAN_MIN, PAN_MAX), 2), \
            round(_clamp(tilt, TILT_MIN, TILT_MAX), 2)


class GestureHeadController:
    """Map a hand target to a bounded, dead-zoned pan/tilt correction."""

    def __init__(self, pan=0.0, tilt=0.0, gain=8.0, max_step_deg=4.0):
        self.pan, self.tilt = HeadLimits.clamp(pan, tilt)
        self.gain = max(0.1, float(gain))
        self.max_step_deg = max(0.1, float(max_step_deg))

    def update(self, x, y, frame_width, frame_height):
        """Return a new ``(pan, tilt)`` or the unchanged pose.

        ``x``/``y`` are pixel coordinates. A target to the right/down of the
        image center produces a right/down head correction. The deadzone is
        evaluated in raw pixels before normalization, so it remains exactly
        10 px at the requested 320x240 input size.
        """
        try:
            fw, fh = float(frame_width), float(frame_height)
            x, y = float(x), float(y)
        except (TypeError, ValueError):
            return self.pan, self.tilt
        if fw <= 0 or fh <= 0:
            return self.pan, self.tilt
        dx, dy = x - fw / 2.0, y - fh / 2.0
        pan_step = 0.0 if abs(dx) <= CENTER_DEADZONE_PX else \
            _clamp((dx / (fw / 2.0)) * self.gain,
                   -self.max_step_deg, self.max_step_deg)
        tilt_step = 0.0 if abs(dy) <= CENTER_DEADZONE_PX else \
            _clamp((dy / (fh / 2.0)) * self.gain,
                   -self.max_step_deg, self.max_step_deg)
        self.pan, self.tilt = HeadLimits.clamp(self.pan + pan_step,
                                                self.tilt + tilt_step)
        return self.pan, self.tilt

    def pose(self):
        return HeadLimits.clamp(self.pan, self.tilt)


class LatestFrame:
    """One-slot frame buffer; put() always drops stale work."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._frame_no = 0

    def put(self, frame):
        with self._lock:
            self._frame_no += 1
            self._frame = (self._frame_no, frame, time.monotonic())

    def take(self):
        with self._lock:
            item, self._frame = self._frame, None
        return item

    def clear(self):
        with self._lock:
            self._frame = None


class CpuThermalMonitor:
    """Fail-soft CPU and temperature reader with no required dependency."""

    def __init__(self, cpu_reader=None, temp_reader=None,
                 high_percent=CPU_HIGH_PERCENT, thermal_limit=THERMAL_LIMIT_C,
                 recover_temp=THERMAL_RECOVER_C):
        self.cpu_reader = cpu_reader or self.read_cpu_percent
        self.temp_reader = temp_reader or self.read_temperature_c
        self.high_percent = float(high_percent)
        self.thermal_limit = float(thermal_limit)
        self.recover_temp = float(recover_temp)
        self._prev_cpu = None
        self._prev_at = None

    def read_cpu_percent(self):
        """Read process/system CPU from procfs when available, else None."""
        try:
            with open("/proc/stat") as f:
                fields = f.readline().split()[1:]
            values = [float(x) for x in fields[:8]]
            idle = values[3] + (values[4] if len(values) > 4 else 0.0)
            total = sum(values)
            now = time.monotonic()
            if self._prev_cpu is None:
                self._prev_cpu, self._prev_at = (total, idle), now
                return None
            ptotal, pidle = self._prev_cpu
            self._prev_cpu, self._prev_at = (total, idle), now
            dt_total, dt_idle = total - ptotal, idle - pidle
            return 100.0 * (1.0 - dt_idle / dt_total) if dt_total > 0 else None
        except (OSError, ValueError, IndexError):
            return None

    @staticmethod
    def read_temperature_c():
        for path in ("/sys/class/thermal/thermal_zone0/temp",):
            try:
                with open(path) as stream:
                    value = float(stream.read().strip())
                return value / 1000.0 if value > 200 else value
            except (OSError, ValueError):
                continue
        return None

    def sample(self):
        try:
            cpu = self.cpu_reader()
        except Exception:
            cpu = None
        try:
            temp = self.temp_reader()
        except Exception:
            temp = None
        return {"cpu_percent": cpu, "temperature_c": temp}


class AdaptiveFrameScheduler:
    """Frame skip and thermal policy for bounded inference cost."""

    def __init__(self, monitor=None, skip_factor=DEFAULT_SKIP_FACTOR,
                 high_hold_sec=CPU_HIGH_HOLD_SEC):
        self.monitor = monitor or CpuThermalMonitor()
        self.skip_factor = max(3, int(skip_factor))
        self.base_skip_factor = self.skip_factor
        self.high_hold_sec = float(high_hold_sec)
        self._high_since = None
        self._cool_since = None
        self._frame_no = 0
        self.last_sample = {"cpu_percent": None, "temperature_c": None}
        self.throttled = False
        self.thermal_stop = False

    def observe(self, now=None):
        now = time.monotonic() if now is None else float(now)
        self.last_sample = self.monitor.sample()
        cpu, temp = self.last_sample["cpu_percent"], self.last_sample["temperature_c"]
        # The Pi's temperature sensor is part of the hardware safety gate.
        # If it cannot be read, do not continue running MediaPipe blindly.
        if temp is None:
            self.thermal_stop = True
            self.throttled = True
            self.skip_factor = max(self.skip_factor, 12)
            return self.last_sample
        if temp is not None and temp >= self.monitor.thermal_limit:
            self.thermal_stop = True
            self.throttled = True
            self.skip_factor = max(self.skip_factor, 12)
            return self.last_sample
        if self.thermal_stop and (temp is None or temp > self.monitor.recover_temp):
            return self.last_sample
        if self.thermal_stop:
            self.thermal_stop = False
        if cpu is not None and cpu >= self.monitor.high_percent:
            self._high_since = self._high_since if self._high_since is not None else now
            self._cool_since = None
            if now - self._high_since >= self.high_hold_sec:
                self.throttled = True
                self.skip_factor = min(12, max(3, self.skip_factor + 2))
        elif cpu is not None and cpu < self.monitor.high_percent - 10 and \
                temp is not None and temp < self.monitor.recover_temp:
            self._high_since = None
            self._cool_since = self._cool_since if self._cool_since is not None else now
            if now - self._cool_since >= self.high_hold_sec:
                self.skip_factor = max(self.base_skip_factor, self.skip_factor - 1)
                self.throttled = self.skip_factor > self.base_skip_factor
        return self.last_sample

    def should_process(self, now=None):
        self._frame_no += 1
        # Sampling is intentionally separate from capture; a monitor failure
        # cannot stop the frame thread or make it accumulate work.
        self.observe(now)
        if self.thermal_stop:
            return False
        return self._frame_no % max(3, self.skip_factor) == 0


def hand_target(results, frame_width, frame_height):
    """Extract one stable pixel target from MediaPipe Hands output.

    The index fingertip is preferred for pointing. For an open/partially
    occluded hand, the palm landmarks provide a less jumpy fallback.
    """
    # Legacy Solutions returns ``multi_hand_landmarks``.  Tasks returns
    # ``hand_landmarks`` as a list of plain landmark lists.  Both contain the
    # same normalized x/y coordinates, so keep the controller independent of
    # which MediaPipe generation supplied the result.
    hands = getattr(results, "multi_hand_landmarks", None)
    if hands is None:
        hands = getattr(results, "hand_landmarks", None)
    hands = hands or []
    if not hands:
        return None
    hand = hands[0]
    landmarks = getattr(hand, "landmark", None)
    if landmarks is None and isinstance(hand, (list, tuple)):
        landmarks = hand
    landmarks = landmarks or []
    if len(landmarks) < 18:
        return None
    point = landmarks[8]  # index fingertip
    if getattr(point, "visibility", 1.0) is not None and \
            getattr(point, "visibility", 1.0) < 0.1:
        point = landmarks[9]
    x = _clamp(float(point.x) * frame_width, 0, frame_width)
    y = _clamp(float(point.y) * frame_height, 0, frame_height)
    return round(x, 2), round(y, 2)


class GestureTracker:
    def __init__(self):
        self.bus = Bus()
        self.enabled = False
        self.running = True
        self.state = "IDLE"
        self.frames = LatestFrame()
        self.controller = GestureHeadController()
        self.scheduler = AdaptiveFrameScheduler()
        self._capture_thread = None
        self._camera = None
        self._hands = None
        self._hands_backend = None
        self._mediapipe = None
        self._model_error = None
        self._last_claim_at = 0.0
        self._last_pose = None
        self._last_hand_at = 0.0
        self._last_camera_wait_at = 0.0

    def _claim(self, force=False):
        now = time.monotonic()
        if not force and now - self._last_claim_at < CLAIM_RENEW_INTERVAL_SEC:
            return
        self._last_claim_at = now
        self.bus.publish(STATE_CLAIM_TOPIC, {
            "owner": OWNER, "state": STATE_NAME, "ttl": 1.5,
            "reason": "gesture tracking enabled", "ts": time.time()})

    def _release(self):
        self.bus.publish(STATE_RELEASE_TOPIC, {"owner": OWNER, "ts": time.time()})

    def on_control(self, payload):
        enabled = bool(payload.get("enabled", False))
        was = self.enabled
        self.enabled = enabled
        if enabled:
            if not was:
                self.controller = GestureHeadController()
                self.scheduler = AdaptiveFrameScheduler()
                self._last_pose = None
                self._last_hand_at = 0.0
                self._last_camera_wait_at = 0.0
            self._claim(force=True)
            self.bus.publish(STATUS_TOPIC, {"enabled": True, "state": "starting",
                                             "ts": time.time()})
        else:
            self._release()
            self.frames.clear()
            self.controller = GestureHeadController()
            self._last_pose = None
            self._last_hand_at = 0.0
            self.bus.publish(STATUS_TOPIC, {"enabled": False, "state": "off",
                                             "ts": time.time()})

    def on_state(self, payload):
        self.state = str(payload.get("state") or "IDLE")
        # Speech/RC/safety can temporarily outrank gesture tracking.  Keep
        # our lease alive while preempted so RobotState hands the resource
        # back as soon as the higher-priority owner releases it.  Without
        # this, the initial TTS confirmation can let the 1.5s gesture lease
        # expire, leaving the module stuck in IDLE with the camera held by a
        # different process.
        if self.enabled and self.state != STATE_NAME:
            self._claim()

    def _capture_loop(self):
        lease = None
        camera = None
        try:
            # RobotState closes the normal race, while this non-blocking
            # kernel lock protects the short MQTT handoff window too.
            while self.running and self.enabled and self.state == STATE_NAME:
                try:
                    lease = CameraLease().acquire()
                    break
                except CameraBusy:
                    # Keep the capture worker alive while vision finishes its
                    # handoff instead of exiting and spawning a new worker on
                    # every 10 ms tick. The one-second log/status throttle
                    # also prevents camera_wait from flooding systemd/MQTT.
                    now = time.monotonic()
                    if now - self._last_camera_wait_at >= 1.0:
                        self._last_camera_wait_at = now
                        self.bus.publish(STATUS_TOPIC, {
                            "enabled": self.enabled, "state": "camera_wait",
                            "error": "camera owned by another process",
                            "lock": CameraLease().path, "ts": time.time()})
                    time.sleep(0.2)
            if lease is None:
                return
            from picamera2 import Picamera2
            camera = Picamera2()
            config = camera.create_preview_configuration(
                main={"format": "RGB888", "size": FRAME_SIZE})
            camera.configure(config)
            camera.start()
            self._camera = camera
            time.sleep(0.3)
            while self.running and self.enabled and self.state == STATE_NAME:
                self.frames.put(camera.capture_array())
        except Exception as e:
            self.bus.publish(STATUS_TOPIC, {"enabled": self.enabled, "state": "camera_error",
                                             "error": str(e)[:200], "ts": time.time()})
        finally:
            camera = self._camera
            self._camera = None
            if camera is not None:
                for method in ("stop", "close"):
                    try:
                        getattr(camera, method)()
                    except Exception:
                        pass
            if lease is not None:
                lease.release()

    def _ensure_capture(self):
        if self._capture_thread is None or not self._capture_thread.is_alive():
            self._capture_thread = threading.Thread(target=self._capture_loop,
                                                    name="gesture-capture", daemon=True)
            self._capture_thread.start()

    def _load_hands(self):
        if self._hands is not None:
            return self._hands
        try:
            import mediapipe as mp
            legacy = getattr(getattr(mp, "solutions", None), "hands", None)
            if legacy is not None:
                self._hands = legacy.Hands(
                    static_image_mode=False, max_num_hands=1,
                    model_complexity=0, min_detection_confidence=0.55,
                    min_tracking_confidence=0.55)
                self._hands_backend = "solutions"
            else:
                # MediaPipe >= 0.10.30 no longer ships mp.solutions.  Use the
                # supported Tasks API instead of requiring users to downgrade
                # the package or install a distro-specific replacement.
                if not self._ensure_hand_model():
                    self._hands = False
                    return self._hands
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision
                options = vision.HandLandmarkerOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=HAND_MODEL_PATH),
                    running_mode=vision.RunningMode.IMAGE,
                    num_hands=1,
                    min_hand_detection_confidence=0.55,
                    min_hand_presence_confidence=0.55,
                    min_tracking_confidence=0.55)
                self._hands = vision.HandLandmarker.create_from_options(options)
                self._hands_backend = "tasks"
            self._mediapipe = mp
            self._model_error = None
        except ModuleNotFoundError as e:
            missing = e.name or "an import"
            self._model_error = (
                f"missing Python package '{missing}'; install the gesture "
                "runtime dependencies with the same interpreter running "
                f"the module ({sys.executable})"
            )
            self.bus.publish(STATUS_TOPIC, {"enabled": self.enabled, "state": "model_error",
                                             "error": self._model_error, "exception": str(e),
                                             "ts": time.time()})
            self._hands = False
        except Exception as e:
            self._model_error = f"{type(e).__name__}: {e}"
            self.bus.publish(STATUS_TOPIC, {"enabled": self.enabled, "state": "model_error",
                                             "error": self._model_error[:200], "ts": time.time()})
            self._hands = False
        return self._hands

    def _ensure_hand_model(self):
        """Ensure the MediaPipe Tasks hand model exists, downloading it once."""
        if os.path.isfile(HAND_MODEL_PATH):
            return True
        try:
            from urllib.request import urlopen
            os.makedirs(HAND_MODEL_DIR, exist_ok=True)
            temporary = HAND_MODEL_PATH + ".tmp"
            try:
                with urlopen(HAND_MODEL_URL, timeout=30) as response, \
                        open(temporary, "wb") as output:
                    while True:
                        chunk = response.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                if os.path.getsize(temporary) < 1024:
                    raise ValueError("downloaded file is unexpectedly small")
                os.replace(temporary, HAND_MODEL_PATH)
            finally:
                try:
                    os.unlink(temporary)
                except FileNotFoundError:
                    pass
            return True
        except Exception as e:
            self._model_error = (
                f"MediaPipe hand model unavailable at {HAND_MODEL_PATH}: {e}. "
                f"Download it from {HAND_MODEL_URL}")
            self.bus.publish(STATUS_TOPIC, {
                "enabled": self.enabled, "state": "model_error",
                "error": self._model_error[:500], "ts": time.time()})
            return False

    def process_once(self, now=None):
        now = time.monotonic() if now is None else float(now)
        if not self.enabled or self.state != STATE_NAME:
            self.frames.clear()
            return None
        self._claim()
        item = self.frames.take()
        if item is None:
            return None
        should_process = self.scheduler.should_process(now)
        if self.scheduler.thermal_stop:
            sample = self.scheduler.last_sample
            self.bus.publish(STATUS_TOPIC, {
                "enabled": True, "state": "thermal_stop",
                "throttle": self.scheduler.skip_factor,
                "cpu_percent": sample.get("cpu_percent"),
                "temperature_c": sample.get("temperature_c"), "ts": time.time()})
            return None
        if not should_process:
            return None
        _, frame, captured_at = item
        if now - captured_at > 1.0:
            self._handle_hand_loss(now)
            return None
        hands = self._load_hands()
        if not hands:
            return None
        try:
            # Picamera2 is configured as RGB888 and MediaPipe expects RGB;
            # avoid an unnecessary conversion (and its extra Pi CPU cost).
            if self._hands_backend == "tasks":
                image = self._mediapipe.Image(
                    image_format=self._mediapipe.ImageFormat.SRGB, data=frame)
                results = hands.detect(image)
            else:
                results = hands.process(frame)
        except Exception as e:
            self.bus.publish(STATUS_TOPIC, {"enabled": self.enabled, "state": "process_error",
                                             "error": str(e)[:200], "ts": time.time()})
            return None
        shape = getattr(frame, "shape", (FRAME_SIZE[1], FRAME_SIZE[0]))
        height, width = int(shape[0]), int(shape[1])
        target = hand_target(results, width, height)
        if target is None:
            self._handle_hand_loss(now)
            return None
        pan, tilt = self.controller.update(target[0], target[1], width, height)
        self._last_hand_at = now
        pose = (pan, tilt)
        if pose != self._last_pose:
            self._last_pose = pose
            self.bus.publish(LOOK_TOPIC, {
                "source": OWNER, "action": {"direction": "look", "pan": pan, "tilt": tilt},
                "ts": time.time()})
        self.bus.publish(STATUS_TOPIC, {
            "enabled": True, "state": "tracking", "target": {"x": target[0], "y": target[1]},
            "pan": pan, "tilt": tilt, "throttle": self.scheduler.skip_factor,
            "cpu_percent": self.scheduler.last_sample.get("cpu_percent"),
            "temperature_c": self.scheduler.last_sample.get("temperature_c"),
            "ts": time.time()})
        return {"target": target, "pan": pan, "tilt": tilt}

    def _handle_hand_loss(self, now):
        """Hold briefly, then recenter while continuing cheap reacquisition."""
        if self._last_hand_at and now - self._last_hand_at >= HAND_LOSS_HOLD_SEC:
            if self._last_pose != (0.0, 0.0):
                self.controller = GestureHeadController()
                self._last_pose = (0.0, 0.0)
                self.bus.publish(LOOK_TOPIC, {
                    "source": OWNER,
                    "action": {"direction": "look", "pan": 0.0, "tilt": 0.0},
                    "reason": "hand lost; recentering", "ts": time.time()})
        self.bus.publish(STATUS_TOPIC, {
            "enabled": True, "state": "searching", "hand_lost_sec":
            round(max(0.0, now - self._last_hand_at), 2) if self._last_hand_at else None,
            "pan": self.controller.pan, "tilt": self.controller.tilt,
            "throttle": self.scheduler.skip_factor,
            "cpu_percent": self.scheduler.last_sample.get("cpu_percent"),
            "temperature_c": self.scheduler.last_sample.get("temperature_c"),
            "ts": time.time()})

    def run(self):
        self.bus.subscribe(CONTROL_TOPIC, self.on_control)
        self.bus.subscribe(STATE_TOPIC, self.on_state)
        print("Gesture tracker active (disabled until explicitly enabled)")
        while self.running:
            try:
                # Do not open Picamera2 merely because the UI toggle arrived;
                # wait for RobotState to acknowledge that this process owns
                # the camera/head. This closes the startup race with
                # vision_basic on brokers without retained state messages.
                if self.enabled:
                    # Renew independently of the currently winning state. A
                    # higher-priority owner may temporarily preempt gesture;
                    # keeping this lower-priority lease alive lets gesture
                    # reclaim the camera when that owner releases it.
                    self._claim()
                    if self.state == STATE_NAME:
                        self._ensure_capture()
                        self.process_once()
                    else:
                        self.frames.clear()
                else:
                    self.frames.clear()
                time.sleep(0.01)
            except Exception as e:
                print(f"Gesture tracker: loop failed ({e})")
                time.sleep(0.2)


if __name__ == "__main__":
    GestureTracker().run()
