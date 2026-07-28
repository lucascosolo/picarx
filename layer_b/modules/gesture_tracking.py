#!/usr/bin/env python3
"""Low-cost MediaPipe Hands head tracking.

The production loop is deliberately split into a camera subscription and a
processing loop.  The camera controller keeps only the newest frame for this
consumer, so
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
import sys
import threading
import time
import importlib.metadata
import platform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus
from camera_client import CameraSubscription, decode_camera_frame

try:
    import robot_config
except Exception:  # pragma: no cover - production import should work
    robot_config = None


PAN_MIN, PAN_MAX = -35.0, 35.0
TILT_MIN, TILT_MAX = -30.0, 30.0
FRAME_SIZE = (320, 240)
CAMERA_FPS = 10.0
CENTER_DEADZONE_PX = 10.0
DEFAULT_SKIP_FACTOR = 3       # process one out of every three captured frames
CPU_HIGH_PERCENT = 90.0
CPU_HIGH_HOLD_SEC = 3.0
THERMAL_LIMIT_C = 80.0
THERMAL_RECOVER_C = 72.0
HAND_LOSS_HOLD_SEC = 1.0
CLAIM_RENEW_INTERVAL_SEC = 0.5
CLAIM_TTL_SEC = 3.0
MODEL_LOAD_TIMEOUT_SEC = max(1.0, float(robot_config.get(
    "gesture_tracking", "model_load_timeout_sec", 45.0,
    env="GESTURE_MODEL_LOAD_TIMEOUT"))) if robot_config is not None else 45.0
MODEL_STATUS_HEARTBEAT_SEC = 1.0
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


def _first_hand_landmarks(results):
    """Return the first hand's landmarks across both MediaPipe APIs."""
    hands = getattr(results, "multi_hand_landmarks", None)
    if hands is None:
        hands = getattr(results, "hand_landmarks", None)
    hands = hands or []
    if not hands:
        return []
    hand = hands[0]
    landmarks = getattr(hand, "landmark", None)
    if landmarks is None and isinstance(hand, (list, tuple)):
        landmarks = hand
    return landmarks or []


def hand_bbox(results, frame_width, frame_height, margin=0.03):
    """Return the first hand's pixel bounding box as ``(x, y, w, h)``."""
    landmarks = _first_hand_landmarks(results)
    if len(landmarks) < 18:
        return None
    try:
        xs = [float(point.x) * float(frame_width) for point in landmarks]
        ys = [float(point.y) * float(frame_height) for point in landmarks]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
        mx, my = (x2 - x1) * float(margin), (y2 - y1) * float(margin)
        x1 = _clamp(x1 - mx, 0, frame_width)
        y1 = _clamp(y1 - my, 0, frame_height)
        x2 = _clamp(x2 + mx, 0, frame_width)
        y2 = _clamp(y2 + my, 0, frame_height)
        return (round(x1, 2), round(y1, 2), round(x2 - x1, 2),
                round(y2 - y1, 2))
    except (AttributeError, TypeError, ValueError):
        return None


def hand_target(results, frame_width, frame_height):
    """Extract one stable pixel target from MediaPipe Hands output.

    The index fingertip is preferred for pointing. For an open/partially
    occluded hand, the palm landmarks provide a less jumpy fallback.
    """
    # Legacy Solutions returns ``multi_hand_landmarks``.  Tasks returns
    # ``hand_landmarks`` as a list of plain landmark lists.  Both contain the
    # same normalized x/y coordinates, so keep the controller independent of
    # which MediaPipe generation supplied the result.
    landmarks = _first_hand_landmarks(results)
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
        self.camera = CameraSubscription(self.bus, OWNER, CAMERA_FPS,
                                         on_frame=self.frames.put)
        self._hands = None
        self._hands_backend = None
        self._mediapipe = None
        self._model_error = None
        self._model_failed = False
        self._model_loading = False
        self._model_load_thread = None
        self._model_load_started_at = 0.0
        self._model_load_last_status_at = 0.0
        self._model_load_phase = "idle"
        self._model_load_frame_age = None
        self._model_load_generation = 0
        self._model_load_cancel = threading.Event()
        self._model_load_lock = threading.Lock()
        self._model_diagnostics = {}
        self._last_claim_at = 0.0
        self._claim_lock = threading.Lock()
        self._claim_stop = threading.Event()
        self._claim_thread = None
        self._last_pose = None
        self._last_hand_at = 0.0

    def _claim(self, force=False):
        with self._claim_lock:
            if not self.enabled or self._model_failed:
                return
            now = time.monotonic()
            if not force and now - self._last_claim_at < CLAIM_RENEW_INTERVAL_SEC:
                return
            self._last_claim_at = now
            self.bus.publish(STATE_CLAIM_TOPIC, {
                "owner": OWNER, "state": STATE_NAME, "ttl": CLAIM_TTL_SEC,
                "reason": "gesture tracking enabled", "ts": time.time()})

    def _claim_loop(self):
        """Renew the lease independently of camera/model processing."""
        while not self._claim_stop.wait(0.1):
            if self.enabled:
                self._claim()

    def _start_claim_loop(self):
        if self._claim_thread is not None and self._claim_thread.is_alive():
            return
        self._claim_stop.clear()
        self._claim_thread = threading.Thread(
            target=self._claim_loop, name="gesture-state-lease", daemon=True)
        self._claim_thread.start()

    def _stop_claim_loop(self):
        self._claim_stop.set()
        thread = self._claim_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=1.0)
        self._claim_thread = None

    def _release(self):
        self.bus.publish(STATE_RELEASE_TOPIC, {"owner": OWNER, "ts": time.time()})

    def _cancel_model_load(self):
        """Invalidate a model worker without blocking on native/network code."""
        with self._model_load_lock:
            self._model_load_generation += 1
            self._model_load_cancel.set()
            self._model_loading = False
            self._model_load_thread = None

    def _model_runtime_diagnostics(self, mediapipe_module=None,
                                   backend=None):
        model_size = None
        model_exists = os.path.isfile(HAND_MODEL_PATH)
        if model_exists:
            try:
                model_size = os.path.getsize(HAND_MODEL_PATH)
            except OSError:
                model_size = None
        version = None
        package_versions = {}
        for distribution in ("mediapipe", "mediapipe-rpi4"):
            try:
                package_versions[distribution] = importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError:
                pass
            except Exception as exc:
                package_versions[distribution] = f"metadata error: {exc}"
        if mediapipe_module is not None:
            version = getattr(mediapipe_module, "__version__", None)
        return {
            "interpreter": sys.executable,
            "python_version": platform.python_version(),
            "mediapipe_version": version,
            "mediapipe_file": (getattr(mediapipe_module, "__file__", None)
                                if mediapipe_module is not None else None),
            "package_versions": package_versions,
            "backend": backend,
            "model_path": HAND_MODEL_PATH,
            "model_exists": model_exists,
            "model_size_bytes": model_size,
        }

    def _publish_model_status(self, state, error=None, exception=None,
                              frame_age=None, cleanup=None, force=False,
                              **fields):
        now = time.time()
        with self._model_load_lock:
            if not force and now - self._model_load_last_status_at < MODEL_STATUS_HEARTBEAT_SEC:
                return
            self._model_load_last_status_at = now
            diagnostics = dict(self._model_diagnostics)
            started = self._model_load_started_at
            phase = self._model_load_phase
        payload = {
            "enabled": self.enabled,
            "state": state,
            "phase": phase,
            "timeout_sec": MODEL_LOAD_TIMEOUT_SEC,
            "elapsed_sec": round(max(0.0, time.monotonic() - started), 3)
            if started else 0.0,
            "frame_age_sec": (round(float(frame_age), 3)
                               if frame_age is not None else self._model_load_frame_age),
            "ts": now,
        }
        payload.update(diagnostics)
        payload.update(fields)
        if error:
            payload["error"] = str(error)[:500]
        if exception:
            payload["exception"] = str(exception)[:500]
        if cleanup is not None:
            payload["cleanup"] = dict(cleanup)
        self.bus.publish(STATUS_TOPIC, payload)

    def _model_load_progress(self, generation, phase, **diagnostics):
        with self._model_load_lock:
            if generation != self._model_load_generation or self._model_load_cancel.is_set():
                return False
            self._model_load_phase = phase
            self._model_diagnostics.update(diagnostics)
        self._publish_model_status("model_loading", force=True)
        return True

    def _create_hands(self, progress=None):
        """Construct a backend without mutating tracker state."""
        progress = progress or (lambda phase, **fields: True)
        progress("import", **self._model_runtime_diagnostics())
        import mediapipe as mp
        progress("imported", **self._model_runtime_diagnostics(mp))
        legacy = getattr(getattr(mp, "solutions", None), "hands", None)
        if legacy is not None:
            progress("constructing", backend="solutions")
            hands = legacy.Hands(
                static_image_mode=False, max_num_hands=1,
                model_complexity=0, min_detection_confidence=0.55,
                min_tracking_confidence=0.55)
            return hands, mp, "solutions", self._model_runtime_diagnostics(mp, "solutions")

        # MediaPipe >= 0.10.30 no longer ships mp.solutions.  Use the
        # supported Tasks API instead of requiring users to downgrade the
        # package or install a distro-specific replacement.
        progress("asset_check", **self._model_runtime_diagnostics(mp, "tasks"))
        if not self._ensure_hand_model(progress=progress):
            raise RuntimeError(self._model_error or "hand model asset unavailable")
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        progress("constructing", backend="tasks",
                 **self._model_runtime_diagnostics(mp, "tasks"))
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55)
        hands = vision.HandLandmarker.create_from_options(options)
        return hands, mp, "tasks", self._model_runtime_diagnostics(mp, "tasks")

    def _close_hands(self, hands):
        if hands and hands is not False:
            try:
                close = getattr(hands, "close", None)
                if close:
                    close()
            except Exception:
                pass

    def _set_model_error(self, error, exception=None, diagnostics=None,
                         timed_out=False):
        self._close_hands(None)
        with self._model_load_lock:
            if diagnostics:
                self._model_diagnostics.update(diagnostics)
            self._model_loading = False
            self._model_failed = True
            self._hands = False
            self._hands_backend = None
            self._model_error = str(error)
            self._model_load_phase = "timeout" if timed_out else "error"
            self._model_load_generation += 1
            self._model_load_cancel.set()
        # A model failure must not retain the camera or RobotState lease while
        # the user investigates it.  Re-enabling (or retry=true) starts fresh.
        self.camera.release()
        self.frames.clear()
        self._release()
        self._publish_model_status(
            "model_error", error=error, exception=exception, force=True,
            timeout=bool(timed_out),
            cleanup={"camera_released": True, "state_released": True})

    def _model_load_worker(self, generation):
        def progress(phase, **fields):
            return self._model_load_progress(generation, phase, **fields)
        try:
            result = self._create_hands(progress=progress)
        except ModuleNotFoundError as exc:
            missing = exc.name or "an import"
            self._finish_model_worker(generation, error=(
                f"missing Python package '{missing}'; install the gesture "
                "runtime dependencies with the same interpreter running "
                f"the module ({sys.executable})"), exception=exc)
            return
        except Exception as exc:
            self._finish_model_worker(generation,
                                      error=f"{type(exc).__name__}: {exc}",
                                      exception=exc)
            return
        self._finish_model_worker(generation, result=result)

    def _finish_model_worker(self, generation, result=None, error=None,
                             exception=None):
        with self._model_load_lock:
            current = (generation == self._model_load_generation and
                       not self._model_load_cancel.is_set())
        if not current:
            if result:
                self._close_hands(result[0])
            return
        if error:
            self._set_model_error(error, exception=exception)
            return
        hands, mediapipe_module, backend, diagnostics = result
        with self._model_load_lock:
            self._hands = hands
            self._mediapipe = mediapipe_module
            self._hands_backend = backend
            self._model_diagnostics = dict(diagnostics)
            self._model_loading = False
            self._model_failed = False
            self._model_load_phase = "ready"
        self._publish_model_status("model_ready", backend=backend, force=True)

    def _start_model_load(self, frame_age=None):
        with self._model_load_lock:
            if self._model_loading or self._model_failed or self._hands is not None:
                return
            self._model_load_generation += 1
            generation = self._model_load_generation
            self._model_load_cancel = threading.Event()
            self._model_loading = True
            self._model_load_started_at = time.monotonic()
            self._model_load_phase = "queued"
            self._model_load_frame_age = frame_age
            self._model_diagnostics = self._model_runtime_diagnostics()
            self._model_load_last_status_at = 0.0
            thread = threading.Thread(target=self._model_load_worker,
                                      args=(generation,),
                                      name="gesture-model-loader", daemon=True)
            self._model_load_thread = thread
        self._publish_model_status("model_loading", frame_age=frame_age, force=True)
        thread.start()

    def _check_model_load_timeout(self):
        with self._model_load_lock:
            if not self._model_loading:
                return
            elapsed = time.monotonic() - self._model_load_started_at
            if elapsed < MODEL_LOAD_TIMEOUT_SEC:
                timed_out = False
            else:
                timed_out = True
                self._model_loading = False
                self._model_load_generation += 1
                self._model_load_cancel.set()
        if not timed_out:
            self._publish_model_status("model_loading")
            return
        self._set_model_error(
            f"MediaPipe model initialization exceeded {MODEL_LOAD_TIMEOUT_SEC:.1f}s",
            timed_out=True)

    def on_control(self, payload):
        enabled = bool(payload.get("enabled", False))
        was = self.enabled
        self.enabled = enabled
        if enabled:
            if not was or bool(payload.get("retry")):
                self._cancel_model_load()
                self.controller = GestureHeadController()
                self.scheduler = AdaptiveFrameScheduler()
                self._last_pose = None
                self._last_hand_at = 0.0
                self._hands = None
                self._hands_backend = None
                self._mediapipe = None
                self._model_error = None
                self._model_failed = False
            self._claim(force=True)
            self.bus.publish(STATUS_TOPIC, {"enabled": True, "state": "starting",
                                             "ts": time.time()})
        else:
            self._cancel_model_load()
            self._model_failed = False
            self._release()
            self.camera.release()
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
        # this, the initial TTS confirmation can let the short gesture lease
        # expire, leaving the module stuck in IDLE with the camera held by a
        # different process.
        if self.enabled and not self._model_failed and self.state != STATE_NAME:
            self._claim()

    def _load_hands(self):
        """Synchronous compatibility helper used by diagnostics/tests.

        Production processing uses ``_start_model_load`` so imports, asset
        downloads, and native model construction cannot block the control loop.
        """
        if self._hands is not None:
            return self._hands
        try:
            self._model_load_started_at = time.monotonic()
            hands, mediapipe_module, backend, diagnostics = self._create_hands()
            self._hands = hands
            self._mediapipe = mediapipe_module
            self._hands_backend = backend
            self._model_diagnostics = diagnostics
            self._model_error = None
            self._model_failed = False
            self._model_load_phase = "ready"
            self._publish_model_status("model_ready", backend=backend, force=True)
        except ModuleNotFoundError as e:
            missing = e.name or "an import"
            self._set_model_error(
                error=(
                f"missing Python package '{missing}'; install the gesture "
                "runtime dependencies with the same interpreter running "
                f"the module ({sys.executable})"), exception=e)
        except Exception as e:
            self._set_model_error(f"{type(e).__name__}: {e}", exception=e)
        return self._hands

    def _ensure_hand_model(self, progress=None):
        """Ensure the MediaPipe Tasks hand model exists, downloading it once."""
        if os.path.isfile(HAND_MODEL_PATH):
            try:
                size = os.path.getsize(HAND_MODEL_PATH)
            except OSError:
                size = 0
            if size >= 1024:
                if progress:
                    progress("asset_ready", **self._model_runtime_diagnostics(
                        backend="tasks"))
                return True
        try:
            from urllib.request import urlopen
            os.makedirs(HAND_MODEL_DIR, exist_ok=True)
            temporary = HAND_MODEL_PATH + ".tmp"
            if progress:
                progress("asset_downloading", **self._model_runtime_diagnostics(
                    backend="tasks"))
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
            raise RuntimeError(
                f"MediaPipe hand model unavailable at {HAND_MODEL_PATH}: {e}. "
                f"Download it from {HAND_MODEL_URL}") from e

    def process_once(self, now=None):
        now = time.monotonic() if now is None else float(now)
        self._check_model_load_timeout()
        if not self.enabled or self.state != STATE_NAME:
            self.frames.clear()
            return None
        if self._model_failed:
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
        _, payload, captured_at = item
        frame = decode_camera_frame(payload) if isinstance(payload, dict) else payload
        if frame is None:
            self._handle_hand_loss(now)
            return None
        if now - captured_at > 1.0:
            self._handle_hand_loss(now)
            return None
        if self._hands is None:
            self._start_model_load(frame_age=max(0.0, now - captured_at))
            self._publish_model_status(
                "model_loading", frame_age=max(0.0, now - captured_at))
            return None
        hands = self._hands
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
        bbox = hand_bbox(results, width, height)
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
            "bbox": {"x": bbox[0], "y": bbox[1], "w": bbox[2], "h": bbox[3]}
            if bbox else None,
            "frame_width": width, "frame_height": height,
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
        self._start_claim_loop()
        print("Gesture tracker active (disabled until explicitly enabled)")
        try:
            while self.running:
                try:
                    self._check_model_load_timeout()
                    # Do not open Picamera2 merely because the UI toggle
                    # arrived; wait for RobotState to acknowledge that this
                    # process owns the camera/head. This closes the startup
                    # race with vision_basic on brokers without retained
                    # state messages.
                    if self.enabled and not self._model_failed:
                        if self.state == STATE_NAME:
                            self.camera.ensure()
                            self.process_once()
                        else:
                            self.camera.release()
                            self.frames.clear()
                    else:
                        self.camera.release()
                        self.frames.clear()
                    time.sleep(0.01)
                except Exception as e:
                    print(f"Gesture tracker: loop failed ({e})")
                    time.sleep(0.2)
        finally:
            self._stop_claim_loop()
            self._cancel_model_load()
            self.camera.release()


if __name__ == "__main__":
    GestureTracker().run()
