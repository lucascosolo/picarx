#!/usr/bin/env python3
# layer_b/modules/imu.py
"""
IMU sensor (Layer B) - reads an MPU-6050 (accelerometer + gyroscope +
temperature) over I2C and publishes the robot's own motion and orientation
on picarx/sensors/imu, so the rest of the system can sense how the body is
actually moving, not just what the wheels were told to do.

MOUNTING NOTE - this matters. The MPU-6050 rides on TOP OF THE PAN/TILT HEAD,
above the camera, not on the chassis. So its readings are in the HEAD frame:
when the head tilts to look down, the measured gravity vector rotates even
though the chassis never moved. The head can also be mounted a touch off
level. Two things handle this:

  * Calibration at startup captures the RESTING gravity vector and gyro bias
    while the robot sits still (head assumed centred, chassis level). That
    resting vector becomes "down" - so an imperfect mount is zeroed out, and
    every orientation reading is measured RELATIVE to rest, not to a perfect
    axis. Gyro bias is subtracted so a still robot reads ~0 deg/s.
  * Head pose is tracked from picarx/intent/look (the only place the head is
    commanded; the angle is not otherwise published). The commanded tilt is
    subtracted from the measured tilt-from-rest to estimate CHASSIS tilt, so a
    look-down sweep isn't mistaken for driving up a ramp.

What we publish is chosen to be ROBUST to the unknown exact axis orientation
of the chip: magnitudes and angles, not raw per-axis body angles.

  moving          - |rotation rate| or |accel deviation from rest| over a
                    threshold. Frame-independent; good for confirming the body
                    really is (or isn't) moving when commanded to.
  impact          - a sudden accel spike well beyond gravity: a bump/collision.
  rotation_rate   - gyro magnitude (deg/s); is the robot turning/being jostled.
  tilt_from_rest  - angle between the current and resting gravity vectors.
  body_tilt       - tilt_from_rest with the commanded head tilt removed: a
                    heuristic chassis tilt (ramp / being tipped or picked up).

Fail-soft and hardware-isolated: if python-mpu6050 or the chip is absent, the
module logs once and idles (it never crash-loops the orchestrator, and never
touches the drive motors or the safety daemon - it only reads I2C and
publishes). All the decision math lives in pure module-level helpers so it is
unit-tested off-robot with a fake sensor.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus
import robot_config

import math
import time
import threading

IMU_TOPIC = "picarx/sensors/imu"
LOOK_TOPIC = "picarx/intent/look"

# I2C address of the MPU-6050 (0x68 = 104 default; 0x69 if AD0 is pulled high).
I2C_ADDRESS = int(robot_config.get("imu", "i2c_address", 0x68, env="IMU_I2C_ADDRESS"))
IMU_HZ = float(robot_config.get("imu", "hz", 20.0, env="IMU_HZ"))
CALIBRATION_SAMPLES = int(robot_config.get("imu", "calibration_samples", 40,
                                           env="IMU_CALIBRATION_SAMPLES"))
# Derived-signal thresholds (all overridable / on the Config page).
MOVE_GYRO_DPS = float(robot_config.get("imu", "move_gyro_dps", 8.0, env="IMU_MOVE_GYRO_DPS"))
MOVE_ACCEL_MS2 = float(robot_config.get("imu", "move_accel_ms2", 0.6, env="IMU_MOVE_ACCEL_MS2"))
IMPACT_MS2 = float(robot_config.get("imu", "impact_ms2", 6.0, env="IMU_IMPACT_MS2"))
TILT_ALERT_DEG = float(robot_config.get("imu", "tilt_alert_deg", 25.0, env="IMU_TILT_ALERT_DEG"))

STANDARD_GRAVITY = 9.80665
EVENT_COOLDOWN_SEC = 1.5      # min seconds between repeats of the same IMU event


# --------------------------------------------------------------------------
# pure vector helpers + derived-signal math (unit-tested, no hardware)
# --------------------------------------------------------------------------

def _vec(d):
    """(x, y, z) tuple from a {'x','y','z'} reading dict, missing axes -> 0."""
    return (float(d.get("x", 0.0)), float(d.get("y", 0.0)), float(d.get("z", 0.0)))


def magnitude(v):
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def angle_between_deg(a, b):
    """Angle (degrees) between two 3-vectors; 0 if either is degenerate."""
    ma, mb = magnitude(a), magnitude(b)
    if ma == 0.0 or mb == 0.0:
        return 0.0
    cos = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (ma * mb)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def mean_vector(vectors):
    """Component-wise mean of a list of 3-vectors; (0,0,0) if empty."""
    n = len(vectors)
    if not n:
        return (0.0, 0.0, 0.0)
    return (sum(v[0] for v in vectors) / n,
            sum(v[1] for v in vectors) / n,
            sum(v[2] for v in vectors) / n)


def calibrate(accel_samples, gyro_samples):
    """Turn resting samples into a calibration: the resting gravity vector
    (captures mount misalignment), its magnitude, and the gyro bias. Pure."""
    accel_rest = mean_vector(accel_samples)
    gyro_bias = mean_vector(gyro_samples)
    return {"accel_rest": accel_rest,
            "g_mag": magnitude(accel_rest) or STANDARD_GRAVITY,
            "gyro_bias": gyro_bias}


def detect_event(prev, derived):
    """Edge-detect a notable transition worth its own message, comparing the
    previous derived dict to the current one. Returns an event kind string or
    None. A high-rate periodic topic can be sampled between snapshots; a brief
    impact or a pickup should never be missed, so they fire on the RISING edge.
    Pure/unit-testable."""
    prev = prev or {}
    if derived.get("impact") and not prev.get("impact"):
        return "impact"
    if derived.get("tilted") and not prev.get("tilted"):
        return "tilted"          # tipped, on a ramp, or picked up
    return None


def body_tilt_deg(tilt_from_rest, head_tilt_cmd):
    """Estimate CHASSIS tilt by removing the commanded head tilt from the
    measured tilt-from-rest. Heuristic (a pan mixes the axes), clamped to >= 0,
    so a level chassis with the head tilted down doesn't read as a ramp."""
    return max(0.0, tilt_from_rest - abs(head_tilt_cmd))


def compute_derived(accel, gyro_corrected, calib, head_tilt_cmd,
                    move_gyro_dps=MOVE_GYRO_DPS, move_accel_ms2=MOVE_ACCEL_MS2,
                    impact_ms2=IMPACT_MS2, tilt_alert_deg=TILT_ALERT_DEG):
    """The robust, frame-independent derived signals from one reading. `accel`
    is the raw accel vector, `gyro_corrected` the bias-removed gyro vector,
    `calib` from calibrate(), `head_tilt_cmd` the last commanded head tilt (deg).
    Pure - the whole behavioural decision surface, unit-tested off-robot."""
    accel_mag = magnitude(accel)
    rotation_rate = magnitude(gyro_corrected)
    accel_dev = magnitude(_sub(accel, calib["accel_rest"]))
    tilt = angle_between_deg(accel, calib["accel_rest"])
    b_tilt = body_tilt_deg(tilt, head_tilt_cmd)
    return {
        "accel_magnitude_ms2": round(accel_mag, 3),
        "rotation_rate_dps": round(rotation_rate, 2),
        "accel_dev_ms2": round(accel_dev, 3),
        "tilt_from_rest_deg": round(tilt, 1),
        "body_tilt_deg": round(b_tilt, 1),
        "moving": rotation_rate > move_gyro_dps or accel_dev > move_accel_ms2,
        "impact": (accel_mag - calib["g_mag"]) > impact_ms2,
        "tilted": b_tilt > tilt_alert_deg,
    }


# --------------------------------------------------------------------------
# the module
# --------------------------------------------------------------------------

class IMU:
    def __init__(self, sensor=None):
        self.bus = Bus()
        self.lock = threading.Lock()
        self.sensor = sensor          # injected in tests; opened in run() otherwise
        self.calib = None
        self.head_pan = 0.0
        self.head_tilt = 0.0
        self._prev_derived = {}       # for edge-triggered events
        self._event_at = {}           # kind -> last-published ts (throttle)

    # ---------- head pose (from the look channel; not otherwise published) ----

    def on_look(self, payload):
        action = payload.get("action") or {}
        if action.get("direction") != "look":
            return
        with self.lock:
            self.head_pan = float(action.get("pan", self.head_pan) or 0.0)
            self.head_tilt = float(action.get("tilt", self.head_tilt) or 0.0)

    # ---------- sensor access (guarded) ----------

    def _read(self):
        """(accel_vec, gyro_vec, temp_c) or None on any read error. The chip
        occasionally NAKs a read; a dropped sample is skipped, not fatal."""
        try:
            accel = _vec(self.sensor.get_accel_data())
            gyro = _vec(self.sensor.get_gyro_data())
            temp = float(self.sensor.get_temperature())
            return accel, gyro, temp
        except Exception as e:
            print(f"IMU: read failed ({e}); skipping this sample")
            return None

    def calibrate_at_rest(self, samples=CALIBRATION_SAMPLES, delay=None):
        """Sample the still robot to learn resting gravity + gyro bias. Returns
        True on success. Assumes the robot is stationary and level with the head
        centred - i.e. run it at startup, or on demand when parked."""
        delay = delay if delay is not None else 1.0 / max(1.0, IMU_HZ)
        accel_samples, gyro_samples = [], []
        for _ in range(max(1, samples)):
            reading = self._read()
            if reading is not None:
                accel_samples.append(reading[0])
                gyro_samples.append(reading[1])
            time.sleep(delay)
        if not accel_samples:
            return False
        calib = calibrate(accel_samples, gyro_samples)
        with self.lock:
            self.calib = calib
        print(f"IMU: calibrated - resting |g|={calib['g_mag']:.2f} m/s^2, "
              f"gyro bias ({calib['gyro_bias'][0]:.1f},{calib['gyro_bias'][1]:.1f},"
              f"{calib['gyro_bias'][2]:.1f}) deg/s")
        return True

    def on_recalibrate(self, _payload):
        """picarx/sensors/imu/recalibrate: re-zero while parked (e.g. after a
        pickup, or a head re-home). Fail-soft - a bad recal just keeps the old."""
        self.calibrate_at_rest()

    # ---------- one publish cycle (pure-ish given a reading) ----------

    def _publish_reading(self, accel, gyro, temp):
        with self.lock:
            calib = self.calib
            head_pan, head_tilt = self.head_pan, self.head_tilt
        if calib is None:
            return
        gyro_corrected = _sub(gyro, calib["gyro_bias"])
        derived = compute_derived(accel, gyro_corrected, calib, head_tilt)
        now = time.time()
        self.bus.publish(IMU_TOPIC, {
            "ts": now,
            "accel": {"x": round(accel[0], 3), "y": round(accel[1], 3),
                      "z": round(accel[2], 3)},
            "gyro": {"x": round(gyro_corrected[0], 2), "y": round(gyro_corrected[1], 2),
                     "z": round(gyro_corrected[2], 2)},
            "temperature_c": round(temp, 1),
            "head_pose": {"pan": head_pan, "tilt": head_tilt},
            "calibrated": True,
            **derived,
        })
        # A brief impact or a pickup can fall between the 2Hz world_state
        # snapshots - fire it as its own edge-triggered, throttled event too.
        event = detect_event(self._prev_derived, derived)
        if event and now - self._event_at.get(event, 0.0) >= EVENT_COOLDOWN_SEC:
            self._event_at[event] = now
            self.bus.publish("picarx/sensors/imu/event", {
                "kind": event, "ts": now,
                "body_tilt_deg": derived["body_tilt_deg"],
                "accel_magnitude_ms2": derived["accel_magnitude_ms2"]})
            print(f"IMU: {event} (body tilt {derived['body_tilt_deg']:.0f}deg, "
                  f"|accel| {derived['accel_magnitude_ms2']:.1f})")
        self._prev_derived = derived

    def _open_sensor(self):
        """Open the MPU-6050. Returns (ok, reason); reason is a human string
        when ok is False. Guarded so the module runs (idle) off-robot / with no
        chip and never crash-loops."""
        if self.sensor is not None:
            return True, "ok"
        try:
            from mpu6050 import mpu6050
        except ImportError as e:
            # Surface the REAL import error - mpu6050-raspberrypi is pure Python
            # and imports `smbus`, which pip does NOT pull in, so the usual
            # failure is a missing smbus (sudo apt install python3-smbus), not a
            # missing mpu6050. Reporting the generic name hid that.
            return False, (f"IMU driver import failed: {e}. Need "
                           "'mpu6050-raspberrypi' AND its I2C backend "
                           "(sudo apt install python3-smbus).")
        try:
            self.sensor = mpu6050(I2C_ADDRESS)
            self.sensor.get_accel_data()   # probe: raises if the chip isn't there
            return True, "ok"
        except Exception as e:
            self.sensor = None
            return False, f"MPU-6050 not found at 0x{I2C_ADDRESS:02x} ({e})"

    def _publish_status(self, available, reason):
        """A bus-visible health beacon (picarx/sensors/imu/status) so the IMU's
        state is discoverable with `mosquitto_sub` - not just buried in a log.
        Republished periodically while idle so a late subscriber still sees it."""
        self.bus.publish("picarx/sensors/imu/status", {
            "available": bool(available), "reason": reason,
            "address": I2C_ADDRESS, "hz": IMU_HZ, "ts": time.time()})

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe(LOOK_TOPIC, self.on_look)
        self.bus.subscribe("picarx/sensors/imu/recalibrate", self.on_recalibrate)

        ok, reason = self._open_sensor()
        if not ok:
            # Stay alive (don't exit -> the orchestrator would just restart us in
            # a loop) but publish nothing on the data topic; a robot without the
            # chip runs normally. Beacon the reason so it's visible on the bus.
            print(f"IMU: {reason} - IMU disabled, idling.")
            while True:
                self._publish_status(False, reason)
                time.sleep(30)

        if not self.calibrate_at_rest():
            reason = "could not calibrate (no readings from the chip)"
            print(f"IMU: {reason} - idling.")
            while True:
                self._publish_status(False, reason)
                time.sleep(30)

        print(f"IMU active on picarx/sensors/imu at {IMU_HZ:.0f}Hz "
              f"(address 0x{I2C_ADDRESS:02x})")
        self._publish_status(True, "ok")
        period = 1.0 / max(1.0, IMU_HZ)
        while True:
            reading = self._read()
            if reading is not None:
                self._publish_reading(*reading)
            time.sleep(period)


if __name__ == "__main__":
    IMU().run()
