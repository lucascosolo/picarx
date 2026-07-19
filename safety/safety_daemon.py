#!/usr/bin/env python3
# /home/picarx/safety/safety_daemon.py
"""
Hardcoded physical safety layer for PicarX.

SCOPE: This handles the machine preservation and smooth motion control.
"""

import socket
import os
import getpass
import time
import json
import threading
import importlib

# Fix for os.getlogin() failing under systemd (no controling TTY)
os.getlogin = getpass.getuser

from picarx import Picarx
from robot_hat import ADC

SOCKET_PATH = "/tmp/picarx_safety.sock"
SAFE_DISTANCE_CM = 15
CLIFF_THRESHOLD = 200

# The cliff/grayscale sensors are FRONT-mounted and there is no
# rear-facing sensor, so nothing can see a drop-off behind the robot
# (it backed off a tabletop at startup). As a backstop, bound how long
# one continuous reverse may run: a normal escape reverse (~1.2s) and
# coach reverse suggestions (<=1.5s) pass, but a sustained runaway gets
# vetoed -> emergency stop. Any non-reverse command resets the timer,
# so bursts of reversing separated by forward/turn/stop are each
# independently bounded. Not a substitute for a real rear sensor.
MAX_CONTINUOUS_REVERSE_SEC = 2.0
_reverse_state = {"since": None}

# Battery monitoring thresholds
BATTERY_ADC_CHANNEL = "A4"
LOW_BATTERY_VOLTAGE = 6.7
CRITICAL_BATTERY_VOLTAGE = 6.4
BATTERY_CHECK_INTERVAL = 10

px = Picarx()
battery_adc = ADC(BATTERY_ADC_CHANNEL)
battery_state = {"voltage": None, "critical": False, "low": False}

hardware_lock = threading.Lock()

# ---- MotionSmoother tuning ----
# Ramp RATES are per-second (not per-tick), so motion is time-consistent
# even when the control loop is jittered by hardware_lock contention with
# the cliff/ultrasonic reads (the old fixed per-tick steps silently
# under-accelerated whenever a tick ran late). These match the previous
# effective rates: the old speed_step 2.0 and angle_step 5.0 at the 50Hz
# tick were 100 units/s and 250 deg/s.
SPEED_RAMP_PER_SEC = 100.0
ANGLE_RAMP_PER_SEC = 250.0
MOTION_TICK_SEC = 0.02          # 50Hz nominal
# A tick that ran late (lock contention, a cliff-veto sample burst) must
# not translate its whole elapsed time into one big ramp jump - clamp the
# per-tick advance so a stall just resumes the normal rate, never lurches.
MOTION_MAX_DT = 0.05
# Servo writes below this change are skipped: they're sub-degree jitter
# that only makes the steering servo buzz and steals the hardware lock.
ANGLE_APPLY_EPSILON = 0.5


def ramp_toward(current, target, rate, dt):
    """Move `current` toward `target` by at most rate*dt (a linear rate
    limit). Pure/hardware-free so the motion ramp is unit-testable off the
    robot; lands exactly on the target once within one step."""
    max_step = rate * dt
    if current < target:
        return min(current + max_step, target)
    if current > target:
        return max(current - max_step, target)
    return current


class MotionSmoother(threading.Thread):
    """
    Background thread that smoothly ramps motor speeds and servo angles
    to prevent hardware stress and wheel slippage. Ramp rates are in
    real time (see SPEED_RAMP_PER_SEC / ANGLE_RAMP_PER_SEC) and only the
    values that actually change are written to hardware, so a steady
    cruise isn't re-sending the same command 50 times a second and
    fighting the safety sensor reads for the hardware lock.
    """
    def __init__(self, hardware):
        super().__init__()
        self.px = hardware
        self.daemon = True

        self.target_speed = 0.0
        self.current_speed = 0.0

        self.target_angle = 0.0
        self.current_angle = 0.0

        self.speed_rate = SPEED_RAMP_PER_SEC
        self.angle_rate = ANGLE_RAMP_PER_SEC

        self.lock = threading.Lock()
        self.running = True

        # Last values actually pushed to hardware, so redundant writes are
        # skipped (None = nothing written yet / force the next write).
        self._applied_speed = None
        self._applied_angle = None

    def update_targets(self, speed=None, angle=None):
        with self.lock:
            if speed is not None:
                # On a direction REVERSAL, snap through zero instead of
                # ramping down first: going from forward-25 to backward-30
                # otherwise spent a big fraction of a ~1s escape step just
                # decelerating - eroding commanded maneuvers to almost no
                # actual displacement (a big part of "announces an escape
                # but barely moves"). Ramping is kept within a direction;
                # only the sign flip is immediate.
                if speed * self.current_speed < 0:
                    self.current_speed = 0.0
                self.target_speed = float(speed)
            if angle is not None:
                self.target_angle = float(angle)

    def emergency_stop(self):
        """Bypasses smoothing for immediate safety halts."""
        with self.lock:
            self.target_speed = 0.0
            self.current_speed = 0.0
            with hardware_lock:
                self.px.stop()
            self._applied_speed = 0.0

    def _apply(self, speed, angle):
        """Push speed/angle to hardware, skipping writes that wouldn't
        change anything (the motor/servo hold their last command). Keeping
        the hardware lock idle between real changes leaves it free for the
        safety daemon's cliff/ultrasonic reads."""
        with hardware_lock:
            if self._applied_speed is None or speed != self._applied_speed:
                if speed > 0:
                    self.px.forward(speed)
                elif speed < 0:
                    self.px.backward(abs(speed))
                else:
                    self.px.stop()
                self._applied_speed = speed
            # Write the servo on a meaningful change, but always land
            # exactly on a freshly-reached target (so the last sub-epsilon
            # step isn't dropped and the wheel settles a hair off-line).
            reached_target = (angle == self.target_angle
                              and angle != self._applied_angle)
            if (self._applied_angle is None
                    or abs(angle - self._applied_angle) >= ANGLE_APPLY_EPSILON
                    or reached_target):
                self.px.set_dir_servo_angle(angle)
                self._applied_angle = angle

    def _tick(self, dt):
        """One ramp+apply step for elapsed time dt. Separated from run()
        so the motion logic is unit-testable without the thread."""
        with self.lock:
            self.current_speed = ramp_toward(
                self.current_speed, self.target_speed, self.speed_rate, dt)
            self.current_angle = ramp_toward(
                self.current_angle, self.target_angle, self.angle_rate, dt)
            speed, angle = self.current_speed, self.current_angle
        self._apply(speed, angle)

    def run(self):
        last = time.time()
        while self.running:
            time.sleep(MOTION_TICK_SEC)
            now = time.time()
            dt = min(now - last, MOTION_MAX_DT)
            last = now
            self._tick(dt)


# Initialize the global motion controller
motion = MotionSmoother(px)


def _manual_battery_voltage():
    """Hand-rolled divider read. 12-bit full-scale is 4095 (not 4096), Vref is
    3.3V, and the Robot HAT A4 pin sits behind a ~3x divider. Kept only as a
    fallback for library versions that don't ship a battery reader."""
    return battery_adc.read() / 4095 * 3.3 * 3


def _resolve_battery_reader():
    """Prefer SunFounder's own calibrated battery reader over the hand-rolled
    divider math.

    The old formula (raw/4096*3.3*3) under-read the pack - enough to trip a
    false 'battery critical' (and an emergency stop) on a near-full battery
    whose Robot HAT charge LEDs were all still lit. The library ships the
    correct scaling for whatever HAT revision is actually installed, so defer
    to it and only fall back to the manual formula if it isn't available (older
    robot_hat), so the daemon still runs everywhere."""
    # 1) SunFounder utils helper - the canonical get_battery_voltage().
    for mod_name in ("robot_hat.utils", "robot_hat"):
        try:
            fn = getattr(importlib.import_module(mod_name), "get_battery_voltage", None)
        except Exception:
            fn = None
        if callable(fn):
            print(f"Safety daemon: battery voltage via {mod_name}.get_battery_voltage()")
            return fn
    # 2) Picarx instance method, if this version exposes one (reuses the HAT we
    #    already own - no second ADC object).
    fn = getattr(px, "get_battery_voltage", None)
    if callable(fn):
        print("Safety daemon: battery voltage via Picarx.get_battery_voltage()")
        return fn
    # 3) Manual divider fallback.
    print("Safety daemon: SunFounder battery reader unavailable; using manual ADC formula")
    return _manual_battery_voltage


_battery_reader = _resolve_battery_reader()


def read_battery_voltage():
    """Battery pack voltage in volts. See _resolve_battery_reader for why this
    prefers the library's reader over hand-rolled divider math."""
    return _battery_reader()

def check_battery():
    try:
        voltage = read_battery_voltage()
        battery_state["voltage"] = voltage
        battery_state["low"] = voltage < LOW_BATTERY_VOLTAGE
        battery_state["critical"] = voltage < CRITICAL_BATTERY_VOLTAGE
        
        if battery_state["critical"]:
            motion.emergency_stop()
            print("CRITICAL BATTERY!")
        elif battery_state["low"]:
            print("BATTERY LOW")
    except Exception as e:
        print(f"Battery read error: {e}")

def is_safe(action):
    direction = action.get("direction")

    # Only FORWARD motion needs sensor checks. This matters for two
    # reasons beyond correctness:
    #
    #  - The cliff sensors are front-mounted IR reflectance sensors.
    #    Running the cliff check on "backward" meant a dark patch of
    #    floor (carpet seam, shadow) that false-reads as a cliff would
    #    veto the robot's own backward escape from that very spot -
    #    the observed "says backing away but never moves" freeze. A
    #    front-detected cliff/obstacle is escaped BY reversing; the
    #    sensors can't see behind the robot either way, so vetoing
    #    reverse never protected anything.
    #  - "stop" must never be vetoable (it IS the safe state), and
    #    "turn" only moves the steering servo. Skipping the 3-sample
    #    grayscale read (~30ms of sleeps + hardware-lock contention
    #    with the 50Hz MotionSmoother) for all of these also removes
    #    most of this daemon's measured CPU load - the arbiter
    #    re-sends the active action at 10Hz, so is_safe used to run
    #    the full sensor suite ten times a second even while stopped.
    if direction == "backward":
        now = time.time()
        if _reverse_state["since"] is None:
            _reverse_state["since"] = now
        if now - _reverse_state["since"] > MAX_CONTINUOUS_REVERSE_SEC:
            return False, "reverse time limit (no rear sensor)"
        return True, "ok"

    # Any non-reverse command ends the current continuous-reverse run.
    _reverse_state["since"] = None

    if direction in ("stop", "turn", "look"):
        return True, "ok"

    with hardware_lock:
        distance = px.ultrasonic.read()

    if direction == "forward" and (0 < distance < SAFE_DISTANCE_CM):
        return False, f"obstacle at {distance}cm"

    # Grayscale/cliff sensors are IR reflectance sensors, not true
    # depth sensors - they infer "cliff" from how much light bounces
    # back off whatever surface is underneath. That makes them
    # sensitive to surface color/texture changes (e.g. a carpet/tile
    # seam, a dark patch of carpet, a shadow) that are NOT actually a
    # drop-off but can momentarily read the same as one. A genuine
    # edge reads low consistently; surface noise typically does not.
    # Take a few quick samples and require most of them to agree
    # before treating it as a real cliff, to filter out that noise
    # without weakening protection against an actual edge.
    CLIFF_SAMPLES = 3
    CLIFF_SAMPLES_REQUIRED = 2  # majority of CLIFF_SAMPLES must agree
    samples_taken = []
    low_readings = 0
    for _ in range(CLIFF_SAMPLES):
        with hardware_lock:
            grayscale = px.get_grayscale_data()
        samples_taken.append(grayscale)
        if min(grayscale) < CLIFF_THRESHOLD:
            low_readings += 1
        time.sleep(0.01)

    if low_readings >= CLIFF_SAMPLES_REQUIRED:
        # Log the actual raw readings that caused this veto - this is
        # the real evidence needed to tell a genuine edge apart from
        # sensor/electrical noise, captured at the moment it happens
        # rather than inferred from a separate stationary test.
        print(f"CLIFF VETO - samples: {samples_taken}, threshold: {CLIFF_THRESHOLD}")
        return False, "cliff detected"

    return True, "ok"

# Camera head servo limits - clamped here (the sole hardware gate)
# so no upstream module can ever command the servos past their
# physical range regardless of what arrives on the socket.
CAM_PAN_RANGE = (-80, 80)
CAM_TILT_RANGE = (-30, 60)


def execute(action):
    """Updates the targets for the motion thread instead of blocking hardware."""
    d = action.get("direction")
    speed = action.get("speed", 30)

    if d == "forward":
        motion.update_targets(speed=speed)
    elif d == "backward":
        motion.update_targets(speed=-speed)
    elif d == "stop":
        motion.update_targets(speed=0)
    elif d == "turn":
        motion.update_targets(angle=action.get("angle", 0))
    elif d == "look":
        # Camera pan/tilt only - never touches drive motors or
        # steering, so it doesn't go through the MotionSmoother.
        pan = max(CAM_PAN_RANGE[0], min(CAM_PAN_RANGE[1], int(action.get("pan", 0))))
        tilt = max(CAM_TILT_RANGE[0], min(CAM_TILT_RANGE[1], int(action.get("tilt", 0))))
        with hardware_lock:
            px.set_cam_pan_angle(pan)
            px.set_cam_tilt_angle(tilt)

def main():
    if os.path.exists(SOCKET_PATH):
        os.remove(SOCKET_PATH)
        
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(SOCKET_PATH)
    os.chmod(SOCKET_PATH, 0o666)
    server.listen(5)
    server.settimeout(2.0)
    
    # Start the background smoothing thread
    motion.start()
    
    print(f"Safety daemon listening on {SOCKET_PATH}")

    last_battery_check = 0

    while True:
        now = time.time()
        if now - last_battery_check > BATTERY_CHECK_INTERVAL:
            check_battery()
            last_battery_check = now

        try:
            conn, _ = server.accept()
        except socket.timeout:
            continue

        try:
            data = conn.recv(1024)
            if not data:
                continue
            action = json.loads(data.decode())

            # --- Queries ---
            if action.get("query") == "battery_status":
                try:
                    conn.sendall(json.dumps(battery_state).encode())
                except Exception as se:
                    print(f"Socket reply error (battery): {se}")
                continue

            if action.get("query") == "distance":
                try:
                    current_distance = px.ultrasonic.read()
                    conn.sendall(json.dumps({"distance_cm": current_distance}).encode())
                except Exception as sensor_err:
                    try:
                        conn.sendall(json.dumps({"error": str(sensor_err)}).encode())
                    except Exception:
                        pass
                continue
            # ---------------

            # --- Movements ---
            safe, reason = is_safe(action)
            if safe:
                execute(action)
                try:
                    conn.sendall(json.dumps({"status": "executed"}).encode())
                except Exception as se:
                    print(f"Socket reply error (executed): {se}")
            else:
                motion.emergency_stop()
                # reason_code is a STABLE machine-readable failure type
                # alongside the human-readable reason - the learning
                # layer keys recovery tactics on it, so these codes must
                # never change once shipped: obstacle | cliff |
                # reverse_limit | unknown.
                code = ("obstacle" if reason.startswith("obstacle")
                        else "cliff" if "cliff" in reason
                        else "reverse_limit" if "reverse" in reason
                        else "unknown")
                try:
                    conn.sendall(json.dumps({"status": "vetoed", "reason": reason,
                                             "reason_code": code}).encode())
                except Exception as se:
                    print(f"Socket reply error (vetoed): {se}")
                    
        except Exception as e:
            print(f"Daemon process handling error: {e}")
            try:
                conn.sendall(json.dumps({"status": "error", "detail": str(e)}).encode())
            except Exception:
                pass 
        finally:
            try:
                conn.close()
            except Exception:
                pass

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        motion.running = False
        motion.join()