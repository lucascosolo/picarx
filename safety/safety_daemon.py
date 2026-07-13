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

class MotionSmoother(threading.Thread):
    """
    Background thread that smoothly ramps motor speeds and servo angles
    to prevent hardware stress and wheel slippage.
    """
    def __init__(self, hardware):
        super().__init__()
        self.px = hardware
        self.daemon = True
        
        self.target_speed = 0
        self.current_speed = 0
        
        self.target_angle = 0
        self.current_angle = 0
        
        # Tuning variables for how fast the robot accelerates
        self.speed_step = 2.0
        self.angle_step = 5.0
        
        self.lock = threading.Lock()
        self.running = True

    def update_targets(self, speed=None, angle=None):
        with self.lock:
            if speed is not None:
                self.target_speed = speed
            if angle is not None:
                self.target_angle = angle

    def emergency_stop(self):
        """Bypasses smoothing for immediate safety halts."""
        with self.lock:
            self.target_speed = 0
            self.current_speed = 0
            self.px.stop()

    def run(self):
        while self.running:
            with self.lock:
                # Smooth the speed
                if self.current_speed < self.target_speed:
                    self.current_speed = min(self.current_speed + self.speed_step, self.target_speed)
                elif self.current_speed > self.target_speed:
                    self.current_speed = max(self.current_speed - self.speed_step, self.target_speed)

                # Smooth the steering angle
                if self.current_angle < self.target_angle:
                    self.current_angle = min(self.current_angle + self.angle_step, self.target_angle)
                elif self.current_angle > self.target_angle:
                    self.current_angle = max(self.current_angle - self.angle_step, self.target_angle)

                # Apply states to hardware
                with hardware_lock:
                    if self.current_speed > 0:
                        self.px.forward(self.current_speed)
                    elif self.current_speed < 0:
                        self.px.backward(abs(self.current_speed))
                    else:
                        self.px.stop()
                    self.px.set_dir_servo_angle(self.current_angle)
                
            time.sleep(0.02)  # Run at 50Hz for buttery smooth adjustments


# Initialize the global motion controller
motion = MotionSmoother(px)


def read_battery_voltage():
    raw = battery_adc.read()
    voltage = raw / 4096 * 3.3 * 3
    return voltage

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
                try:
                    conn.sendall(json.dumps({"status": "vetoed", "reason": reason}).encode())
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