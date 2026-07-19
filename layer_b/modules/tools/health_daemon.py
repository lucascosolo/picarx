#!/usr/bin/env python3
# layer_b/modules/tools/health_daemon.py
"""
Health daemon (Layer B tool) - the robot's homeostatic self-monitoring.

Watches the robot's own physical "vital stats" - battery, CPU temperature,
free disk - and drives a self-preservation LOW-POWER state.

  - Publishes picarx/health/state at a fixed rate so anything (companion's
    check_vital_stats tool, the web console, ...) can read them.
  - Owns picarx/health/low_power (sole publisher). It enters low power on a
    battery hysteresis (LOW_BATTERY_V in, RECOVER_BATTERY_V out) OR when the
    LLM asks via picarx/tools/lowpower/request (companion's
    register_low_power_intent). On entering it announces once; consumers
    that honor the topic curtail high-power work - vision_basic backs its
    heavy SSD/YOLO pass right off, for instance.
  - This deterministic auto-trigger matters: self-preservation must not
    depend on the LLM being spoken to. The register_low_power_intent tool is
    an ADDITIONAL, proactive lever, not the only line of defense.

Battery source: it CONSUMES the voltage already on picarx/state/world (the
safety daemon reads the battery ADC and world_state.py republishes it).
Re-reading the robot_hat ADC from a second process would contend with the
safety daemon on the I2C bus, so we don't by default. A direct ADC read
(the SunFounder A4 snippet) is available behind HEALTH_BATTERY_ADC=1 as a
fallback for setups that don't run world_state.

Fail-soft throughout: an unreadable sensor just reports None. Issues no
motion and writes no database.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from broker_client import Bus
import robot_config

import shutil
import threading
import time

WORLD_TOPIC = "picarx/state/world"
STATE_TOPIC = "picarx/health/state"
LOW_POWER_TOPIC = "picarx/health/low_power"
LOWPOWER_REQUEST_TOPIC = "picarx/tools/lowpower/request"
SPEAK_TOPIC = "picarx/audio/speak"

THERMAL_PATH = "/sys/class/thermal/thermal_zone0/temp"
DISK_PATH = "/"

HEALTH_INTERVAL = 30.0          # seconds between vital-stat publishes

# PiCar-X runs 2x 18650 Li-ion in series: ~8.4V full, ~6.0V empty.
BATT_FULL_V = 8.4
BATT_EMPTY_V = 6.0
LOW_BATTERY_V = 6.6             # enter low power at/below this...
RECOVER_BATTERY_V = 7.0        # ...and only leave once back above this (hysteresis)

# Glitch rejection band. The pack is 2x 18650 in series: a genuine reading
# while the robot is powered and running lives in the ~6.0-8.4V range, and
# the Pi/Robot HAT brown out and lose power long before the pack could ever
# actually sit near zero. So a reading at/below GLITCH_FLOOR_V (classically a
# momentary 0.0V from an I2C/ADC hiccup or a dropped safety-daemon frame) is
# physically impossible while we're alive to read it - it is a sensor glitch,
# not a dead battery, and must NOT be allowed to trip the low-power state. A
# reading above GLITCH_CEILING_V is likewise impossible for this pack. Both
# are dropped: we keep the last good voltage rather than acting on a spike.
# The floor sits far below LOW_BATTERY_V/BATT_EMPTY_V, so this can never mask
# a real low battery - it only discards readings a real battery can't produce.
GLITCH_FLOOR_V = 3.0
GLITCH_CEILING_V = 9.0


def plausible_voltage(voltage):
    """True if `voltage` is a physically-possible pack reading (see the
    GLITCH_FLOOR_V/GLITCH_CEILING_V note). None (no reading) is not a glitch -
    it's honest 'unknown' - so it returns False here and is handled separately
    by callers (kept as-is / left to the ADC fallback)."""
    return voltage is not None and GLITCH_FLOOR_V <= voltage <= GLITCH_CEILING_V


def battery_percent(voltage):
    """Rough state-of-charge % from pack voltage (linear, clamped)."""
    if voltage is None:
        return None
    pct = (voltage - BATT_EMPTY_V) / (BATT_FULL_V - BATT_EMPTY_V) * 100.0
    return int(max(0.0, min(100.0, round(pct))))


def read_cpu_temp_c(path=THERMAL_PATH):
    """CPU core temp in C from /sys (a cheap file read, no subprocess).
    None if unavailable (e.g. off-robot)."""
    try:
        with open(path) as f:
            return round(int(f.read().strip()) / 1000.0, 1)
    except (OSError, ValueError):
        return None


def read_disk(path=DISK_PATH):
    """(free_gb, used_pct) for the filesystem at `path`, or (None, None)."""
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None, None
    free_gb = round(usage.free / 1e9, 1)
    used_pct = round(usage.used / usage.total * 100.0, 1) if usage.total else None
    return free_gb, used_pct


def read_battery_adc():
    """Direct robot_hat ADC battery read (SunFounder A4 divider). OFF by
    default (see module docstring - I2C contention with the safety daemon).
    Enable with HEALTH_BATTERY_ADC=1 for setups without world_state."""
    try:
        from robot_hat import ADC
        v = round(ADC("A4").read() * 3.3 / 4095 * 3, 2)
    except Exception:
        return None
    # A momentary 0.0V (or otherwise impossible) ADC sample is a glitch, not a
    # flat pack - report None ("unknown") so it can't drive low power.
    if not plausible_voltage(v):
        print(f"Health daemon: ignoring implausible ADC battery reading ({v}V)")
        return None
    return v


def summarize(vitals):
    """One spoken-friendly line from a vitals dict."""
    if not vitals:
        return "I don't have my vital stats yet."
    parts = []
    v, pct = vitals.get("battery_v"), vitals.get("battery_pct")
    if v is not None and pct is not None:
        parts.append(f"battery {v:.1f} volts, about {pct} percent")
    elif v is not None:
        parts.append(f"battery {v:.1f} volts")
    else:
        parts.append("battery reading unavailable")
    if vitals.get("temp_c") is not None:
        parts.append(f"CPU {vitals['temp_c']:.0f} degrees")
    if vitals.get("disk_free_gb") is not None:
        parts.append(f"{vitals['disk_free_gb']:.1f} gigabytes of disk free")
    if vitals.get("low_power"):
        parts.append("I'm in low-power mode")
    return ". ".join(p[0].upper() + p[1:] for p in parts) + "."


class HealthDaemon:
    def __init__(self):
        self.bus = Bus()
        self.lock = threading.Lock()
        # _evaluate runs from both the MQTT callback thread (on_world_state,
        # on_lowpower_request) and the main loop; serializing it keeps the
        # low_power read-modify-write from racing and double-publishing (or
        # missing) a transition.
        self._eval_lock = threading.Lock()
        self.battery_v = None
        self.battery_critical = False
        self.battery_low = False          # battery-driven component (hysteresis)
        self.low_power = False            # combined published state
        self.low_power_critical = False   # published critical sub-level (drives
                                          # vision's face-throttle 1s->2s step)
        self.manual_latch = False         # LLM/manual low-power request
        self.use_adc = robot_config.get_bool("health", "battery_adc", False,
                                             env="HEALTH_BATTERY_ADC")

    # ---------- inbound ----------

    def on_world_state(self, payload):
        battery = payload.get("battery") or {}
        voltage = battery.get("voltage")
        # Drop a glitch reading (e.g. a spurious 0.0V) outright: don't store it,
        # don't re-evaluate, and DON'T honor a 'critical' flag arriving with it
        # - that flag was computed downstream from the very same bad sample, so
        # trusting it would let the glitch trip low power through the back door.
        # We simply keep the last good state until a plausible reading lands.
        if voltage is not None and not plausible_voltage(voltage):
            print(f"Health daemon: ignoring implausible battery reading ({voltage}V)")
            return
        with self.lock:
            if voltage is not None:
                self.battery_v = voltage
            self.battery_critical = bool(battery.get("critical"))
        # React to a battery change immediately, not only on the 30s loop.
        self._evaluate(time.time())

    def on_lowpower_request(self, payload):
        """companion.register_low_power_intent lands here. A request to enter
        low power latches until the battery is healthy again; an explicit
        {active:false} clears the manual latch."""
        want = bool(payload.get("active", True))
        with self.lock:
            self.manual_latch = want
        print(f"Health daemon: manual low-power request active={want}")
        self._evaluate(time.time())

    # ---------- vitals ----------

    def _battery_voltage(self):
        with self.lock:
            v = self.battery_v
        if v is None and self.use_adc:
            v = read_battery_adc()
            if v is not None:
                with self.lock:
                    self.battery_v = v
        return v

    def _collect(self):
        voltage = self._battery_voltage()
        free_gb, used_pct = read_disk()
        return {
            "battery_v": voltage,
            "battery_pct": battery_percent(voltage),
            "temp_c": read_cpu_temp_c(),
            "disk_free_gb": free_gb,
            "disk_used_pct": used_pct,
            "low_power": self.low_power,
            "ts": time.time(),
        }

    # ---------- low-power state machine ----------

    def _battery_low_now(self, voltage):
        """Battery-only low state with hysteresis: enter at/below
        LOW_BATTERY_V, leave only above RECOVER_BATTERY_V. A safety-daemon
        'critical' flag forces low. Unknown voltage keeps the battery
        component as-is (kept separate from the manual latch so clearing the
        latch can actually turn low power off)."""
        with self.lock:
            critical = self.battery_critical
        if critical:
            return True
        if voltage is None:
            return self.battery_low
        if self.battery_low:
            return voltage < RECOVER_BATTERY_V
        return voltage <= LOW_BATTERY_V

    def _evaluate(self, now, voltage=None):
        """Recompute low-power = (battery low) OR (manual latch), publish +
        announce on any transition. Manual latch auto-clears once healthy."""
        with self._eval_lock:
            self._evaluate_locked(now, voltage)

    def _evaluate_locked(self, now, voltage=None):
        if voltage is None:
            voltage = self._battery_voltage()
        with self.lock:
            if self.manual_latch and voltage is not None and voltage >= RECOVER_BATTERY_V:
                self.manual_latch = False
            manual = self.manual_latch
        self.battery_low = self._battery_low_now(voltage)
        active = self.battery_low or manual
        with self.lock:
            # Critical is only meaningful while we're actually curtailing; it's
            # the safety daemon's own sub-threshold flag, surfaced so vision can
            # step its face throttle 1s->2s.
            critical = bool(self.battery_critical) and active
        prev_active, prev_critical = self.low_power, self.low_power_critical
        # Republish on a critical-level change too, not only on the active edge,
        # so the deeper throttle actually reaches consumers within low power.
        if active == prev_active and critical == prev_critical:
            return
        self.low_power = active
        self.low_power_critical = critical
        reason = ("battery" if self.battery_low else "requested") if active else "recovered"
        self.bus.publish(LOW_POWER_TOPIC, {"active": active, "critical": critical,
                                           "reason": reason, "ts": now})
        # Speak only on the active edge - a critical-only change is a silent
        # deeper throttle, not a new announcement.
        if active == prev_active:
            return
        if active:
            print(f"Health daemon: ENTERING low power ({reason}, {voltage}V)")
            self.bus.publish(SPEAK_TOPIC, {
                "text": "My battery is getting low, so I'm conserving power.",
                "ts": now})
        else:
            print("Health daemon: leaving low power")
            self.bus.publish(SPEAK_TOPIC, {
                "text": "My power is back to normal.", "ts": now})

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe(WORLD_TOPIC, self.on_world_state)
        self.bus.subscribe(LOWPOWER_REQUEST_TOPIC, self.on_lowpower_request)
        print(f"Health daemon active, publishing vitals to {STATE_TOPIC} "
              f"every {HEALTH_INTERVAL:.0f}s")
        while True:
            try:
                self._evaluate(time.time())
                self.bus.publish(STATE_TOPIC, self._collect())
            except Exception as e:
                print(f"Health daemon: cycle error: {e}")
            time.sleep(HEALTH_INTERVAL)


if __name__ == "__main__":
    HealthDaemon().run()
