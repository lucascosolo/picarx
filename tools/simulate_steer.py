#!/usr/bin/env python3
"""
Off-robot steering-controller inspection harness.

Feeds synthetic world snapshots through SteeringController at the real
5Hz tick and prints the resulting steering-angle/speed time series, so
tuning changes (area_distance_k, clearance_m, steering rate, ...) can
be eyeballed without a robot. Pure stdout, no MQTT, no hardware.

    python3 tools/simulate_steer.py

Scenarios:
  1. table-leg-right : a table leg off to the right grows as the robot
                       approaches, then falls away - expect a smooth
                       left arc that deepens, then relaxes to straight.
  2. flanked-gap     : two chair legs flanking a gap - expect ~0 angle
                       (threading) with speed easing off, no flip-flop.
  3. sudden-appear   : a person steps in half-left at close range -
                       expect a rate-limited swing right, never a jump.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "layer_b"))
sys.path.insert(0, os.path.join(REPO, "layer_b", "modules"))

from steering_controller import SteeringController  # noqa: E402

FRAME_W = 320
DT = 0.2   # field_agent's real 5Hz tick


def obj(label, area, offset, approaching=False):
    return {"id": f"{label}_{offset}", "label": label, "area_ratio": area,
            "center_offset": offset, "frame_width": FRAME_W,
            "approaching": approaching}


def world(items, distance=None):
    return {"distance_cm": distance, "distance_stale": distance is None,
            "objects": {"stale": False, "items": items,
                        "close_object": False, "overhead": None}}


def scenario_table_leg_right(t):
    if t < 4.0:   # approach: leg looms bigger and drifts central
        frac = t / 4.0
        return world([obj("table leg", 0.06 + 0.30 * frac,
                          int(110 - 40 * frac), approaching=frac > 0.4)],
                     distance=180 - 130 * frac)
    if t < 5.6:   # passing: slides right out of the cone
        frac = (t - 4.0) / 1.6
        return world([obj("table leg", 0.30 - 0.1 * frac, int(70 + 90 * frac))],
                     distance=60 + 100 * frac)
    return world([], distance=200)   # clear road


def scenario_flanked_gap(t):
    if t < 5.0:
        area = 0.10 + 0.04 * t
        return world([obj("left leg", area, -70), obj("right leg", area, 70)],
                     distance=120 - 15 * t)
    return world([], distance=200)


def scenario_sudden_appear(t):
    if t < 1.0:
        return world([], distance=200)
    if t < 5.0:
        return world([obj("person", 0.40, -50, approaching=True)], distance=55)
    return world([], distance=200)


def run(name, scenario, seconds=7.0):
    c = SteeringController()
    print(f"\n=== {name} ===")
    print(f"{'t(s)':>5} {'angle(deg)':>11} {'speed':>6} {'active':>6}  reason")
    ticks = int(seconds / DT)
    for i in range(ticks):
        t = i * DT
        cmd = c.compute_command(scenario(t), now=1000.0 + t)
        bar = "#" * int(abs(cmd["steering_angle_deg"]))
        side = bar.rjust(22) + "|" + " " * 22 if cmd["steering_angle_deg"] < 0 \
            else " " * 22 + "|" + bar.ljust(22)
        print(f"{t:5.1f} {cmd['steering_angle_deg']:11.2f} {cmd['speed']:6.1f} "
              f"{'yes' if cmd['active'] else 'no':>6}  {side}  {cmd['reason']}")


if __name__ == "__main__":
    run("table-leg-right (smooth left arc, then relax)", scenario_table_leg_right)
    run("flanked-gap (thread straight, slow down)", scenario_flanked_gap)
    run("sudden-appear (rate-limited swing, no jump)", scenario_sudden_appear)
