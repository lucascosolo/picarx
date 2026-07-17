import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import field_agent  # noqa: E402
from steering_controller import SteeringController  # noqa: E402

FRAME_W = 320


def _obj(label="chair", area=0.2, offset=0, approaching=False):
    return {"id": f"object_{label}_{offset}", "label": label,
            "area_ratio": area, "center_offset": offset,
            "frame_width": FRAME_W, "approaching": approaching}


def _world(items=None, distance=100, distance_stale=False):
    return {
        "distance_cm": distance,
        "distance_stale": distance_stale,
        "objects": {"stale": False, "items": items or [],
                    "close_object": False, "overhead": None},
    }


class ControllerTickIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.fa = field_agent.FieldAgent()
        self.assertIsNotNone(self.fa.steering)   # real controller loaded
        self.fa.state = "CRUISING"
        self.fa.last_scan_at = time.time()
        self.fa.last_wander = time.time()

    def _drive(self, world, ticks=1):
        stamps = []
        for _ in range(ticks):
            self.fa.latest_world = world
            stamps.append(time.time())
            self.fa.explore_tick()
        return stamps

    def _actions(self):
        return [p["action"] for p in self.fa.bus.of("picarx/intent/move")]

    def test_alternates_turn_and_forward_primitives(self):
        # The arbiter keeps ONE intent per source, so the agent must
        # never overwrite a turn with a forward in the same tick - and
        # never starve forward (whose safety checks matter) either.
        self._drive(_world([_obj(area=0.3, offset=80)]), ticks=6)
        kinds = [a["direction"] for a in self._actions()]
        self.assertIn("turn", kinds)
        self.assertIn("forward", kinds)
        for a, b in zip(kinds, kinds[1:]):
            self.assertFalse(a == b == "turn", f"two consecutive turns in {kinds}")

    def test_turn_angles_are_floats_away_from_object(self):
        self._drive(_world([_obj(area=0.3, offset=80)]), ticks=6)
        turns = [a for a in self._actions() if a["direction"] == "turn"]
        self.assertTrue(turns)
        for t in turns:
            self.assertIsInstance(t["angle"], float)
            self.assertLess(t["angle"], 0)       # object right -> steer left

    def test_forward_speed_is_controller_scaled(self):
        self._drive(_world([_obj(area=0.3, offset=80)]), ticks=4)
        forwards = [a for a in self._actions() if a["direction"] == "forward"]
        self.assertTrue(forwards)
        c = self.fa.steering
        for f in forwards:
            self.assertLessEqual(f["speed"], c.cruise_speed)
            self.assertGreaterEqual(f["speed"],
                                    c.cruise_speed * c.min_speed_factor - 1e-6)

    def test_successive_turn_commands_are_rate_limited(self):
        stamps = self._drive(
            _world([_obj(area=0.45, offset=60, approaching=True)]), ticks=12)
        c = self.fa.steering
        published = []   # (angle, tick timestamp)
        idx = 0
        for p in self.fa.bus.of("picarx/intent/move"):
            if p["action"]["direction"] == "turn":
                published.append(p["action"]["angle"])
        elapsed = stamps[-1] - stamps[0]
        # Between consecutive published turns at most 2 compute calls ran;
        # each is bounded by rate * max(dt_min, real elapsed).
        bound = c.steering_rate * (2 * c.dt_min + elapsed) + 1e-6
        for a, b in zip(published, published[1:]):
            self.assertLessEqual(abs(b - a), bound)

    def test_fewer_abrupt_recommands_than_baseline(self):
        # Sensor noise: the object's size/offset jitters every tick. The
        # discrete law re-commands a jumped angle on nearly every tick;
        # the filtered controller glides.
        worlds = []
        for i in range(20):
            if i % 2 == 0:
                worlds.append(_world([_obj(area=0.12, offset=40)]))
            else:
                worlds.append(_world([_obj(area=0.35, offset=80)]))

        # Baseline: the discrete law + its resend semantics.
        baseline_sent, last_sent = [], None
        for w in worlds:
            out = field_agent._steer_away_angle(w)
            if out is None:
                continue
            if last_sent is None or abs(out["angle"] - last_sent) >= \
                    field_agent.AVOID_RESEND_DELTA:
                baseline_sent.append(out["angle"])
                last_sent = out["angle"]
        baseline_abrupt = sum(
            1 for a, b in zip(baseline_sent, baseline_sent[1:])
            if abs(b - a) >= field_agent.AVOID_RESEND_DELTA)

        controller = SteeringController()
        angles = [controller.compute_command(w, now=1000.0 + i * 0.2)
                  ["steering_angle_deg"] for i, w in enumerate(worlds)]
        controller_abrupt = sum(
            1 for a, b in zip(angles, angles[1:])
            if abs(b - a) >= field_agent.AVOID_RESEND_DELTA)

        self.assertGreater(baseline_abrupt, 0)
        self.assertLess(controller_abrupt, baseline_abrupt)

    def test_emergency_evasion_still_outranks_controller(self):
        self.fa.last_probe_at = time.time()   # sensor probe on cooldown
        self._drive(_world([_obj(area=0.3, offset=80)], distance=10))
        self.assertEqual(self.fa.state, "EVADING")

    def test_overhead_emergency_still_outranks_controller(self):
        world = _world([_obj(area=0.3, offset=80)], distance=None,
                       distance_stale=True)
        world["objects"]["overhead"] = {"area_ratio": 0.5, "approaching": True}
        self._drive(world)
        self.assertEqual(self.fa.state, "EVADING")

    def test_wheels_straighten_after_hold_expires(self):
        self._drive(_world([_obj(area=0.3, offset=80)]), ticks=2)
        self.fa.bus.clear()
        self.fa.steering_active_until = time.time() - 0.1
        self._drive(_world([]))
        actions = self._actions()
        self.assertIn({"direction": "turn", "angle": 0}, actions)

    def test_fallback_to_discrete_law_without_controller(self):
        self.fa.steering = None
        self._drive(_world([_obj(area=0.3, offset=80)]))
        actions = self._actions()
        turns = [a for a in actions if a["direction"] == "turn"]
        forwards = [a for a in actions if a["direction"] == "forward"]
        # Old semantics: turn + forward in the same tick, discrete speed.
        self.assertTrue(turns and turns[-1]["angle"] < 0)
        self.assertTrue(forwards)
        self.assertEqual(forwards[-1]["speed"], field_agent.AVOID_SPEED)


if __name__ == "__main__":
    unittest.main()
