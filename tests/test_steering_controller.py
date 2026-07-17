import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

from steering_controller import SteeringController  # noqa: E402

FRAME_W = 320
T0 = 1000.0
DT = 0.2


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


def _run(controller, world, ticks, start=T0):
    """Feed the same world for `ticks` calls at DT spacing; return the
    list of command dicts."""
    return [controller.compute_command(world, now=start + i * DT)
            for i in range(ticks)]


class SignAndMagnitudeTest(unittest.TestCase):
    def test_object_right_steers_left(self):
        cmds = _run(SteeringController(), _world([_obj(area=0.3, offset=80)]), 8)
        final = cmds[-1]
        self.assertTrue(final["active"])
        self.assertLess(final["steering_angle_deg"], 0)
        self.assertIn("chair", final["labels"])
        self.assertIn("left", final["reason"])

    def test_object_left_steers_right(self):
        cmds = _run(SteeringController(), _world([_obj(area=0.3, offset=-80)]), 8)
        self.assertGreater(cmds[-1]["steering_angle_deg"], 0)

    def test_angle_is_float_and_within_limits(self):
        c = SteeringController()
        crowd = _world([_obj(label=f"o{i}", area=0.5, offset=30 + i) for i in range(5)])
        for cmd in _run(c, crowd, 12):
            self.assertIsInstance(cmd["steering_angle_deg"], float)
            self.assertLessEqual(abs(cmd["steering_angle_deg"]), c.avoid_max_deg)
            self.assertLessEqual(c.avoid_max_deg, c.max_steer_deg)

    def test_flanking_gap_threads_straight(self):
        world = _world([_obj(label="left leg", area=0.2, offset=-70),
                        _obj(label="right leg", area=0.2, offset=70)])
        cmds = _run(SteeringController(), world, 8)
        final = cmds[-1]
        self.assertTrue(final["active"])
        self.assertEqual(final["steering_angle_deg"], 0.0)   # deadbanded: no flip-flop
        self.assertIn("threading", final["reason"])

    def test_close_frontal_object_slows_but_does_not_pick_a_side(self):
        world = _world([_obj(area=0.4, offset=0)], distance=30)
        cmds = _run(SteeringController(), world, 6)
        final = cmds[-1]
        self.assertTrue(final["active"])
        self.assertEqual(final["steering_angle_deg"], 0.0)
        self.assertLess(final["speed"], SteeringController().cruise_speed)
        self.assertIn("dead ahead", final["reason"])

    def test_speck_ignored_unless_approaching(self):
        quiet = _run(SteeringController(), _world([_obj(area=0.01, offset=60)]), 4)
        self.assertFalse(quiet[-1]["active"])
        urgent = _run(SteeringController(),
                      _world([_obj(area=0.01, offset=60, approaching=True)]), 4)
        self.assertTrue(urgent[-1]["active"])

    def test_far_off_path_ignored(self):
        edge = int((FRAME_W / 2) * SteeringController().cone_frac) + 10
        cmds = _run(SteeringController(), _world([_obj(area=0.4, offset=edge)]), 4)
        self.assertFalse(cmds[-1]["active"])

    def test_approaching_object_steers_harder(self):
        calm = _run(SteeringController(),
                    _world([_obj(area=0.15, offset=80)]), 8)
        urgent = _run(SteeringController(),
                      _world([_obj(area=0.15, offset=80, approaching=True)]), 8)
        self.assertGreater(abs(urgent[-1]["steering_angle_deg"]),
                           abs(calm[-1]["steering_angle_deg"]))


class SmoothnessTest(unittest.TestCase):
    def test_per_tick_delta_respects_steering_rate(self):
        c = SteeringController()
        world = _world([_obj(area=0.45, offset=60, approaching=True)])
        cmds = _run(c, world, 10)
        angles = [cmd["steering_angle_deg"] for cmd in cmds]
        # First call uses dt_max (no previous timestamp); later calls DT.
        bounds = [c.steering_rate * c.dt_max] + [c.steering_rate * DT] * len(angles)
        prev = 0.0
        for angle, bound in zip(angles, bounds):
            self.assertLessEqual(abs(angle - prev), bound + 1e-6)
            prev = angle

    def test_magnitude_grows_smoothly_as_object_approaches(self):
        c = SteeringController()
        angles = []
        for i, area in enumerate((0.08, 0.12, 0.18, 0.26, 0.36, 0.45)):
            cmd = c.compute_command(_world([_obj(area=area, offset=70)]),
                                    now=T0 + i * DT)
            angles.append(cmd["steering_angle_deg"])
        self.assertLess(angles[-1], 0)
        self.assertGreater(abs(angles[-1]), abs(angles[0]))
        deltas = [abs(b - a) for a, b in zip(angles, angles[1:])]
        self.assertLessEqual(max(deltas), c.steering_rate * DT + 1e-6)

    def test_angle_decays_toward_straight_when_path_clears(self):
        c = SteeringController()
        _run(c, _world([_obj(area=0.4, offset=70, approaching=True)]), 8)
        mags = []
        for i in range(30):
            cmd = c.compute_command(_world([]), now=T0 + (8 + i) * DT)
            self.assertFalse(cmd["active"])
            mags.append(abs(cmd["steering_angle_deg"]))
        self.assertEqual(mags[-1], 0.0)
        self.assertTrue(all(b <= a + 1e-9 for a, b in zip(mags, mags[1:])))


class SpeedPolicyTest(unittest.TestCase):
    def test_tighter_curve_means_lower_speed(self):
        gentle = _run(SteeringController(),
                      _world([_obj(area=0.10, offset=90)], distance=200), 8)[-1]
        tight = _run(SteeringController(),
                     _world([_obj(area=0.45, offset=60, approaching=True)],
                            distance=200), 8)[-1]
        self.assertGreater(abs(tight["steering_angle_deg"]),
                           abs(gentle["steering_angle_deg"]))
        self.assertLess(tight["speed"], gentle["speed"])

    def test_speed_floor_holds(self):
        c = SteeringController()
        world = _world([_obj(area=0.5, offset=40, approaching=True)], distance=18)
        for cmd in _run(c, world, 8):
            self.assertGreaterEqual(cmd["speed"],
                                    c.cruise_speed * c.min_speed_factor - 1e-6)

    def test_clear_path_is_cruise_speed(self):
        cmd = SteeringController().compute_command(_world([]), now=T0)
        self.assertFalse(cmd["active"])
        self.assertEqual(cmd["speed"], SteeringController().cruise_speed)


class UltrasonicFusionTest(unittest.TestCase):
    def test_fresh_short_ultrasonic_is_authoritative(self):
        # Same near-center object; a fresh 35cm sonar reading should pull
        # the distance estimate in and slow the robot down vs. no sonar.
        blind = _run(SteeringController(),
                     _world([_obj(area=0.15, offset=48)],
                            distance=None, distance_stale=True), 6)[-1]
        sonar = _run(SteeringController(),
                     _world([_obj(area=0.15, offset=48)], distance=35), 6)[-1]
        self.assertLess(sonar["nearest_cm"], blind["nearest_cm"])
        self.assertLess(sonar["speed"], blind["speed"])

    def test_long_ultrasonic_does_not_inflate_distance(self):
        c = SteeringController()
        cmd = _run(c, _world([_obj(area=0.3, offset=48)], distance=300), 4)[-1]
        # Vision says ~64cm; a clear-long sonar must not override that.
        self.assertLess(cmd["nearest_cm"], 100)

    def test_offcenter_object_not_fused(self):
        c = SteeringController()
        est_off = c._estimate_distance_cm(_obj(area=0.15, offset=120), ultra_cm=30)
        est_center = c._estimate_distance_cm(_obj(area=0.15, offset=10), ultra_cm=30)
        self.assertEqual(est_center, 30)          # near-center: sonar wins
        self.assertGreater(est_off, 30)           # off to the side: vision only


if __name__ == "__main__":
    unittest.main()
