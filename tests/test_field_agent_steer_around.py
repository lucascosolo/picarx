import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import field_agent  # noqa: E402

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


class SteerAwayAngleTest(unittest.TestCase):
    """Pure perception->heading law (_steer_away_angle)."""

    def test_none_when_empty_or_stale(self):
        self.assertIsNone(field_agent._steer_away_angle(None))
        self.assertIsNone(field_agent._steer_away_angle(_world()))
        stale = _world([_obj(area=0.3, offset=80)])
        stale["objects"]["stale"] = True
        self.assertIsNone(field_agent._steer_away_angle(stale))

    def test_steers_left_away_from_object_on_right(self):
        out = field_agent._steer_away_angle(_world([_obj(area=0.3, offset=80)]))
        self.assertLess(out["angle"], 0)
        self.assertIn("chair", out["labels"])

    def test_steers_right_away_from_object_on_left(self):
        out = field_agent._steer_away_angle(_world([_obj(area=0.3, offset=-80)]))
        self.assertGreater(out["angle"], 0)

    def test_central_object_steers_harder_than_peripheral(self):
        near_center = field_agent._steer_away_angle(_world([_obj(area=0.3, offset=30)]))
        peripheral = field_agent._steer_away_angle(_world([_obj(area=0.3, offset=110)]))
        self.assertGreater(abs(near_center["angle"]), abs(peripheral["angle"]))

    def test_flanking_pair_cancels_and_threads_the_gap(self):
        # Equal objects both sides of a gap: contributions cancel, drive
        # between them instead of ping-ponging.
        out = field_agent._steer_away_angle(_world([
            _obj(label="left leg", area=0.2, offset=-70),
            _obj(label="right leg", area=0.2, offset=70),
        ]))
        self.assertIsNone(out)

    def test_speck_ignored_unless_approaching(self):
        self.assertIsNone(field_agent._steer_away_angle(
            _world([_obj(area=0.01, offset=60)])))
        out = field_agent._steer_away_angle(
            _world([_obj(area=0.01, offset=60, approaching=True)]))
        self.assertIsNotNone(out)   # approaching counts regardless of size

    def test_far_off_path_ignored(self):
        # |offset| beyond AVOID_CONE_FRAC of half-frame isn't in the way.
        edge = int((FRAME_W / 2) * field_agent.AVOID_CONE_FRAC) + 10
        self.assertIsNone(field_agent._steer_away_angle(
            _world([_obj(area=0.4, offset=edge)])))

    def test_angle_capped(self):
        out = field_agent._steer_away_angle(_world([
            _obj(label=f"o{i}", area=0.5, offset=20 + i) for i in range(5)
        ]))
        self.assertLessEqual(abs(out["angle"]), field_agent.AVOID_MAX_ANGLE)

    def test_dead_center_contributes_nothing(self):
        # No side to prefer - that's the evasion reflex's call.
        self.assertIsNone(field_agent._steer_away_angle(
            _world([_obj(area=0.4, offset=0)])))


class SteerAroundTickTest(unittest.TestCase):
    """The cruising tick bends the heading away from a looming object and
    keeps rolling, instead of driving straight until the reflex trips."""

    def setUp(self):
        self.fa = field_agent.FieldAgent()
        self.bus = self.fa.bus
        self.fa.state = "CRUISING"
        self.fa.last_scan_at = time.time()   # keep the periodic glance quiet
        self.fa.last_wander = time.time()    # and the wander timer

    def _drive(self, world):
        self.fa.latest_world = world
        self.fa.explore_tick()

    def _intents(self):
        return [p["action"] for p in self.bus.of("picarx/intent/move")]

    def test_steers_around_and_keeps_moving(self):
        # The smooth controller alternates primitives (steer tick, then
        # drive tick) through the arbiter's one-intent-per-source channel,
        # so both appear across TWO ticks, with a float angle and a
        # curvature/proximity-scaled speed.
        self._drive(_world([_obj(area=0.3, offset=80)]))
        self._drive(_world([_obj(area=0.3, offset=80)]))
        self.assertEqual(self.fa.state, "CRUISING")        # no evasion
        turns = [a for a in self._intents() if a.get("direction") == "turn"]
        forwards = [a for a in self._intents() if a.get("direction") == "forward"]
        self.assertTrue(turns and turns[-1]["angle"] < 0)  # away from the right
        self.assertIsInstance(turns[-1]["angle"], float)
        self.assertTrue(forwards)
        c = self.fa.steering
        self.assertLessEqual(forwards[-1]["speed"], c.cruise_speed)
        self.assertGreater(forwards[-1]["speed"], 0)
        self.assertIsNotNone(self.fa.avoid_active_angle)

    def test_journal_entry_on_activation_only(self):
        self._drive(_world([_obj(area=0.3, offset=80)]))
        self._drive(_world([_obj(area=0.3, offset=80)]))
        kinds = [p["kind"] for p in self.bus.of("picarx/decision")]
        self.assertEqual(kinds.count("steer_around"), 1)

    def test_clears_when_path_empties(self):
        self._drive(_world([_obj(area=0.3, offset=80)]))
        self.assertIsNotNone(self.fa.avoid_active_angle)
        self._drive(_world([]))
        self.assertIsNone(self.fa.avoid_active_angle)

    def test_emergency_paths_still_outrank_it(self):
        # An approaching object with the ultrasonic NOT clear is the
        # emergency evade's territory, not a gentle steer-around.
        self._drive(_world([_obj(area=0.3, offset=80, approaching=True)],
                           distance=None, distance_stale=True))
        self.assertEqual(self.fa.state, "EVADING")


class FluidSteerBeforeReverseTest(unittest.TestCase):
    """A looming but not-point-blank object with lateral room should be steered
    AROUND (stay CRUISING) rather than triggering a stop-and-reverse - fluid
    driving instead of bump-and-back-out. It still reverses when it's close or
    the distance is unknown/stale."""

    def setUp(self):
        self.fa = field_agent.FieldAgent()
        self.fa.state = "CRUISING"
        self.fa.last_scan_at = time.time()
        self.fa.last_wander = time.time()

    def _drive(self, world):
        self.fa.latest_world = world
        self.fa.explore_tick()

    def test_looming_object_with_room_is_steered_not_reversed(self):
        # Fresh reading in the steer band (30 < d <= 60), object off to the
        # right with room -> a smooth arc, no evasion.
        world = _world([_obj(area=0.3, offset=80, approaching=True)],
                       distance=45, distance_stale=False)
        self._drive(world)
        self.assertEqual(self.fa.state, "CRUISING")           # steered, not EVADING
        turns = [p["action"] for p in self.fa.bus.of("picarx/intent/move")
                 if p["action"].get("direction") == "turn"]
        self.assertTrue(turns and turns[-1]["angle"] < 0)     # away from the right

    def test_close_object_still_reverses(self):
        # Same object but point-blank (< STEER_COMMIT_CM) -> reverse reflex wins.
        self._drive(_world([_obj(area=0.3, offset=80, approaching=True)],
                           distance=22, distance_stale=False))
        self.assertEqual(self.fa.state, "EVADING")

    def test_stale_distance_still_reverses(self):
        # No trustworthy distance -> emergency territory, reverse (don't assume
        # there's room to arc).
        self._drive(_world([_obj(area=0.3, offset=80, approaching=True)],
                           distance=None, distance_stale=True))
        self.assertEqual(self.fa.state, "EVADING")


class EscapeSideHintTest(unittest.TestCase):
    """Evasion swings away from the side the obstacle was seen on."""

    def setUp(self):
        self.fa = field_agent.FieldAgent()
        self.fa.state = "CRUISING"
        self.fa.last_scan_at = time.time()
        self.fa.last_wander = time.time()

    def _stage1_angle(self):
        # Drive the EVADING machine one transition: stage 0 -> 1 picks the
        # escape angle.
        self.fa.state_until = 0.0
        self.fa.evade_stage = 0
        self.fa.explore_tick()
        return self.fa.evade_angle

    def test_vision_evade_sets_hint_away_from_obstacle(self):
        self.fa.latest_world = _world(
            [_obj(area=0.3, offset=80, approaching=True)],
            distance=None, distance_stale=True)
        self.fa.explore_tick()
        self.assertEqual(self.fa.state, "EVADING")
        self.assertEqual(self.fa.evade_away_hint, -30)   # obstacle right -> swing left
        self.assertEqual(self._stage1_angle(), -30)

    def test_hint_is_one_shot(self):
        self.fa._begin_evasion("vision", away_hint=30)
        self.assertEqual(self._stage1_angle(), 30)
        self.assertIsNone(self.fa.evade_away_hint)       # consumed
        # A later evasion without side info falls back to scan bias.
        self.fa.preferred_escape_angle = -30
        self.fa._begin_evasion("ultrasonic")
        self.assertEqual(self._stage1_angle(), -30)

    def test_near_center_obstacle_gives_no_hint(self):
        self.fa.latest_world = _world(
            [_obj(area=0.3, offset=5, approaching=True)],
            distance=None, distance_stale=True)
        self.fa.explore_tick()
        self.assertEqual(self.fa.state, "EVADING")
        self.assertIsNone(self.fa.evade_away_hint)


if __name__ == "__main__":
    unittest.main()
