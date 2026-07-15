import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

from spatial_store import SpatialStore  # noqa: E402
import location_graph  # noqa: E402


class LocationGraphLoopTest(unittest.TestCase):
    """The map half of the hypothesis loop: a VetoProneLocationProbe that
    resolves 'maybe_clear' should ease that location's veto_count."""

    def setUp(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.w = SpatialStore(readonly=False, db_path=self.db)
        self.w.conn.execute(
            "INSERT INTO locations (label, fingerprint_json, discovered_at, "
            "last_visited_at, visit_count, veto_count) VALUES (?,?,?,?,?,?)",
            ("place 4 (chair)", "{}", 1.0, 2.0, 3, 5))
        self.w.conn.commit()
        # Build a LocationGraph without its default-path DB / real Bus.
        self.lg = location_graph.LocationGraph.__new__(location_graph.LocationGraph)
        self.lg.store = self.w
        self.lg.lock = threading.Lock()
        self.lg.bus = harness.FakeBus()

    def tearDown(self):
        try:
            os.remove(self.db)
        except OSError:
            pass

    def _veto(self):
        return self.w.get_location(1)["veto_count"]

    def _maybe_clear(self, **overrides):
        payload = {"question": "is_veto_prone_area_still_blocked",
                   "resolution": "maybe_clear", "location_id": 1}
        payload.update(overrides)
        self.lg.on_hypothesis(payload)

    def test_maybe_clear_decrements(self):
        self._maybe_clear()
        self.assertEqual(self._veto(), 4)

    def test_repeated_maybe_clear_falls_below_threshold(self):
        for _ in range(3):
            self._maybe_clear()
        self.assertLess(self._veto(), 3)  # VETO_PRONE_THRESHOLD

    def test_still_blocked_ignored(self):
        self._maybe_clear(resolution="still_blocked")
        self.assertEqual(self._veto(), 5)

    def test_other_hypothesis_type_ignored(self):
        self.lg.on_hypothesis({
            "question": "ultrasonic_obstacle_vs_empty_vision",
            "resolution": "phantom_reading", "location_id": 1})
        self.assertEqual(self._veto(), 5)

    def test_location_id_falls_back_to_location_block(self):
        self._maybe_clear(location_id=None, location={"id": 1, "label": "x"})
        self.assertEqual(self._veto(), 4)

    def test_missing_location_id_is_noop(self):
        self._maybe_clear(location_id=None)
        self.assertEqual(self._veto(), 5)

    def test_maybe_clear_journals_a_map_update(self):
        self._maybe_clear()
        decisions = self.lg.bus.of("picarx/decision")
        self.assertEqual(len(decisions), 1)
        d = decisions[0]
        self.assertEqual(d["source"], "location_graph")
        self.assertEqual(d["kind"], "map_update")
        self.assertEqual(d["choice"]["change"], "veto_relaxed")
        self.assertEqual(d["choice"]["veto_count"], 4)
        self.assertIn("place 4 (chair)", d["reason"])   # human-readable why
        self.assertEqual(d["location"], {"id": 1, "label": "place 4 (chair)"})

    def test_no_journal_when_already_floored(self):
        # A location already at 0 has nothing to relax -> no journal spam.
        self.w.conn.execute("UPDATE locations SET veto_count = 0 WHERE id = 1")
        self.w.conn.commit()
        self._maybe_clear()
        self.assertEqual(self.lg.bus.of("picarx/decision"), [])
        self.assertEqual(self._veto(), 0)

    def test_still_blocked_writes_no_journal(self):
        self._maybe_clear(resolution="still_blocked")
        self.assertEqual(self.lg.bus.of("picarx/decision"), [])

    def test_delivered_through_bus_subscription(self):
        # Wire the real subscription and deliver like the broker would.
        bus = harness.FakeBus()
        self.lg.bus = bus
        bus.subscribe("picarx/exploration/hypothesis", self.lg.on_hypothesis)
        bus.deliver("picarx/exploration/hypothesis", {
            "question": "is_veto_prone_area_still_blocked",
            "resolution": "maybe_clear", "location_id": 1})
        self.assertEqual(self._veto(), 4)


if __name__ == "__main__":
    unittest.main()
