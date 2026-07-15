import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402  - stubs + sys.path

from spatial_store import SpatialStore  # noqa: E402


class SpatialStoreVetoTest(unittest.TestCase):
    def setUp(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.w = SpatialStore(readonly=False, db_path=self.db)
        self.w.conn.execute(
            "INSERT INTO locations (label, fingerprint_json, discovered_at, "
            "last_visited_at, visit_count, veto_count) VALUES (?,?,?,?,?,?)",
            ("place 4 (chair)", "{}", 1.0, 2.0, 3, 5))
        self.w.conn.commit()

    def tearDown(self):
        try:
            os.remove(self.db)
        except OSError:
            pass

    def _veto(self):
        return self.w.get_location(1)["veto_count"]

    def test_note_veto_increments(self):
        self.w.note_veto(1)
        self.assertEqual(self._veto(), 6)

    def test_relax_veto_decrements_by_one(self):
        self.w.relax_veto(1)
        self.assertEqual(self._veto(), 4)

    def test_relax_veto_eventually_below_threshold(self):
        # VETO_PRONE_THRESHOLD is 3; from 5 it takes 3 clean re-tests.
        for _ in range(3):
            self.w.relax_veto(1)
        self.assertEqual(self._veto(), 2)
        self.assertLess(self._veto(), 3)

    def test_relax_veto_floors_at_zero(self):
        for _ in range(20):
            self.w.relax_veto(1)
        self.assertEqual(self._veto(), 0)

    def test_relax_veto_custom_amount(self):
        self.w.relax_veto(1, amount=2)
        self.assertEqual(self._veto(), 3)

    def test_relax_veto_unknown_location_is_noop(self):
        self.w.relax_veto(999)  # must not raise
        self.assertEqual(self._veto(), 5)

    def test_readonly_store_refuses_relax(self):
        ro = SpatialStore(readonly=True, db_path=self.db)
        with self.assertRaises(RuntimeError):
            ro.relax_veto(1)

    def test_readonly_reader_sees_writes(self):
        self.w.relax_veto(1)
        ro = SpatialStore(readonly=True, db_path=self.db)
        self.assertEqual(ro.get_location(1)["veto_count"], 4)


if __name__ == "__main__":
    unittest.main()
