import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402  - stubs + sys.path

from spatial_store import (SpatialStore, fingerprint_from_scan,
                           fingerprint_similarity)  # noqa: E402


class FingerprintRichnessTest(unittest.TestCase):
    def _scan(self, order=(0, 1), pans=(-35, 35)):
        sightings = [
            {"pan": pans[0], "labels": [{"name": "chair", "confidence": 0.9,
                                       "area_ratio": 0.20}]},
            {"pan": pans[1], "labels": [{"name": "lamp", "confidence": 0.8,
                                      "area_ratio": 0.10}]},
        ]
        return fingerprint_from_scan([sightings[i] for i in order], 100)

    def test_reordered_rotated_sweep_is_identical(self):
        first = self._scan()
        rotated = self._scan((1, 0), pans=(-55, 55))
        self.assertEqual(first, rotated)
        self.assertEqual(fingerprint_similarity(first, rotated), 1.0)

    def test_metadata_keeps_lateral_and_detector_features(self):
        fp = fingerprint_from_scan([{
            "pan": -10,
            "labels": [{"name": "chair", "confidence": 0.72,
                         "area_ratio": 0.18}],
        }], 100)
        self.assertEqual(fp["labels"], ["l:chair"])
        self.assertEqual(fp["metadata"]["labels"], [{
            "name": "chair", "conf": 0.72, "bin": -1, "area": 0.18,
        }])

    def test_range_change_separates_otherwise_same_scan(self):
        first = self._scan()
        far = fingerprint_from_scan([
            {"pan": -35, "labels": [{"name": "chair", "confidence": 0.9,
                                       "area_ratio": 0.20}]},
            {"pan": 35, "labels": [{"name": "lamp", "confidence": 0.8,
                                      "area_ratio": 0.10}]},
        ], 250)
        self.assertLess(fingerprint_similarity(first, far), 0.70)

    def test_range_change_creates_a_distinct_location(self):
        db = tempfile.mktemp(suffix=".db")
        store = SpatialStore(readonly=False, db_path=db)
        try:
            self.assertTrue(store.match_or_create(self._scan(), now=1.0)["is_new"])
            far = fingerprint_from_scan([
                {"pan": -35, "labels": ["chair"]},
                {"pan": 35, "labels": ["lamp"]},
            ], 250)
            result = store.match_or_create(far, now=2.0)
            self.assertTrue(result["is_new"])
            self.assertEqual(store.location_count(), 2)
        finally:
            try:
                os.remove(db)
            except OSError:
                pass


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


class LocationConfidenceTest(unittest.TestCase):
    def setUp(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.w = SpatialStore(readonly=False, db_path=self.db)

    def tearDown(self):
        try:
            os.remove(self.db)
        except OSError:
            pass

    def test_weak_scan_does_not_identify_an_existing_place(self):
        first = self.w.match_or_create(
            {"labels": ["l:sofa", "r:lamp"], "range": "mid"}, now=1.0)
        result = self.w.match_or_create({"labels": ["l:sofa"], "range": "mid"}, now=2.0)
        self.assertTrue(first["resolved"])
        self.assertEqual(first["candidate_scores"], [])
        self.assertFalse(result["resolved"])
        self.assertEqual(result["reason"], "insufficient_landmarks")
        self.assertEqual(result["candidate_scores"], [{
            "location_id": 1, "similarity": fingerprint_similarity(
                {"labels": ["l:sofa"], "range": "mid"},
                first["fingerprint"]),
        }])
        self.assertEqual(self.w.location_count(), 1)

    def test_near_tied_known_places_remain_unlocalized(self):
        self.w.match_or_create({"labels": ["l:sofa", "r:lamp"], "range": "mid"}, now=1.0)
        self.w.match_or_create({"labels": ["l:sofa", "r:table"], "range": "mid"}, now=2.0)
        result = self.w.match_or_create(
            {"labels": ["l:sofa", "r:lamp", "r:table"], "range": "mid"}, now=3.0)
        self.assertFalse(result["resolved"])
        self.assertEqual(result["reason"], "ambiguous_match")
        self.assertEqual(
            [candidate["location_id"] for candidate in result["candidate_scores"]],
            [1, 2])
        self.assertEqual(
            result["candidate_scores"],
            sorted(result["candidate_scores"],
                   key=lambda candidate: (-candidate["similarity"],
                                          candidate["location_id"])))
        self.assertEqual(self.w.location_count(), 2)

    def test_resolved_result_has_bounded_ordered_candidate_scores(self):
        fingerprints = [
            {"labels": ["l:sofa", "r:lamp"], "range": "mid"},
            {"labels": ["l:sofa", "r:table"], "range": "mid"},
            {"labels": ["l:chair", "r:plant"], "range": "far"},
            {"labels": ["l:book", "r:cup"], "range": "near"},
        ]
        for index, fingerprint in enumerate(fingerprints):
            self.w.match_or_create(fingerprint, now=float(index + 1))

        result = self.w.match_or_create(fingerprints[0], now=10.0)

        self.assertTrue(result["resolved"])
        self.assertEqual(result["reason"], "matched")
        self.assertLessEqual(len(result["candidate_scores"]), 3)
        self.assertEqual(
            [set(candidate) for candidate in result["candidate_scores"]],
            [{"location_id", "similarity"}] * 3)
        self.assertEqual(
            result["candidate_scores"],
            sorted(result["candidate_scores"],
                   key=lambda candidate: (-candidate["similarity"],
                                          candidate["location_id"])))


if __name__ == "__main__":
    unittest.main()
