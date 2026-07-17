import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import speech_match  # noqa: E402
from spatial_store import SpatialStore  # noqa: E402


class SightingsStoreTest(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "spatial.db")
        self.w = SpatialStore(readonly=False, db_path=self.db)
        self.loc = self.w.match_or_create({"labels": ["c:sofa"], "range": "mid"},
                                          now=1000.0)

    def test_note_and_query_sightings(self):
        self.w.note_sightings(self.loc["id"], ["bottle", "sofa"], now=1000.0)
        self.w.note_sightings(self.loc["id"], ["bottle"], now=2000.0)
        places = self.w.object_locations("bottle")
        self.assertEqual(len(places), 1)
        self.assertEqual(places[0]["times_seen"], 2)
        self.assertEqual(places[0]["last_seen"], 2000.0)
        self.assertEqual(sorted(self.w.sighting_labels()), ["bottle", "sofa"])

    def test_location_objects_orders_by_frequency(self):
        self.w.note_sightings(self.loc["id"], ["bottle", "chair"], now=1000.0)
        self.w.note_sightings(self.loc["id"], ["chair"], now=2000.0)
        objs = self.w.location_objects(self.loc["id"])
        self.assertEqual(objs[0]["label"], "chair")
        self.assertEqual(objs[0]["times_seen"], 2)

    def test_unknown_label_returns_empty(self):
        self.assertEqual(self.w.object_locations("unicorn"), [])

    def test_reader_failsoft_without_db(self):
        r = SpatialStore(readonly=True, db_path="/nonexistent/nowhere.db")
        self.assertEqual(r.object_locations("bottle"), [])
        self.assertEqual(r.sighting_labels(), [])
        self.assertIsNone(r.find_location_by_name("kitchen"))


class RenameAndLookupTest(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "spatial.db")
        self.w = SpatialStore(readonly=False, db_path=self.db)
        self.loc = self.w.match_or_create({"labels": ["c:sofa"], "range": "mid"},
                                          now=1000.0)

    def test_rename_location(self):
        new = self.w.rename_location(self.loc["id"], "the kitchen")
        self.assertEqual(new, "the kitchen")
        self.assertEqual(self.w.get_location(self.loc["id"])["label"], "the kitchen")

    def test_rename_unknown_location_is_none(self):
        self.assertIsNone(self.w.rename_location(999, "nowhere"))

    def test_rename_empty_name_is_none(self):
        self.assertIsNone(self.w.rename_location(self.loc["id"], "  "))

    def test_find_by_name_exact_and_substring(self):
        self.w.rename_location(self.loc["id"], "kitchen")
        self.assertEqual(self.w.find_location_by_name("kitchen")["id"], self.loc["id"])
        # spoken query with trailing words still finds it (substring)
        self.assertEqual(
            self.w.find_location_by_name("kitchen please")["id"], self.loc["id"])
        self.assertIsNone(self.w.find_location_by_name("garage"))

    def test_find_matches_auto_label(self):
        # auto labels look like "place 1 (sofa)" - "sofa" should find it
        self.assertEqual(
            self.w.find_location_by_name("sofa")["id"], self.loc["id"])


class BestLabelMatchTest(unittest.TestCase):
    LABELS = ["bottle", "diningtable", "tvmonitor"]

    def test_exact(self):
        self.assertEqual(speech_match.best_label_match("bottle", self.LABELS), "bottle")

    def test_substring(self):
        self.assertEqual(speech_match.best_label_match("dining table"[:6], self.LABELS),
                         "diningtable")

    def test_fuzzy(self):
        self.assertEqual(speech_match.best_label_match("bottel", self.LABELS), "bottle")

    def test_unknown_is_none(self):
        self.assertIsNone(speech_match.best_label_match("giraffe", self.LABELS))
        self.assertIsNone(speech_match.best_label_match("", self.LABELS))
        self.assertIsNone(speech_match.best_label_match("bottle", []))


if __name__ == "__main__":
    unittest.main()
