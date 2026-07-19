"""Multi-page web console: the server-side helpers behind the new pages -
objects-in-view (Training relabelling), the semantic-memory facts view, the
Config page's tree/help, and the page/asset route table."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import robot_config  # noqa: E402
import web_console  # noqa: E402
from semantic_store import SemanticStore  # noqa: E402


class ObjectsSnapshotTest(unittest.TestCase):
    def _world(self, items, stale=False):
        return {"objects": {"items": items, "stale": stale}}

    def test_lists_fresh_tracked_objects_with_ids(self):
        world = self._world([
            {"id": "object_1", "label": "chair", "confidence": 0.82,
             "alt_label": "sofa", "area_ratio": 0.2},
            {"id": "object_2", "label": "bottle", "confidence": 0.6},
        ])
        objs = web_console.objects_snapshot(world)
        self.assertEqual([o["id"] for o in objs], ["object_1", "object_2"])
        self.assertEqual(objs[0]["label"], "chair")
        self.assertEqual(objs[0]["alt_label"], "sofa")

    def test_stale_objects_give_nothing(self):
        self.assertEqual(web_console.objects_snapshot(
            self._world([{"id": "o1", "label": "chair"}], stale=True)), [])

    def test_items_without_id_or_label_are_skipped(self):
        world = self._world([{"label": "chair"}, {"id": "o2"},
                             {"id": "o3", "label": "mug", "confidence": 0.9}])
        objs = web_console.objects_snapshot(world)
        self.assertEqual([o["id"] for o in objs], ["o3"])

    def test_missing_objects_is_safe(self):
        self.assertEqual(web_console.objects_snapshot({}), [])


class FactsSnapshotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        store = SemanticStore(readonly=False,
                              db_path=os.path.join(self.tmp, "semantic.db"))
        store.upsert_fact("chair", "I have seen a chair", confidence=0.6)
        store.upsert_fact("kitchen", "the kitchen has a table", confidence=0.7)
        self._orig = web_console.SEMANTIC
        web_console.SEMANTIC = store

    def tearDown(self):
        web_console.SEMANTIC = self._orig

    def test_recent_facts_when_no_query(self):
        snap = web_console.facts_snapshot()
        self.assertEqual(snap["count"], 2)
        subjects = {f["subject"] for f in snap["facts"]}
        self.assertEqual(subjects, {"chair", "kitchen"})

    def test_search_filters(self):
        snap = web_console.facts_snapshot("chair")
        self.assertTrue(all("chair" in (f["subject"] + f["fact"]) for f in snap["facts"]))
        self.assertTrue(snap["facts"])

    def test_failsoft_on_broken_store(self):
        web_console.SEMANTIC = object()   # no methods -> caught
        self.assertEqual(web_console.facts_snapshot(), {"facts": [], "count": 0})


class ConfigDataTest(unittest.TestCase):
    def setUp(self):
        self._orig_path = robot_config.CONFIG_PATH
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "config.json")
        with open(self.path, "w") as f:
            json.dump({
                "_readme": ["General note about the file.",
                            "audio.gain: mic amplification for quiet USB mics"],
                "audio": {"gain": 12.0, "espeak_voice": "mb-us1"},
                "web_console": {"port": 8088},
            }, f)
        robot_config.CONFIG_PATH = self.path
        robot_config.reload()

    def tearDown(self):
        robot_config.CONFIG_PATH = self._orig_path
        robot_config.reload()

    def test_config_data_excludes_readme_and_keeps_sections(self):
        d = web_console.config_data()
        self.assertIn("audio", d["config"])
        self.assertNotIn("_readme", d["config"])
        self.assertEqual(d["config"]["web_console"]["port"], 8088)
        self.assertTrue(d["note"])

    def test_help_is_parsed_from_readme_knob_lines(self):
        help_map = web_console._config_help()
        self.assertIn("audio.gain", help_map)
        self.assertIn("amplification", help_map["audio.gain"])
        # The general note (no 'section.key:' shape) is not a knob doc.
        self.assertNotIn("General note about the file.", help_map.values())


class RouteTableTest(unittest.TestCase):
    def test_every_page_and_asset_file_exists(self):
        # WEB_UI_DIR is the robot's absolute deploy path; the files live under
        # the repo here, so resolve against that instead.
        web_ui = os.path.join(harness.LAYER_B, "web_ui")
        for fname in list(web_console.PAGES.values()) + \
                [a[0] for a in web_console.ASSETS.values()]:
            self.assertTrue(os.path.exists(os.path.join(web_ui, fname)),
                            f"{fname} referenced by a route but missing on disk")

    def test_core_pages_are_routed(self):
        for path in ("/", "/drive", "/training", "/people", "/audio", "/config"):
            self.assertIn(path, web_console.PAGES)


if __name__ == "__main__":
    unittest.main()
