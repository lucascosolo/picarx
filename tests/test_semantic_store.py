import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

from semantic_store import SemanticStore  # noqa: E402


class SemanticStoreTest(unittest.TestCase):
    def setUp(self):
        self.db = tempfile.mktemp(suffix=".db")
        self.s = SemanticStore(readonly=False, db_path=self.db)

    def tearDown(self):
        try:
            os.remove(self.db)
        except OSError:
            pass

    # ---- upsert_fact / belief revision ----

    def test_upsert_returns_id_and_dedupes(self):
        a = self.s.upsert_fact("kitchen", "the floor is tile", 0.6)
        b = self.s.upsert_fact("kitchen", "the floor is tile", 0.9)  # same -> reinforce
        self.assertEqual(a, b)
        rows = self.s.facts_for("kitchen")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["seen_count"], 2)
        self.assertAlmostEqual(rows[0]["confidence"], 0.9)  # MAX(confidence)

    def test_supersede_retires_old_fact(self):
        old = self.s.upsert_fact("kitchen", "the door to the hall is open", 0.8)
        new = self.s.upsert_fact("kitchen", "the door to the hall is closed", 0.9,
                                 supersedes=old)
        active = self.s.facts_for("kitchen")
        self.assertEqual([f["fact"] for f in active], ["the door to the hall is closed"])
        history = self.s.facts_for("kitchen", include_superseded=True)
        retired = [f for f in history if f["status"] == "superseded"]
        self.assertEqual(len(retired), 1)
        self.assertEqual(retired[0]["id"], old)
        self.assertEqual(retired[0]["superseded_by_id"], new)

    def test_reinforcing_reactivates_superseded_fact(self):
        old = self.s.upsert_fact("kitchen", "door open", 0.8)
        self.s.upsert_fact("kitchen", "door closed", 0.9, supersedes=old)
        # world re-asserts the old fact -> it comes back active
        self.s.upsert_fact("kitchen", "door open", 0.7)
        active = sorted(f["fact"] for f in self.s.facts_for("kitchen"))
        self.assertIn("door open", active)

    def test_supersede_self_is_noop(self):
        fid = self.s.upsert_fact("x", "fact", 0.5)
        # asking a fact to supersede itself must not retire it
        self.s.upsert_fact("x", "fact", 0.5, supersedes=fid)
        self.assertEqual(len(self.s.facts_for("x")), 1)

    # ---- replace_subject (self-model snapshot semantics) ----

    def test_replace_subject_swaps_snapshot(self):
        self.s.replace_subject("self", [("I map places.", 0.6),
                                        ("I back away first.", 0.75)])
        self.assertEqual(len(self.s.facts_for("self")), 2)
        # next pass drops one, adds one
        self.s.replace_subject("self", [("I map places.", 0.6),
                                        ("I get stuck often.", 0.7)])
        active = sorted(f["fact"] for f in self.s.facts_for("self"))
        self.assertEqual(active, ["I get stuck often.", "I map places."])
        # the dropped one is retired, not deleted
        history = self.s.facts_for("self", include_superseded=True)
        self.assertTrue(any(f["fact"] == "I back away first."
                            and f["status"] == "superseded" for f in history))

    def test_replace_subject_empty_is_noop(self):
        self.s.replace_subject("self", [("I map places.", 0.6)])
        self.s.replace_subject("self", [])  # transient empty synthesis
        self.assertEqual(len(self.s.facts_for("self")), 1)

    def test_replace_subject_returns_kept_ids(self):
        kept = self.s.replace_subject("self", [("a", 0.5), ("b", 0.5)])
        self.assertEqual(len(kept), 2)

    # ---- reader fail-soft ----

    def test_reader_on_missing_db_is_empty(self):
        ro = SemanticStore(readonly=True, db_path=tempfile.mktemp(suffix=".db"))
        self.assertEqual(ro.facts_for("self"), [])
        self.assertEqual(ro.recent_facts(), [])

    def test_readonly_refuses_writes(self):
        ro = SemanticStore(readonly=True, db_path=self.db)
        with self.assertRaises(RuntimeError):
            ro.upsert_fact("x", "y")
        with self.assertRaises(RuntimeError):
            ro.replace_subject("self", [("a", 0.5)])


if __name__ == "__main__":
    unittest.main()
