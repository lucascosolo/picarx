"""Importing a picarx-training knowledge pack must MERGE into the robot's
own learning, never clobber it: bandit arm records add together, unseen
situations/arms are adopted, real-world-only knowledge is left alone, and
transferable facts/patterns land in semantic.db through the normal dedup."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402  - stubs + sys.path

import import_training  # noqa: E402
from semantic_store import SemanticStore  # noqa: E402


def _arm(direction, s, f, dur=1.0, **extra):
    steps = [{"action": {"direction": direction}, "duration": dur}]
    return json.dumps(steps), {"steps": [{"action": {"direction": direction,
                                                      "speed": 25}, "duration": dur}],
                               "rationale": f"{direction} arm", "successes": s,
                               "failures": f, "last_updated": 100, **extra}


def _situation(*arms):
    return {"arms": dict(arms)}


class MergePolicyTest(unittest.TestCase):
    def test_fresh_robot_adopts_everything(self):
        incoming = {"novel_object:bottle": _situation(_arm("turn", 3, 0))}
        merged, stats = import_training.merge_policy({}, incoming)
        self.assertEqual(stats["situations_added"], 1)
        self.assertEqual(stats["arms_added"], 1)
        self.assertIn("novel_object:bottle", merged)

    def test_shared_arm_sums_win_loss_records(self):
        sig, arm = _arm("backward", 7, 1)
        base = {"k": _situation((sig, {**arm, "successes": 2, "failures": 1,
                                        "rationale": "real reverse"}))}
        merged, stats = import_training.merge_policy(base, {"k": _situation((sig, arm))})
        got = merged["k"]["arms"][sig]
        self.assertEqual((got["successes"], got["failures"]), (9, 2))
        # base's steps/rationale are kept, not overwritten by the incoming arm
        self.assertEqual(got["rationale"], "real reverse")
        self.assertEqual(stats["arms_reinforced"], 1)
        self.assertEqual(stats["situations_merged"], 1)

    def test_real_world_only_arm_is_untouched_and_new_arm_added(self):
        base_sig, base_arm = _arm("stop", 4, 0)
        inc_sig, inc_arm = _arm("forward", 1, 5)
        base = {"k": _situation((base_sig, base_arm))}
        merged, stats = import_training.merge_policy(
            base, {"k": _situation((inc_sig, inc_arm))})
        arms = merged["k"]["arms"]
        self.assertEqual(arms[base_sig]["successes"], 4)   # untouched
        self.assertIn(inc_sig, arms)                       # adopted
        self.assertEqual(stats["arms_added"], 1)

    def test_does_not_mutate_base(self):
        sig, arm = _arm("backward", 3, 0)
        base = {"k": _situation((sig, {**arm, "successes": 2}))}
        import_training.merge_policy(base, {"k": _situation((sig, arm))})
        self.assertEqual(base["k"]["arms"][sig]["successes"], 2)

    def test_base_embedding_wins_incoming_fills_gap(self):
        sig, arm = _arm("backward", 1, 0)
        base = {"k": {**_situation((sig, arm)), "embedding": [0.1, 0.2]}}
        inc = {"k": {**_situation((sig, arm)), "embedding": [0.9, 0.9]},
               "k2": {**_situation((sig, arm)), "embedding": [0.5, 0.5]}}
        merged, _ = import_training.merge_policy(base, inc)
        self.assertEqual(merged["k"]["embedding"], [0.1, 0.2])   # base kept
        self.assertEqual(merged["k2"]["embedding"], [0.5, 0.5])  # adopted

    def test_malformed_entries_skipped(self):
        incoming = {"legacy": {"no_arms": True}, "bad": 42,
                    "ok": _situation(_arm("turn", 1, 0))}
        merged, stats = import_training.merge_policy({}, incoming)
        self.assertNotIn("legacy", merged)
        self.assertNotIn("bad", merged)
        self.assertEqual(stats["situations_added"], 1)

    def test_demonstrations_concatenated_and_capped(self):
        base = {"_demonstrations": [{"ts": i, "steps": []} for i in range(6)]}
        incoming = {"_demonstrations": [{"ts": 100 + i, "steps": []} for i in range(9)]}
        merged, stats = import_training.merge_policy(base, incoming)
        demos = merged["_demonstrations"]
        self.assertEqual(len(demos), import_training.MAX_DEMONSTRATIONS)
        # freshest kept (highest ts survive the cap)
        self.assertEqual(demos[-1]["ts"], 108)
        self.assertEqual(stats["demonstrations_added"], 9)


class SemanticImportTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.pack = tempfile.mkdtemp()

    def _write_pack(self, facts, patterns):
        with open(os.path.join(self.pack, "navigation_facts.json"), "w") as f:
            json.dump({"schema": 1, "facts": facts, "patterns": patterns}, f)

    def test_facts_and_patterns_land_in_semantic_db(self):
        self._write_pack(
            facts=[{"subject": "escape tactics", "fact": "reverse first works",
                    "confidence": 0.7, "source": "training"}],
            patterns=[{"condition": "veto:obstacle", "outcome": "bursts",
                       "frequency": 5, "confidence": 0.8}])
        import_training.import_navigation_facts(self.pack, self.dir)
        store = SemanticStore(readonly=True, db_path=os.path.join(self.dir, "semantic.db"))
        facts = store.search_facts("reverse")
        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0]["subject"], "escape tactics")
        pats = store.top_patterns()
        self.assertEqual(pats[0]["condition"], "veto:obstacle")

    def test_dry_run_writes_nothing(self):
        self._write_pack(facts=[{"subject": "s", "fact": "f", "confidence": 0.5}],
                         patterns=[])
        import_training.import_navigation_facts(self.pack, self.dir, dry_run=True)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "semantic.db")))

    def test_reimport_reinforces_not_duplicates(self):
        self._write_pack(facts=[{"subject": "s", "fact": "f", "confidence": 0.6}],
                         patterns=[])
        import_training.import_navigation_facts(self.pack, self.dir)
        import_training.import_navigation_facts(self.pack, self.dir)
        store = SemanticStore(readonly=True, db_path=os.path.join(self.dir, "semantic.db"))
        self.assertEqual(store.fact_count(), 1)   # deduped on (subject, fact)

    def test_missing_pack_file_is_soft(self):
        # no navigation_facts.json in an empty dir - must not raise
        import_training.import_navigation_facts(self.pack, self.dir)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "semantic.db")))


class CoachPolicyFileImportTest(unittest.TestCase):
    def test_merge_written_atomically_to_data_dir(self):
        pack = tempfile.mkdtemp()
        data = tempfile.mkdtemp()
        sig, arm = _arm("backward", 5, 1)
        with open(os.path.join(pack, "coach_policy.json"), "w") as f:
            json.dump({"k": _situation((sig, arm))}, f)
        import_training.import_coach_policy(pack, data)
        with open(os.path.join(data, "coach_policy.json")) as f:
            written = json.load(f)
        self.assertEqual(written["k"]["arms"][sig]["successes"], 5)


if __name__ == "__main__":
    unittest.main()
