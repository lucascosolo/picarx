"""Online learning intake: newly-trained learning is folded into the LIVE
robot WITHOUT the trainer becoming a second writer - it's routed through the
owning modules over the bus. coach.py stays the sole writer of
coach_policy.json (picarx/coach/adopt), reflection.py the sole writer of
semantic.db (picarx/memory/pattern). The reality gap is respected: sim learning
may add or refresh an arm, never retire a real one."""
import json
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import coach  # noqa: E402
import reflection  # noqa: E402
from semantic_store import SemanticStore  # noqa: E402

T0 = 7000.0


def _arm(direction, s, f, dur=1.0):
    """(signature, arm_dict) shaped like the coach's own arms."""
    sig = json.dumps([{"action": {"direction": direction}, "duration": dur}],
                     sort_keys=True)
    return sig, {"steps": [{"action": {"direction": direction, "speed": 25},
                            "duration": dur}], "rationale": f"{direction} arm",
                 "successes": s, "failures": f, "last_updated": T0}


def _situation(*arms):
    return {"arms": dict(arms)}


class CoachAdoptIntakeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._orig = (coach.DATA_DIR, coach.COACH_POLICY_PATH)
        coach.DATA_DIR = self.tmp
        coach.COACH_POLICY_PATH = os.path.join(self.tmp, "coach_policy.json")
        self.c = coach.Coach.__new__(coach.Coach)
        self.c.bus = harness.FakeBus()
        self.c.lock = threading.Lock()
        self.c.policy = {}

    def tearDown(self):
        coach.DATA_DIR, coach.COACH_POLICY_PATH = self._orig

    def _adopt(self, policy, **extra):
        self.c.on_adopt({"coach_policy": policy, **extra})

    # ---- persistence ----

    def test_adopt_updates_and_persists_live_policy(self):
        sig, seed = _arm("backward", 7, 1)          # what the robot exported
        self.c.policy = {"k": _situation((sig, seed))}
        _, refined = _arm("backward", 10, 2)         # sim sharpened it
        self._adopt({"k": _situation((sig, refined))})   # default mode = adopt
        got = self.c.policy["k"]["arms"][sig]
        self.assertEqual((got["successes"], got["failures"]), (10, 2))  # replaced
        self.assertTrue(got["trained_in_sim"])
        # and it hit disk - coach is the sole writer, straight through _save_policy
        reloaded = coach.Coach._load_policy(self.c)
        self.assertEqual(reloaded["k"]["arms"][sig]["successes"], 10)

    def test_default_mode_is_adopt(self):
        sig, seed = _arm("backward", 7, 1)
        self.c.policy = {"k": _situation((sig, seed))}
        _, refined = _arm("backward", 10, 2)
        self._adopt({"k": _situation((sig, refined))})   # no "mode" key
        self.assertEqual(self.c.policy["k"]["arms"][sig]["failures"], 2)  # not 3

    def test_merge_mode_sums_shared_arm(self):
        sig, base = _arm("backward", 2, 1)
        self.c.policy = {"k": _situation((sig, base))}
        _, inc = _arm("backward", 7, 1)
        self._adopt({"k": _situation((sig, inc))}, mode="merge")
        got = self.c.policy["k"]["arms"][sig]
        self.assertEqual((got["successes"], got["failures"]), (9, 2))     # summed

    def test_adopt_adds_unseen_situations_and_arms(self):
        self.c.policy = {}
        self._adopt({"novel_object:bottle": _situation(_arm("turn", 3, 0))})
        self.assertIn("novel_object:bottle", self.c.policy)
        arm = list(self.c.policy["novel_object:bottle"]["arms"].values())[0]
        self.assertTrue(arm["trained_in_sim"])

    def test_malformed_payload_is_soft(self):
        self.c.policy = {"k": _situation(_arm("stop", 1, 0))}
        before = json.dumps(self.c.policy, sort_keys=True)
        self._adopt(None)                      # no coach_policy
        self.c.on_adopt({"coach_policy": 42})  # wrong type
        self.c.on_adopt({})                    # missing key
        self.assertEqual(json.dumps(self.c.policy, sort_keys=True), before)

    # ---- reality-gap guard: sim never retires a real arm ----

    def test_sim_arm_is_never_retired(self):
        # An arm with a terrible record that WOULD normally be retired, but it
        # carries sim counts, so retirement must leave it in place.
        real_sig, real = _arm("forward", 5, 0)     # a keeper, keeps >= MIN_ARMS
        sim_sig, sim = _arm("backward", 0, 9)      # awful, but sim-derived
        sim["trained_in_sim"] = True
        entry = _situation((real_sig, real), (sim_sig, sim))
        self.c.policy = {"k": entry}
        with self.c.lock:
            self.c._maybe_retire_arm(self.c.policy["k"], sim_sig)
        self.assertIn(sim_sig, self.c.policy["k"]["arms"])   # survived

    def test_real_arm_still_retires(self):
        # Same shape, but the failing arm is purely real -> it IS culled, so the
        # guard is specific to sim arms, not a blanket "never retire".
        # (Retirement needs MIN_ARMS_BEFORE_EXPLOIT arms to REMAIN, so start
        # with two keepers alongside the bad one.)
        k1_sig, k1 = _arm("forward", 5, 0)
        k2_sig, k2 = _arm("stop", 4, 0)
        bad_sig, bad = _arm("backward", 0, 9)      # no trained_in_sim tag
        self.c.policy = {"k": _situation((k1_sig, k1), (k2_sig, k2),
                                         (bad_sig, bad))}
        with self.c.lock:
            self.c._maybe_retire_arm(self.c.policy["k"], bad_sig)
        self.assertNotIn(bad_sig, self.c.policy["k"]["arms"])  # retired

    def test_adopt_over_a_real_arm_shields_it_from_later_retirement(self):
        # End to end: a real arm (5/0) is refined-in-sim to a poor 0/9 and
        # adopted. Even after a real failure it must NOT be retired, because the
        # adopted counts are sim-tagged - sim can't delete real learning.
        sig, real = _arm("backward", 5, 0)
        keeper_sig, keeper = _arm("forward", 4, 0)
        self.c.policy = {"k": _situation((sig, real), (keeper_sig, keeper))}
        _, refined = _arm("backward", 0, 9)
        self._adopt({"k": _situation((sig, refined))})
        with self.c.lock:
            arm = self.c.policy["k"]["arms"][sig]
            arm["failures"] += 1                    # a subsequent real failure
            self.c._maybe_retire_arm(self.c.policy["k"], sig)
        self.assertIn(sig, self.c.policy["k"]["arms"])   # shielded


class ReflectionPatternIntakeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.r = reflection.Reflection.__new__(reflection.Reflection)
        self.r.bus = harness.FakeBus()
        self.r.lock = threading.Lock()
        self.r.store = SemanticStore(
            readonly=False, db_path=os.path.join(self.tmp, "semantic.db"))

    def test_pattern_intake_persists_via_reflection(self):
        self.r.on_pattern({"condition": "stuck:box_corner",
                           "outcome": "reverse-first works",
                           "frequency": 5, "confidence": 0.8})
        pats = self.r.store.top_patterns()
        self.assertEqual(pats[0]["condition"], "stuck:box_corner")
        self.assertEqual(pats[0]["frequency"], 5)

    def test_pattern_intake_replaces_not_duplicates(self):
        self.r.on_pattern({"condition": "c", "outcome": "o",
                           "frequency": 3, "confidence": 0.7})
        self.r.on_pattern({"condition": "c", "outcome": "o",
                           "frequency": 9, "confidence": 0.9})
        pats = self.r.store.top_patterns()
        self.assertEqual(len(pats), 1)              # (condition, outcome) unique
        self.assertEqual(pats[0]["frequency"], 9)   # re-mine replaced it

    def test_malformed_pattern_is_soft(self):
        self.r.on_pattern({"condition": "", "outcome": ""})   # empty
        self.r.on_pattern({"condition": "c", "outcome": "o",
                           "frequency": "oops", "confidence": 0.5})  # bad type
        self.assertEqual(self.r.store.top_patterns(), [])


if __name__ == "__main__":
    unittest.main()
