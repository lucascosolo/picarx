"""The coach-episode miner must name the actual escape TACTIC (full maneuver
shape), not just its first move, so a robot that learned to reverse-then-turn
isn't reported as merely 'starting with backward'. Frequency stays keyed on the
first move so multi-step shapes don't fragment below MIN_FREQUENCY."""
import json
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import pattern_miner  # noqa: E402


def _steps(*dirs):
    return [{"action": {"direction": d}, "duration": 1.0} for d in dirs]


class ManeuverPhraseTest(unittest.TestCase):
    def test_single_step(self):
        self.assertEqual(pattern_miner.maneuver_phrase(_steps("backward")), "reverse")

    def test_multi_step_named_in_order(self):
        self.assertEqual(
            pattern_miner.maneuver_phrase(_steps("backward", "turn", "forward")),
            "reverse, turn out then pull forward")

    def test_consecutive_dupes_collapse(self):
        self.assertEqual(
            pattern_miner.maneuver_phrase(_steps("backward", "backward", "turn")),
            "reverse then turn out")

    def test_empty(self):
        self.assertIsNone(pattern_miner.maneuver_phrase([]))


class MineShapeTest(unittest.TestCase):
    def _db(self, episodes):
        path = os.path.join(tempfile.mkdtemp(), "events.db")
        c = sqlite3.connect(path)
        c.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                  "ts REAL, topic TEXT, payload_json TEXT)")
        t = 1000.0
        for key, dirs, success in episodes:
            c.execute("INSERT INTO events (ts,topic,payload_json) VALUES (?,?,?)",
                      (t, "picarx/coach/episode",
                       json.dumps({"situation_key": key, "steps": _steps(*dirs),
                                   "success": success})))
            t += 1
        c.commit(); c.close()
        return path

    def test_winning_pattern_names_the_full_shape(self):
        # 4 reverse-then-turn escapes that worked -> the tactic is named, not
        # just "backward", and it stays one high-frequency pattern.
        db = self._db([("collision_loop:box", ["backward", "turn"], True)] * 4)
        pats = pattern_miner.mine_patterns(db)
        escape = [p for p in pats if p["condition"] == "stuck:collision_loop:box"]
        self.assertEqual(len(escape), 1)
        self.assertIn("reverse then turn out", escape[0]["outcome"])
        self.assertEqual(escape[0]["frequency"], 4)

    def test_first_move_keeps_frequency_together(self):
        # Mixed continuations after a backward first move still aggregate into
        # one pattern (keyed on the first move), clearing MIN_FREQUENCY.
        db = self._db([("k", ["backward", "turn"], True),
                       ("k", ["backward", "forward"], True),
                       ("k", ["backward"], True)])
        escape = [p for p in pattern_miner.mine_patterns(db)
                  if p["condition"] == "stuck:k"]
        self.assertEqual(len(escape), 1)
        self.assertEqual(escape[0]["frequency"], 3)


if __name__ == "__main__":
    unittest.main()
