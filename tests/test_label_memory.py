"""On-board visual label memory: similarity, teaching/merging, matching,
eviction, persistence, and the detector->memory resolution tier."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import label_memory  # noqa: E402
from label_memory import LabelMemory, cosine, resolve_label  # noqa: E402


class CosineTest(unittest.TestCase):
    def test_identical_is_one(self):
        self.assertAlmostEqual(cosine([1, 2, 3], [1, 2, 3]), 1.0)

    def test_orthogonal_is_zero(self):
        self.assertEqual(cosine([1, 0], [0, 1]), 0.0)

    def test_degenerate_inputs_are_zero(self):
        self.assertEqual(cosine([], [1]), 0.0)
        self.assertEqual(cosine([1, 2], [1, 2, 3]), 0.0)
        self.assertEqual(cosine([0, 0], [1, 1]), 0.0)


class LabelMemoryTest(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(tempfile.mkdtemp(), "label_memory.json")
        self.m = LabelMemory(path=self.path, match_threshold=0.9)

    def test_remember_then_match(self):
        self.m.remember([1.0, 0.0, 0.0], "mug", "user")
        hit = self.m.match([1.0, 0.01, 0.0])
        self.assertIsNotNone(hit)
        self.assertEqual(hit[0], "mug")
        self.assertEqual(hit[2], "user")

    def test_weak_resemblance_is_not_a_match(self):
        self.m.remember([1.0, 0.0, 0.0], "mug", "user")
        self.assertIsNone(self.m.match([0.3, 1.0, 0.0]))

    def test_reteaching_same_object_merges(self):
        self.m.remember([1.0, 0.0], "mug", "user")
        self.m.remember([0.98, 0.02], "mug", "user")  # near-identical, same label
        self.assertEqual(len(self.m), 1)
        self.assertEqual(self.m.entries[0]["count"], 2)

    def test_more_trusted_source_upgrades_on_merge(self):
        self.m.remember([1.0, 0.0], "mug", "llm")
        self.m.remember([1.0, 0.0], "mug", "user")
        self.assertEqual(self.m.entries[0]["source"], "user")

    def test_empty_label_or_sig_is_ignored(self):
        self.assertFalse(self.m.remember([1.0, 0.0], "  ", "user"))
        self.assertFalse(self.m.remember([], "mug", "user"))
        self.assertEqual(len(self.m), 0)

    def test_persists_across_instances(self):
        self.m.remember([1.0, 0.0, 0.0], "watering can", "user")
        reloaded = LabelMemory(path=self.path, match_threshold=0.9)
        self.assertEqual(reloaded.match([1.0, 0.0, 0.0])[0], "watering can")

    def test_eviction_keeps_trusted_entries(self):
        label_memory_orig = label_memory.MAX_ENTRIES
        label_memory.MAX_ENTRIES = 2
        try:
            self.m.remember([1, 0, 0], "a", "llm")
            self.m.remember([0, 1, 0], "b", "user")
            self.m.remember([0, 0, 1], "c", "user")  # forces an eviction
            labels = {e["label"] for e in self.m.entries}
            self.assertEqual(len(self.m), 2)
            self.assertNotIn("a", labels)  # the least-trusted (llm) went first
        finally:
            label_memory.MAX_ENTRIES = label_memory_orig


class ResolveLabelTest(unittest.TestCase):
    def setUp(self):
        self.m = LabelMemory(path=os.path.join(tempfile.mkdtemp(), "m.json"),
                             match_threshold=0.9)
        self.m.remember([1.0, 0.0, 0.0], "mug", "user")

    def test_confident_detection_is_never_overridden(self):
        label, source, alt = resolve_label(
            self.m, [1.0, 0.0, 0.0], "cup", 0.95, None, 0.6)
        self.assertEqual((label, source, alt), ("cup", "detector", None))

    def test_low_confidence_defers_to_memory(self):
        label, source, alt = resolve_label(
            self.m, [1.0, 0.0, 0.0], "cup", 0.55, None, 0.6)
        self.assertEqual((label, source), ("mug", "memory"))
        self.assertIsNone(alt)  # ambiguity resolved

    def test_contested_defers_to_memory_even_if_confident(self):
        label, source, alt = resolve_label(
            self.m, [1.0, 0.0, 0.0], "cup", 0.95, "bowl", 0.6)
        self.assertEqual((label, source), ("mug", "memory"))

    def test_uncertain_but_unknown_look_stays_with_detector(self):
        label, source, alt = resolve_label(
            self.m, [0.0, 1.0, 0.0], "cup", 0.55, "bowl", 0.6)
        self.assertEqual((label, source, alt), ("cup", "detector", "bowl"))

    def test_no_signature_stays_with_detector(self):
        label, source, alt = resolve_label(self.m, None, "cup", 0.55, None, 0.6)
        self.assertEqual((label, source), ("cup", "detector"))


if __name__ == "__main__":
    unittest.main()
