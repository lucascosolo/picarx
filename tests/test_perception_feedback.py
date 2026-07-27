import json
import os
import tempfile
import unittest

from layer_b.perception_feedback import PerceptionFeedbackStore


class FakeFrame:
    shape = (240, 320, 3)


class PerceptionFeedbackTests(unittest.TestCase):
    def test_record_writes_image_and_annotation(self):
        with tempfile.TemporaryDirectory() as root:
            store = PerceptionFeedbackStore(root, encoder=lambda frame: b"jpeg")
            result = store.record(FakeFrame(), (2, 3, 40, 50), "Coffee Mug", "human",
                                  observed_at=12.0, now=13.0)
            self.assertEqual(result["label"], "coffee mug")
            directory = os.path.join(root, "coffee_mug")
            files = os.listdir(directory)
            self.assertEqual(len(files), 2)
            annotation_path = os.path.join(
                directory, next(f for f in files if f.endswith(".json")))
            with open(annotation_path, encoding="utf-8") as stream:
                annotation = json.load(stream)
            self.assertEqual(annotation["bbox"], [2, 3, 40, 50])

    def test_bad_frame_is_fail_soft(self):
        with tempfile.TemporaryDirectory() as root:
            store = PerceptionFeedbackStore(root, encoder=lambda frame: None)
            self.assertIsNone(store.record(None, (0, 0, 1, 1), "x"))
            self.assertIsNone(store.record(FakeFrame(), (0, 0, 1, 1), "x"))


if __name__ == "__main__":
    unittest.main()
