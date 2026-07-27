import importlib.util
import json
import os
import tempfile
import unittest


def module():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tools", "export_perception_feedback.py")
    spec = importlib.util.spec_from_file_location("export_feedback_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ExportFeedbackTests(unittest.TestCase):
    def test_valid_samples_export_and_bad_samples_skip(self):
        with tempfile.TemporaryDirectory() as root:
            label_dir = os.path.join(root, "person")
            os.makedirs(label_dir)
            with open(os.path.join(label_dir, "a.jpg"), "wb") as f:
                f.write(b"jpeg")
            with open(os.path.join(label_dir, "a.json"), "w") as f:
                json.dump({"image": "a.jpg", "label": "person",
                           "bbox": [1, 2, 10, 20], "frame_size": [100, 80],
                           "source": "human"}, f)
            with open(os.path.join(label_dir, "bad.json"), "w") as f:
                json.dump({"image": "missing.jpg", "label": "person",
                           "bbox": [0, 0, 1, 1], "frame_size": [2, 2]}, f)
            output = os.path.join(root, "dataset.json")
            result = module().export_dataset(root, output)
            self.assertEqual(len(result["images"]), 1)
            self.assertEqual(result["categories"][0]["name"], "person")
            self.assertEqual(len(result["skipped"]), 1)


if __name__ == "__main__":
    unittest.main()
