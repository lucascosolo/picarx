import importlib.util
import json
import os
import tempfile
import unittest


def module():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "tools", "train_perception_model.py")
    spec = importlib.util.spec_from_file_location("train_perception_model_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class PerceptionTrainingTests(unittest.TestCase):
    def test_prepare_bundle_is_deterministic_and_writes_darknet_labels(self):
        mod = module()
        with tempfile.TemporaryDirectory() as root:
            image_root = os.path.join(root, "images")
            os.makedirs(image_root)
            images = []
            annotations = []
            for ident in range(1, 5):
                name = f"frame{ident}.jpg"
                with open(os.path.join(image_root, name), "wb") as stream:
                    stream.write(b"not-a-real-jpeg-but-a-bounded-fixture")
                images.append({"id": ident, "file_name": name,
                               "width": 320, "height": 240})
                annotations.append({"id": ident, "image_id": ident,
                                    "category_id": 1,
                                    "bbox": [32, 24, 64, 48]})
            coco = os.path.join(root, "dataset.json")
            with open(coco, "w", encoding="utf-8") as stream:
                json.dump({"images": images, "annotations": annotations,
                           "categories": [{"id": 1, "name": "person"}]}, stream)
            bundle = os.path.join(root, "bundle")
            result = mod.prepare_bundle(coco, image_root, bundle,
                                        val_fraction=0.25, seed="test")
            self.assertEqual(result["images"], {"total": 4, "train": 3,
                                                 "validation": 1})
            self.assertEqual(len(os.listdir(os.path.join(bundle, "images", "train"))), 3)
            labels = os.listdir(os.path.join(bundle, "labels", "train"))
            self.assertEqual(len(labels), 3)
            with open(os.path.join(bundle, "labels", "train", labels[0]),
                      encoding="utf-8") as stream:
                fields = stream.read().strip().split()
            self.assertEqual(fields[0], "0")
            self.assertEqual(fields[1:], ["0.200000", "0.200000",
                                          "0.200000", "0.200000"])
            with open(os.path.join(bundle, "classes.names"), encoding="utf-8") as stream:
                self.assertEqual(stream.read().strip(), "person")

    def test_cfg_rewrite_updates_each_detection_head_and_batch_limit(self):
        mod = module()
        with tempfile.TemporaryDirectory() as root:
            template = os.path.join(root, "base.cfg")
            output = os.path.join(root, "training.cfg")
            with open(template, "w", encoding="utf-8") as stream:
                stream.write("[net]\nmax_batches = 5000\n"
                             "[convolutional]\nfilters=255\n[yolo]\nclasses=80\n"
                             "[convolutional]\nfilters=255\n[yolo]\nclasses=80\n")
            result = mod.rewrite_yolo_cfg(template, output, 2, max_batches=100)
            self.assertEqual(result["heads"], 2)
            with open(output, encoding="utf-8") as stream:
                text = stream.read()
            self.assertEqual(text.count("classes=2"), 2)
            self.assertEqual(text.count("filters=21"), 2)
            self.assertIn("max_batches=100", text)

    def test_promotion_requires_metrics_and_keeps_rollback_copy(self):
        mod = module()
        with tempfile.TemporaryDirectory() as root:
            candidate = []
            for name, data in (("candidate.weights", b"weights"),
                               ("candidate.cfg", b"cfg"),
                               ("candidate.names", b"person\n")):
                path = os.path.join(root, name)
                with open(path, "wb") as stream:
                    stream.write(data)
                candidate.append(path)
            model_dir = os.path.join(root, "model")
            os.makedirs(model_dir)
            for name in ("yolov4-tiny.weights", "yolov4-tiny.cfg", "coco.names"):
                with open(os.path.join(model_dir, name), "wb") as stream:
                    stream.write(b"old")
            metrics = os.path.join(root, "metrics.json")
            with open(metrics, "w", encoding="utf-8") as stream:
                json.dump({"precision": 0.9, "recall": 0.85}, stream)
            record = mod.promote_candidate(*candidate, model_dir, metrics,
                                           min_precision=0.8, min_recall=0.8)
            self.assertTrue(os.path.isdir(record["backup"]))
            with open(os.path.join(model_dir, "coco.names"),
                      encoding="utf-8") as stream:
                self.assertEqual(stream.read(), "person\n")

    def test_promotion_gate_rejects_regression(self):
        mod = module()
        allowed, reasons = mod.promotion_gate(
            {"precision": 0.70, "recall": 0.90}, baseline={"precision": 0.80,
                                                              "recall": 0.90})
        self.assertFalse(allowed)
        self.assertTrue(any("precision regresses" in reason for reason in reasons))


if __name__ == "__main__":
    unittest.main()
