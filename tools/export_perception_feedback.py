#!/usr/bin/env python3
"""Validate and export robot human-correction samples as COCO JSON.

This is intentionally an offline tool. It prepares data for a detector
training pipeline without pretending that the Pi's label-memory overlay has
changed MobileNet-SSD/YOLO weights.
"""
import argparse
import json
import os
from collections import OrderedDict


def export_dataset(root, output):
    root = os.path.abspath(os.path.expanduser(root))
    images, annotations, categories = [], [], OrderedDict()
    next_image, next_annotation = 1, 1
    skipped = []
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if not name.endswith(".json"):
                continue
            path = os.path.join(dirpath, name)
            try:
                with open(path, encoding="utf-8") as stream:
                    sample = json.load(stream)
                image_path = os.path.join(dirpath, sample["image"])
                label = str(sample["label"]).strip().lower()
                x, y, w, h = [int(v) for v in sample["bbox"]]
                width, height = [int(v) for v in sample["frame_size"]]
                if not os.path.isfile(image_path) or not label or w <= 0 or h <= 0:
                    raise ValueError("missing image, label, or invalid bbox")
                if x < 0 or y < 0 or x + w > width or y + h > height:
                    raise ValueError("bbox outside frame")
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                skipped.append({"path": path, "reason": str(e)[:200]})
                continue
            if label not in categories:
                categories[label] = len(categories) + 1
            image_id = next_image
            next_image += 1
            images.append({"id": image_id,
                           "file_name": os.path.relpath(image_path, root),
                           "width": width, "height": height})
            annotations.append({"id": next_annotation, "image_id": image_id,
                                "category_id": categories[label],
                                "bbox": [x, y, w, h], "area": w * h,
                                "iscrowd": 0, "source": sample.get("source")})
            next_annotation += 1
    dataset = {"info": {"description": "PiCar-X human perception corrections"},
               "images": images, "annotations": annotations,
               "categories": [{"id": ident, "name": label}
                              for label, ident in categories.items()],
               "skipped": skipped}
    with open(output, "w", encoding="utf-8") as stream:
        json.dump(dataset, stream, indent=2)
        stream.write("\n")
    return dataset


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("output")
    args = parser.parse_args(argv)
    dataset = export_dataset(args.root, args.output)
    print(f"exported {len(dataset['annotations'])} samples, "
          f"skipped {len(dataset['skipped'])}")


if __name__ == "__main__":
    main()
