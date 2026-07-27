#!/usr/bin/env python3
"""Prepare, train, and safely promote a corrected YOLOv4-tiny detector.

Human corrections are collected on the robot and exported by
``export_perception_feedback.py`` as COCO JSON.  This tool is deliberately an
offline/host workflow: it converts that export to Darknet's YOLO layout,
optionally invokes an installed Darknet binary, and refuses to replace the
robot's active model without explicit evaluation metrics and a rollback copy.

The live detector in ``vision_basic.py`` uses OpenCV's Darknet reader, so a
candidate consists of a matching ``.cfg``, ``.weights``, and ``.names`` file.
The robot is never retrained in its hot path and ``promote`` is never implied
by ``train``.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


MAX_IMAGES = 100000
MAX_ANNOTATIONS = 500000
MAX_METRIC_REGRESSION = 0.02


class TrainingError(Exception):
    """A user-correctable dataset, training, or promotion error."""


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inside(root, value):
    root = Path(root).expanduser().resolve()
    path = (root / str(value)).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise TrainingError(f"dataset path escapes image root: {value}")
    return path


def _load_coco(path):
    try:
        with open(path, encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"could not read COCO export: {exc}")
    if not isinstance(data, dict):
        raise TrainingError("COCO export must be an object")
    images = data.get("images")
    annotations = data.get("annotations")
    categories = data.get("categories")
    if not isinstance(images, list) or not isinstance(annotations, list) or \
            not isinstance(categories, list):
        raise TrainingError("COCO export is missing images, annotations, or categories")
    if len(images) > MAX_IMAGES or len(annotations) > MAX_ANNOTATIONS:
        raise TrainingError("COCO export exceeds the offline dataset limits")
    return data


def split_image_ids(image_ids, val_fraction=0.2, seed="picarx"):
    """Return deterministic ``(train_ids, validation_ids)`` partitions.

    Hashing IDs rather than shuffling the input makes repeated exports stable
    even when directory traversal order changes.  A dataset with at least two
    images always receives at least one validation image and one train image.
    """
    ids = list(dict.fromkeys(image_ids))
    try:
        fraction = float(val_fraction)
    except (TypeError, ValueError):
        raise TrainingError("validation fraction must be a number")
    if not 0.0 <= fraction < 1.0:
        raise TrainingError("validation fraction must be in [0, 1)")
    ranked = sorted(ids, key=lambda ident: hashlib.sha256(
        f"{seed}:{ident}".encode("utf-8")).hexdigest())
    if len(ranked) < 2 or fraction == 0.0:
        return ranked, []
    count = max(1, min(len(ranked) - 1, round(len(ranked) * fraction)))
    validation = ranked[:count]
    val_set = set(validation)
    return [ident for ident in ranked if ident not in val_set], validation


def _validate_categories(categories):
    result = []
    seen = set()
    for category in categories:
        try:
            ident = int(category["id"])
            name = str(category["name"]).strip()
        except (KeyError, TypeError, ValueError):
            raise TrainingError("invalid category in COCO export")
        if ident in seen or not name:
            raise TrainingError("categories must have unique IDs and names")
        seen.add(ident)
        result.append((ident, name))
    if not result:
        raise TrainingError("COCO export contains no categories")
    return result


def _yolo_box(annotation, width, height, category_index):
    try:
        x, y, box_w, box_h = [float(v) for v in annotation["bbox"]]
        category_id = int(annotation["category_id"])
    except (KeyError, TypeError, ValueError):
        raise TrainingError("invalid annotation")
    if width <= 0 or height <= 0 or box_w <= 0 or box_h <= 0:
        raise TrainingError("annotation has non-positive dimensions")
    if x < 0 or y < 0 or x + box_w > width or y + box_h > height:
        raise TrainingError("annotation bbox lies outside its image")
    if category_id not in category_index:
        raise TrainingError(f"annotation references unknown category {category_id}")
    center_x = (x + box_w / 2.0) / width
    center_y = (y + box_h / 2.0) / height
    return (category_index[category_id], center_x, center_y,
            box_w / width, box_h / height)


def prepare_bundle(coco_path, image_root, output_dir, val_fraction=0.2,
                   seed="picarx", copy_images=True):
    """Create a self-contained Darknet dataset bundle from a COCO export."""
    coco_path = Path(coco_path).expanduser().resolve()
    image_root = Path(image_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    data = _load_coco(coco_path)
    categories = _validate_categories(data["categories"])
    category_index = {ident: index for index, (ident, _) in enumerate(categories)}

    images = {}
    for image in data["images"]:
        try:
            ident = int(image["id"])
            name = str(image["file_name"])
            width, height = int(image["width"]), int(image["height"])
        except (KeyError, TypeError, ValueError):
            raise TrainingError("invalid image in COCO export")
        if ident in images or not name or width <= 0 or height <= 0:
            raise TrainingError("images must have unique IDs and valid dimensions")
        source = _inside(image_root, name)
        if not source.is_file():
            raise TrainingError(f"image is missing: {name}")
        images[ident] = {"name": name, "width": width, "height": height,
                         "source": source}

    by_image = {ident: [] for ident in images}
    for annotation in data["annotations"]:
        try:
            image_id = int(annotation["image_id"])
        except (KeyError, TypeError, ValueError):
            raise TrainingError("annotation has an invalid image ID")
        if image_id not in images:
            raise TrainingError(f"annotation references unknown image {image_id}")
        by_image[image_id].append(annotation)

    train_ids, val_ids = split_image_ids(images, val_fraction, seed)
    if not train_ids:
        raise TrainingError("training split is empty")
    if output_dir.exists():
        # Never merge into an old bundle: stale labels are easy to mistake for
        # fresh corrections.  The caller can remove/recreate it explicitly.
        if any(output_dir.iterdir()):
            raise TrainingError(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    image_lists = {}
    for split, ids in (("train", train_ids), ("val", val_ids)):
        paths = []
        for image_id in ids:
            image = images[image_id]
            source = image["source"]
            suffix = source.suffix.lower() or ".jpg"
            destination_name = f"{image_id}_{source.stem}{suffix}"
            destination = output_dir / "images" / split / destination_name
            if copy_images:
                shutil.copy2(source, destination)
            else:
                destination.symlink_to(source)
            label_path = output_dir / "labels" / split / f"{image_id}_{source.stem}.txt"
            rows = [_yolo_box(ann, image["width"], image["height"], category_index)
                    for ann in by_image[image_id]]
            with open(label_path, "w", encoding="utf-8") as stream:
                for class_id, cx, cy, width, height in rows:
                    stream.write(f"{class_id} {cx:.6f} {cy:.6f} "
                                 f"{width:.6f} {height:.6f}\n")
            paths.append(str(destination.resolve()))
        image_lists[split] = paths
        with open(output_dir / f"{split}.txt", "w", encoding="utf-8") as stream:
            stream.write("\n".join(paths))
            if paths:
                stream.write("\n")

    names_path = output_dir / "classes.names"
    with open(names_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(name for _, name in categories) + "\n")
    data_path = output_dir / "dataset.data"
    with open(data_path, "w", encoding="utf-8") as stream:
        stream.write(f"classes = {len(categories)}\n")
        stream.write(f"train = {output_dir / 'train.txt'}\n")
        stream.write(f"valid = {output_dir / 'val.txt'}\n")
        stream.write(f"names = {names_path}\n")
        stream.write(f"backup = {output_dir / 'backup'}\n")
    (output_dir / "backup").mkdir(exist_ok=True)

    manifest = {
        "format": "picarx-yolov4-tiny-training-bundle-v1",
        "source_coco": str(coco_path),
        "source_sha256": _sha256(coco_path),
        "image_root": str(image_root),
        "seed": str(seed),
        "validation_fraction": float(val_fraction),
        "copied_images": bool(copy_images),
        "categories": [{"id": ident, "name": name}
                       for ident, name in categories],
        "images": {"total": len(images), "train": len(train_ids),
                   "validation": len(val_ids)},
        "annotations": sum(len(v) for v in by_image.values()),
        "created_at": time.time(),
        "files": {"data": str(data_path), "names": str(names_path)},
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest


def rewrite_yolo_cfg(template, output, classes, max_batches=None):
    """Adapt a YOLOv4-tiny cfg's detection heads to the exported classes."""
    try:
        classes = int(classes)
    except (TypeError, ValueError):
        raise TrainingError("classes must be an integer")
    if classes < 1:
        raise TrainingError("classes must be positive")
    try:
        lines = Path(template).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TrainingError(f"could not read cfg template: {exc}")
    section = None
    pending_filters = None
    yolo_filters = None
    replaced_heads = 0
    result = list(lines)
    for index, raw in enumerate(lines):
        stripped = raw.strip().lower()
        if stripped.startswith("[") and stripped.endswith("]"):
            previous_section, previous_filters = section, pending_filters
            section = stripped[1:-1]
            yolo_filters = previous_filters if previous_section == "convolutional" else None
            pending_filters = None
            continue
        if section == "convolutional" and stripped.startswith("filters") and "=" in raw:
            pending_filters = index
        if section == "yolo" and stripped.startswith("classes") and "=" in raw:
            result[index] = raw[:len(raw) - len(raw.lstrip())] + f"classes={classes}"
            if yolo_filters is not None:
                result[yolo_filters] = (
                    lines[yolo_filters][:len(lines[yolo_filters]) -
                                        len(lines[yolo_filters].lstrip())] +
                    f"filters={(classes + 5) * 3}")
            replaced_heads += 1
    if replaced_heads == 0:
        raise TrainingError("cfg template contains no YOLO detection head")
    if max_batches is not None:
        try:
            max_batches = int(max_batches)
        except (TypeError, ValueError):
            raise TrainingError("max_batches must be an integer")
        if max_batches < 1:
            raise TrainingError("max_batches must be positive")
        section = None
        found = False
        for index, raw in enumerate(lines):
            stripped = raw.strip().lower()
            if stripped.startswith("[") and stripped.endswith("]"):
                section = stripped[1:-1]
            elif section == "net" and stripped.startswith("max_batches") and "=" in raw:
                result[index] = raw[:len(raw) - len(raw.lstrip())] + \
                    f"max_batches={max_batches}"
                found = True
                break
        if not found:
            raise TrainingError("cfg template has no max_batches setting")
    Path(output).write_text("\n".join(result) + "\n", encoding="utf-8")
    return {"classes": classes, "heads": replaced_heads,
            "max_batches": max_batches}


def _bounded_output(value, limit=32000):
    value = value or ""
    return value if len(value) <= limit else value[:limit] + "\n[truncated]"


def run_darknet(darknet, bundle, cfg_template, base_weights=None,
                max_batches=None, timeout=None):
    """Run an explicitly selected Darknet binary against a prepared bundle."""
    bundle = Path(bundle).expanduser().resolve()
    manifest_path = bundle / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"invalid training bundle: {exc}")
    cfg_path = bundle / "training.cfg"
    rewrite_yolo_cfg(cfg_template, cfg_path, len(manifest["categories"]), max_batches)
    data_path = bundle / "dataset.data"
    command = [str(darknet), "detector", "train", str(data_path),
               str(cfg_path)]
    if base_weights:
        command.append(str(Path(base_weights).expanduser().resolve()))
    command += ["-dont_show", "-map"]
    try:
        proc = subprocess.run(command, cwd=bundle, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True,
                              timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TrainingError(f"Darknet training failed to start or timed out: {exc}")
    candidates = sorted((bundle / "backup").glob("*.weights"),
                        key=lambda path: path.stat().st_mtime, reverse=True)
    return {"returncode": proc.returncode, "command": command,
            "output": _bounded_output(proc.stdout),
            "candidate_weights": str(candidates[0]) if candidates else None,
            "cfg": str(cfg_path), "data": str(data_path)}


def promotion_gate(metrics, min_precision=0.0, min_recall=0.0,
                   baseline=None, max_regression=MAX_METRIC_REGRESSION):
    """Return ``(allowed, reasons)`` for explicit candidate metrics."""
    reasons = []
    for key, minimum in (("precision", min_precision), ("recall", min_recall)):
        try:
            value = float(metrics[key])
        except (KeyError, TypeError, ValueError):
            reasons.append(f"missing numeric {key}")
            continue
        if not 0.0 <= value <= 1.0:
            reasons.append(f"{key} must be between 0 and 1")
        elif value < float(minimum):
            reasons.append(f"{key} {value:.3f} is below required {float(minimum):.3f}")
        if baseline and key in baseline:
            try:
                old = float(baseline[key])
            except (TypeError, ValueError):
                reasons.append(f"baseline {key} is not numeric")
            else:
                if value < old - float(max_regression):
                    reasons.append(f"{key} regresses from {old:.3f} to {value:.3f}")
    return not reasons, reasons


def promote_candidate(weights, cfg, names, model_dir, metrics_path,
                      min_precision=0.0, min_recall=0.0, baseline_path=None):
    """Install a measured candidate beside the live OpenCV-DNN model."""
    paths = [Path(value).expanduser().resolve() for value in (weights, cfg, names)]
    if any(not path.is_file() or path.stat().st_size == 0 for path in paths):
        raise TrainingError("candidate cfg, weights, and names must be non-empty files")
    try:
        metrics = json.loads(Path(metrics_path).expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TrainingError(f"could not read evaluation metrics: {exc}")
    baseline = None
    if baseline_path:
        try:
            baseline = json.loads(Path(baseline_path).expanduser().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingError(f"could not read baseline metrics: {exc}")
    allowed, reasons = promotion_gate(metrics, min_precision, min_recall, baseline)
    if not allowed:
        raise TrainingError("promotion gate failed: " + "; ".join(reasons))
    model_dir = Path(model_dir).expanduser().resolve()
    model_dir.mkdir(parents=True, exist_ok=True)
    backup = model_dir / "backups" / time.strftime("%Y%m%d-%H%M%S")
    backup.mkdir(parents=True, exist_ok=False)
    destinations = [model_dir / "yolov4-tiny.weights",
                    model_dir / "yolov4-tiny.cfg",
                    model_dir / "coco.names"]
    for destination in destinations:
        if destination.exists():
            shutil.copy2(destination, backup / destination.name)
    for source, destination in zip(paths, destinations):
        temporary = destination.with_suffix(destination.suffix + ".candidate")
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    record = {"promoted_at": time.time(), "metrics": metrics,
              "baseline": baseline, "backup": str(backup),
              "weights": str(destinations[0]), "cfg": str(destinations[1]),
              "names": str(destinations[2])}
    with open(model_dir / "model_manifest.json", "w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
        stream.write("\n")
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("coco")
    prepare.add_argument("image_root")
    prepare.add_argument("output_dir")
    prepare.add_argument("--val-fraction", type=float, default=0.2)
    prepare.add_argument("--seed", default="picarx")
    prepare.add_argument("--symlink-images", action="store_true")
    train = sub.add_parser("train")
    train.add_argument("bundle")
    train.add_argument("--darknet", required=True)
    train.add_argument("--cfg-template", required=True)
    train.add_argument("--base-weights")
    train.add_argument("--max-batches", type=int)
    train.add_argument("--timeout", type=float)
    promote = sub.add_parser("promote")
    promote.add_argument("--weights", required=True)
    promote.add_argument("--cfg", required=True)
    promote.add_argument("--names", required=True)
    promote.add_argument("--model-dir", required=True)
    promote.add_argument("--metrics", required=True)
    promote.add_argument("--baseline-metrics")
    promote.add_argument("--min-precision", type=float, default=0.0)
    promote.add_argument("--min-recall", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            result = prepare_bundle(args.coco, args.image_root, args.output_dir,
                                    args.val_fraction, args.seed,
                                    not args.symlink_images)
        elif args.action == "train":
            result = run_darknet(args.darknet, args.bundle, args.cfg_template,
                                 args.base_weights, args.max_batches, args.timeout)
            if result["returncode"]:
                raise TrainingError(f"Darknet exited with {result['returncode']}")
        else:
            result = promote_candidate(args.weights, args.cfg, args.names,
                                       args.model_dir, args.metrics,
                                       args.min_precision, args.min_recall,
                                       args.baseline_metrics)
    except TrainingError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
