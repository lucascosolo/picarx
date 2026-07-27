#!/usr/bin/env python3
"""Persist human perception corrections as a small, offline training set.

The current on-robot ``label_memory`` is a signature-to-label overlay; it
does not modify MobileNet-SSD/YOLO weights. This store makes that distinction
explicit and preserves the evidence needed for actual detector retraining:
the full frame, the corrected class, the detector bbox, and provenance.
Weight training remains an offline/host operation, not a CPU-heavy Pi hot
path. The writer is the vision process, so no database ownership boundary is
introduced.
"""
import json
import os
import re
import time
import uuid


MAX_SAMPLES = 10000
MAX_FRAME_AGE_SEC = 5.0


class PerceptionFeedbackStore:
    def __init__(self, root, encoder=None):
        self.root = os.path.abspath(os.path.expanduser(root))
        self.encoder = encoder
        os.makedirs(self.root, exist_ok=True)

    @staticmethod
    def safe_label(label):
        value = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(label or "unknown").strip().lower())
        return value[:80] or "unknown"

    def record(self, frame, bbox, label, source="human", observed_at=None, now=None):
        """Write a JPEG + JSON annotation; return metadata or None fail-soft."""
        now = time.time() if now is None else float(now)
        if frame is None or not bbox or len(bbox) != 4:
            return None
        try:
            x, y, w, h = [int(v) for v in bbox]
        except (TypeError, ValueError):
            return None
        if w <= 0 or h <= 0:
            return None
        try:
            encoded = self.encoder(frame) if self.encoder else None
            if encoded is None:
                return None
            if isinstance(encoded, tuple):
                ok, encoded = encoded
                if not ok:
                    return None
            data = encoded.tobytes() if hasattr(encoded, "tobytes") else bytes(encoded)
            if not data or len(data) > 4 * 1024 * 1024:
                return None
            sample_id = uuid.uuid4().hex
            dirname = os.path.join(self.root, self.safe_label(label))
            os.makedirs(dirname, exist_ok=True)
            image_name = sample_id + ".jpg"
            image_path = os.path.join(dirname, image_name)
            annotation_path = os.path.join(dirname, sample_id + ".json")
            height, width = int(frame.shape[0]), int(frame.shape[1])
            annotation = {
                "sample_id": sample_id, "image": image_name,
                "label": str(label).strip().lower()[:80], "source": str(source)[:40],
                "bbox": [max(0, x), max(0, y), min(w, width), min(h, height)],
                "frame_size": [width, height], "observed_at": observed_at,
                "recorded_at": now,
            }
            tmp_image, tmp_json = image_path + ".tmp", annotation_path + ".tmp"
            with open(tmp_image, "wb") as f:
                f.write(data)
            with open(tmp_json, "w", encoding="utf-8") as f:
                json.dump(annotation, f, separators=(",", ":"))
                f.write("\n")
            os.replace(tmp_image, image_path)
            os.replace(tmp_json, annotation_path)
            return annotation
        except (OSError, ValueError, TypeError):
            for path in locals().get("tmp_image", ""), locals().get("tmp_json", ""):
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass
            return None
