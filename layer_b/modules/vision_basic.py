#!/usr/bin/env python3
# /home/picarx/layer_b/modules/vision_basic.py
"""
Vision module (Layer B) - publishes face detection and tracked/labeled
object data to MQTT.

Two detectors share one camera feed:

  - Face detection (Haar cascade, cheap, runs every capture tick) is
    debounced: a face must be seen on FACE_CONFIRM_FRAMES consecutive
    frames before we report "detected": True, and a single missed
    frame clears it immediately. A raw single-frame Haar cascade call
    throws a false positive on all sorts of textured, non-face objects
    (that's the "randomly says it sees a face" bug) - requiring a
    short run of consecutive hits is a cheap, effective filter for
    that class of noise without meaningfully delaying real detections.

  - Object detection (MobileNet-SSD via OpenCV's DNN module, heavier,
    throttled to OBJECT_DETECT_INTERVAL) feeds a small centroid
    tracker so each physical object keeps one persistent id/label
    across frames instead of being reported as a fresh, unrelated
    blob every time the detector fires. New tracks are only published
    once they've been confirmed on OBJECT_CONFIRM_HITS separate
    detection passes, for the same reason face detection is debounced.

Published:
  picarx/vision/faces   - {"detected": bool, "x","y","w","h",
                           "frame_width", "frame_center_offset"}
                           (bbox fields only present if detected)
  picarx/vision/objects - {"objects": [ {"id", "label", "confidence",
                           "x","y","w","h","frame_width","frame_height",
                           "area_ratio","center_offset","first_seen",
                           "last_seen"}, ... ], "close_object": bool}
                           area_ratio = bbox area / frame area, a cheap
                           stand-in for "how close/big this looks,"
                           since it needs no depth sensor - world_state
                           tracks how this changes over time per id to
                           flag an object that's rapidly filling the
                           frame (i.e. approaching).

                           close_object is separate from the labeled
                           items list and doesn't require confident
                           classification: the SSD assigns a confidence
                           score and bounding box to anything vaguely
                           object-shaped even when it can't name it
                           confidently, and that gets discarded below
                           OBJECT_CONFIDENCE_THRESHOLD for the labeled
                           list. close_object instead just asks "is
                           there ANY detection, confidently classified
                           or not, whose box covers most of the frame" -
                           i.e. something is right in front of the
                           camera, regardless of what it is. This is
                           what catches obstacles that aren't one of
                           the SSD's 20 trained categories (a cabinet,
                           say) which the labeled-object list can never
                           recognize no matter how close it gets.

---------------------------------------------------------------------
CPU footprint (Pi 4, running alongside audio_nodes.py's STT decoder)
---------------------------------------------------------------------
This module and audio_nodes.py are the two heaviest processes in the
whole pipeline and directly compete for the same CPU - a slow vision
loop doesn't just look sluggish, it steals cycles the STT decoder
needed to keep up with live audio in real time. Several changes here
exist specifically to bound that, beyond just "make the constants
smaller":

  - CAPTURE_SIZE is 320x240, not 640x480. Every per-pixel operation
    (grayscale conversion, the Haar cascade's sliding window, the
    resize into the SSD's fixed 300x300 input) scales with pixel
    count, so this alone is a ~4x reduction across the board - and
    since the SSD input is a fixed 300x300 regardless, shrinking
    capture to 320x240 costs it almost no real detail.
  - The SSD's color conversion happens via blobFromImage's swapRB
    AFTER resizing to 300x300, instead of running a full-frame
    RGB->BGR conversion first and immediately throwing most of that
    work away in the resize that follows.
  - cv2.setNumThreads caps how many cores OpenCV's internal threading
    (cascade + DNN) is allowed to grab, so this process can't
    opportunistically eat every core and starve the STT process.
  - MOTION-GATED object detection: the SSD forward pass is the single
    most expensive thing this module does. Instead of running it
    unconditionally every OBJECT_DETECT_INTERVAL, a cheap frame-diff
    on a tiny (MOTION_CHECK_SIZE) grayscale thumbnail decides whether
    the scene has actually changed enough to be worth a fresh SSD
    pass at all. If the robot (or the scene) is holding still, most
    of those ticks get skipped entirely - existing tracked objects
    just age normally in the meantime. FORCE_DETECT_INTERVAL still
    guarantees a fresh look periodically regardless, so a new object
    that appears with zero motion (e.g. while the robot is parked)
    doesn't go unnoticed forever.
"""
import os

# Cap BLAS/OpenMP-level threading before cv2 is imported (some OpenCV
# builds respect these at load time) - belt-and-suspenders alongside
# cv2.setNumThreads() below, so this process leaves real headroom for
# audio_nodes.py's decoder instead of grabbing every core it can.
THREAD_LIMIT = 2
os.environ.setdefault("OMP_NUM_THREADS", str(THREAD_LIMIT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(THREAD_LIMIT))

import sys
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus
from picamera2 import Picamera2
import cv2
import time

cv2.setNumThreads(THREAD_LIMIT)

CAPTURE_SIZE = (320, 240)      # was 640x480 - biggest single lever, see notes above

DETECT_INTERVAL = 0.3          # base capture/face-check tick (was 0.2)
OBJECT_DETECT_INTERVAL = 1.5   # candidate ticks for a fresh SSD pass (was 1.0)

FACE_CONFIRM_FRAMES = 3        # consecutive frames before reporting a face

OBJECT_CONFIRM_HITS = 2        # separate detection passes before publishing a track
OBJECT_MAX_DISAPPEARED_SEC = 3.0   # how long to keep a track alive with no match
OBJECT_MATCH_MAX_DIST = 120        # pixels; centroid match distance cap
OBJECT_CONFIDENCE_THRESHOLD = 0.5

# Class-agnostic "something huge is right in front of the camera"
# signal - see the close_object note in the module docstring. Deliberately
# not debounced across multiple ticks the way faces/tracked objects are:
# this is meant to be as immediate as the ultrasonic's own point-blank
# check, which also reacts on a single reading.
CLOSE_OBJECT_MIN_CONFIDENCE = 0.2   # much lower bar - we don't care what it is
CLOSE_OBJECT_AREA_RATIO = 0.45      # bbox covers this fraction of the frame

# Motion gate: only bother running the SSD at all if the scene looks
# like it's actually changed since the last time we ran it, checked on
# a tiny thumbnail so the check itself costs almost nothing.
MOTION_CHECK_SIZE = (80, 60)
MOTION_DIFF_THRESHOLD = 6.0     # mean abs pixel difference (0-255 scale) to count as "changed"
FORCE_DETECT_INTERVAL = 6.0     # always refresh at least this often, motion or not

MODEL_DIR = "/home/picarx/layer_b/modules/models/mobilenet_ssd"
SSD_PROTOTXT = f"{MODEL_DIR}/deploy.prototxt"
SSD_WEIGHTS = f"{MODEL_DIR}/mobilenet_iter_73000.caffemodel"
SSD_INPUT_SIZE = (300, 300)

# Class order this particular Caffe model (chuanqi305/MobileNet-SSD,
# trained on PASCAL VOC) was trained with - index 0 is background.
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse",
    "motorbike", "person", "pottedplant", "sheep", "sofa", "train",
    "tvmonitor",
]


class CentroidTracker:
    """
    Minimal multi-object tracker: matches new detections to existing
    tracks by nearest centroid, ages out tracks that stop matching, and
    only exposes a track once it's been confirmed on more than one
    detection pass. Small object counts are expected here (a handful
    at most), so plain O(n*m) greedy matching is enough - no need for
    scipy/Hungarian matching.
    """

    def __init__(self):
        self.next_id = 0
        self.tracks = {}  # id -> track dict

    @staticmethod
    def _centroid(bbox):
        x, y, w, h = bbox
        return (x + w / 2.0, y + h / 2.0)

    @staticmethod
    def _dist(a, b):
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def update(self, detections, now):
        """detections: list of {"label", "confidence", "bbox": (x,y,w,h)}"""
        unmatched_tracks = set(self.tracks.keys())
        unmatched_detections = list(range(len(detections)))

        # Greedily pair the closest (track, detection) below the
        # distance cap first, then keep pairing whatever's left.
        candidate_pairs = []
        for tid in unmatched_tracks:
            t_centroid = self.tracks[tid]["centroid"]
            for di in unmatched_detections:
                d_centroid = self._centroid(detections[di]["bbox"])
                d = self._dist(t_centroid, d_centroid)
                if d <= OBJECT_MATCH_MAX_DIST:
                    candidate_pairs.append((d, tid, di))
        candidate_pairs.sort(key=lambda p: p[0])

        matched_tracks = set()
        matched_detections = set()
        for _, tid, di in candidate_pairs:
            if tid in matched_tracks or di in matched_detections:
                continue
            matched_tracks.add(tid)
            matched_detections.add(di)

            det = detections[di]
            track = self.tracks[tid]
            track["bbox"] = det["bbox"]
            track["centroid"] = self._centroid(det["bbox"])
            track["confidence"] = det["confidence"]
            track["last_seen"] = now
            track["disappeared_since"] = None
            track["hits"] += 1

            # Stabilize the reported label against frame-to-frame
            # flicker between visually similar classes by voting
            # instead of just taking whatever this frame said.
            label_counts = track["label_counts"]
            label_counts[det["label"]] = label_counts.get(det["label"], 0) + 1
            track["label"] = max(label_counts, key=label_counts.get)

        for tid in unmatched_tracks - matched_tracks:
            track = self.tracks[tid]
            if track["disappeared_since"] is None:
                track["disappeared_since"] = now
            if now - track["disappeared_since"] > OBJECT_MAX_DISAPPEARED_SEC:
                del self.tracks[tid]

        for di in set(unmatched_detections) - matched_detections:
            det = detections[di]
            tid = f"object_{self.next_id}"
            self.next_id += 1
            self.tracks[tid] = {
                "bbox": det["bbox"],
                "centroid": self._centroid(det["bbox"]),
                "confidence": det["confidence"],
                "label": det["label"],
                "label_counts": {det["label"]: 1},
                "first_seen": now,
                "last_seen": now,
                "disappeared_since": None,
                "hits": 1,
            }

    def confirmed_tracks(self):
        return {
            tid: t for tid, t in self.tracks.items()
            if t["hits"] >= OBJECT_CONFIRM_HITS
        }


def run():
    bus = Bus()
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"format": "RGB888", "size": CAPTURE_SIZE})
    picam2.configure(config)
    picam2.start()
    time.sleep(1)

    face_cascade = cv2.CascadeClassifier(
        "/home/picarx/layer_b/modules/cascades/cascades.xml"
    )
    net = cv2.dnn.readNetFromCaffe(SSD_PROTOTXT, SSD_WEIGHTS)
    tracker = CentroidTracker()

    print("vision basic module running, publishing to picarx/vision/faces and picarx/vision/objects")

    face_streak = 0
    last_object_detect = 0.0
    last_forced_detect = 0.0
    last_motion_thumb = None
    close_object = False   # persists between ticks where the SSD didn't run

    while True:
        frame = picam2.capture_array()
        frame_h, frame_w = frame.shape[:2]

        # ---------- face detection (debounced) ----------
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.2, minNeighbors=6,
            minSize=(int(frame_w * 0.08), int(frame_h * 0.08)),
        )

        if len(faces) > 0:
            face_streak += 1
        else:
            face_streak = 0

        if face_streak >= FACE_CONFIRM_FRAMES:
            x, y, w, h = max(faces.tolist(), key=lambda f: f[2])
            bus.publish("picarx/vision/faces", {
                "detected": True,
                "x": x, "y": y, "w": w, "h": h,
                "frame_width": frame_w,
                "frame_center_offset": (x + w // 2) - (frame_w // 2),
            })
        else:
            bus.publish("picarx/vision/faces", {"detected": False})

        # ---------- object detection (motion-gated + throttled) ----------
        now = time.time()
        if now - last_object_detect >= OBJECT_DETECT_INTERVAL:
            last_object_detect = now

            # Cheap motion check on a tiny thumbnail - reuses the gray
            # frame we already computed for the face cascade above, so
            # this costs almost nothing extra.
            thumb = cv2.resize(gray, MOTION_CHECK_SIZE)
            if last_motion_thumb is None:
                should_run_ssd = True
            else:
                diff = cv2.absdiff(thumb, last_motion_thumb)
                moved = float(diff.mean()) > MOTION_DIFF_THRESHOLD
                should_run_ssd = moved or (now - last_forced_detect) > FORCE_DETECT_INTERVAL
            last_motion_thumb = thumb

            if should_run_ssd:
                last_forced_detect = now
                # Resize first (cheap - CAPTURE_SIZE is already small),
                # then let blobFromImage's swapRB handle the RGB->BGR
                # conversion on the now-tiny 300x300 image instead of
                # converting the full frame beforehand.
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(frame, SSD_INPUT_SIZE), 0.007843, SSD_INPUT_SIZE, 127.5, swapRB=True
                )
                net.setInput(blob)
                detections = net.forward()

                found = []
                close_object = False
                for i in range(detections.shape[2]):
                    confidence = float(detections[0, 0, i, 2])
                    if confidence < CLOSE_OBJECT_MIN_CONFIDENCE:
                        continue
                    class_id = int(detections[0, 0, i, 1])
                    if class_id <= 0:
                        continue
                    box = detections[0, 0, i, 3:7] * [frame_w, frame_h, frame_w, frame_h]
                    # int(...) here, not just .astype(int): numpy int64
                    # scalars aren't JSON-serializable, and json.dumps()
                    # would otherwise blow up the instant any object is
                    # actually detected.
                    x1, y1, x2, y2 = (int(v) for v in box.astype(int))
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame_w, x2), min(frame_h, y2)
                    if x2 <= x1 or y2 <= y1:
                        continue

                    area_ratio = ((x2 - x1) * (y2 - y1)) / float(frame_w * frame_h)
                    if area_ratio > CLOSE_OBJECT_AREA_RATIO:
                        close_object = True

                    if class_id >= len(VOC_CLASSES) or confidence < OBJECT_CONFIDENCE_THRESHOLD:
                        continue
                    found.append({
                        "label": VOC_CLASSES[class_id],
                        "confidence": confidence,
                        "bbox": (x1, y1, x2 - x1, y2 - y1),
                    })

                tracker.update(found, now)

        objects_payload = []
        for tid, t in tracker.confirmed_tracks().items():
            x, y, w, h = t["bbox"]
            objects_payload.append({
                "id": tid,
                "label": t["label"],
                "confidence": t["confidence"],
                "x": x, "y": y, "w": w, "h": h,
                "frame_width": frame_w,
                "frame_height": frame_h,
                "area_ratio": (w * h) / float(frame_w * frame_h),
                "center_offset": (x + w // 2) - (frame_w // 2),
                "first_seen": t["first_seen"],
                "last_seen": t["last_seen"],
            })
        bus.publish("picarx/vision/objects", {"objects": objects_payload, "close_object": close_object})

        time.sleep(DETECT_INTERVAL)


if __name__ == "__main__":
    run()
