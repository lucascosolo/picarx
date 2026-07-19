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
  picarx/vision/objects - {"objects": [ {"id", "label", "alt_label",
                           "label_source", "confidence",
                           "x","y","w","h","frame_width","frame_height",
                           "area_ratio","center_offset","first_seen",
                           "last_seen"}, ... ], "close_object": bool}
                           alt_label is the runner-up class when the label
                           vote is a genuine two-way tie (else None) - the
                           "chair or speaker?" ambiguity signal. label_source
                           is "detector" for the model's own label, or
                           "memory" when an UNCERTAIN detection was relabeled
                           from the on-board label memory (label_memory.py,
                           taught by human/LLM corrections). See that module
                           and curiosity.py for the recognition cascade.
                           area_ratio = bbox area / frame area, a cheap
                           stand-in for "how close/big this looks,"
                           since it needs no depth sensor - world_state
                           tracks how this changes over time per id to
                           flag an object that's rapidly filling the
                           frame (i.e. approaching).

                           overhead ({"area_ratio","y_center_frac"} or
                           None) is the largest class-agnostic mass that
                           looks like a head-height OVERHANG - big and high
                           in the frame with open space below it (a counter
                           lip, a table edge). It exists because the low
                           bumper ultrasonic is blind above its beam: an
                           overhang reads as clear air underneath while the
                           camera head is about to hit it. See the
                           OVERHEAD_* notes and field_agent's cross-check.

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
import label_memory
from picamera2 import Picamera2
import base64
import cv2
import numpy as np
import time

cv2.setNumThreads(THREAD_LIMIT)

CAPTURE_SIZE = (640, 480)      # raised from 320x240 for finer detail; note the
                               # detector still resizes to its fixed 300/320 input,
                               # so this mainly sharpens face crops, the console
                               # view, and label signatures - at ~4x the per-pixel
                               # CPU on the face/motion path (watch the STT budget).

DETECT_INTERVAL = 0.3          # base capture/face-check tick (was 0.2)
OBJECT_DETECT_INTERVAL = 1.5   # candidate ticks for a fresh SSD pass (was 1.0)
# Homeostatic self-preservation: when health_daemon signals low battery on
# picarx/health/low_power, the expensive SSD/YOLO forward pass is throttled
# way back to conserve power. The Haar face pass - actually the single most
# expensive per-tick op - is also backed off (gently, so faces are still
# picked up reasonably promptly): to 1s in low power, and 2s once the battery
# is critical. At full power it runs every DETECT_INTERVAL tick as before.
LOW_POWER_TOPIC = "picarx/health/low_power"
LOW_POWER_OBJECT_DETECT_INTERVAL = 15.0
LOW_POWER_FACE_DETECT_INTERVAL = 1.0
CRITICAL_FACE_DETECT_INTERVAL = 2.0

FACE_CONFIRM_FRAMES = 3        # consecutive frames before reporting a face

# ---- face crops for person recognition (person_memory.py) ----
# While a face is confirmed, a small grayscale crop of it is published so
# person_memory can identify (or learn) WHO it is. Throttled and tiny
# (100x100 JPEG about once a second) so it costs effectively nothing on
# top of the Haar pass we already ran; nothing is published while no face
# is confirmed, and if person_memory isn't running the topic just goes
# nowhere - same fail-soft optionality as every other consumer.
FACE_CROP_TOPIC = "picarx/vision/face_crop"
FACE_CROP_INTERVAL = 1.0
FACE_CROP_SIZE = (100, 100)
FACE_CROP_MARGIN = 0.15        # fraction of bbox padded on each side

OBJECT_CONFIRM_HITS = 2        # separate detection passes before publishing a track
OBJECT_MAX_DISAPPEARED_SEC = 3.0   # how long to keep a track alive with no match
OBJECT_MATCH_MAX_DIST = 120        # pixels; centroid match distance cap
OBJECT_CONFIDENCE_THRESHOLD = 0.5

# A track's label is the majority vote over its per-frame classifications
# (see CentroidTracker.label_counts) - that stabilizes the reported label
# but HIDES a real two-way ambiguity, exactly the "is that a chair or a
# speaker?" case worth asking a human about. If the runner-up label has at
# least this fraction of the winner's votes, the classification is genuinely
# contested and we publish the runner-up as alt_label so curiosity.py can ask.
LABEL_CONTEST_RATIO = 0.6

# ---- on-board label memory (the recognition tier between the detector and
# the cloud LLM; see label_memory.py) ----
# When a detection is UNCERTAIN - contested vote, or confidence below
# LABEL_LOW_CONF - we consult the visual-signature memory before trusting the
# detector's nearest-of-N guess. Human/LLM labels (picarx/perception/label,
# carrying the object id) teach it: we look up the object's cached signature
# and store signature -> label. The label the console/curiosity feedback
# reaches us on is the SAME topic reflection.py writes as a fact, so one
# correction both remembers the look AND records the fact.
LABEL_TOPIC = "picarx/perception/label"
LABEL_LOW_CONF = 0.6            # below this a single guess is "uncertain"
SIG_CACHE_TTL = 60.0           # keep a track's signature this long after last seen,
                               # so a label answered seconds later still finds it
SIG_THUMB = (16, 16)           # grayscale shape thumbnail side of the signature


def visual_signature(crop_rgb):
    """A cheap, fixed-length visual fingerprint of an object crop: a small
    normalized grayscale shape thumbnail concatenated with an HSV hue/sat
    color histogram, the whole thing L2-normalized. Deterministic, ~320
    floats, computed only on detection passes for a handful of objects, so
    it adds little to the CPU budget this module guards. Approximate by
    design - it disambiguates a small set of taught objects when the
    detector is already unsure, not general re-identification. Returns a
    plain list (JSON-serializable for label_memory), or None on a bad crop.

    crop_rgb is an RGB uint8 array (picamera2 frames are RGB888)."""
    try:
        if crop_rgb is None or crop_rgb.size == 0:
            return None
        gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
        thumb = cv2.resize(gray, SIG_THUMB).astype("float32").flatten()
        n = float(np.linalg.norm(thumb))
        thumb = thumb / n if n else thumb
        hsv = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256]).flatten()
        n = float(np.linalg.norm(hist))
        hist = hist / n if n else hist
        vec = np.concatenate([thumb * 0.5, hist.astype("float32") * 0.5])
        n = float(np.linalg.norm(vec))
        if n:
            vec = vec / n
        return [float(x) for x in vec]
    except Exception as e:
        print(f"vision: signature failed ({e})")
        return None

# Class-agnostic "something huge is right in front of the camera"
# signal - see the close_object note in the module docstring.
#
# Field data (7.5min of debug_monitor logging) showed the original
# single-pass 0.2-confidence/0.45-area version false-positived
# constantly - dozens of "evasion_loop:vision" fail states while the
# ultrasonic simultaneously read 60-300cm of clear air, completely
# derailing exploration. The SSD hands out low-confidence frame-filling
# boxes for walls, floors, and furniture across the room, not just
# point-blank obstacles. Three changes, each keeping most of the
# original intent:
#   - confidence 0.2 -> 0.3 and area 0.45 -> 0.55 (still far below the
#     0.5 labeled-object bar, still class-agnostic)
#   - debounced to CLOSE_OBJECT_CONFIRM_PASSES consecutive SSD passes,
#     like every other detection in this module - a real point-blank
#     obstacle stays in frame across passes; single-frame texture
#     flukes don't
# field_agent additionally cross-checks against the ultrasonic before
# treating it as an obstacle (see VISION_OBSTACLE_ULTRASONIC_CLEAR_CM
# there).
CLOSE_OBJECT_MIN_CONFIDENCE = 0.3
CLOSE_OBJECT_AREA_RATIO = 0.55
CLOSE_OBJECT_CONFIRM_PASSES = 2

# ---- overhead / head-height obstacle signal (vertical blind-spot fix) ----
# The ultrasonic rides low on the front bumper; the camera rides high on the
# pan/tilt head. That vertical gap is a real blind spot: an OVERHANGING
# obstacle - a counter lip, a table edge, a shelf - sits above the ultrasonic
# beam, so the beam passes UNDER it into clear air and reads long even as the
# head is about to smack the object's side. close_object can't disambiguate
# this from a wall across the room (both fill the frame), so field_agent
# rightly dismisses a frame-filler when the ultrasonic reads clearly long.
#
# To catch the overhang case BEFORE the head hits, we publish the geometry of
# the largest class-agnostic detection that looks like a head-height mass:
# big enough to be imminent AND sitting high in the frame (its vertical center
# in the upper portion), with the lower frame - the gap the base is driving
# INTO - relatively open. A wall across the room fills top-to-bottom; a looming
# overhang is high with clear floor below, which is exactly this signature.
# Threshold is lower than close_object's (0.55) because the extra "high in
# frame" requirement makes it specific, so it can fire earlier - the point is
# to stop before contact, not after. world_state tracks how this mass grows
# to tell a real closing overhang from a distant high wall; field_agent then
# trusts vision over a clear ultrasonic ONLY for this (see the module there).
OVERHEAD_MIN_AREA_RATIO = 0.30   # smaller than close_object: high-in-frame is the extra filter
OVERHEAD_MAX_Y_CENTER = 0.55     # bbox vertical center must sit in the upper ~half of the frame
OVERHEAD_MAX_BOTTOM = 0.90       # ...and not run to the very floor (that's a wall, not an overhang)
OVERHEAD_CONFIRM_PASSES = 2      # debounced like every other detection here


def pick_overhead(boxes):
    """From this pass's class-agnostic detections, return the largest that
    looks like a head-height overhang, or None.

    boxes: iterable of (area_ratio, y_center_frac, y_bottom_frac), all in
    0..1 frame-relative units. Pure/hardware-free so it's unit-testable off
    the robot. A qualifying box is big (>= OVERHEAD_MIN_AREA_RATIO), centered
    high (<= OVERHEAD_MAX_Y_CENTER), and doesn't extend to the floor (bottom
    <= OVERHEAD_MAX_BOTTOM) - i.e. an object looming at head height with open
    space below it, not a full-height wall."""
    best = None
    for area_ratio, y_center, y_bottom in boxes:
        if (area_ratio >= OVERHEAD_MIN_AREA_RATIO
                and y_center <= OVERHEAD_MAX_Y_CENTER
                and y_bottom <= OVERHEAD_MAX_BOTTOM):
            if best is None or area_ratio > best[0]:
                best = (area_ratio, y_center)
    if best is None:
        return None
    return {"area_ratio": round(best[0], 3), "y_center_frac": round(best[1], 3)}

# Motion gate: only bother running the SSD at all if the scene looks
# like it's actually changed since the last time we ran it, checked on
# a tiny thumbnail so the check itself costs almost nothing.
MOTION_CHECK_SIZE = (80, 60)
MOTION_DIFF_THRESHOLD = 6.0     # mean abs pixel difference (0-255 scale) to count as "changed"
FORCE_DETECT_INTERVAL = 6.0     # always refresh at least this often, motion or not

# ---- on-demand MJPEG-style stream for the web console ----
# The Pi camera is a single-owner device and this module holds it, so
# the web console cannot open it directly. Instead, WHEN A VIEWER IS
# ACTUALLY WATCHING, we JPEG-encode the frame we already captured and
# publish it on picarx/vision/frame; web_console.py caches the latest
# one and serves it over HTTP. This stays off by default and is gated by
# picarx/vision/stream_control {"enabled": bool} (the console turns it on
# only while its live view is open, and a watchdog there turns it back
# off when nobody's looking), so the encode cost is paid only during
# hands-on debugging - never during autonomous operation, preserving the
# CPU budget this module guards everywhere else.
STREAM_CONTROL_TOPIC = "picarx/vision/stream_control"
STREAM_FRAME_TOPIC = "picarx/vision/frame"
STREAM_MIN_INTERVAL = 0.2       # cap publish rate (~5 fps ceiling; loop tick bounds it lower)
STREAM_JPEG_QUALITY = 60        # small over MQTT/base64; a debug view doesn't need more

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

# ---- optional upgraded detector: YOLOv4-tiny trained on COCO ----
# The VOC model above only knows 20 classes, which is why half the
# house gets reported as "bottle"/"sofa"/"tvmonitor" - the SSD is
# forced to pick its nearest of 20 labels for everything object-shaped.
# If the three files below exist (run layer_b/setup_coco_detector.sh
# once to download them - models/ is gitignored, weights don't live in
# git), vision switches to YOLOv4-tiny with the 80-class COCO label set
# (adds e.g. couch, bed, laptop, cell phone, book, cup, remote, sink,
# refrigerator, scissors, teddy bear...). A YOLOv4-tiny pass at 320x320
# costs roughly 2-3x the SSD pass on a Pi 4 - acceptable because the
# motion gate + FORCE_DETECT_INTERVAL already keep passes rare, and the
# per-pass budget is the right place to spend for 4x the vocabulary.
# Fail-soft: files missing or cv2 too old -> the VOC model runs as before.
YOLO_DIR = "/home/picarx/layer_b/modules/models/yolov4-tiny"
YOLO_CFG = f"{YOLO_DIR}/yolov4-tiny.cfg"
YOLO_WEIGHTS = f"{YOLO_DIR}/yolov4-tiny.weights"
YOLO_NAMES = f"{YOLO_DIR}/coco.names"
YOLO_INPUT_SIZE = (320, 320)   # multiple of 32; 320 over 416 to protect the CPU budget
YOLO_NMS_THRESHOLD = 0.4


class CaffeSsdDetector:
    """The original MobileNet-SSD/VOC pipeline, behind the shared
    detector contract: detect() -> (found, close_hit) where found is
    the confidently-labeled boxes and close_hit means ANY detection
    (labeled confidently or not) fills most of the frame."""
    name = "MobileNet-SSD (Caffe, 20 VOC classes)"

    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(SSD_PROTOTXT, SSD_WEIGHTS)

    def detect(self, frame, frame_w, frame_h):
        # Resize first (cheap - CAPTURE_SIZE is already small), then let
        # blobFromImage's swapRB handle the RGB->BGR conversion on the
        # now-tiny 300x300 image instead of converting the full frame.
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, SSD_INPUT_SIZE), 0.007843, SSD_INPUT_SIZE, 127.5, swapRB=True
        )
        self.net.setInput(blob)
        detections = self.net.forward()

        found = []
        close_hit = False
        overhead_boxes = []   # (area_ratio, y_center_frac, y_bottom_frac)
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
                close_hit = True
            overhead_boxes.append((area_ratio,
                                   ((y1 + y2) / 2.0) / frame_h,
                                   y2 / float(frame_h)))

            if class_id >= len(VOC_CLASSES) or confidence < OBJECT_CONFIDENCE_THRESHOLD:
                continue
            found.append({
                "label": VOC_CLASSES[class_id],
                "confidence": confidence,
                "bbox": (x1, y1, x2 - x1, y2 - y1),
            })
        return found, close_hit, pick_overhead(overhead_boxes)


class YoloTinyDetector:
    """YOLOv4-tiny (Darknet) on COCO's 80 classes, same contract as
    CaffeSsdDetector so run() doesn't care which one is loaded."""

    def __init__(self):
        with open(YOLO_NAMES) as f:
            self.classes = [line.strip() for line in f if line.strip()]
        self.net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
        self.out_names = self.net.getUnconnectedOutLayersNames()
        self.name = f"YOLOv4-tiny (Darknet, {len(self.classes)} COCO classes)"

    def detect(self, frame, frame_w, frame_h):
        # Darknet models are trained on RGB and the picamera2 frame is
        # already RGB888, so no channel swap here (unlike the Caffe path).
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, YOLO_INPUT_SIZE, swapRB=False, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.out_names)

        boxes, confidences, class_ids = [], [], []
        for out in outs:
            for row in out:
                scores = row[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence < CLOSE_OBJECT_MIN_CONFIDENCE:
                    continue
                cx, cy, w, h = row[0] * frame_w, row[1] * frame_h, \
                    row[2] * frame_w, row[3] * frame_h
                x1 = max(0, int(cx - w / 2))
                y1 = max(0, int(cy - h / 2))
                x2 = min(frame_w, int(cx + w / 2))
                y2 = min(frame_h, int(cy + h / 2))
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                confidences.append(confidence)
                class_ids.append(class_id)

        found = []
        close_hit = False
        overhead_boxes = []   # (area_ratio, y_center_frac, y_bottom_frac)
        kept = cv2.dnn.NMSBoxes(boxes, confidences,
                                CLOSE_OBJECT_MIN_CONFIDENCE, YOLO_NMS_THRESHOLD)
        for i in np.array(kept).flatten():
            x, y, w, h = boxes[i]
            if (w * h) / float(frame_w * frame_h) > CLOSE_OBJECT_AREA_RATIO:
                close_hit = True
            overhead_boxes.append(((w * h) / float(frame_w * frame_h),
                                   (y + h / 2.0) / frame_h,
                                   (y + h) / float(frame_h)))
            if confidences[i] < OBJECT_CONFIDENCE_THRESHOLD or \
                    class_ids[i] >= len(self.classes):
                continue
            found.append({
                "label": self.classes[class_ids[i]],
                "confidence": confidences[i],
                "bbox": (x, y, w, h),
            })
        return found, close_hit, pick_overhead(overhead_boxes)


def _make_detector():
    if all(os.path.exists(p) for p in (YOLO_CFG, YOLO_WEIGHTS, YOLO_NAMES)):
        try:
            return YoloTinyDetector()
        except Exception as e:
            print(f"vision: YOLO COCO model present but failed to load ({e}) - "
                  f"falling back to the VOC model")
    else:
        print("vision: COCO model not found (run layer_b/setup_coco_detector.sh "
              "for 80-class detection) - using the 20-class VOC model")
    return CaffeSsdDetector()


def contested_label(label_counts):
    """The runner-up label when a track's classification is a genuine two-way
    tie (the winner isn't dominant, a specific rival is close behind), else
    None. This is the ambiguity signal curiosity.py turns into "is that a X
    or a Y?" - pure/hardware-free so it's unit-testable off the robot.

    label_counts: {label: votes} accumulated over the track's lifetime."""
    if not label_counts or len(label_counts) < 2:
        return None
    ranked = sorted(label_counts.items(), key=lambda kv: kv[1], reverse=True)
    (top_label, top_n), (alt_label, alt_n) = ranked[0], ranked[1]
    if top_n > 0 and alt_label != top_label and alt_n >= LABEL_CONTEST_RATIO * top_n:
        return alt_label
    return None


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
    detector = _make_detector()
    tracker = CentroidTracker()

    # On-demand console stream state. Toggled from the MQTT callback
    # thread; the loop only reads it, so a plain dict is enough (a stale
    # read just means one extra/fewer frame, which is harmless).
    stream = {"enabled": False}
    def on_stream_control(payload):
        stream["enabled"] = bool(payload.get("enabled", False))
        print(f"vision stream {'ENABLED' if stream['enabled'] else 'disabled'} (console live view)")
    bus.subscribe(STREAM_CONTROL_TOPIC, on_stream_control)

    # Low-power curtailment (same mutable-dict-from-callback pattern as the
    # stream flag above): while active, the heavy SSD pass runs far less often.
    low_power = {"active": False, "critical": False}
    def on_low_power(payload):
        low_power["active"] = bool(payload.get("active", False))
        low_power["critical"] = bool(payload.get("critical", False))
        state = ("CRITICAL - throttling object + face detection" if low_power["critical"]
                 else "ON - throttling object + face detection" if low_power["active"]
                 else "off")
        print(f"vision: low-power {state}")
    bus.subscribe(LOW_POWER_TOPIC, on_low_power)

    # On-board label memory (recognition tier 2): visual signatures of tracked
    # objects, cached by track id on each detection pass, plus the persistent
    # signature->label store a human/LLM teaches. A label arriving on
    # LABEL_TOPIC (from curiosity or the console, carrying the object id) is
    # remembered against that object's cached look, so an uncertain detection
    # that resembles it later gets the taught label before we ask again or
    # phone the cloud. All fail-soft - any error here degrades to plain
    # detector labels and never disturbs the real-time loop.
    memory = label_memory.LabelMemory()
    sig_cache = {}   # track id -> (signature, last_seen_ts)
    def on_label(payload):
        oid = payload.get("object_id")
        correct = (payload.get("correct_label") or "").strip().lower()
        if not oid or not correct:
            return
        entry = sig_cache.get(oid)
        if entry is None:
            print(f"vision: label '{correct}' for {oid} but its signature has aged out")
            return
        source = {"web": "user", "voice": "user", "llm": "llm",
                  "coach": "coach"}.get(payload.get("origin"), "user")
        try:
            if memory.remember(entry[0], correct, source):
                print(f"vision: learned '{correct}' by sight "
                      f"(source {source}, {len(memory)} memories)")
        except Exception as e:
            print(f"vision: failed to remember '{correct}': {e}")
    bus.subscribe(LABEL_TOPIC, on_label)

    print(f"vision basic module running ({detector.name}, {len(memory)} label "
          f"memories), publishing to picarx/vision/faces and picarx/vision/objects")

    face_streak = 0
    last_face_crop = 0.0
    last_face_detect = 0.0
    last_object_detect = 0.0
    last_forced_detect = 0.0
    last_motion_thumb = None
    last_stream_pub = 0.0
    close_object = False   # persists between ticks where the SSD didn't run
    close_streak = 0       # consecutive SSD passes with a frame-filling detection
    overhead_object = None # persists between ticks; last confirmed head-height mass geometry
    overhead_streak = 0    # consecutive SSD passes with an overhead-looking mass
    scene_motion = None    # last motion-thumb mean abs diff (for stuck detection downstream)
    objects_payload = []   # rebuilt only on SSD passes (see below); cached and
                           # republished on the intervening throttled ticks

    while True:
        frame = picam2.capture_array()
        frame_h, frame_w = frame.shape[:2]

        # ---------- face detection (debounced; throttled in low power) ----------
        # gray is computed every tick regardless - the object-detection motion
        # check and the face crops both reuse it - but the expensive Haar pass
        # itself backs off to 1s/2s while conserving power (see the FACE_DETECT
        # interval constants).
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        face_now = time.time()
        face_interval = (CRITICAL_FACE_DETECT_INTERVAL if low_power["critical"]
                         else LOW_POWER_FACE_DETECT_INTERVAL if low_power["active"]
                         else 0.0)
        if face_now - last_face_detect >= face_interval:
            last_face_detect = face_now
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
                # Face crop for person_memory (throttled - see FACE_CROP_*).
                crop_now = time.time()
                if crop_now - last_face_crop >= FACE_CROP_INTERVAL:
                    last_face_crop = crop_now
                    mx, my = int(w * FACE_CROP_MARGIN), int(h * FACE_CROP_MARGIN)
                    x1, y1 = max(0, x - mx), max(0, y - my)
                    x2, y2 = min(frame_w, x + w + mx), min(frame_h, y + h + my)
                    if x2 > x1 and y2 > y1:
                        crop = cv2.resize(gray[y1:y2, x1:x2], FACE_CROP_SIZE)
                        ok, buf = cv2.imencode(
                            ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        if ok:
                            bus.publish(FACE_CROP_TOPIC, {
                                "jpeg": base64.b64encode(buf.tobytes()).decode("ascii"),
                                "w": FACE_CROP_SIZE[0], "h": FACE_CROP_SIZE[1],
                                "ts": crop_now,
                            })
            else:
                bus.publish("picarx/vision/faces", {"detected": False})

        # ---------- object detection (motion-gated + throttled) ----------
        now = time.time()
        detect_interval = (LOW_POWER_OBJECT_DETECT_INTERVAL if low_power["active"]
                           else OBJECT_DETECT_INTERVAL)
        if now - last_object_detect >= detect_interval:
            last_object_detect = now

            # Cheap motion check on a tiny thumbnail - reuses the gray
            # frame we already computed for the face cascade above, so
            # this costs almost nothing extra.
            thumb = cv2.resize(gray, MOTION_CHECK_SIZE)
            if last_motion_thumb is None:
                should_run_ssd = True
            else:
                diff = cv2.absdiff(thumb, last_motion_thumb)
                scene_motion = float(diff.mean())
                moved = scene_motion > MOTION_DIFF_THRESHOLD
                should_run_ssd = moved or (now - last_forced_detect) > FORCE_DETECT_INTERVAL
            last_motion_thumb = thumb

            if should_run_ssd:
                last_forced_detect = now
                found, close_hit_this_pass, overhead_this_pass = \
                    detector.detect(frame, frame_w, frame_h)

                close_streak = close_streak + 1 if close_hit_this_pass else 0
                close_object = close_streak >= CLOSE_OBJECT_CONFIRM_PASSES

                # Debounce the overhead mass exactly like close_object: a real
                # overhang stays in frame across passes, a single-frame texture
                # fluke doesn't. We keep the latest geometry once confirmed.
                overhead_streak = overhead_streak + 1 if overhead_this_pass else 0
                overhead_object = (overhead_this_pass
                                   if overhead_streak >= OVERHEAD_CONFIRM_PASSES else None)

                tracker.update(found, now)

                # Refresh the visual signature of each confirmed track (only on
                # detection passes, so it rides the same throttle the SSD does)
                # and age out signatures for tracks long gone.
                for tid, t in tracker.confirmed_tracks().items():
                    x, y, w, h = t["bbox"]
                    crop = frame[max(0, y):y + h, max(0, x):x + w]
                    sig = visual_signature(crop)
                    if sig is not None:
                        sig_cache[tid] = (sig, now)
                for tid in [k for k, (_, ts) in sig_cache.items()
                            if now - ts > SIG_CACHE_TTL]:
                    del sig_cache[tid]

                # Rebuild the objects payload only here, on the SSD pass, when
                # the tracker state actually changed. confirmed_tracks() is a
                # pure filtered view - it never ages tracks on its own - so the
                # confirmed set is identical between passes and the payload can
                # be safely cached and republished on the intervening 0.3s
                # ticks. This keeps the costly resolve_label/cosine match off
                # the ~5 idle ticks per detection update.
                objects_payload = []
                for tid, t in tracker.confirmed_tracks().items():
                    x, y, w, h = t["bbox"]
                    alt = contested_label(t.get("label_counts"))
                    # Recognition tiers 1->2: keep a confident detector label,
                    # but let the on-board memory relabel an UNCERTAIN one (see
                    # resolve_label). label_source tells consumers who decided:
                    # detector or memory.
                    sig_entry = sig_cache.get(tid)
                    try:
                        label, label_source, alt = label_memory.resolve_label(
                            memory, sig_entry[0] if sig_entry else None,
                            t["label"], t["confidence"], alt, LABEL_LOW_CONF)
                    except Exception as e:
                        print(f"vision: label resolve failed ({e})")
                        label, label_source = t["label"], "detector"
                    objects_payload.append({
                        "id": tid,
                        "label": label,
                        "label_source": label_source,   # "detector" | "memory"
                        # Runner-up label when the vote is genuinely split (else
                        # None), so downstream can ask a human to disambiguate
                        # this object. Cleared once memory resolves the object.
                        "alt_label": alt,
                        "confidence": t["confidence"],
                        "x": x, "y": y, "w": w, "h": h,
                        "frame_width": frame_w,
                        "frame_height": frame_h,
                        "area_ratio": (w * h) / float(frame_w * frame_h),
                        "center_offset": (x + w // 2) - (frame_w // 2),
                        "first_seen": t["first_seen"],
                        "last_seen": t["last_seen"],
                    })

        bus.publish("picarx/vision/objects", {
            "objects": objects_payload,
            "close_object": close_object,
            "overhead": overhead_object,   # None, or {"area_ratio","y_center_frac"}
            "scene_motion": scene_motion,
        })

        # ---------- on-demand console stream (only while a viewer watches) ----------
        if stream["enabled"] and now - last_stream_pub >= STREAM_MIN_INTERVAL:
            last_stream_pub = now
            # picamera2 gives RGB888; cv2.imencode assumes BGR, so convert
            # first or the preview shows swapped red/blue channels.
            ok, buf = cv2.imencode(
                ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, STREAM_JPEG_QUALITY])
            if ok:
                bus.publish(STREAM_FRAME_TOPIC, {
                    "jpeg": base64.b64encode(buf.tobytes()).decode("ascii"),
                    "w": frame_w, "h": frame_h, "ts": now,
                })

        time.sleep(DETECT_INTERVAL)


if __name__ == "__main__":
    run()
