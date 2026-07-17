#!/usr/bin/env python3
# /home/picarx/layer_b/modules/person_memory.py
"""
Person memory (Layer B) - learning WHO people are, so the robot can tell
two people apart and greet them by name.

Pipeline:
  vision_basic.py publishes small grayscale face crops on
  picarx/vision/face_crop while a face is confirmed (throttled to ~1/s).
  This module runs an OpenCV LBPH face recognizer over each crop and,
  once the same identity has been predicted STABLE_HITS times in a row,
  publishes it on:

    picarx/vision/person  {"name", "confidence", "ts"}

  Consumers: field_agent greets by name and answers "who am I?";
  world_state folds it into the snapshot; companion addresses the person
  by name in conversation. All of them fall back to plain anonymous
  face detection when this module is absent - it is pure enrichment.

Enrollment is by voice, on picarx/audio/heard (no wake phrase needed -
the phrases below are unmistakably addressed to the robot):

  "remember me, I am lucas" / "remember my face, I'm lucas" /
  "my name is lucas"            -> collects ENROLL_SAMPLES face crops
                                   over a few seconds, trains, confirms.
  "forget lucas" / "forget me"  -> deletes that person's samples.

Why LBPH: it's the only face recognizer that ships inside OpenCV itself
(cv2.face, in the contrib modules Debian's python3-opencv includes), it
trains in milliseconds on a handful of 100x100 crops, and prediction is
microseconds - no new model downloads, no neural net fighting the SSD
for CPU. It is NOT a deep face embedding: it distinguishes the handful
of people in a household under reasonable lighting, which is exactly
this robot's job - not biometric security. MATCH_MAX_DISTANCE is the
honesty knob: predictions worse than it are reported as nobody rather
than the nearest wrong name.

Fail-soft: if cv2.face is missing (an opencv build without contrib),
the module stays up, says so once when someone tries to enroll, and
publishes nothing - everything downstream behaves exactly as before.

Storage (this module is the sole writer):
  data/people/<name>/*.png   - enrolled face samples
  data/people/people.json    - {"names": {"<label int>": "<name>"}}
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus

import base64
import json
import re
import threading
import time

PEOPLE_DIR = "/home/picarx/layer_b/data/people"
PEOPLE_JSON = f"{PEOPLE_DIR}/people.json"

FACE_CROP_TOPIC = "picarx/vision/face_crop"
PERSON_TOPIC = "picarx/vision/person"
SPEAK_TOPIC = "picarx/audio/speak"

def known_people(people_dir=PEOPLE_DIR):
    """Names of enrolled people (one directory per person). Module-level
    so light consumers (field_agent's reports, the web console) can ask
    without constructing a recognizer. Fail-soft to []."""
    try:
        return sorted(d for d in os.listdir(people_dir)
                      if os.path.isdir(os.path.join(people_dir, d)))
    except OSError:
        return []


ENROLL_SAMPLES = 8          # face crops collected per enrollment
ENROLL_TIMEOUT = 25.0       # give up if the face wanders off mid-enrollment
MAX_SAMPLES_PER_PERSON = 40 # rolling cap; newest samples win
MATCH_MAX_DISTANCE = 70.0   # LBPH distance above this = "I don't know you"
STABLE_HITS = 2             # consecutive same-name predictions before publishing
REPUBLISH_INTERVAL = 3.0    # re-assert a stable identity at most this often


# ---------------------------------------------------------------------
# Voice command parsing (pure, unit-testable)
# ---------------------------------------------------------------------
# Names are a single word (Vosk emits lowercase words; a first name is
# what a greeting needs). Parsed from RAW text - canonicalization is
# lossy on names.

_ENROLL_PATTERNS = (
    re.compile(r"\bremember (?:me|my face)\b[,.]?\s*"
               r"(?:i(?:'m| am)|my name is|it(?:'s| is)|this is)\s+([a-z]+)"),
    re.compile(r"\bmy name is ([a-z]+)"),
    re.compile(r"\bi(?:'m| am) ([a-z]+)[,.]?\s*remember (?:me|my face)\b"),
)

# Words that end up where a name goes but never are one ("my name is
# actually..."), so a mis-parse doesn't enroll garbage.
_NOT_NAMES = {"not", "actually", "really", "the", "a", "still", "now"}

_FORGET_PATTERN = re.compile(r"\bforget (?:about )?(me|[a-z]+)\b")


def parse_enroll_command(text):
    """Name to enroll from an utterance, or None."""
    text = (text or "").lower()
    for pattern in _ENROLL_PATTERNS:
        m = pattern.search(text)
        if m and m.group(1) not in _NOT_NAMES:
            return m.group(1)
    return None


def parse_forget_command(text):
    """Name to forget (or the literal 'me') from an utterance, or None.
    Requires the utterance to be ABOUT forgetting a person, so 'don't
    forget to charge' doesn't wipe anyone."""
    text = (text or "").lower()
    if "forget" not in text or "don't forget" in text or "do not forget" in text:
        return None
    m = _FORGET_PATTERN.search(text)
    if not m:
        return None
    name = m.group(1)
    return None if name in _NOT_NAMES or name in ("to", "it", "that") else name


# ---------------------------------------------------------------------
# LBPH recognizer wrapper (fail-soft, same pattern as embedding_util)
# ---------------------------------------------------------------------

class FaceRecognizer:
    """Owns the sample store and the LBPH model. available=False when
    OpenCV's contrib face module isn't installed - callers then skip
    recognition entirely and the robot behaves as before."""

    def __init__(self, people_dir=PEOPLE_DIR):
        self.people_dir = people_dir
        self.available = False
        self._cv2 = None
        self._np = None
        self._model = None
        self._names = {}          # int label -> name
        try:
            import cv2
            import numpy as np
        except ImportError as e:
            print(f"Person memory: disabled (missing dependency: {e}).")
            return
        if not hasattr(cv2, "face"):
            print("Person memory: disabled (this OpenCV build has no cv2.face - "
                  "install opencv-contrib / Debian python3-opencv).")
            return
        self._cv2 = cv2
        self._np = np
        self.available = True
        self.retrain()

    # ----- storage -----

    def _people(self):
        """Sorted list of enrolled names (one directory per person)."""
        return known_people(self.people_dir)

    def known_names(self):
        return self._people()

    def decode_crop(self, jpeg_b64):
        """Grayscale numpy image from a face_crop payload, or None."""
        if not self.available or not jpeg_b64:
            return None
        try:
            raw = base64.b64decode(jpeg_b64)
            arr = self._np.frombuffer(raw, dtype=self._np.uint8)
            img = self._cv2.imdecode(arr, self._cv2.IMREAD_GRAYSCALE)
            return img if img is not None and img.size else None
        except Exception:
            return None

    def add_sample(self, name, gray):
        """Persist one face sample for `name` (rolling cap, newest win)."""
        person_dir = os.path.join(self.people_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        path = os.path.join(person_dir, f"{time.time():.3f}.png")
        self._cv2.imwrite(path, gray)
        samples = sorted(os.listdir(person_dir))
        for stale in samples[:-MAX_SAMPLES_PER_PERSON]:
            try:
                os.remove(os.path.join(person_dir, stale))
            except OSError:
                pass

    def forget(self, name):
        """Delete a person's samples. Returns True if they existed."""
        person_dir = os.path.join(self.people_dir, name)
        if not os.path.isdir(person_dir):
            return False
        for f in os.listdir(person_dir):
            try:
                os.remove(os.path.join(person_dir, f))
            except OSError:
                pass
        try:
            os.rmdir(person_dir)
        except OSError:
            pass
        if self.available:
            self.retrain()
        return True

    # ----- model -----

    def retrain(self):
        """Rebuild the LBPH model from every stored sample. Fast (ms) at
        household scale; called after each enrollment/forget."""
        if not self.available:
            return
        images, labels, names = [], [], {}
        for label, name in enumerate(self._people()):
            names[label] = name
            person_dir = os.path.join(self.people_dir, name)
            for fname in os.listdir(person_dir):
                img = self._cv2.imread(os.path.join(person_dir, fname),
                                       self._cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
                    labels.append(label)
        self._names = names
        if not images:
            self._model = None
            return
        model = self._cv2.face.LBPHFaceRecognizer_create()
        model.train(images, self._np.array(labels))
        self._model = model
        self._save_names()

    def _save_names(self):
        try:
            os.makedirs(self.people_dir, exist_ok=True)
            final = os.path.join(self.people_dir, "people.json")
            tmp = f"{final}.tmp"
            with open(tmp, "w") as f:
                json.dump({"names": {str(k): v for k, v in self._names.items()}}, f)
            os.replace(tmp, final)
        except OSError as e:
            print(f"Person memory: couldn't persist names index: {e}")

    def predict(self, gray):
        """(name, distance) for a face crop, or None when the model is
        empty or the match is worse than MATCH_MAX_DISTANCE (unknown
        person - honesty beats a wrong name)."""
        if not self.available or self._model is None or gray is None:
            return None
        try:
            label, distance = self._model.predict(gray)
        except Exception:
            return None
        name = self._names.get(label)
        if name is None or distance > MATCH_MAX_DISTANCE:
            return None
        return name, float(distance)


# ---------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------

class PersonMemory:
    def __init__(self, recognizer=None):
        self.bus = Bus()
        self.lock = threading.Lock()
        self.recognizer = recognizer if recognizer is not None else FaceRecognizer()
        self._warned_unavailable = False
        # Enrollment session, or None: {"name", "collected", "deadline"}
        self.enrolling = None
        # Stable-identity debounce: publish only after STABLE_HITS
        # consecutive crops agree, so one lookalike frame can't misname.
        self._streak_name = None
        self._streak = 0
        self._last_published_name = None
        self._last_published_at = 0.0

    def _say(self, text):
        self.bus.publish(SPEAK_TOPIC, {"text": text, "ts": time.time()})

    # ---------- inbound: voice ----------

    def on_heard(self, payload):
        text = (payload.get("text") or "").lower()
        if not text:
            return
        name = parse_enroll_command(text)
        if name:
            self._start_enrollment(name)
            return
        target = parse_forget_command(text)
        if target:
            self._forget(target)

    def _start_enrollment(self, name):
        if not self.recognizer.available:
            if not self._warned_unavailable:
                self._warned_unavailable = True
                self._say("I'd love to learn your face, but my face memory "
                          "isn't installed on this hardware.")
            return
        with self.lock:
            self.enrolling = {"name": name, "collected": 0,
                              "deadline": time.time() + ENROLL_TIMEOUT}
        print(f"Person memory: enrolling '{name}'")
        self._say(f"Nice to meet you, {name}. Look at me for a few seconds "
                  f"while I memorize your face.")

    def _forget(self, target):
        if target == "me":
            with self.lock:
                target = self._last_published_name
            if not target:
                self._say("I'm not sure who you are right now, so tell me the "
                          "name to forget.")
                return
        if self.recognizer.forget(target):
            with self.lock:
                self._streak_name, self._streak = None, 0
                if self._last_published_name == target:
                    self._last_published_name = None
            print(f"Person memory: forgot '{target}'")
            self._say(f"Okay, I've forgotten {target}.")
        else:
            self._say(f"I don't know anyone called {target}.")

    # ---------- inbound: face crops ----------

    def on_face_crop(self, payload):
        gray = self.recognizer.decode_crop(payload.get("jpeg"))
        if gray is None:
            return
        now = time.time()
        with self.lock:
            session = dict(self.enrolling) if self.enrolling else None
        if session is not None:
            self._enroll_step(session, gray, now)
            return
        self._identify(gray, now)

    def _enroll_step(self, session, gray, now):
        if now > session["deadline"]:
            with self.lock:
                self.enrolling = None
            self._say("I lost your face before I finished memorizing it. "
                      "Try again when you're in front of me.")
            return
        self.recognizer.add_sample(session["name"], gray)
        session["collected"] += 1
        with self.lock:
            self.enrolling = session if session["collected"] < ENROLL_SAMPLES else None
        if session["collected"] >= ENROLL_SAMPLES:
            self.recognizer.retrain()
            print(f"Person memory: enrolled '{session['name']}' "
                  f"({ENROLL_SAMPLES} samples)")
            self._say(f"Got it. I'll recognize you now, {session['name']}.")
            self.bus.publish("picarx/decision", {
                "source": "person_memory", "kind": "person_enrolled",
                "choice": {"name": session["name"]},
                "reason": "they introduced themselves and asked me to remember them",
                "ts": now})

    def _identify(self, gray, now):
        result = self.recognizer.predict(gray)
        name = result[0] if result else None
        publish = False
        with self.lock:
            if name is not None and name == self._streak_name:
                self._streak += 1
            else:
                self._streak_name, self._streak = name, 1
            stable = name is not None and self._streak >= STABLE_HITS
            if stable and (name != self._last_published_name
                           or now - self._last_published_at >= REPUBLISH_INTERVAL):
                self._last_published_name = name
                self._last_published_at = now
                publish = True
        if publish:
            self.bus.publish(PERSON_TOPIC, {
                "name": name, "confidence": round(result[1], 1), "ts": now})

    # ---------- main loop ----------

    def run(self):
        self.bus.subscribe("picarx/audio/heard", self.on_heard)
        self.bus.subscribe(FACE_CROP_TOPIC, self.on_face_crop)
        known = self.recognizer.known_names()
        print(f"Person memory active (recognizer "
              f"{'ready' if self.recognizer.available else 'UNAVAILABLE - degraded'}, "
              f"{len(known)} people known: {', '.join(known) or 'none'})")
        while True:
            time.sleep(5)


if __name__ == "__main__":
    PersonMemory().run()
