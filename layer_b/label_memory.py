#!/usr/bin/env python3
# /home/picarx/layer_b/label_memory.py
"""
On-board visual label memory - the middle tier of object recognition,
between the fixed-vocabulary detector and the cloud LLM.

The camera detector (vision_basic.py) only ever emits its trained classes
(20 VOC, or 80 COCO once setup_coco_detector.sh has run), so it MUST map
everything to its nearest known label - that's why a foot becomes a "cat"
and half the house becomes a "bottle". This store lets a human (or, as a
last resort, the LLM) teach the robot what a specific thing actually is:
a cheap visual SIGNATURE of the object (computed in vision_basic, since
that's the only process with pixels) is stored against the taught label.
Next time the detector is UNSURE about something that looks similar, we
prefer the remembered label instead of the detector's nearest guess or a
paid LLM call.

Deliberately simple and hardware-light (pure Python, no numpy/cv2, so it
unit-tests off-robot and adds negligible CPU to the vision loop):
  - signatures are compared by cosine similarity; a match must clear
    MATCH_THRESHOLD to be trusted, so a weak resemblance falls through to
    the next tier rather than confidently mislabeling.
  - re-teaching the same object (a near-duplicate signature with the same
    label) MERGES into the existing entry as a running average and bumps
    its count, so repetition sharpens a memory instead of cluttering it.
  - entries carry a source (user > coach > llm) and, when the store is
    full, the least-trusted and oldest are evicted first - a human's
    label outlives an LLM guess.

Ownership: vision_basic.py is the sole writer (it holds the pixels and the
signatures); it also reads to relabel uncertain detections. Persisted as
plain JSON in data/ (gitignored, like every other on-robot store).
"""
import json
import math
import os
import threading
import time

DEFAULT_PATH = "/home/picarx/layer_b/data/label_memory.json"

# Cosine similarity a candidate signature must reach to adopt a remembered
# label. High on purpose: a wrong confident relabel is worse than falling
# through to asking a human / the LLM, so we only trust a strong match.
MATCH_THRESHOLD = 0.86
# Above this, a new signature for the SAME label is treated as the same
# object seen again and merged (running average) rather than stored twice.
MERGE_THRESHOLD = 0.95
MAX_ENTRIES = 200

# Whose label wins when the store is full and something must be evicted,
# and which source may overwrite another on a merge. A person outranks the
# coach, which outranks an LLM guess, which outranks the raw detector.
SOURCE_TRUST = {"user": 3, "coach": 2, "llm": 1, "detector": 0}


def cosine(a, b):
    """Cosine similarity of two equal-length vectors (lists of floats).
    0.0 for empty, length-mismatched, or zero-norm inputs - a non-match,
    which is the safe default (fall through to the next recognition tier)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class LabelMemory:
    def __init__(self, path=DEFAULT_PATH, match_threshold=MATCH_THRESHOLD):
        self.path = path
        self.match_threshold = match_threshold
        self.lock = threading.Lock()
        self.entries = self._load()   # [{sig, label, source, count, updated_at}, ...]

    def __len__(self):
        return len(self.entries)

    # ---------- persistence (fail-soft) ----------

    def _load(self):
        try:
            with open(self.path) as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return []

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = f"{self.path}.tmp"
            with open(tmp, "w") as f:
                json.dump(self.entries, f)
            os.replace(tmp, self.path)
        except OSError as e:
            print(f"LabelMemory: could not persist ({e})")

    # ---------- writer ----------

    def remember(self, sig, label, source="user"):
        """Teach (or reinforce) that `sig` looks like `label`. A near-identical
        signature already stored under the same label is merged (averaged,
        count bumped, source upgraded if the new source is more trusted);
        otherwise a new entry is added and the store is capped by evicting the
        least-trusted, oldest entries. Returns True if anything was stored."""
        label = (label or "").strip().lower()
        sig = [float(x) for x in (sig or [])]
        if not label or not sig:
            return False
        now = time.time()
        with self.lock:
            for e in self.entries:
                if e["label"] == label and cosine(e["sig"], sig) >= MERGE_THRESHOLD:
                    n = e.get("count", 1)
                    e["sig"] = [(x * n + y) / (n + 1) for x, y in zip(e["sig"], sig)]
                    e["count"] = n + 1
                    e["updated_at"] = now
                    if SOURCE_TRUST.get(source, 0) > SOURCE_TRUST.get(e["source"], 0):
                        e["source"] = source
                    self._save()
                    return True
            self.entries.append({"sig": sig, "label": label, "source": source,
                                 "count": 1, "updated_at": now})
            if len(self.entries) > MAX_ENTRIES:
                # Keep the most trusted, then the most recent.
                self.entries.sort(
                    key=lambda e: (SOURCE_TRUST.get(e["source"], 0), e["updated_at"]))
                self.entries = self.entries[-MAX_ENTRIES:]
            self._save()
            return True

    # ---------- reader ----------

    def match(self, sig):
        """Best remembered (label, score, source) whose signature is at least
        match_threshold similar to `sig`, or None. Highest similarity wins."""
        sig = [float(x) for x in (sig or [])]
        if not sig:
            return None
        best = None
        with self.lock:
            for e in self.entries:
                score = cosine(e["sig"], sig)
                if score >= self.match_threshold and (best is None or score > best[1]):
                    best = (e["label"], score, e["source"])
        return best


def resolve_label(memory, sig, label, confidence, alt_label, low_conf_threshold):
    """One detection's label after the on-board memory tier.

    A confident, unambiguous detection is kept as-is (the detector tier
    already won - never spend a memory lookup or override it). Only when the
    detection is UNCERTAIN - a contested vote (alt_label) or confidence below
    low_conf_threshold - do we consult memory; a trusted match adopts the
    remembered label and clears the ambiguity, otherwise the detector's guess
    stands and stays flagged uncertain for the human/LLM tiers downstream.

    Returns (label, source, alt_label). Pure and hardware-free."""
    uncertain = bool(alt_label) or (confidence is not None and confidence < low_conf_threshold)
    if not uncertain or sig is None or memory is None:
        return label, "detector", alt_label
    hit = memory.match(sig)
    if hit:
        return hit[0], "memory", None
    return label, "detector", alt_label
