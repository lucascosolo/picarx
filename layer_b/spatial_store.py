#!/usr/bin/env python3
# layer_b/spatial_store.py
"""
Topological spatial memory - the robot's map of WHERE things happen,
as opposed to WHAT it has learned (semantic.db) or the raw event
stream (events.db).

There is no odometry or SLAM on this platform, so a "location" is a
*perceptual fingerprint*, not a coordinate: the object labels and stable
detector features a look-around head sweep saw, plus a coarse open-space
bucket from the ultrasonic. Two sweeps that see the same things in the same
relative arrangement are treated as the same place. That is honest about
the hardware - it can be fooled by two identical-looking corners, but it
can never claim centimeter positions it doesn't have.

Ownership rules (mirrors semantic_store.py's convention):
  - location_graph.py is the SOLE writer to spatial.db. It opens the
    store with readonly=False, which creates the schema.
  - Everything else (explorer.py, goal_manager.py, field_agent.py)
    opens readonly=True and degrades to "no map yet" if the DB is
    missing.

The roadmap sketched these tables inside semantic.db; they live in
their own file instead so the one-writer-per-database rule that the
rest of the codebase is built on stays intact (reflection.py keeps
sole ownership of semantic.db).
"""
import json
import os
import sqlite3
import time

import robot_config

DB_DIR = robot_config.data_path()
DB_PATH = f"{DB_DIR}/spatial.db"

# A location is a *belief*, not an answer the robot must always produce.
# These deliberately leave a little room between "some overlap" and a
# confident recognition.  In particular, a sofa and a range bucket are not
# enough to tell two living-room corners apart.
MATCH_THRESHOLD = 0.70
MATCH_MARGIN = 0.15
MIN_DISTINCT_LANDMARKS = 2
# A re-scan within this many seconds of the last visit refreshes the
# location but doesn't count as a new "visit" (one wander session
# re-scanning every 25s isn't ten visits).
REVISIT_GAP_SEC = 120.0
MAX_CANDIDATE_SCORES = 3

_SCHEMA = """
CREATE TABLE IF NOT EXISTS locations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    label TEXT NOT NULL,
    fingerprint_json TEXT NOT NULL,
    discovered_at REAL NOT NULL,
    last_visited_at REAL NOT NULL,
    visit_count INTEGER NOT NULL DEFAULT 1,
    veto_count INTEGER NOT NULL DEFAULT 0,
    coach_wins INTEGER NOT NULL DEFAULT 0,
    coach_losses INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS edges (
    a INTEGER NOT NULL,
    b INTEGER NOT NULL,
    traversals INTEGER NOT NULL DEFAULT 1,
    last_traversed_at REAL NOT NULL,
    PRIMARY KEY (a, b)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS sightings (
    location_id INTEGER NOT NULL,
    label TEXT NOT NULL,
    times_seen INTEGER NOT NULL DEFAULT 1,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    PRIMARY KEY (location_id, label)
);
"""


# ---------- pure fingerprint logic (no DB, unit-testable) ----------

def _clamp01(value, default=0.0):
    """Return a finite confidence/area value in the JSON-safe 0..1 range."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    if value != value:  # NaN
        return default
    return max(0.0, min(1.0, value))


def _lateral_bin(sighting):
    """Quantize a sighting's lateral position to left/centre/right.

    The scan code historically supplied pan angles in degrees.  Newer
    producers may provide a normalized ``center_offset`` (or an explicit
    ``bin``); accepting all three keeps this function useful to old logs and
    makes the resulting fingerprint deterministic across camera resolutions.
    """
    explicit = sighting.get("bin", sighting.get("lateral_bin"))
    if isinstance(explicit, str):
        side = explicit.strip().lower()
        if side in ("l", "left", "-1"):
            return -1
        if side in ("r", "right", "+1", "1"):
            return 1
        if side in ("c", "center", "centre", "0"):
            return 0
    try:
        if explicit is not None:
            return -1 if float(explicit) < 0 else (1 if float(explicit) > 0 else 0)
    except (TypeError, ValueError):
        pass

    value = sighting.get("center_offset", sighting.get("normalized_offset"))
    if value is None:
        value = sighting.get("pan", 0)
        # The stock scan has a +/-70 degree mechanical range.  Dividing by
        # 70 makes the quantizer independent of the actual angle used by a
        # quick or a full sweep.  Values already in [-1, 1] are normalized.
        try:
            value = float(value)
            if abs(value) > 1.0:
                value /= 70.0
        except (TypeError, ValueError):
            value = 0.0
    else:
        try:
            value = float(value)
            # Pixel offsets are normalized when frame width is available.
            width = sighting.get("frame_width")
            if width and abs(value) > 1.0:
                value /= (float(width) / 2.0)
        except (TypeError, ValueError):
            value = 0.0
    # Keep the sign semantics of the historical side tag: centre is the
    # optical axis, while any measurable left/right offset remains useful
    # evidence for distinguishing two otherwise similar corners.
    return -1 if value < 0 else (1 if value > 0 else 0)


def _scan_imu_delta(sightings, imu_delta):
    """Find an optional short-term pose delta without requiring a new caller."""
    if imu_delta is not None:
        return _normalize_imu(imu_delta)
    for sighting in sightings or []:
        if not isinstance(sighting, dict):
            continue
        for key in ("imu_delta", "pose_delta", "imu"):
            if sighting.get(key) is not None:
                return _normalize_imu(sighting[key])
    return None


def _normalize_imu(value):
    """Make a pose delta compact, JSON-safe, and order-independent."""
    if isinstance(value, dict):
        out = {}
        for key in sorted(value, key=str):
            item = value[key]
            try:
                item = float(item)
            except (TypeError, ValueError):
                if isinstance(item, (dict, list, tuple)):
                    item = _normalize_imu(item)
            out[str(key)] = item
        return out
    if isinstance(value, (list, tuple)):
        return [_normalize_imu(item) for item in value]
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def fingerprint_from_scan(sightings, distance_cm=None, imu_delta=None):
    """Collapse a room scan into a backwards-compatible rich fingerprint.

    ``labels`` remains the historical list of side-tagged strings.  The
    ``metadata`` member carries stable matching features: confidence, a
    quantized lateral bin, apparent object area, and an optional short-term
    IMU/pose delta.  Inputs may contain either the old ``labels: ["chair"]``
    form or detector records (``{"name": ..., "confidence": ...}``).
    """
    sightings = list(sightings or [])
    labels = set()
    rich_labels = []
    for sighting in sightings:
        if not isinstance(sighting, dict):
            continue
        lateral = _lateral_bin(sighting)
        # Preserve the exact historical pan-sign label tag.  Matching uses
        # the quantized bin below, but consumers that display/parse ``labels``
        # should see the same l:/c:/r: value they saw before this refactor.
        try:
            pan = float(sighting.get("pan", 0))
        except (TypeError, ValueError):
            pan = 0.0
        legacy_side = "l" if pan < 0 else ("r" if pan > 0 else "c")
        # A detector-rich producer generally calls these ``objects``; a
        # lightweight scan supplies ``labels``.  Prefer objects when present
        # so confidence and area are not discarded by a parallel label list.
        raw_labels = sighting.get("objects")
        if not raw_labels:
            raw_labels = sighting.get("labels")
        if raw_labels is None and sighting.get("label") is not None:
            raw_labels = [sighting.get("label")]
        if raw_labels is None:
            raw_labels = []
        if isinstance(raw_labels, (str, dict)):
            raw_labels = [raw_labels]
        for raw in raw_labels:
            if isinstance(raw, dict):
                name = raw.get("name", raw.get("label"))
                conf = raw.get("conf", raw.get(
                    "confidence", raw.get("score", sighting.get(
                        "conf", sighting.get("confidence", sighting.get(
                            "confidences", 1.0))))))
                area = raw.get("area", raw.get(
                    "area_ratio", sighting.get("area", sighting.get(
                        "area_ratio", sighting.get("areas", sighting.get(
                            "area_ratios", 0.0))))))
                item_bin = raw.get("bin", raw.get("lateral_bin", lateral))
                # Reuse the quantizer for explicit object-level offsets.
                if item_bin != lateral:
                    item_bin = _lateral_bin({**sighting, **raw})
            else:
                name = raw
                conf = sighting.get("conf", sighting.get("confidence",
                                      sighting.get("confidences", 1.0)))
                area = sighting.get("area", sighting.get("area_ratio",
                                      sighting.get("areas", sighting.get("area_ratios", 0.0))))
                item_bin = lateral
            # Some scan adapters retain per-label values in a mapping rather
            # than converting each label into an object record.
            if isinstance(conf, dict):
                conf = conf.get(name, conf.get(str(name), 1.0))
            if isinstance(area, dict):
                area = area.get(name, area.get(str(name), 0.0))
            if name is None:
                continue
            name = str(name).strip()
            if not name:
                continue
            try:
                item_bin = float(item_bin)
            except (TypeError, ValueError):
                item_bin = lateral
            item_bin = -1 if item_bin < 0 else (1 if item_bin > 0 else 0)
            conf = _clamp01(conf, 1.0)
            area = _clamp01(area, 0.0)
            labels.add(f"{legacy_side}:{name}")
            rich_labels.append({"name": name, "conf": conf,
                                "bin": item_bin, "area": area})

    # Sorting makes the JSON fingerprint stable even when detector/tracker
    # iteration order changes (or a scan's sightings arrive rotated/reordered).
    rich_labels.sort(key=lambda item: (item["name"], item["bin"],
                                       item["conf"], item["area"]))
    try:
        distance_cm = float(distance_cm)
    except (TypeError, ValueError):
        distance_cm = None
    if distance_cm is None or distance_cm <= 0 or distance_cm != distance_cm:
        rng = "unknown"
    elif distance_cm < 50:
        rng = "near"
    elif distance_cm < 150:
        rng = "mid"
    else:
        rng = "far"
    imu = _scan_imu_delta(sightings, imu_delta)
    return {"labels": sorted(labels), "range": rng,
            # Keep the pose delta at the top level as a convenient read-only
            # alias for callers using the compact fingerprint shape; matching
            # consumes the copy in metadata so all new features stay grouped.
            "imu_delta": imu,
            "metadata": {"labels": rich_labels, "imu_delta": imu}}


def _fingerprint_records(fp):
    """Return rich label records, upgrading legacy side-tagged labels."""
    def record_bin(record):
        value = record.get("bin", record.get("lateral_bin", 0))
        if isinstance(value, str):
            value = {"left": -1, "l": -1, "center": 0, "centre": 0,
                     "c": 0, "right": 1, "r": 1}.get(value.lower(), value)
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = 0
        return -1 if value < 0 else (1 if value > 0 else 0)

    metadata = fp.get("metadata") or fp.get("meta") or {}
    records = metadata.get("labels") if isinstance(metadata, dict) else None
    if records:
        out = []
        for record in records:
            if not isinstance(record, dict) or not record.get("name"):
                continue
            out.append({"name": str(record["name"]),
                        "conf": _clamp01(record.get("conf", record.get("confidence", 1.0)), 1.0),
                        "bin": record_bin(record),
                        "area": _clamp01(record.get("area", record.get("area_ratio", 0.0)))})
        if out:
            return out
    out = []
    for label in fp.get("labels") or []:
        if isinstance(label, dict):
            name = label.get("name", label.get("label"))
            if name:
                out.append({"name": str(name), "conf": _clamp01(label.get("conf", 1.0), 1.0),
                            "bin": record_bin(label), "area": _clamp01(label.get("area", 0.0))})
            continue
        text = str(label)
        side, sep, name = text.partition(":")
        if not sep:
            side, name = "c", text
        out.append({"name": name, "conf": 1.0,
                    "bin": -1 if side == "l" else (1 if side == "r" else 0),
                    "area": 0.0})
    return out


def _imu_similarity(fp_a, fp_b):
    metadata_a = fp_a.get("metadata") or fp_a.get("meta") or {}
    metadata_b = fp_b.get("metadata") or fp_b.get("meta") or {}
    a = metadata_a.get("imu_delta") if isinstance(metadata_a, dict) else None
    b = metadata_b.get("imu_delta") if isinstance(metadata_b, dict) else None
    if a is None:
        a = fp_a.get("imu_delta")
    if b is None:
        b = fp_b.get("imu_delta")
    if a is None or b is None:
        return None
    if isinstance(a, dict) and isinstance(b, dict):
        keys = sorted(set(a) & set(b), key=str)
        try:
            delta = sum((float(a[k]) - float(b[k])) ** 2 for k in keys) ** 0.5
        except (TypeError, ValueError):
            return 1.0 if a == b else 0.0
    else:
        try:
            va = list(a) if isinstance(a, (list, tuple)) else [float(a)]
            vb = list(b) if isinstance(b, (list, tuple)) else [float(b)]
            n = min(len(va), len(vb))
            delta = sum((float(va[i]) - float(vb[i])) ** 2 for i in range(n)) ** 0.5
        except (TypeError, ValueError):
            return 1.0 if a == b else 0.0
    return 1.0 / (1.0 + delta)


def fingerprint_similarity(fp_a, fp_b):
    """Return a weighted 0..1 similarity for two perceptual fingerprints.

    Matching labels contribute their average confidence, an exact lateral-bin
    match, and (when known) a similar apparent area.  Legacy fingerprints are
    upgraded on the fly, so maps written before the richer metadata existed
    remain readable.
    """
    a, b = _fingerprint_records(fp_a), _fingerprint_records(fp_b)
    if not a and not b:
        label_similarity = 1.0
    elif not a or not b:
        label_similarity = 0.0
    else:
        # Greedy one-to-one pairing avoids duplicate labels inflating a score.
        pairs = []
        for ia, left in enumerate(a):
            for ib, right in enumerate(b):
                if left["name"] != right["name"]:
                    continue
                pos = (1.0 if left["bin"] == right["bin"] else
                       0.5 if abs(left["bin"] - right["bin"]) == 1 else 0.0)
                if left["area"] and right["area"]:
                    area = max(0.0, 1.0 - abs(left["area"] - right["area"]) /
                               max(left["area"], right["area"]))
                else:
                    area = 1.0  # old records have no area feature
                score = ((left["conf"] + right["conf"]) / 2.0) * pos * area
                pairs.append((score, ia, ib))
        matched = 0.0
        used_a, used_b = set(), set()
        for score, ia, ib in sorted(pairs, reverse=True):
            if ia not in used_a and ib not in used_b:
                used_a.add(ia)
                used_b.add(ib)
                matched += score
        total = max(sum(item["conf"] for item in a),
                    sum(item["conf"] for item in b))
        label_similarity = matched / total if total else 0.0

    range_match = 1.0 if fp_a.get("range") == fp_b.get("range") else 0.0
    imu_match = _imu_similarity(fp_a, fp_b)
    if imu_match is None:
        # Range is deliberately strong enough that otherwise-identical scans
        # in near versus far space do not merge into one location.
        return 0.65 * label_similarity + 0.35 * range_match
    return 0.55 * label_similarity + 0.25 * range_match + 0.20 * imu_match


def fingerprint_is_distinctive(fingerprint):
    """Whether a scan contains enough independent visual evidence to name
    a place.  Range alone, and one common detector label, are useful hints
    but are far too easy to alias in a house."""
    landmarks = {name for name in
                 ((entry.get("name") if isinstance(entry, dict)
                   else str(entry).split(":", 1)[-1])
                  for entry in (fingerprint.get("labels") or []))
                 if name}
    if not landmarks:
        landmarks = {entry["name"] for entry in _fingerprint_records(fingerprint)
                     if entry.get("name")}
    return len(landmarks) >= MIN_DISTINCT_LANDMARKS


def label_for_fingerprint(fp, location_id):
    seen = sorted({name for name in
                   ((l.get("name") if isinstance(l, dict)
                     else str(l).split(":", 1)[-1])
                    for l in fp.get("labels") or [])
                   if name})
    if not seen:
        seen = sorted({entry["name"] for entry in _fingerprint_records(fp)
                       if entry.get("name")})
    if seen:
        return f"place {location_id} ({', '.join(seen[:3])})"
    return f"place {location_id} (open {fp.get('range', 'unknown')} area)"


class SpatialStore:
    def __init__(self, readonly=True, db_path=None):
        self.readonly = readonly
        self.db_path = db_path if db_path is not None else DB_PATH
        self.conn = None
        if not readonly:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self.conn.executescript(_SCHEMA)
            self.conn.commit()

    # ---------- reader side (fail-soft) ----------

    def _query(self, sql, params=()):
        try:
            conn = self.conn or sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            try:
                return conn.execute(sql, params).fetchall()
            finally:
                if self.readonly:
                    conn.close()
        except sqlite3.Error:
            return []  # no map yet - readers degrade gracefully

    def all_locations(self):
        rows = self._query(
            "SELECT id, label, fingerprint_json, discovered_at, last_visited_at,"
            " visit_count, veto_count, coach_wins, coach_losses FROM locations")
        return [self._row_to_location(r) for r in rows]

    def get_location(self, location_id):
        rows = self._query(
            "SELECT id, label, fingerprint_json, discovered_at, last_visited_at,"
            " visit_count, veto_count, coach_wins, coach_losses FROM locations WHERE id = ?",
            (location_id,))
        return self._row_to_location(rows[0]) if rows else None

    def neighbors(self, location_id):
        """Location ids connected to this one by at least one traversal."""
        rows = self._query(
            "SELECT a, b FROM edges WHERE a = ? OR b = ?", (location_id, location_id))
        out = set()
        for a, b in rows:
            out.add(b if a == location_id else a)
        return sorted(out)

    def edge_list(self):
        return self._query("SELECT a, b, traversals FROM edges")

    def location_count(self):
        rows = self._query("SELECT COUNT(*) FROM locations")
        return rows[0][0] if rows else 0

    def find_location_by_name(self, name):
        """Location dict whose label best matches a spoken place name
        ("kitchen" matches both a renamed "kitchen" and the auto label
        "place 3 (kitchen table)"). Exact match beats substring; None if
        nothing matches."""
        want = (name or "").strip().lower()
        if not want:
            return None
        locations = self.all_locations()
        for loc in locations:
            if loc["label"].lower() == want:
                return loc
        for loc in locations:
            label = loc["label"].lower()
            if want in label or label in want:
                return loc
        return None

    def object_locations(self, label, limit=3):
        """Where a given object label has been seen: most recent first.
        [{'location_id','place','times_seen','last_seen'}]"""
        rows = self._query(
            "SELECT s.location_id, l.label, s.times_seen, s.last_seen"
            " FROM sightings s JOIN locations l ON l.id = s.location_id"
            " WHERE s.label = ? ORDER BY s.last_seen DESC LIMIT ?",
            (label, limit))
        return [{"location_id": lid, "place": place,
                 "times_seen": times, "last_seen": last}
                for lid, place, times, last in rows]

    def location_objects(self, location_id, limit=8):
        """What's been seen at a place, most-often-seen first.
        [{'label','times_seen','last_seen'}]"""
        rows = self._query(
            "SELECT label, times_seen, last_seen FROM sightings"
            " WHERE location_id = ? ORDER BY times_seen DESC, last_seen DESC LIMIT ?",
            (location_id, limit))
        return [{"label": lb, "times_seen": t, "last_seen": ls}
                for lb, t, ls in rows]

    def sighting_labels(self):
        """Every object label ever recorded at any place."""
        return [r[0] for r in self._query("SELECT DISTINCT label FROM sightings")]

    @staticmethod
    def _row_to_location(r):
        return {
            "id": r[0], "label": r[1], "fingerprint": json.loads(r[2]),
            "discovered_at": r[3], "last_visited_at": r[4], "visit_count": r[5],
            "veto_count": r[6], "coach_wins": r[7], "coach_losses": r[8],
        }

    # ---------- writer side (location_graph.py only) ----------

    def _assert_writer(self):
        if self.readonly:
            raise RuntimeError("SpatialStore opened readonly - only location_graph.py writes")

    def match_or_create(self, fingerprint, now=None):
        """Resolve a scan fingerprint to a location, creating one if
        nothing known is similar enough. Returns the location dict plus
        'is_new' (just discovered) and 'new_visit' (revisit after being
        away, vs. a same-session re-scan), and a bounded ranked
        ``candidate_scores`` list for telemetry."""
        self._assert_writer()
        now = now if now is not None else time.time()
        # Do not turn a nearly empty sweep into either a false recognition
        # or a durable "open floor" node.  The caller receives an explicit
        # unlocalized result and can keep exploring/scanning.
        # Permit the very first weak scan to seed a named place for backwards
        # compatibility and human labelling, but never use weak evidence to
        # identify an already-known place.  It is a hypothesis until a later,
        # richer scan confirms it.
        known_locations = self.all_locations()
        candidates = sorted(
            ((fingerprint_similarity(fingerprint, loc["fingerprint"]), loc)
             for loc in known_locations),
            key=lambda item: (-item[0], item[1]["id"]))
        candidate_scores = [
            {"location_id": loc["id"], "similarity": similarity}
            for similarity, loc in candidates[:MAX_CANDIDATE_SCORES]
        ]
        if not fingerprint_is_distinctive(fingerprint) and known_locations:
            return {"id": None, "label": None, "fingerprint": fingerprint,
                    "is_new": False, "new_visit": False, "similarity": None,
                    "resolved": False, "reason": "insufficient_landmarks",
                    "ambiguity": None, "candidate_scores": candidate_scores}

        best_sim, best = candidates[0] if candidates else (None, None)
        second_sim = candidates[1][0] if len(candidates) > 1 else None
        # Two known places that explain this scan almost equally well are an
        # ambiguity, not a license to pick whichever SQLite row happened to
        # win a tie.  Avoid minting a third duplicate as well.
        if best is not None and best_sim >= MATCH_THRESHOLD and \
                second_sim is not None and best_sim - second_sim < MATCH_MARGIN:
            return {"id": None, "label": None, "fingerprint": fingerprint,
                    "is_new": False, "new_visit": False, "similarity": best_sim,
                    "resolved": False, "reason": "ambiguous_match",
                    "ambiguity": {"best_location_id": best["id"],
                                  "second_location_id": candidates[1][1]["id"],
                                  "margin": best_sim - second_sim},
                    "candidate_scores": candidate_scores}
        if best is not None and best_sim >= MATCH_THRESHOLD:
            new_visit = (now - best["last_visited_at"]) > REVISIT_GAP_SEC
            self.conn.execute(
                "UPDATE locations SET last_visited_at = ?, visit_count = visit_count + ? WHERE id = ?",
                (now, 1 if new_visit else 0, best["id"]))
            self.conn.commit()
            best.update(last_visited_at=now,
                        visit_count=best["visit_count"] + (1 if new_visit else 0))
            return {**best, "is_new": False, "new_visit": new_visit,
                    "similarity": best_sim, "resolved": True,
                    "reason": "matched", "ambiguity": None,
                    "candidate_scores": candidate_scores}

        cur = self.conn.execute(
            "INSERT INTO locations (label, fingerprint_json, discovered_at, last_visited_at)"
            " VALUES (?, ?, ?, ?)",
            ("pending", json.dumps(fingerprint), now, now))
        loc_id = cur.lastrowid
        label = label_for_fingerprint(fingerprint, loc_id)
        self.conn.execute("UPDATE locations SET label = ? WHERE id = ?", (label, loc_id))
        self.conn.commit()
        return {"id": loc_id, "label": label, "fingerprint": fingerprint,
                "discovered_at": now, "last_visited_at": now, "visit_count": 1,
                "veto_count": 0, "coach_wins": 0, "coach_losses": 0,
                "is_new": True, "new_visit": True, "similarity": None,
                "resolved": True, "reason": "new_distinctive_place",
                "ambiguity": None, "candidate_scores": candidate_scores}

    def note_edge(self, a, b, now=None):
        self._assert_writer()
        if a == b:
            return
        now = now if now is not None else time.time()
        lo, hi = (a, b) if a < b else (b, a)  # undirected, stored once
        self.conn.execute(
            "INSERT INTO edges (a, b, traversals, last_traversed_at) VALUES (?, ?, 1, ?)"
            " ON CONFLICT(a, b) DO UPDATE SET traversals = traversals + 1,"
            " last_traversed_at = excluded.last_traversed_at",
            (lo, hi, now))
        self.conn.commit()

    def note_veto(self, location_id):
        self._assert_writer()
        self.conn.execute(
            "UPDATE locations SET veto_count = veto_count + 1 WHERE id = ?", (location_id,))
        self.conn.commit()

    def relax_veto(self, location_id, amount=1):
        """Ease a place's veto_count back DOWN (floored at 0), the inverse
        of note_veto. Called when field_agent's VetoProneLocationProbe
        physically re-tests a once-blocked spot and the safety daemon stays
        silent ('maybe_clear') - the robot unlearning its fear of a place as
        the environment changes. Gradual on purpose: it takes a few clean
        re-tests to drop a place below the veto-prone threshold, so one
        lucky pass never erases a real, recurring hazard. Writer-only, same
        single-writer rule as note_veto (location_graph.py only)."""
        self._assert_writer()
        self.conn.execute(
            "UPDATE locations SET veto_count = MAX(0, veto_count - ?) WHERE id = ?",
            (int(amount), location_id))
        self.conn.commit()

    def note_sightings(self, location_id, labels, now=None):
        """Record that these object labels were visible at a place (one
        completed room scan). Repeat sightings bump times_seen/last_seen,
        so 'where is the bottle' can answer with the place it's seen
        most recently and most reliably."""
        self._assert_writer()
        now = now if now is not None else time.time()
        for label in set(labels or []):
            if not label:
                continue
            self.conn.execute(
                "INSERT INTO sightings (location_id, label, times_seen, first_seen, last_seen)"
                " VALUES (?, ?, 1, ?, ?)"
                " ON CONFLICT(location_id, label) DO UPDATE SET"
                " times_seen = times_seen + 1, last_seen = excluded.last_seen",
                (location_id, label, now, now))
        self.conn.commit()

    def rename_location(self, location_id, name):
        """Give a place a human name ('the kitchen'). Returns the new
        label, or None if the location doesn't exist. The name replaces
        the auto-generated 'place N (...)' label everywhere it's spoken."""
        self._assert_writer()
        name = (name or "").strip()[:60]
        if not name or self.get_location(location_id) is None:
            return None
        self.conn.execute("UPDATE locations SET label = ? WHERE id = ?",
                          (name, location_id))
        self.conn.commit()
        return name

    def note_coach_outcome(self, location_id, success):
        self._assert_writer()
        column = "coach_wins" if success else "coach_losses"
        self.conn.execute(
            f"UPDATE locations SET {column} = {column} + 1 WHERE id = ?", (location_id,))
        self.conn.commit()
