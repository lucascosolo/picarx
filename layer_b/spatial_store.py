#!/usr/bin/env python3
# /home/picarx/layer_b/spatial_store.py
"""
Topological spatial memory - the robot's map of WHERE things happen,
as opposed to WHAT it has learned (semantic.db) or the raw event
stream (events.db).

There is no odometry or SLAM on this platform, so a "location" is a
*perceptual fingerprint*, not a coordinate: the set of object labels a
look-around head sweep saw, plus a coarse open-space bucket from the
ultrasonic. Two sweeps that see the same things are treated as the
same place. That is honest about the hardware - it can be fooled by
two identical-looking corners, but it can never claim centimeter
positions it doesn't have.

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

DB_DIR = "/home/picarx/layer_b/data"
DB_PATH = f"{DB_DIR}/spatial.db"

# Two fingerprints at least this similar are the same place.
MATCH_THRESHOLD = 0.60
# A re-scan within this many seconds of the last visit refreshes the
# location but doesn't count as a new "visit" (one wander session
# re-scanning every 25s isn't ten visits).
REVISIT_GAP_SEC = 120.0

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
"""


# ---------- pure fingerprint logic (no DB, unit-testable) ----------

def fingerprint_from_scan(sightings, distance_cm=None):
    """Collapse a room_scan payload into a comparable fingerprint.
    labels: which object labels were visible and on which side
    (left/center/right by pan sign) - side matters, "sofa on the left"
    and "sofa on the right" are different corners of the room.
    range: coarse forward open-space bucket from the ultrasonic."""
    labels = set()
    for s in sightings or []:
        pan = s.get("pan", 0)
        side = "l" if pan < 0 else ("r" if pan > 0 else "c")
        for label in s.get("labels") or []:
            labels.add(f"{side}:{label}")
    if distance_cm is None or distance_cm <= 0:
        rng = "unknown"
    elif distance_cm < 50:
        rng = "near"
    elif distance_cm < 150:
        rng = "mid"
    else:
        rng = "far"
    return {"labels": sorted(labels), "range": rng}


def fingerprint_similarity(fp_a, fp_b):
    """0..1. Jaccard over side-tagged labels, with the open-space
    bucket as a weighted tie-breaker so featureless scans (empty label
    sets - very common with the SSD seeing nothing) still separate a
    tight corner from open floor instead of all collapsing into one
    giant 'nowhere' node."""
    a, b = set(fp_a.get("labels") or []), set(fp_b.get("labels") or [])
    if a or b:
        jaccard = len(a & b) / len(a | b)
    else:
        jaccard = 1.0  # both featureless - rely on the range bucket
    range_match = 1.0 if fp_a.get("range") == fp_b.get("range") else 0.0
    return 0.8 * jaccard + 0.2 * range_match


def label_for_fingerprint(fp, location_id):
    seen = sorted({l.split(":", 1)[1] for l in fp.get("labels") or []})
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
        away, vs. a same-session re-scan)."""
        self._assert_writer()
        now = now if now is not None else time.time()
        best, best_sim = None, MATCH_THRESHOLD
        for loc in self.all_locations():
            sim = fingerprint_similarity(fingerprint, loc["fingerprint"])
            if sim >= best_sim:
                best, best_sim = loc, sim
        if best is not None:
            new_visit = (now - best["last_visited_at"]) > REVISIT_GAP_SEC
            self.conn.execute(
                "UPDATE locations SET last_visited_at = ?, visit_count = visit_count + ? WHERE id = ?",
                (now, 1 if new_visit else 0, best["id"]))
            self.conn.commit()
            best.update(last_visited_at=now,
                        visit_count=best["visit_count"] + (1 if new_visit else 0))
            return {**best, "is_new": False, "new_visit": new_visit, "similarity": best_sim}

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
                "is_new": True, "new_visit": True, "similarity": None}

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

    def note_coach_outcome(self, location_id, success):
        self._assert_writer()
        column = "coach_wins" if success else "coach_losses"
        self.conn.execute(
            f"UPDATE locations SET {column} = {column} + 1 WHERE id = ?", (location_id,))
        self.conn.commit()
