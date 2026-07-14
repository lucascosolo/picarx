#!/usr/bin/env python3
# /home/picarx/layer_b/semantic_store.py
"""
Semantic memory store - durable facts the robot has learned about its
world ("the corner with the tvmonitor causes repeated vetoes", "a
person named X visits in the evening"), as opposed to the raw episodic
event stream in events.db.

Ownership rules (mirrors the events.db convention):
  - reflection.py is the SOLE writer to semantic.db. It opens the
    store with readonly=False, which creates the schema.
  - Everything else (companion.py, field_agent.py, future mapping
    code) opens readonly=True. Reads are fail-soft: if the DB doesn't
    exist yet (reflection has never run), readers just get [] back -
    a robot with no memories yet is fine, a crashed consumer is not.

Facts are deduplicated on (subject, fact): re-learning the same thing
bumps seen_count/updated_at instead of inserting a duplicate row, so
repeated reflections converge instead of accumulating noise.
"""
import os
import sqlite3
import time

DB_DIR = "/home/picarx/layer_b/data"
DB_PATH = f"{DB_DIR}/semantic.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    fact TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    source TEXT NOT NULL DEFAULT 'reflection',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    seen_count INTEGER NOT NULL DEFAULT 1,
    UNIQUE(subject, fact)
);
CREATE INDEX IF NOT EXISTS idx_facts_subject ON facts(subject);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
CREATE TABLE IF NOT EXISTS patterns (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    condition TEXT NOT NULL,
    outcome TEXT NOT NULL,
    frequency INTEGER NOT NULL,
    confidence REAL NOT NULL,
    first_seen REAL NOT NULL,
    last_seen REAL NOT NULL,
    UNIQUE(condition, outcome)
);
"""


class SemanticStore:
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

    def _read_conn(self):
        """Fresh read-only connection per call - cheap for our volumes,
        and it means a reader never holds a handle across the writer's
        transactions or blocks on a stale one."""
        return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)

    def _query(self, sql, params=()):
        try:
            conn = self._read_conn() if self.readonly else self.conn
            try:
                return conn.execute(sql, params).fetchall()
            finally:
                if self.readonly:
                    conn.close()
        except sqlite3.Error:
            return []  # DB missing/locked/etc - reader degrades to "no memories"

    def recent_facts(self, limit=5):
        """Most recently reinforced facts, best first."""
        rows = self._query(
            "SELECT subject, fact, confidence, seen_count FROM facts "
            "ORDER BY updated_at DESC LIMIT ?", (limit,))
        return [{"subject": s, "fact": f, "confidence": c, "seen_count": n}
                for s, f, c, n in rows]

    def facts_for(self, subject, limit=5):
        rows = self._query(
            "SELECT subject, fact, confidence, seen_count FROM facts "
            "WHERE subject = ? ORDER BY confidence DESC, updated_at DESC LIMIT ?",
            (subject, limit))
        return [{"subject": s, "fact": f, "confidence": c, "seen_count": n}
                for s, f, c, n in rows]

    def fact_count(self):
        rows = self._query("SELECT COUNT(*) FROM facts")
        return rows[0][0] if rows else 0

    def top_patterns(self, limit=5, max_age_sec=7 * 86400):
        """Mined event-sequence patterns, freshest + most confident
        first. Old patterns age out of the results (the world changes;
        the roadmap's spurious-correlation mitigation) but stay stored."""
        rows = self._query(
            "SELECT condition, outcome, frequency, confidence FROM patterns "
            "WHERE last_seen >= ? ORDER BY confidence DESC, frequency DESC LIMIT ?",
            (time.time() - max_age_sec, limit))
        return [{"condition": c, "outcome": o, "frequency": f, "confidence": conf}
                for c, o, f, conf in rows]

    # ---------- writer side (reflection.py only) ----------

    def upsert_fact(self, subject, fact, confidence=0.5, source="reflection"):
        if self.readonly:
            raise RuntimeError("SemanticStore opened readonly - only reflection.py writes")
        now = time.time()
        self.conn.execute(
            """INSERT INTO facts (subject, fact, confidence, source, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(subject, fact) DO UPDATE SET
                 seen_count = seen_count + 1,
                 confidence = MAX(confidence, excluded.confidence),
                 updated_at = excluded.updated_at""",
            (subject.strip()[:80], fact.strip()[:300], float(confidence), source, now, now))
        self.conn.commit()

    def upsert_pattern(self, condition, outcome, frequency, confidence):
        """Patterns are aggregate statistics over a window, so a re-mine
        REPLACES frequency/confidence rather than accumulating them."""
        if self.readonly:
            raise RuntimeError("SemanticStore opened readonly - only reflection.py writes")
        now = time.time()
        self.conn.execute(
            """INSERT INTO patterns (condition, outcome, frequency, confidence, first_seen, last_seen)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(condition, outcome) DO UPDATE SET
                 frequency = excluded.frequency,
                 confidence = excluded.confidence,
                 last_seen = excluded.last_seen""",
            (condition.strip()[:120], outcome.strip()[:200],
             int(frequency), float(confidence), now, now))
        self.conn.commit()

    def get_meta(self, key, default=None):
        rows = self._query("SELECT value FROM meta WHERE key = ?", (key,))
        return rows[0][0] if rows else default

    def set_meta(self, key, value):
        if self.readonly:
            raise RuntimeError("SemanticStore opened readonly")
        self.conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, str(value)))
        self.conn.commit()
