#!/usr/bin/env python3
# layer_b/semantic_store.py
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

Fact lifecycle: rows are never deleted. `status` is 'active' until a
later, contradicting fact supersedes it, at which point it flips to
'superseded' and `superseded_by_id` points at the replacement row.
Readers only see active facts unless they ask for history with
include_superseded=True.

Schema migrations are additive-only and run when the writer opens the
store; readers on an un-migrated DB fail-soft to [] until reflection.py
has restarted once, so deploy the writer first (in practice the
orchestrator restarts everything together).
"""
import os
import sqlite3
import time

import robot_config

DB_DIR = robot_config.data_path()
DB_PATH = f"{DB_DIR}/semantic.db"

# Passive time-decay, applied at READ time only (never written back):
# a fact or pattern untouched for a full week loses 10% of its returned
# confidence per week of staleness. Re-learning it (which bumps
# updated_at / last_seen) restores full confidence for free.
DECAY_WEEK_SEC = 7 * 86400
DECAY_PER_WEEK = 0.9

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
    status TEXT NOT NULL DEFAULT 'active',
    superseded_by_id INTEGER,
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
            self._migrate()
            self.conn.commit()

    def _migrate(self):
        """Additive, idempotent upgrades for DBs created before a column
        existed in _SCHEMA (CREATE TABLE IF NOT EXISTS won't touch them).
        ALTER TABLE ADD COLUMN is O(1) in SQLite - no table rewrite."""
        cols = {row[1] for row in
                self.conn.execute("PRAGMA table_info(facts)").fetchall()}
        if "status" not in cols:
            self.conn.execute(
                "ALTER TABLE facts ADD COLUMN status TEXT NOT NULL DEFAULT 'active'")
        if "superseded_by_id" not in cols:
            self.conn.execute(
                "ALTER TABLE facts ADD COLUMN superseded_by_id INTEGER")

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

    @staticmethod
    def _decayed(confidence, last_touched, now):
        """Read-time confidence after passive staleness decay."""
        weeks_stale = int(max(0.0, now - last_touched) // DECAY_WEEK_SEC)
        if weeks_stale <= 0:
            return confidence
        return confidence * (DECAY_PER_WEEK ** weeks_stale)

    _FACT_COLS = ("id, subject, fact, confidence, seen_count, updated_at, "
                  "status, superseded_by_id")

    @classmethod
    def _fact_dict(cls, row, now):
        fid, subject, fact, confidence, seen, updated_at, status, sup = row
        return {"id": fid, "subject": subject, "fact": fact,
                "confidence": cls._decayed(confidence, updated_at, now),
                "seen_count": seen, "status": status, "superseded_by_id": sup}

    def recent_facts(self, limit=5, include_superseded=False):
        """Most recently reinforced facts, best first."""
        where = "" if include_superseded else "WHERE status = 'active' "
        rows = self._query(
            f"SELECT {self._FACT_COLS} FROM facts "
            f"{where}ORDER BY updated_at DESC LIMIT ?", (limit,))
        now = time.time()
        return [self._fact_dict(r, now) for r in rows]

    def facts_for(self, subject, limit=5, include_superseded=False):
        # No LIMIT in SQL: decay can reorder, so rank on decayed
        # confidence in Python (stable sort keeps the SQL updated_at
        # tiebreak). Per-subject volumes are small.
        status_sql = "" if include_superseded else "AND status = 'active' "
        rows = self._query(
            f"SELECT {self._FACT_COLS} FROM facts "
            f"WHERE subject = ? {status_sql}"
            f"ORDER BY confidence DESC, updated_at DESC", (subject,))
        now = time.time()
        facts = [self._fact_dict(r, now) for r in rows]
        facts.sort(key=lambda f: f["confidence"], reverse=True)
        return facts[:limit]

    def search_facts(self, query, limit=5, include_superseded=False):
        """Facts whose subject OR text contains `query` (case-insensitive
        LIKE), freshest first. Powers conversational memory recall ("what
        do you remember about the kitchen?") without needing the exact
        subject string."""
        like = f"%{(query or '').strip()[:60]}%"
        status_sql = "" if include_superseded else "AND status = 'active' "
        rows = self._query(
            f"SELECT {self._FACT_COLS} FROM facts "
            f"WHERE (subject LIKE ? OR fact LIKE ?) {status_sql}"
            f"ORDER BY updated_at DESC LIMIT ?", (like, like, limit))
        now = time.time()
        return [self._fact_dict(r, now) for r in rows]

    def fact_count(self):
        rows = self._query("SELECT COUNT(*) FROM facts")
        return rows[0][0] if rows else 0

    def top_patterns(self, limit=5, max_age_sec=7 * 86400):
        """Mined event-sequence patterns, freshest + most confident
        first. Old patterns age out of the results (the world changes;
        the roadmap's spurious-correlation mitigation) but stay stored.
        Confidence is staleness-decayed at read time like facts are
        (only visible when callers widen max_age_sec past the decay
        grace week). Patterns have no status column: a re-mine REPLACES
        the row outright, so there is never history to supersede."""
        rows = self._query(
            "SELECT condition, outcome, frequency, confidence, last_seen "
            "FROM patterns WHERE last_seen >= ? "
            "ORDER BY confidence DESC, frequency DESC",
            (time.time() - max_age_sec,))
        now = time.time()
        patterns = [{"condition": c, "outcome": o, "frequency": f,
                     "confidence": self._decayed(conf, last_seen, now)}
                    for c, o, f, conf, last_seen in rows]
        patterns.sort(key=lambda p: (p["confidence"], p["frequency"]), reverse=True)
        return patterns[:limit]

    # ---------- writer side (reflection.py only) ----------

    def upsert_fact(self, subject, fact, confidence=0.5, source="reflection",
                    supersedes=None):
        """Insert or reinforce a fact and return its row id.

        Belief revision: pass `supersedes=<old fact id>` when this new fact
        CONTRADICTS an existing active one (reflection.py decides this from
        the LLM's verdict). The old row is flipped status='superseded' and
        its superseded_by_id is pointed at the new row, in the same
        transaction, so readers stop seeing the retired belief while its
        history stays queryable via include_superseded=True. A no-op if the
        old id is missing, already superseded, or equal to the new row (a
        fact never supersedes itself)."""
        if self.readonly:
            raise RuntimeError("SemanticStore opened readonly - only reflection.py writes")
        now = time.time()
        subject = subject.strip()[:80]
        fact = fact.strip()[:300]
        self.conn.execute(
            """INSERT INTO facts (subject, fact, confidence, source, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)
               ON CONFLICT(subject, fact) DO UPDATE SET
                 seen_count = seen_count + 1,
                 confidence = MAX(confidence, excluded.confidence),
                 updated_at = excluded.updated_at,
                 status = 'active',
                 superseded_by_id = NULL""",
            (subject, fact, float(confidence), source, now, now))
        # UNIQUE(subject, fact) makes this lookup the reliable way to get the
        # id back for both the insert and the ON CONFLICT update path.
        row = self.conn.execute(
            "SELECT id FROM facts WHERE subject = ? AND fact = ?",
            (subject, fact)).fetchone()
        new_id = row[0] if row else None
        if (supersedes is not None and new_id is not None
                and int(supersedes) != new_id):
            self.conn.execute(
                "UPDATE facts SET status = 'superseded', superseded_by_id = ? "
                "WHERE id = ? AND status = 'active'",
                (new_id, int(supersedes)))
        self.conn.commit()
        return new_id

    def replace_subject(self, subject, facts, source="reflection"):
        """Make `facts` the COMPLETE active set for `subject`, atomically.

        For fully-recomputed subjects like the self-model ("self"), whose
        every fact is re-derived from scratch each pass: upsert (reinforce
        / reactivate) each given fact, then retire any other still-active
        fact under the same subject so a stale snapshot never lingers next
        to the fresh one. Retired rows follow the normal lifecycle
        (status='superseded'); superseded_by_id stays NULL because a whole
        snapshot has no single successor row. No-op guard: an EMPTY facts
        list leaves the subject untouched (a transient empty synthesis must
        not wipe a good self-model).

        facts: iterable of (fact_text, confidence). Returns the kept ids."""
        if self.readonly:
            raise RuntimeError("SemanticStore opened readonly - only reflection.py writes")
        kept = []
        for fact, confidence in facts:
            fid = self.upsert_fact(subject, fact, confidence, source=source)
            if fid is not None:
                kept.append(fid)
        if not kept:
            return kept  # nothing synthesized this pass - don't disturb existing
        placeholders = ",".join("?" * len(kept))
        self.conn.execute(
            f"UPDATE facts SET status = 'superseded' "
            f"WHERE subject = ? AND status = 'active' AND id NOT IN ({placeholders})",
            (subject.strip()[:80], *kept))
        self.conn.commit()
        return kept

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
