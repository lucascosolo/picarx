#!/usr/bin/env python3
# /home/picarx/layer_b/pattern_miner.py
"""
Temporal pattern mining over events.db - pure Python, no LLM call.

Called by reflection.py during its idle window (never on the hot
path); results go into semantic.db's patterns table, which coach.py
folds into its prompts ("based on learned patterns, ..."). Read-only
on events.db per the single-writer convention.

Two mines, both aggregate statistics rather than one-off coincidences:

  1. Coach-episode outcomes: for each situation_key, does the FIRST
     move of the escape maneuver predict success? Emits both "works"
     and "keeps failing" patterns - knowing what reliably fails is as
     coaching-relevant as knowing what works.

  2. Veto burstiness: after a safety veto of a given type, how often
     does another veto follow within a few seconds? A high rate means
     that failure type traps the robot in loops (retreat further);
     a low rate means single vetoes there resolve themselves.

Spurious-correlation guard (roadmap mitigation): a pattern is only
returned with frequency >= MIN_FREQUENCY and confidence at/beyond
CONFIDENCE_BAND on either side.
"""
import json
import sqlite3

MIN_FREQUENCY = 3
CONFIDENCE_BAND = 0.70     # emit "works" at >= this, "fails" at <= 1 - this
VETO_BURST_WINDOW = 5.0    # seconds - a follow-up veto within this = a loop
MAX_ROWS_PER_TOPIC = 4000  # bound the scan; recent history is what matters


def _fetch(conn, topic, limit=MAX_ROWS_PER_TOPIC):
    rows = conn.execute(
        "SELECT ts, payload_json FROM events WHERE topic = ? ORDER BY id DESC LIMIT ?",
        (topic, limit)).fetchall()
    return list(reversed(rows))


def _mine_coach_first_moves(conn):
    stats = {}  # (situation_key, first_direction) -> [wins, total]
    for _ts, payload_json in _fetch(conn, "picarx/coach/episode"):
        try:
            p = json.loads(payload_json)
            steps = p.get("steps") or []
            first = (steps[0].get("action") or {}).get("direction") if steps else None
            key = p.get("situation_key")
        except (json.JSONDecodeError, AttributeError, IndexError):
            continue
        if not key or not first:
            continue
        wins, total = stats.setdefault((key, first), [0, 0])
        stats[(key, first)] = [wins + (1 if p.get("success") else 0), total + 1]

    out = []
    for (key, first), (wins, total) in stats.items():
        if total < MIN_FREQUENCY:
            continue
        rate = wins / total
        if rate >= CONFIDENCE_BAND:
            out.append({"condition": f"stuck:{key}",
                        "outcome": f"escapes starting with '{first}' usually work",
                        "frequency": total, "confidence": rate})
        elif rate <= 1.0 - CONFIDENCE_BAND:
            out.append({"condition": f"stuck:{key}",
                        "outcome": f"escapes starting with '{first}' keep failing",
                        "frequency": total, "confidence": 1.0 - rate})
    return out


def _mine_veto_bursts(conn):
    vetoes = []  # (ts, code)
    for ts, payload_json in _fetch(conn, "picarx/action/result"):
        try:
            p = json.loads(payload_json)
            result = p.get("result") or {}
        except (json.JSONDecodeError, AttributeError):
            continue
        if result.get("status") != "vetoed":
            continue
        reason = result.get("reason") or ""
        code = result.get("reason_code") or (
            "obstacle" if reason.startswith("obstacle")
            else "cliff" if "cliff" in reason
            else "reverse_limit" if "reverse" in reason else "unknown")
        vetoes.append((ts, code))

    stats = {}  # code -> [followed, total]
    for i, (ts, code) in enumerate(vetoes):
        followed = i + 1 < len(vetoes) and (vetoes[i + 1][0] - ts) <= VETO_BURST_WINDOW
        f, t = stats.setdefault(code, [0, 0])
        stats[code] = [f + (1 if followed else 0), t + 1]

    out = []
    for code, (followed, total) in stats.items():
        if total < MIN_FREQUENCY:
            continue
        rate = followed / total
        if rate >= CONFIDENCE_BAND:
            out.append({"condition": f"veto:{code}",
                        "outcome": "another veto usually follows within seconds - "
                                   "single small retreats don't clear it",
                        "frequency": total, "confidence": rate})
        elif rate <= 1.0 - CONFIDENCE_BAND:
            out.append({"condition": f"veto:{code}",
                        "outcome": "usually a one-off - a single small retreat clears it",
                        "frequency": total, "confidence": 1.0 - rate})
    return out


def mine_patterns(events_db_path):
    """Returns a list of {condition, outcome, frequency, confidence}.
    Fail-soft: any DB problem returns []."""
    try:
        conn = sqlite3.connect(f"file:{events_db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        return []
    try:
        return _mine_coach_first_moves(conn) + _mine_veto_bursts(conn)
    except sqlite3.Error:
        return []
    finally:
        conn.close()


if __name__ == "__main__":
    # Inspect what the miner currently sees, without writing anything.
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/home/picarx/layer_b/data/events.db"
    for p in mine_patterns(path):
        print(f"[{p['confidence']:.2f} x{p['frequency']}] {p['condition']} -> {p['outcome']}")
