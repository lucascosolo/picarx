#!/usr/bin/env python3
# layer_b/ab_report.py
"""
A/B report over behavior_metrics.jsonl - does the learning round-trip help?

behavior_metrics.py writes a rolling checkpoint per session, tagged with the A/B
condition coach chose (adopt = sim-trained arms in play, control = held out).
This reads those, keeps the LATEST checkpoint per session, groups by condition,
and prints the real-world collision/veto rates side by side so you can see
whether "adopt" actually beats "control" - and refuses to over-claim until
enough sessions have accumulated under each condition.

Pure aggregation (latest_per_session / aggregate_by_condition / format_report)
is unit-tested; __main__ just points them at the metrics file.

    python3 layer_b/ab_report.py [path/to/behavior_metrics.jsonl]
"""
import json
import os
import sys

MIN_SESSIONS_PER_CONDITION = 5   # below this, say "not enough data yet"


def read_records(path):
    """Every JSON object (one per line) in the metrics file; skips blanks and
    any corrupt line. Missing file -> []."""
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        return []
    return records


def latest_per_session(records):
    """Keep only the newest checkpoint (max ts) for each session_id - the
    cumulative end-of-session counts."""
    latest = {}
    for r in records:
        sid = r.get("session_id")
        if sid is None:
            continue
        if sid not in latest or r.get("ts", 0) > latest[sid].get("ts", 0):
            latest[sid] = r
    return list(latest.values())


def aggregate_by_condition(sessions):
    """Sum sessions per condition into rate metrics. Returns
    {condition: {sessions, attempts, vetoes, impacts, fail_loops, veto_rate,
    impacts_per_100_moves}}."""
    groups = {}
    for s in sessions:
        cond = s.get("condition") or "unknown"
        g = groups.setdefault(cond, {"sessions": 0, "attempts": 0, "vetoes": 0,
                                     "impacts": 0, "fail_loops": 0})
        g["sessions"] += 1
        g["attempts"] += s.get("move_attempts", 0) or 0
        g["vetoes"] += s.get("vetoes", 0) or 0
        g["impacts"] += s.get("impacts", 0) or 0
        g["fail_loops"] += s.get("fail_loops", 0) or 0
    for g in groups.values():
        a = g["attempts"]
        g["veto_rate"] = round(g["vetoes"] / a, 4) if a else None
        g["impacts_per_100_moves"] = round(100.0 * g["impacts"] / a, 3) if a else None
    return groups


def _verdict(groups, min_sessions=MIN_SESSIONS_PER_CONDITION):
    """A one-line honest read on adopt vs control, or a 'not enough data' note."""
    adopt, control = groups.get("adopt"), groups.get("control")
    if not adopt or not control:
        return "Not enough data: need sessions under BOTH adopt and control."
    if adopt["sessions"] < min_sessions or control["sessions"] < min_sessions:
        return (f"Not enough data yet: {adopt['sessions']} adopt / "
                f"{control['sessions']} control sessions "
                f"(want >= {min_sessions} each before trusting the round-trip).")
    av, cv = adopt["veto_rate"], control["veto_rate"]
    if av is None or cv is None:
        return "Not enough motion under one condition to compute a veto rate."
    if av < cv:
        drop = round(100.0 * (cv - av) / cv, 1) if cv else 0.0
        return (f"Adopt looks BETTER: veto rate {av:.3f} vs control {cv:.3f} "
                f"({drop}% lower). Keep collecting to confirm.")
    if av > cv:
        return (f"Adopt looks WORSE than control ({av:.3f} vs {cv:.3f}) - the "
                f"round-trip is NOT earning its keep yet. Investigate before trusting.")
    return f"Adopt and control are even ({av:.3f}). No measurable benefit yet."


def format_report(groups, min_sessions=MIN_SESSIONS_PER_CONDITION):
    lines = ["A/B learning-loop report (real-world collision/veto rates)", ""]
    if not groups:
        lines.append("No behavior metrics recorded yet.")
        return "\n".join(lines)
    header = f"{'condition':>10} {'sessions':>9} {'moves':>8} {'veto_rate':>10} {'impacts/100':>12} {'fail_loops':>11}"
    lines.append(header)
    lines.append("-" * len(header))
    for cond in sorted(groups):
        g = groups[cond]
        vr = f"{g['veto_rate']:.3f}" if g["veto_rate"] is not None else "n/a"
        ip = f"{g['impacts_per_100_moves']:.2f}" if g["impacts_per_100_moves"] is not None else "n/a"
        lines.append(f"{cond:>10} {g['sessions']:>9} {g['attempts']:>8} {vr:>10} {ip:>12} {g['fail_loops']:>11}")
    lines.append("")
    lines.append(_verdict(groups, min_sessions))
    return "\n".join(lines)


def main(argv):
    default = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "data", "behavior_metrics.jsonl")
    path = argv[1] if len(argv) > 1 else default
    sessions = latest_per_session(read_records(path))
    print(format_report(aggregate_by_condition(sessions)))
    print(f"\n({len(sessions)} session(s) from {path})")


if __name__ == "__main__":
    main(sys.argv)
