#!/usr/bin/env python3
# layer_b/import_training.py
"""
Import a KNOWLEDGE PACK produced by the picarx-training simulator into this
robot, so it starts real-world operation already knowing how to get itself
unstuck instead of learning every escape from scratch on the carpet.

A knowledge pack is a directory the trainer writes (see picarx-training's
sim/knowledge.py). It holds up to three files, each optional:

    coach_policy.json     learned escape maneuvers (bandit arms + records)
    navigation_facts.json transferable facts + mined behavioural patterns
    knowledge_pack.json   a manifest describing how it was trained

This tool MERGES that pack into the robot's own data dir rather than
overwriting it, so a robot that has already learned things in the real
world keeps them:

  * coach_policy.json - per situation, arms are unioned; a maneuver the
    robot already knows has the trained win/loss counts ADDED to its own,
    so simulated and real experience reinforce the same UCB1 statistics.
  * navigation facts/patterns - upserted into semantic.db through the store's
    normal dedup (same fact just reinforces, higher confidence wins), tagged
    source 'training' so their origin stays honest. Only robot-dynamics
    knowledge is in the pack; place-specific memories are never exported,
    because the sim's rooms are not this house.

RUN IT OFFLINE. reflection.py is normally the sole writer to semantic.db and
coach.py to coach_policy.json; this tool writes both. Stop Layer B first
(Ctrl-C the orchestrator), import, then start it again - the modules read
these files at startup and will pick the merged versions up.

    # on the robot, orchestrator stopped:
    python3 layer_b/import_training.py /path/to/training_data
    python3 layer_b/import_training.py /path/to/training_data --dry-run

Fail-soft and idempotent: a missing file in the pack is skipped, and
re-importing the same pack only reinforces (it never double-counts arms it
already merged... it does add counts again, so import a given pack ONCE -
--dry-run first if unsure). Nothing here talks to the bus or hardware.
"""
import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robot_config
from semantic_store import SemanticStore

# Mirrors coach.MAX_DEMONSTRATIONS - kept local so this tool imports without
# pulling in coach.py's broker/embedding stack (it runs on a stopped robot).
MAX_DEMONSTRATIONS = 10


# --------------------------------------------------------------------------
# coach policy merge  (pure function - unit-tested)
# --------------------------------------------------------------------------

def merge_policy(base, incoming):
    """Merge an incoming coach policy into base WITHOUT losing either side's
    learning. Returns (merged_policy, stats). Pure: does not mutate `base`.

    Per situation key:
      - unseen situation      -> adopted wholesale
      - shared situation      -> arms unioned; an arm present on both sides has
                                 its successes/failures SUMMED (so UCB1 sees the
                                 combined evidence) and keeps base's steps/rationale
      - base's embedding wins; incoming's is adopted only if base has none
    Reserved '_'-prefixed list sections (e.g. _demonstrations) are concatenated
    and capped to the freshest MAX_DEMONSTRATIONS by timestamp."""
    merged = copy.deepcopy(base) if isinstance(base, dict) else {}
    stats = {"situations_added": 0, "situations_merged": 0,
             "arms_added": 0, "arms_reinforced": 0, "demonstrations_added": 0}
    if not isinstance(incoming, dict):
        return merged, stats

    for key, entry in incoming.items():
        if key.startswith("_"):
            _merge_reserved(merged, key, entry, stats)
            continue
        if not (isinstance(entry, dict) and isinstance(entry.get("arms"), dict)):
            continue  # legacy / malformed entry - skip like coach._load_policy does
        if key not in merged or not isinstance(merged.get(key), dict):
            merged[key] = copy.deepcopy(entry)
            stats["situations_added"] += 1
            stats["arms_added"] += len(entry["arms"])
            continue
        stats["situations_merged"] += 1
        base_entry = merged[key]
        base_arms = base_entry.setdefault("arms", {})
        for sig, arm in entry["arms"].items():
            if sig in base_arms and isinstance(base_arms[sig], dict):
                b = base_arms[sig]
                b["successes"] = int(b.get("successes", 0)) + int(arm.get("successes", 0))
                b["failures"] = int(b.get("failures", 0)) + int(arm.get("failures", 0))
                b["last_updated"] = max(b.get("last_updated", 0) or 0,
                                        arm.get("last_updated", 0) or 0)
                stats["arms_reinforced"] += 1
            else:
                base_arms[sig] = copy.deepcopy(arm)
                stats["arms_added"] += 1
        if not base_entry.get("embedding") and entry.get("embedding"):
            base_entry["embedding"] = entry["embedding"]
    return merged, stats


def _merge_reserved(merged, key, entry, stats):
    """Merge a reserved list section (only _demonstrations today)."""
    if not isinstance(entry, list):
        merged.setdefault(key, copy.deepcopy(entry))
        return
    dst = merged.get(key)
    if not isinstance(dst, list):
        dst = []
        merged[key] = dst
    dst.extend(copy.deepcopy(entry))
    dst.sort(key=lambda d: (d or {}).get("ts", 0) if isinstance(d, dict) else 0)
    del dst[:-MAX_DEMONSTRATIONS]
    if key == "_demonstrations":
        stats["demonstrations_added"] += len(entry)


# --------------------------------------------------------------------------
# file-level helpers
# --------------------------------------------------------------------------

def _load_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except (json.JSONDecodeError, OSError) as e:
        print(f"  ! could not read {path}: {e}")
        return default


def import_coach_policy(pack_dir, data_dir, dry_run=False):
    incoming = _load_json(os.path.join(pack_dir, "coach_policy.json"), None)
    if incoming is None:
        print("  coach policy:   (none in pack, skipped)")
        return
    dest = os.path.join(data_dir, "coach_policy.json")
    base = _load_json(dest, {})
    merged, stats = merge_policy(base, incoming)
    print(f"  coach policy:   +{stats['situations_added']} new situations, "
          f"{stats['situations_merged']} merged; "
          f"+{stats['arms_added']} new arms, "
          f"{stats['arms_reinforced']} reinforced"
          + (f", +{stats['demonstrations_added']} demonstrations"
             if stats['demonstrations_added'] else ""))
    if dry_run:
        return
    os.makedirs(data_dir, exist_ok=True)
    tmp = dest + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2)
    os.replace(tmp, dest)


def import_navigation_facts(pack_dir, data_dir, dry_run=False):
    pack = _load_json(os.path.join(pack_dir, "navigation_facts.json"), None)
    if pack is None:
        print("  navigation:     (none in pack, skipped)")
        return
    facts = pack.get("facts") or []
    patterns = pack.get("patterns") or []
    print(f"  navigation:     {len(facts)} facts, {len(patterns)} patterns "
          f"-> semantic.db")
    for f in facts:
        subj, fact = (f.get("subject") or "").strip(), (f.get("fact") or "").strip()
        if subj and fact:
            print(f"      fact    [{subj}] {fact}")
    for p in patterns:
        print(f"      pattern {p.get('condition')} -> {p.get('outcome')} "
              f"({float(p.get('confidence', 0)):.0%} x{p.get('frequency')})")
    if dry_run:
        return
    store = SemanticStore(readonly=False,
                          db_path=os.path.join(data_dir, "semantic.db"))
    for f in facts:
        subj, fact = (f.get("subject") or "").strip(), (f.get("fact") or "").strip()
        if not subj or not fact:
            continue
        try:
            conf = max(0.0, min(1.0, float(f.get("confidence", 0.5))))
        except (TypeError, ValueError):
            conf = 0.5
        store.upsert_fact(subj, fact, confidence=conf,
                          source=(f.get("source") or "training"))
    for p in patterns:
        try:
            store.upsert_pattern(p["condition"], p["outcome"],
                                 int(p.get("frequency", 0)),
                                 float(p.get("confidence", 0.0)))
        except (KeyError, TypeError, ValueError):
            continue


def _print_manifest(pack_dir):
    manifest = _load_json(os.path.join(pack_dir, "knowledge_pack.json"), None)
    if not isinstance(manifest, dict):
        return
    t = manifest.get("training") or {}
    scenarios = t.get("scenarios") or []
    when = manifest.get("created_at_iso", "?")
    print(f"  pack trained {t.get('episodes', '?')} episode(s) "
          f"over {len(scenarios)} scenario(s) [{when}]"
          + (f": {', '.join(scenarios)}" if scenarios else ""))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Merge a picarx-training knowledge pack into this robot.")
    ap.add_argument("pack_dir",
                    help="directory holding coach_policy.json / "
                         "navigation_facts.json / knowledge_pack.json")
    ap.add_argument("--data-dir", default=robot_config.data_path(),
                    help="robot data dir to merge into "
                         "(default: layer_b/data/)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show exactly what would be merged, write nothing")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.pack_dir):
        ap.error(f"pack dir not found: {args.pack_dir}")

    print(f"Importing knowledge pack: {args.pack_dir}")
    print(f"  into robot data dir:    {args.data_dir}"
          + ("   [DRY RUN - nothing written]" if args.dry_run else ""))
    _print_manifest(args.pack_dir)
    import_coach_policy(args.pack_dir, args.data_dir, dry_run=args.dry_run)
    import_navigation_facts(args.pack_dir, args.data_dir, dry_run=args.dry_run)
    if args.dry_run:
        print("Dry run complete - re-run without --dry-run to apply. "
              "Import each pack only ONCE (arm counts accumulate).")
    else:
        print("Done. Restart Layer B (orchestrator) so the modules reload "
              "the merged files.")


if __name__ == "__main__":
    main()
