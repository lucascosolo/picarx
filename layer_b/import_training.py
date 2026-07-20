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

This tool folds that pack into the robot's own data dir rather than
overwriting it, so a robot that has already learned things in the real
world keeps them. Coach arms combine in one of two MODES:

  * merge (default) - per situation, arms are unioned; a maneuver the robot
    already knows has the trained win/loss counts ADDED to its own, so two
    INDEPENDENT learners' evidence reinforces the same UCB1 statistics. This
    is right when the robot and the trainer learned separately (e.g. a pack
    built on a dev machine, or a cold-started sim).
  * adopt (--adopt) - a shared arm's counts/steps/rationale are REPLACED by
    the incoming pack's. This is right when the pack was seeded from THIS
    robot's own policy and refined in sim (a self-training round-trip):
    summing would double-count the shared seed (an arm that left the robot
    at 7/1 and came back 10/2 would otherwise import as 17/3).

Both modes still adopt unseen situations/arms wholesale and preserve the
robot's own embedding, adopting the pack's only to fill a gap.

  * navigation facts/patterns - upserted into semantic.db through the store's
    normal dedup (same fact just reinforces, higher confidence wins), tagged
    source 'training' so their origin stays honest. Only robot-dynamics
    knowledge is in the pack; place-specific memories are never exported,
    because the sim's rooms are not this house.

RUN IT OFFLINE. reflection.py is normally the sole writer to semantic.db and
coach.py to coach_policy.json; this tool writes both. Stop Layer B first
(Ctrl-C the orchestrator), import, then start it again - the modules read
these files at startup and will pick the combined versions up.

    # on the robot, orchestrator stopped:
    python3 layer_b/import_training.py /path/to/training_data
    python3 layer_b/import_training.py /path/to/training_data --dry-run
    python3 layer_b/import_training.py /path/to/training_data --adopt

Fail-soft: a missing file in the pack is skipped. In merge mode, importing
the same pack twice adds its counts twice, so import a given pack ONCE
(--dry-run first if unsure); adopt mode replaces shared arms, so it is
idempotent for those. If the pack's manifest carries a lineage id matching
this robot's own policy, the tool flags it and suggests --adopt. Nothing
here talks to the bus or hardware.
"""
import argparse
import copy
import hashlib
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

def combine_policy(base, incoming, mode="merge"):
    """Combine an incoming coach policy into base WITHOUT losing base's own
    real-world learning. Returns (combined_policy, stats). Pure: does not
    mutate `base` (or `incoming`).

    An arm signature present on BOTH sides is handled per `mode`:
      - "merge" : successes/failures are SUMMED (so UCB1 sees the combined
                  evidence of two INDEPENDENT learners) and base keeps its
                  steps/rationale. Right when robot and trainer learned apart.
      - "adopt" : the incoming pack's counts/steps/rationale REPLACE base's.
                  Right when the pack was seeded from THIS robot's own policy
                  and refined in sim - summing would double-count the seed.

    Everything NOT shared is identical in both modes:
      - unseen situation -> adopted wholesale
      - unseen arm       -> adopted into the shared situation
      - base's embedding wins; incoming's is adopted only if base has none
    Reserved '_'-prefixed list sections (e.g. _demonstrations) are concatenated
    and capped to the freshest MAX_DEMONSTRATIONS by timestamp (mode-agnostic)."""
    if mode not in ("merge", "adopt"):
        raise ValueError(f"unknown combine mode: {mode!r} (want 'merge' or 'adopt')")
    merged = copy.deepcopy(base) if isinstance(base, dict) else {}
    stats = {"situations_added": 0, "situations_merged": 0, "arms_added": 0,
             "arms_reinforced": 0, "arms_replaced": 0, "demonstrations_added": 0}
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
                if mode == "adopt":
                    # the pack refined this robot's own seed: take it verbatim
                    # (counts/steps/rationale) instead of summing the seed twice
                    base_arms[sig] = copy.deepcopy(arm)
                    stats["arms_replaced"] += 1
                else:
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


def merge_policy(base, incoming):
    """Back-compat alias for combine_policy in the default 'merge' (sum) mode."""
    return combine_policy(base, incoming, mode="merge")


def policy_lineage(policy):
    """A short, stable fingerprint of a coach policy's learned situations/arms.

    Mirrors picarx-training sim/knowledge.py.policy_lineage EXACTLY so the two
    repos agree: canonical JSON of the non-reserved ('_'-excluded) situations,
    sha256, first 12 hex. Reserved sections (churny demonstration logs) are left
    out so the id tracks learned behaviour, not bookkeeping. Empty/absent -> 'cold'.

    Used only as a hint: a pack whose manifest lineage equals this robot's own
    policy lineage was seeded from this robot (a self-training round-trip) and
    wants --adopt. Kept local (no cross-repo import) like MAX_DEMONSTRATIONS."""
    if not isinstance(policy, dict):
        return "cold"
    core = {k: v for k, v in policy.items() if not k.startswith("_")}
    if not core:
        return "cold"
    blob = json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


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


def import_coach_policy(pack_dir, data_dir, dry_run=False, mode="merge"):
    incoming = _load_json(os.path.join(pack_dir, "coach_policy.json"), None)
    if incoming is None:
        print("  coach policy:   (none in pack, skipped)")
        return
    dest = os.path.join(data_dir, "coach_policy.json")
    base = _load_json(dest, {})
    _lineage_hint(base, _pack_lineage(pack_dir), mode)
    merged, stats = combine_policy(base, incoming, mode=mode)
    shared_label = "replaced" if mode == "adopt" else "reinforced"
    shared_n = stats["arms_replaced"] if mode == "adopt" else stats["arms_reinforced"]
    print(f"  coach policy [{mode}]: +{stats['situations_added']} new situations, "
          f"{stats['situations_merged']} merged; "
          f"+{stats['arms_added']} new arms, "
          f"{shared_n} shared {shared_label}"
          + (f", +{stats['demonstrations_added']} demonstrations"
             if stats['demonstrations_added'] else ""))
    if dry_run:
        return
    os.makedirs(data_dir, exist_ok=True)
    tmp = dest + ".tmp"
    with open(tmp, "w") as f:
        json.dump(merged, f, indent=2)
    os.replace(tmp, dest)


def _lineage_hint(base, pack_lineage, mode):
    """If the pack descends from this robot's OWN policy (same lineage), nudge
    toward the right mode: adopt for a self-training round-trip (so the shared
    seed isn't summed twice), merge for an independently-trained pack."""
    if not pack_lineage or pack_lineage == "cold":
        return
    if pack_lineage == policy_lineage(base):
        if mode == "merge":
            print(f"  ! this pack shares your robot's policy lineage "
                  f"({pack_lineage}) - it looks seeded from this robot's own "
                  "data. Re-run with --adopt so the shared seed isn't "
                  "double-counted.")
    elif mode == "adopt":
        print(f"  ! --adopt on a pack whose lineage ({pack_lineage}) differs "
              "from this robot's - adopt is meant for this robot's own "
              "round-trip; merge may be the safer choice here.")


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


def _pack_lineage(pack_dir):
    """The lineage id the trainer stamped into the manifest, or None."""
    manifest = _load_json(os.path.join(pack_dir, "knowledge_pack.json"), None)
    if isinstance(manifest, dict):
        return manifest.get("lineage")
    return None


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
    lineage = manifest.get("lineage")
    if lineage:
        print(f"  pack lineage:   {lineage}"
              + ("   (cold - not seeded from a robot; merge is right)"
                 if lineage == "cold"
                 else "   (seeded from a robot's policy; --adopt if it's THIS one)"))


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Combine a picarx-training knowledge pack into this robot.")
    ap.add_argument("pack_dir",
                    help="directory holding coach_policy.json / "
                         "navigation_facts.json / knowledge_pack.json")
    ap.add_argument("--data-dir", default=robot_config.data_path(),
                    help="robot data dir to combine into "
                         "(default: layer_b/data/)")
    ap.add_argument("--adopt", action="store_true",
                    help="a shared arm takes the pack's counts/steps/rationale "
                         "instead of SUMMING them. Use for a pack seeded from "
                         "THIS robot's own policy (self-training round-trip), "
                         "so the shared seed isn't double-counted. Default: "
                         "merge (sum - for independently-trained packs).")
    ap.add_argument("--dry-run", action="store_true",
                    help="show exactly what would change, write nothing")
    args = ap.parse_args(argv)

    if not os.path.isdir(args.pack_dir):
        ap.error(f"pack dir not found: {args.pack_dir}")

    mode = "adopt" if args.adopt else "merge"
    print(f"Importing knowledge pack: {args.pack_dir}")
    print(f"  into robot data dir:    {args.data_dir}   [{mode} mode]"
          + ("   [DRY RUN - nothing written]" if args.dry_run else ""))
    _print_manifest(args.pack_dir)
    import_coach_policy(args.pack_dir, args.data_dir, dry_run=args.dry_run, mode=mode)
    import_navigation_facts(args.pack_dir, args.data_dir, dry_run=args.dry_run)
    if args.dry_run:
        tail = ("Adopt mode replaces shared arms, so re-importing is idempotent."
                if mode == "adopt"
                else "In merge mode arm counts accumulate, so import each pack ONCE.")
        print(f"Dry run complete ({mode} mode) - re-run without --dry-run to "
              f"apply. {tail}")
    else:
        print("Done. Restart Layer B (orchestrator) so the modules reload "
              "the combined files.")


if __name__ == "__main__":
    main()
