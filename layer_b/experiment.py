#!/usr/bin/env python3
# layer_b/experiment.py
"""
A/B experiment scaffolding - PROVE the learning loop before trusting it.

The self_trainer -> coach adopt round-trip is SUPPOSED to improve the robot's
real-world behaviour, but until it's measured that's a hypothesis, not a fact.
This assigns each session to one of two conditions and alternates them across
sessions so behavior_metrics.py can compare collision/veto rates between them:

  - "adopt"   : the sim-trained coach arms are in play (the round-trip is live).
  - "control" : those arms are HELD OUT, so the robot runs the pre-adoption
                baseline for this session. (Coach still STORES adopted packs in
                control sessions; it just doesn't SELECT the trained arms - see
                coach._select_arm - so the arms stay comparable for later.)

Deterministic and persisted-counter based (even session -> adopt, odd ->
control), so it's reproducible and unit-testable. Fail-soft: any IO problem
falls back to a fresh rotation rather than raising.

Once the round-trip is trusted, set observability/experiment `enabled=false`
(coach then always adopts and no control sessions run).
"""
import json
import os

ADOPT = "adopt"
CONTROL = "control"
CONDITIONS = (ADOPT, CONTROL)


def assign_condition(counter):
    """Even session index -> adopt, odd -> control. Pure."""
    return ADOPT if int(counter) % 2 == 0 else CONTROL


def load_state(path):
    try:
        with open(path) as f:
            state = json.load(f)
        return state if isinstance(state, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def save_state(path, state):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, path)
    except OSError as e:
        print(f"experiment: could not persist rotation ({e})")


def rotate(path):
    """Advance the persisted session rotation and return (condition, counter),
    where `counter` is the pre-increment session index. Fail-soft."""
    state = load_state(path)
    counter = int(state.get("counter", 0))
    condition = assign_condition(counter)
    save_state(path, {"counter": counter + 1, "last_condition": condition})
    return condition, counter
