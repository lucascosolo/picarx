# Exploration & Memory Roadmap — Implementation Status

All four phases of the roadmap are implemented, plus the three
overarching goals (emergence, introspection, practical tools) woven
through them. Everything is additive, MQTT-first, and fail-soft: any
new module can crash or be disabled in `module_registry.json` and the
robot degrades to exactly its previous behavior. The safety daemon's
veto logic is untouched (its veto *replies* gained a machine-readable
`reason_code`, nothing else).

## What was built

| Capability | Where | Notes |
|---|---|---|
| 1. Topological location graph | `spatial_store.py`, `modules/location_graph.py` | Places = perceptual fingerprints of head sweeps (side-tagged labels + ultrasonic range bucket) — honest for a platform with no odometry. Edges from consecutive distinct scans. |
| 2. Curiosity-driven exploration | `modules/explorer.py`, `field_agent.py` | Per-place uncertainty (novelty/staleness/veto-trouble/coach-unpredictability). Wander leans away from well-understood spots; uniform random when modules absent. |
| 3. Temporal pattern mining | `pattern_miner.py`, `modules/reflection.py`, `semantic.db patterns` table | Pure Python, no LLM, runs in idle windows even without an API key. ≥3 occurrences and ≥70% confidence required; patterns age out of coach prompts after 7 days. |
| 4. Failure-mode-specific recovery | `safety/safety_daemon.py`, `field_agent.py`, `coach.py` | Stable `reason_code` (obstacle/cliff/reverse_limit) → situation keys split per failure mode; old keys warm-start new ones via embedding transfer. |
| 5. Multi-modal situation context | `coach.py` | Embedding text now includes co-present objects, tight/open space, battery, place. Per-situation-type transfer thresholds (novelty 0.70, collision 0.78). |
| 6. Active hypothesis testing | `field_agent.py` (probe state) | Ultrasonic-vs-vision disagreement triggers a bounded creep (speed 12 ≤ the 15 cap, 0.7 s < 1 s, ≥16 cm, ≤1/min; safety veto still underneath). Outcomes → `picarx/exploration/hypothesis` → reflection. |
| 7. Long-horizon goals | `modules/goal_manager.py`, `field_agent.py` | Subgoal = least-understood reachable place; reached/abandoned episodes; 3 abandonments ⇒ unreachable (persisted). Purely advisory — never publishes movement intents. |
| 8. Spatial uncertainty output | `modules/explorer.py` → `data/uncertainty_map.json` + `picarx/exploration/uncertainty_map` | Rich per-place components for human review; published only on material change. Voice: "map" / "where are you". |

## The three overarching goals

- **Emergence:** coach runs deliberate experiments on 10% of
  non-urgent queries ("suggest something unlike anything tried here"),
  tagged `experimental` and narrated as experiments. Surprises (proven
  arm fails / written-off arm succeeds) publish `picarx/coach/surprise`;
  reflection is prompted to turn them into concrete "what if" *idea*
  facts, which flow back into future coach prompts automatically.
- **Introspection:** a decision journal (`picarx/decision`, persisted
  in events.db) records every wander choice, probe, goal adoption, and
  coach strategy pick **with the reason at decision time**. Voice
  command "why" / "why did you do that" reads it back. Coach
  suggestions carry honest confidence (observed success rate or
  "guess"), and the spoken narration is calibrated to it.
- **Practical tools:** `modules/tools_registry.py` is a data-driven
  voice→topic router (`picarx/tools/available` for discovery);
  `modules/radio.py` streams internet radio (SomaFM defaults in
  `data/radio_stations.json`, editable) through the existing speaker
  via mpv/ffplay/mplayer — degrading to a spoken "no radio capability"
  when no player/network exists. field_agent ignores tool keywords so
  "stop radio" never trips the robot-wide "stop".

## Deliberate deviations from the roadmap text

- **`locations`/`patterns` tables:** locations live in a new
  `spatial.db` (owner: location_graph) rather than inside
  `semantic.db`, preserving the codebase's documented
  one-writer-per-database rule. `patterns` DID go into `semantic.db`
  because reflection (its sole writer) is also the miner.
- **Hypothesis probes don't round-trip through the coach.** The probe
  is a fixed, hard-bounded template in field_agent — a deterministic
  micro-test needs no LLM call (resource-light goal) and no new
  arbiter mode. Outcomes still feed coach indirectly via
  reflection-mined facts.
- **Goal episodes aren't bandit outcomes.** goal_progress events are
  digested by reflection into facts instead of feeding coach arms —
  goals are advisory biases, not maneuvers, so success attribution to
  a specific arm would be noise.
- **No web dashboard.** `data/uncertainty_map.json` + the spoken map
  report cover human review without a web stack on the Pi.

## Follow-up: mic reliability + radio dial tuning

- **Voice-band noise filter** (`modules/audio_nodes.py`): two cascaded
  biquads (high-pass ~150 Hz, low-pass ~4 kHz, Butterworth Q) run on
  each raw capture chunk *before* gain, so steady out-of-band room
  noise (fan/HVAC/traffic rumble, hiss) no longer inflates the energy
  gate's noise floor or clips under gain. A few multiply-adds per
  sample — negligible next to one decode. Env-toggleable
  (`AUDIO_BANDPASS=0`) and cutoffs tunable (`AUDIO_BANDPASS_HP` /
  `_LP`); fail-soft to pass-through. Measured: 60 Hz rumble knocked to
  ~16 % of voice level, 1 kHz speech preserved within 0.1 %.
- **Radio dial tuning**: stations may carry a `"dial"` (e.g. `"98.7"`)
  in `data/radio_stations.json`. "tune to ninety-eight point seven"
  (or "98.7", "one oh two point five", etc.) is parsed by
  `tools_registry.parse_dial` — handling both grouped ("ninety eight")
  and digit-by-digit ("one oh two") spoken forms plus literal digits —
  and the radio plays the stream mapped to that dial. No FM tuner
  exists on this hardware; a dial is an alias to an internet stream you
  supply. New voice commands: "what's playing", "list stations". An
  unknown dial says so instead of silently defaulting.

## Follow-up: live station search + web console + mic kill-switch

- **Live radio search** (`radio_browser.py`, stdlib-only client for the
  free radio-browser.info directory, per its API guidelines: DNS server
  discovery, speaking User-Agent, play-click reporting). "radio find
  soft rock" searches by tag then name (top-voted, broken streams
  filtered), plays the first hit, and "next station" cycles the
  RESULTS until you like one. Saved dials/names still work and switch
  back to the saved list.
- **Web console** (`modules/web_console.py` + `web_ui/console.html`,
  stdlib HTTP server on port 8088): phone/laptop control panel for
  loud rooms. Every button/text box submits the exact phrase to
  `picarx/audio/heard` — the same bytes the mic would publish — so the
  web path exercises the identical pipeline as voice. Live status
  (battery, distance, location, mission, radio) plus a heard/spoken
  log, polled every 2 s. LAN-only, no auth: don't port-forward it.
- **Mic kill-switch**: `picarx/audio/mic_control {"enabled": bool}` —
  audio_nodes keeps draining the capture pipe but decodes nothing, so
  a loud room (or the radio itself) can't fire false commands; state
  echoes on `picarx/audio/mic_state` and the console shows a toggle.

## Resource footprint

Steady-state additions are three mostly-sleeping processes
(location_graph, explorer, goal_manager — each wakes on rare events or
a 15–60 s timer), one scoring pass over a tiny SQLite table per
minute, and zero new LLM calls in the hot path. Pattern mining runs
only in idle windows. The only new LLM cost is the ~10% experiment
rounds, which *replace* cache hits, and those are capped by the
existing fail-state cooldowns. No new Python dependencies; mpv (or
ffplay/mplayer) is optional and only for radio.

## Verifying on the robot

1. **Location graph:** say "explore"; after a few scans check
   `spatial.db` (`SELECT label, visit_count FROM locations`) and
   listen for "I think this is somewhere new."
2. **Curiosity:** watch stdout for wander reasons ("...is already well
   understood, drifting toward the clearer side").
3. **Patterns:** after ≥30 min including some vetoes, check
   `semantic.db`: `SELECT * FROM patterns` (or run
   `python3 pattern_miner.py` directly — read-only, prints findings).
4. **Failure modes:** trigger an ultrasonic veto vs. a cliff veto;
   coach_policy.json grows keys like `collision_loop:...:obstacle`.
5. **Hypothesis probe:** put a thin/angled obstacle the camera can't
   classify in front; expect "Testing carefully" then a resolution.
6. **Goals:** once ≥2 places are known, expect "New mission: find my
   way back to place N"; `data/goal_state.json` records failures.
7. **Introspection:** ask "why" after any of the above; ask "map".
8. **Tools:** "what tools do you have", "play radio", "next station",
   "stop radio" (needs mpv: `sudo apt install mpv`).

Note: the orchestrator runs `layer_b/modules/`; the root-level
`field_agent.py`/`coach.py`/`audio_nodes.py` copies are kept in sync
with `modules/` as of this work (the modules/ copies had fallen behind
the newer root copies — both now match).

## Smooth Ackermann steering controller (2026-07-17)

`layer_b/modules/steering_controller.py` replaces the discrete
steer-around law with a continuous local-arc planner: vision objects
become (bearing, distance) estimates, opposite-side threats sum and
cancel (gap threading), and a pure-pursuit arc (`kappa = 2*sin(alpha)/Ld`,
`angle = atan(wheelbase * kappa)`) produces FLOAT steering angles that
are exponentially filtered and hard rate-limited, with speed scaled down
by curvature and proximity. field_agent publishes its output as ordinary
vetoable intents, alternating one primitive per tick (steer/drive) so
both reach the safety daemon through the arbiter's one-intent-per-source
channel; emergencies (evade/coach/hypothesis) still preempt it, and if
the module fails to import the old discrete law runs unchanged.

Tuning (config.json): `kinematics.wheelbase_mm` (measure your chassis),
`kinematics.steering_rate_deg_per_sec` (lower = smoother arcs),
`steering.area_distance_k` (area->distance calibration: put an obstacle
at a known distance and set `k = distance_cm * sqrt(area_ratio)`),
`steering.clearance_m` (lateral passing clearance),
`steering.curve_slowdown_gain` (speed drop with steering angle).
Inspect behaviour off-robot with `python3 tools/simulate_steer.py`.

## RC mode + camera overlay + speaker toggle (2026-07-17)

The web console can now hand the wheel to a human. RC mode publishes
ordinary vetoable intents (source "rc", priority 10 - above every AI
source, so manual input preempts queued AI motion) while
picarx/rc/mode tells the AI side to stand down; the safety daemon's
veto authority is untouched. Drive with WASD/arrows or the on-screen
D-pad; layered fail-safes (0.5s intent TTL, 4Hz client keep-alive,
0.8s dead-man stop, 60s mode timeout) mean a closed laptop lid stops
the robot, not the ceiling.

While the human drives, field_agent passively records "demonstrations":
when an obstacle-like situation appears, it snapshots the context and
collects the human's (deduped) maneuver until the path clears - one
picarx/rc/demonstration event per episode, rate-limited, persisted to
events.db and fed to reflection, which is prompted to distill repeated
demonstrations into durable tactics. A human coach, learned offline.

The live camera view draws real-time labeled bounding boxes (objects,
faces, recognized people by name) scaled over the JPEG feed, and the
Audio card gains a speaker kill-switch: off silences TTS, and the
off->on press re-runs `robot_hat enable_speaker` so the amp is always
re-asserted before speech resumes.

## Idle self-training round-trip (2026-07-20)

`modules/self_trainer.py` (disabled by default in `module_registry.json`)
closes the loop between the sibling **picarx-training** simulator and the
live robot: while idle it refines the robot's OWN learning in the sim and
folds the result back in — without ever becoming a second writer of the
learning stores.

One eligible idle window does: (1) snapshot the live data dir
(`coach_policy.json` + `events.db`/`semantic.db`, via the SQLite backup API
so a live write can't tear it) into a scratch dir; (2) run
`picarx-training/run_training.py <scenario> --knowledge-dir <scratch>
--seed-from <scratch> --speedf <low> --quiet` as a `nice`-d subprocess,
seeded from the robot's own policy so the pack is a same-lineage refinement
(Steps A/B1); (3) on clean success, publish the produced pack to the online
intakes the owning modules expose — `picarx/coach/adopt` (coach folds arms
into `coach_policy.json` in **adopt** mode, so this robot's own round-trip
never double-counts the seed), `picarx/memory/note` and
`picarx/memory/pattern` (reflection persists facts + patterns). The trainer
never writes `coach_policy.json` or `semantic.db` itself — **single-writer
ownership is preserved**.

Guardrails, all preserved from the surrounding design:
- **Live behaviour always wins.** Any `picarx/intent/move`,
  `picarx/audio/heard`, or `picarx/coach/query` resets the idle clock and, if
  a session is running, SIGTERMs it instantly (run_training tears down
  cleanly on SIGTERM). A hard `max_session_sec` wall-clock caps every run.
- **Reality-gap guard.** Adopted arms are tagged `trained_in_sim`;
  `combine_policy` never deletes an arm, and `coach._maybe_retire_arm` refuses
  to retire a sim-tagged arm — sim learning may add or refresh, never retire a
  real maneuver.
- **Safety isolation inherited.** The subprocess only ever talks to the sim's
  private `/tmp/picarx_train_<port>.sock` and an ephemeral bus port — never
  `/tmp/picarx_safety.sock` or `localhost:1883`.
- **Fail-soft.** A missing sibling repo, or a crashed/killed/timed-out
  session, degrades to "no self-training this window" — never stuck, never a
  direct DB write.

Tunables (Config page / `self_trainer.*`): `idle_after_sec`, `cooldown_sec`,
`speedf`, `max_session_sec`, `scenario_source`, `charging_only` (only train on
a healthy/topped-up battery — a proxy for being docked). Enable the module
only on a robot with picarx-training checked out alongside (or
`PICARX_TRAINING_REPO` set).
