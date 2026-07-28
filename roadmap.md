# PiCar-X Unified Roadmap

Generated: 2026-07-26
Repository: <https://github.com/lucascosolo/picarx>
Baseline: `master` at `fb16ffc`

This is the single source of truth for planned work and delivered roadmap
features. The previous `ROADMAP_STATUS.md` build log has been folded into the
completed-work inventory below. The database ownership boundary remains
mandatory: `location_graph.py` is the only writer of `spatial.db`,
`reflection.py` is the only writer of `semantic.db`, and `event_logger.py` is
the only writer of `events.db`. UI and other modules request work over the bus
and read databases fail-soft.

## Current direction

The immediate goal is to make the robot a more useful and playful companion,
not to expand navigation sophistication first:

1. **Remote project helper:** give the robot a host IP address, establish an
   authenticated SSH session, and have the robot copy/run its own helper on
   the host to inspect files, propose/apply code changes, run approved
   commands, and return debugging output. No separate host-side installation
   should be required.
2. **Gesture-responsive head:** use MediaPipe Hands so the robot follows a
   pointing or moving hand while keeping the tracked hand near the middle of
   the camera frame, without overheating the Pi or exceeding servo limits.
3. **Notes and reminders:** make “remind me”, “take a note”, and “take notes”
   first-class, persistent interactions. A one-off note should land in the
   memory bank; a meeting session should continuously append timestamped
   transcript segments until the user stops it. The robot must be able to
   list and delete reminders and notes, announce pending reminders after boot,
   and speak each reminder at its scheduled time.
4. **Reliable LLM access:** put all Claude calls behind one complexity-aware
   helper and use an optional OpenAI API fallback when Claude is unavailable,
   without moving tool or safety authority out of the robot modules.

The existing navigation and memory work remains valuable, but it follows these
user-facing capabilities unless a dependency or safety issue moves it forward.

## Implementation snapshot — 2026-07-26

The new work is underway in the current worktree:

- **RobotState foundation:** implemented as lease-based exclusive claims with
  `IDLE`, `GESTURE_TRACKING`, `OBJECT_DETECTION`, `SPEAKING`, `REMOTE_ASSIST`,
  `RC`, and `SAFETY_STOP`; TTS, RC, vision camera handoff, gesture, remote
  sessions, and the head-intent arbiter now participate in the state channel.
- **Gesture tracking:** bounded controller, threaded latest-frame capture,
  320×240 input, 10px deadzone, hard pan/tilt limits, adaptive frame skipping,
  CPU/thermal guard, and optional MediaPipe integration are implemented. Target
  Pi hardware and thermal validation remain outstanding.
- **Remote assist:** voice and `/tools` web-console controls, scoped JSON-lines
  helper, SSH host validation, typed companion tools for remote project
  operations, robot-side helper bootstrapping, bounded session logs, rollback
  of the last applied patch, and session-scoped write authorization are
  implemented. A real provisioned-host end-to-end test and host-key/user setup
  validation remain outstanding.
- **Follow/perception feedback:** producer timestamps, bounded head
  reacquisition, stale-track handling, human-correction dataset capture, and
  COCO export are implemented. A deterministic Darknet training-bundle
  builder, optional offline training runner, evaluation-metric promotion gate,
  and rollback-backed candidate promotion tool are implemented. No candidate
  weights have been trained or promoted in this environment; current on-device
  correction remains `label_memory`, not weight training.
- **Notes/reminders:** relative and clock-time reminders persist, re-arm after
  restart, list/delete through voice, companion, and web console controls, and
  speak when due. Single notes, auditable deletion, consented continuous meeting
  logging, transcript export, and the bounded boot briefing are implemented in
  the current worktree. Full target-Pi speech/privacy validation remains.
- **Hardware boundary:** the safety daemon globally clamps pan to
  `[-75°, +75°]` and tilt to `[-35°, +35°]`, while gesture tracking remains
  intentionally narrower at pan `[-35°, +35°]` and tilt `[-30°, +30°]`.
  Normal scan/expression producers stay within the global envelope. The
  arbiter also suppresses competing head and drive intents while gesture, RC,
  remote-assist, speaking, or safety-stop states own the robot.

## Priority order

### P0 — Shared safety and resource design gate

This is the only foundation that should precede live integration of the new
features. Add a small, explicit resource/state contract without changing the
safety daemon's veto authority:

- Specify the `RobotState` enum and transition table for `IDLE`,
  `GESTURE_TRACKING`, `OBJECT_DETECTION`, `SPEAKING`, `REMOTE_ASSIST`, `RC`,
  and `SAFETY_STOP`. Safety stop remains independently enforceable by the
  safety daemon even if state management fails.
- Specify camera, head-servo, speaker, and remote-shell ownership
  or lease semantics. Only one mode may own a resource at a time; stale leases
  expire to a safe idle/stop behavior.
- Specify transition telemetry on the bus and in the decision/event journal.
  Existing modules must continue to work when the state module is disabled.
- Establish the test seam first: fake camera frames, fake MediaPipe results,
  fake CPU/temperature readings, fake servos, and a fake SSH helper.

The interfaces and safety tests are agreed here; the active implementation is
Task 3 below. Tasks 1 and 2 may be developed with fakes, but neither is enabled
on the robot until the state manager and its preemption tests land.

### P0 — Remote project helper

This track may proceed in parallel with gesture development. It must never
turn unauthenticated speech into unrestricted remote code execution.

1. **Robot-owned helper protocol.** Ship the helper in the robot repository
   (`tools/picarx_host_helper.py`). It should expose bounded operations for
   connection health, directory listing, file search, file read, unified-diff
   preview, patch apply, command execution, test execution, and log retrieval.
   Return structured JSON with exit status, stdout, stderr, truncation
   markers, and a request ID.
2. **Connection and IP intake.** Accept a validated IPv4/IPv6 or hostname
   through the existing voice/web command pipeline (for example, “connect to
   192.168.1.20”). Do not ask the user to speak a password. Use a
   pre-provisioned SSH key, verify the host key, show the target and scope,
   and require explicit confirmation before the first connection and before
   writes or destructive commands. A confirmed “grant remote write access”
   action may authorize patch/rollback writes for the connected session;
   revoke or disconnect clears that grant. Individual remote commands remain
   separately confirmed.
3. **Self-bootstrap over SSH.** After authentication, open a short-lived SSH
   bootstrap command that streams the helper source from the robot to a
   private, random host temp path, then start that file as the JSON-lines
   helper in a second SSH channel. Require only a host Python 3 runtime; do
   not require package installation, copying files by hand, shell-profile
   edits, or a separately launched host service. Remove the temp helper on
   disconnect when the host permits it.
4. **Scoped filesystem exploration.** Start each session in a configured
   project root. Support bounded tree/list, search, read, and metadata calls;
   reject path traversal, symlink escapes, oversized files, and binary/image
   reads unless explicitly requested through a bounded artifact path.
5. **Code-change and debug loop.** Default to previewing a patch, then apply
   only an approved diff. Permit an allowlisted command set for common project
   workflows, with timeouts, output caps, cancellation, and a metadata-only
   session log. Add rollback by retaining the last applied patch or using a
   host-side git worktree/commit boundary.
6. **Robot integration.** Add a `REMOTE_ASSIST` mode, spoken progress/error
   summaries, web-console session controls, disconnect/kill commands, and
   persistence of the last target without persisting private keys or secrets.
   The robot should degrade to local explanation when SSH, the helper, or the
   network is unavailable.

**Acceptance:** a user can connect to a provisioned test host, explore a
project, inspect a file, preview and approve a small patch, run its tests, and
receive bounded results. An unapproved or malformed request cannot write or
execute remotely.

**Estimated effort:** 18–30 hours, excluding host-specific packaging.

### P0 — Automatic repository update and restart

This is an intentionally small, high-value deployment loop: once enabled, the
robot should pull the configured repository branch and restart its managed
services without requiring a manual login to the Pi.

- Pull only from the configured repository and branch with a fast-forward-only
  update; record the previous/new commit, trigger, and result in the event
  journal.
- Run only while safely idle, with no RC session, active drive intent, camera
  capture, meeting recording, or remote write. Use an update lock so requests
  cannot overlap, and report progress on the bus.
- Restart the managed services after the pull, run bounded import/health checks,
  and automatically restore the previous commit if startup fails. The safety
  daemon must remain independently available throughout the process.
- Keep polling or update requests explicitly configured; never pull an
  arbitrary ref from a voice phrase, and retain a local rollback record when
  the network or repository is unavailable.

**Acceptance:** with the policy enabled, the robot fetches the configured
branch, restarts itself, reports the commit transition, and returns to the
last known-good revision after a failed health check without moving.

**Estimated effort:** 8–14 hours, plus field validation of rollback and power
loss during an update.

### P0 — Follow-me reliability and perception learning

The existing follow daemon already routes movement through the arbiter and
safety daemon, but its behavior needs a measured reliability pass before more
follow features are added.

1. **Reproduce the sit-still failure.** Correlate follow enablement, fresh
   face/person detections, detector-pass timestamps, target age, selected
   intent, arbiter winner, and safety result. Distinguish “no person in the
   frame” from “vision is alive but the cached track is stale” and from “a
   valid target exists but another intent wins.”
2. **Make reacquisition explicit.** Keep the robot stationary when the target
   is absent, slowly sweep the camera through bounded pan offsets, prefer a
   fresh person track, and use a confirmed face as a centering fallback. Never
   drive blind just because an old payload is still arriving. Add hysteresis,
   target confidence/age telemetry, and a deterministic stop/reacquire/give-up
   state machine.
3. **Fix producer freshness contracts.** Vision must publish the time of the
   actual detector/face pass separately from the time a cached payload is
   republished. Follow must consume those timestamps and tolerate normal Pi
   inference/MQTT jitter without accepting stale tracks indefinitely.
4. **Audit the training claim.** Human corrections currently teach the
   on-device visual-signature `label_memory` overlay; they do **not** update
   MobileNet-SSD or YOLO weights. The system must say which tier learned the
   correction instead of implying detector retraining occurred.
5. **Capture real detector-training examples.** On a fresh human correction,
   save a bounded, consented full-frame image, corrected class, detector
   bounding box, timestamp, and provenance in an append-only dataset outside
   the hot path. Add export/validation tooling and an offline training job
   that can produce evaluated replacement weights. Never retrain on the Pi
   while it is driving, and never replace production weights without a
   precision/regression check and rollback copy.
6. **Close the loop.** Measure whether new weights improve person recall and
   reduce false labels against a held-out set. Keep `label_memory` as a fast
   overlay even when a detector model is retrained; a correction should be
   reflected immediately in memory and become a model-training sample for a
   later offline round.

**Acceptance:** a follow session either acquires a fresh person and produces
bounded, safety-vetoable motion or reports why it is waiting; it does not sit
silently because of a stale timestamp or hidden competing intent. A human
correction can be traced to immediate label-memory behavior and to a durable
training example, while model-weight changes are separately benchmarked and
reversible.

**Estimated effort:** 18–32 hours, plus field data collection and offline
model training.

### P0 — Gesture-responsive head

Implement this in exactly three deployable tasks. All motion remains bounded,
rate-limited, and independent of drive intents; the safety daemon remains the
final motion veto.

#### Task 1 — Tracking pipeline and bounded head control

- Add a MediaPipe Hands adapter with a separate frame-capture thread and a
  latest-frame/latest-result buffer. The capture path must not block the bus,
  speech, safety, or servo-control loops.
- Track one selected hand, derive a stable palm/pointing target from
  landmarks, and map target displacement to pan/tilt corrections so the hand
  is brought toward the frame center. Handle hand loss by holding briefly,
  then returning to a neutral head pose or relinquishing control.
- Enforce hard hardware limits of **pan −35° to +35°** and **tilt −30° to
  +30°** at every command boundary. Add servo rate limiting, stale-result
  expiry, and a **10-pixel center deadzone** so small movements do not cause
  jitter.
- Keep all camera and MediaPipe imports optional. If the model, camera, or
  servo is unavailable, report a fail-soft capability error and leave existing
  head behavior intact.

**Tests:** landmark-to-angle mapping, exact limit clamping, deadzone behavior,
hand loss, stale frames, capture-thread shutdown, and mocked servo commands.

#### Task 2 — Pi performance and thermal protection

- Downscale camera input to **320 × 240** before inference. Use a bounded
  latest-frame queue rather than accumulating frames.
- Skip every **third frame or more** when rate-limited; make the skip interval
  adaptive to measured inference time and frame age.
- Add dynamic CPU and thermal monitoring. If CPU usage remains above **90%**
  for more than a few seconds, progressively increase frame skipping and/or
  reduce inference frequency. Restore quality gradually after sustained
  recovery rather than oscillating at the threshold.
- Add configurable temperature thresholds, monitoring failure behavior, and
  telemetry for effective resolution, skip rate, inference time, CPU, and
  temperature. On an overheating or unreadable sensor condition, stop gesture
  processing and release head ownership safely.
- Benchmark on the target Pi with camera capture, MediaPipe, object detection,
  and TTS separately and together. Do not assume desktop timings represent Pi
  behavior.

**Tests:** adaptive throttling, sustained-over-90% behavior, thermal shutdown,
recovery hysteresis, queue bounds, and graceful absence of monitoring tools.

#### Task 3 — Active exclusive `RobotState`

- Implement the enum and transition manager described in P0. Gesture mode
  owns the head/camera budget and turns off heavy object detection; object
  detection mode cannot simultaneously run gesture inference.
- While `SPEAKING`, drop or pause camera frames rather than letting a backlog
  build. Resume with a fresh frame after TTS completes.
- Make `RC`, `REMOTE_ASSIST`, and `SAFETY_STOP` preempt gesture tracking. RC
  and safety behavior must continue to use ordinary vetoable intents and the
  existing safety daemon.
- Add mode-specific heartbeat/status output, transition reasons, timeout
  recovery, and a single cleanup path that releases camera, servo, speaker,
  and remote-session resources.
- Gate the feature behind configuration and default it off until target-Pi
  thermal, servo-limit, and recovery tests pass.
- **Outstanding model-loading diagnosis:** when gesture status reaches
  `model_loading` and then disappears or the state lease returns to `IDLE`,
  capture the exact service interpreter, MediaPipe package/version/backend,
  hand-model asset path and size, camera-frame age, initialization duration,
  exception, and cleanup transition. Add a bounded timeout and terminal
  `model_error`/recovery status so a failed Tasks/legacy backend cannot look
  like a silent camera or state-arbitration failure.

**Acceptance:** no two heavy camera modes run concurrently, speech does not
create an unbounded frame backlog, every head command stays inside the stated
angles, and loss of a module or heartbeat returns the robot to a safe state.

**Estimated effort:** 30–48 hours across the three tasks, plus Pi testing.

### P0 — Notes, meeting logging, and reminder lifecycle

This is the next user-facing utility track after the shared state/safety gate.
The first implementation is now in the worktree: `notes_store.py` owns bounded
durable JSON records, `notes_daemon.py` owns meeting capture and mirrors only
finalized notes to the semantic-memory writer, and reminder/notes controls are
available through voice, typed companion tools, and the web console. Remaining
work is field validation and the hardening items in the acceptance tests below.
Every new command must have both a voice path and an equivalent typed web
console control; the console must publish typed, allowlisted requests rather
than arbitrary bus messages.

1. **Reminder lifecycle.** Preserve the existing example flow — “remind me in
   10 minutes to take out the trash” — while adding deterministic parsing for
   relative and exact local times, a bounded list/status view, stable IDs, and
   delete/cancel operations that disarm the timer and persist immediately.
   Re-arm surviving reminders after restart, announce a short list of pending
   reminders once the robot is ready after boot, and continue speaking each
   reminder at its due time. Invalid, duplicate, or over-capacity requests
   must fail visibly instead of silently creating timers.
2. **Single notes.** Add “take a note …” / “remember …” as an explicit user
   command that writes a timestamped, user-authored memory entry. Provide
   list/search, inspect, and delete/archive operations. Deletion should be a
   reversible or auditable lifecycle state where practical, rather than
   pretending that an immutable event-journal row disappeared.
3. **Continuous meeting notes.** Add explicit `start`, `pause`, `resume`, and
   `stop` controls with a session ID and a consent acknowledgement before
   recording. While active, append bounded transcript segments from the speech
   recognizer with timestamps, speaker/source metadata when available, and
   periodic flushes so a crash loses only the current small segment. Do not
   send every segment through the LLM or let an unbounded meeting consume the
   semantic-fact table; finalize into a searchable note or exportable text
   artifact, with retention and delete controls.
4. **Boot and notification behavior.** Announce at most a small, configurable
   number of due-soon/pending reminders at boot, suppress or defer the briefing
   while RC, safety stop, or another higher-priority speech owner is active,
   and expose notification state in telemetry. A fired reminder should be
   acknowledged as delivered and never repeat after a restart.
5. **Web console and verification.** Add a Notes/Reminders page or a clearly
   scoped section in `/tools` with create/list/search/delete, meeting-session
   controls, transcript preview/export, and explicit confirmation for deletes
   or recording. Test parser edge cases, timer cancellation and persistence,
   boot idempotence, crash-safe segment flushing, deletion semantics, privacy
   boundaries, and parity between voice and web requests.

**Acceptance:** a user can say or submit the example reminder, see and cancel
it, reboot without losing it, and hear it exactly once at the requested time.
They can create one memory note, remove it, and start/stop a consented meeting
log whose bounded transcript is searchable afterward. No note or transcript is
silently sent to the cloud merely because it was recorded.

**Estimated effort:** 14–24 hours, excluding a full speech-recognition quality
pass.

### P0 — Unified LLM gateway with Claude routing and OpenAI fallback

The current LLM users (`companion.py`, `coach.py`, and `reflection.py`) each
own an Anthropic client and call Claude directly. Replace those provider
boundaries with one small, fail-soft helper while keeping prompts, safety
policy, and tool authorization in the calling modules.

1. **Common request contract.** Create a shared LLM helper that accepts a
   request ID, task name, complexity (`low`, `standard`, or `high`), system
   and user content, optional images, optional tool schemas, token/time limits,
   and a privacy classification. Return normalized text, structured/tool-call
   content, usage, provider, model, latency, and failure metadata. The helper
   must not execute a tool or publish motion; it only selects a provider and
   returns a response.
2. **Preserve the present Claude behavior.** Route simple/fast intent,
   reflection, and coaching work to the existing Claude Haiku configuration;
   route open-ended companion work and other high-complexity requests to the
   existing Claude Sonnet configuration. Keep model IDs configurable and do
   not silently upgrade or downgrade a task merely because a provider failed.
   Modules declare complexity rather than embedding provider/model selection.
3. **OpenAI fallback.** Add an optional OpenAI adapter using only the
   `OPENAI_API_KEY` environment variable, with configurable fallback model IDs
   and no secrets in `config.json`, logs, MQTT payloads, or the web console.
   If Claude reports quota/usage exhaustion, token exhaustion, timeout, or a
   transient API/package failure, retry the same idempotent request through
   OpenAI and mark the response as a fallback. Do not duplicate side effects
   from tool calls, and do not fall back around a safety/policy refusal without
   an explicit policy decision.
4. **Dependency and privacy behavior.** Keep both SDK imports optional. If
   neither provider is usable, callers retain their current local fail-soft
   behavior. Make image/audio/transcript forwarding explicit by privacy class,
   expose provider/model/fallback/error telemetry, and bound retained prompts
   and response logs.
5. **Migration and verification.** Migrate conversation/tool loops, intent
   repair, object identification, coach planning, and reflection extraction
   without changing their output parsers. Add fake-provider tests for
   complexity routing, Claude success, quota/error fallback, missing keys or
   SDKs, normalized tool/image responses, timeout bounds, and no duplicate
   tool execution. Surface gateway health/configuration in the web console;
   no new provider command may bypass the helper.

**Acceptance:** every production LLM call passes through the shared helper;
the helper selects the existing Claude model family from declared task
complexity, and a configured `OPENAI_API_KEY` transparently handles eligible
Claude failures with an auditable provider/fallback marker. Missing providers
still leave the robot operational in its existing degraded mode.

**Estimated effort:** 16–26 hours, including provider-adapter tests and a
careful tool-loop migration.

### P1 — High-confidence LLM tool disambiguation and recovery

This track follows the unified LLM gateway. The existing companion intent
arbiter repairs a small set of unparsed phrases and caches successful mappings,
but it does not systematically inspect failed or misinterpreted tool calls.
Expand it only after all provider calls use the shared gateway above.

1. **Tool catalog and evidence.** Give the disambiguator a machine-readable
   catalog of tool names, required fields, safety class, expected result shape,
   and valid voice/web aliases. Feed it the original utterance, the router's
   attempted interpretation, tool-result errors/timeouts, and current robot
   state. Never infer a motion primitive from free-form model text.
2. **Confidence-gated correction.** Ask the LLM for a structured candidate
   tool call, confidence, rationale, and whether clarification is required.
   Auto-retry only when confidence clears a high threshold, the candidate is
   non-motion or read-only, the request is idempotent, and the failed attempt
   provides matching evidence. For writes, deletes, remote commands, meeting
   recording, or any movement-affecting tool, require explicit confirmation
   rather than treating confidence as consent. At most one repair attempt is
   allowed per request to prevent loops and duplicate side effects.
3. **Failure learning.** Record redacted attempted route, correction, outcome,
   and confidence in the existing phrase cache/decision journal. A human
   correction may promote a deterministic alias; an uncertain or conflicting
   correction must remain a clarification rather than silently changing a
   safety-relevant command. Expire stale aliases and provide an operator
   review/delete path.
4. **Integration and tests.** Route failures from tools registry, companion
   tool execution, reminders/notes, remote assist, radio, and future clip
   tools through one bounded recovery path. Add fake-provider tests for high
   confidence, low confidence, ambiguity, provider fallback, failed retries,
   duplicate prevention, state/safety gating, cache promotion, and web-console
   telemetry. The disambiguator must degrade to the existing local behavior
   when no LLM provider is available.

**Acceptance:** a malformed or failed read-only tool request can be repaired
once when evidence and confidence are strong; risky actions stop for explicit
approval; no repair can emit raw drive/safety commands or execute twice; and
the operator can see why a repair happened.

**Estimated effort:** 12–20 hours after the LLM gateway migration.

### P1 — Short local media clips

Add a small media tool with commands such as “record a 5 second clip” and
“playback that clip.” The first implementation should make the media type
explicit (`video` from the Pi camera and/or `audio` from the microphone) rather
than silently assuming what “clip” means, with a documented default. Store one
or a small bounded number of clips locally, rotate old files, and never upload
them automatically.

- Use a dedicated capture/playback owner coordinated with `RobotState`; cap
  duration at five seconds, reject concurrent capture, and release camera,
  microphone, and speaker resources on timeout, cancellation, or crash.
- Add privacy feedback and explicit recording confirmation where a live
  microphone/camera is involved. Playback must be local, bounded, and
  interruptible by stop/safety/TTS controls.
- Expose voice aliases, typed companion tools, and web-console record/play/stop
  controls. Show clip age, duration, media type, storage usage, and delete
  controls; add tests with fake camera/audio devices and bounded file storage.

**Acceptance:** a user can record exactly one bounded five-second local clip,
play it back, stop it, and delete it through voice or the web console without
an unbounded queue or accidental network transfer.

**Estimated effort:** 6–12 hours, depending on whether audio and video ship in
the first slice.

## Navigation and memory backlog

The following items were validated against the existing architecture. Items 1
and 7 are complete. They remain below the new P0 work because neither blocks
the remote helper, and the gesture feature now supplies its own resource-state
priority.

### 7. Rich location-resolution telemetry — **complete**

`SpatialStore.match_or_create()` returns at most three ranked candidate scores.
Room scans and location changes carry bounded `scan_id`, `resolution_id`,
`probe_id`, and `evidence_ids` correlation fields. Ambiguous matches publish a
single probe request; `field_agent.py` owns the deduplicated quick-scan FSM;
`event_logger.py` persists the request and terminal outcome.

### 1. Veto evidence and provenance — **complete**

Vetoes atomically update the location aggregate and retain bounded,
redacted scan/action/result context in `spatial.db`. Reflection receives a
recent evidence catalog, caps uncited LLM facts at 0.70, and stores validated
citations in additive `fact_evidence` rows.

### 2. Runtime configuration for fingerprint matching — **next after P0**

Expose validated runtime knobs for match threshold, ambiguity margin, minimum
distinct landmarks, and revisit gap through `robot_config` and the Config page.
Defaults must preserve current behavior; malformed values fall back safely.

**Estimated effort:** 4–6 hours.

### 3. Directed edges and traversal timestamps

Add directed aggregates and append-only traversal history while preserving the
legacy undirected `neighbors()` and `edge_list()` APIs. Record transitions only
for distinct resolved places; unresolved or same-place scans create no edge.

**Estimated effort:** 8–14 hours.

### 5. Conservative IMU/motion-quality policy

Do not integrate the head-mounted IMU into pose estimates. Cache a fresh,
bounded motion summary, reject or retain continuity for scans taken during
impact/motion/stale data, and store motion only as optional evidence. Preserve
legacy matching when no motion context exists.

**Estimated effort:** 12–20 hours.

### 6. Audited operator merge/split tooling

Add read-only place details and review proposals to the web console, then a
disabled-by-default, confirmation-protected request path owned by
`location_graph.py`. Begin with transactional merge, aliases, backups, and an
append-only audit trail. Defer automated split until retained observations can
support it. Never let the UI write SQLite directly.

**Estimated effort:** 28–48 hours. High compatibility risk.

### 4. Optional embeddings

First add versioned, fail-soft fact embeddings owned by `reflection.py`. Defer
location embeddings until an evaluation corpus exists; deterministic visual
matching remains authoritative, and embeddings may only shadow-rerank or
prefilter candidates after measured validation.

**Estimated effort:** 24–40 hours. Gated on evidence and benchmarks.

## Completed capabilities from the earlier exploration/memory roadmap

These features were implemented before this unified roadmap and are retained
as delivered scope rather than separate planning documents:

- Topological location graph, curiosity-driven exploration, temporal pattern
  mining, failure-mode-specific recovery, multi-modal situation context,
  bounded hypothesis probes, long-horizon advisory goals, and uncertainty-map
  output.
- Emergence experiments, introspection/decision journaling, confidence-aware
  coach suggestions, practical voice tools, internet radio and live station
  search, the LAN web console, and the microphone kill-switch.
- Smooth Ackermann steering, RC mode with dead-man protections, camera
  overlays, speaker toggling, idle self-training with safety isolation, fluid
  obstacle driving, and richer escape tactics.
- Voice-band filtering and configurable radio dial aliases.

The safety daemon's veto logic is unchanged; new behavior must continue to
publish ordinary vetoable intents and must fail-soft when optional modules or
dependencies are missing.

## Verification and delivery rules

- Keep one writer per SQLite database and make migrations idempotent.
- Test all new behavior off-robot with fakes, then validate on the target Pi
  with CPU, thermal, camera, servo, and TTS contention measurements.
- Record resource-state transitions, remote commands, approved patches, and
  gesture throttling reasons without storing credentials, private keys, or raw
  camera frames by default.
- Current baseline verification: `python3 -m unittest discover -s tests -p
  'test_*.py'` passes 891 tests. `pytest` is not currently installed.
- Field calibration, Pi thermal measurements, MediaPipe model packaging, and
  host-helper installation are excluded from the engineering-hour estimates.
