# PiCar-X roadmap / engineering handoff

Updated 2026-07-28. Repository: `lucascosolo/picarx`, branch `master`.
This file is the compact source of truth for future coding sessions. Prefer
small, scoped commits; run the full test suite before pushing. Never weaken the
safety daemon or let an LLM execute arbitrary tools, movement, or shell code.

## Production facts

- The production services are `picarx-safety.service` and
  `picarx-orchestrator.service`; the safety daemon remains an independent
  motion veto and must stay available during orchestrator updates.
- Production currently reports `/usr/bin/python3` 3.13.5 with
  `/opt/picarx/venv` injected through `VIRTUAL_ENV`, `PATH`, `PYTHONPATH`, and
  `PYTHONNOUSERSITE`. The system Python is PEP-668 externally managed; project
  packages belong in a venv, not apt or `--break-system-packages`.
- Only these modules write their SQLite databases: `location_graph.py` owns
  `spatial.db`, `reflection.py` owns `semantic.db`, and `event_logger.py` owns
  `events.db`. All other modules use bus requests and fail-soft reads.
- `camera_controller.py` is the single camera owner. Consumers use bounded
  latest-frame subscriptions with requested FPS; no module may open
  Picamera2 directly. RobotState leases coordinate camera/head/speaker state.
- RobotState states include `IDLE`, `GESTURE_TRACKING`,
  `OBJECT_DETECTION`, `SPEAKING`, `REMOTE_ASSIST`, `RC`, and `SAFETY_STOP`.
  Safety, RC, and higher-priority claims preempt gesture; stale claims expire.
- The SunFounder Robot HAT package is the git `2.5.x` branch. The unrelated
  PyPI package named `robot_hat` lacks `ADC` and breaks `picarx-safety`.

## P0 — do next, in this order

### 1. Make gesture following work on the Pi 4

The bounded head controller, camera subscription, 320×240 processing,
latest-frame buffer, 10-pixel dead zone, adaptive throttling, thermal guard,
RobotState claim, hand target, and hand bounding-box telemetry are implemented.
Production validation and the native runtime fix remain.

Root cause established from production telemetry:

1. Camera contention was a real race between gesture and object detection;
   camera ownership/state leases and the central camera controller now prevent
   both modules from opening Picamera2 concurrently.
2. Gesture loading reaches `imported`, `asset_check`, `asset_ready`, and then
   `constructing`. The model file exists at
   `layer_b/data/models/mediapipe/hand_landmarker.task` (7,819,105 bytes).
3. The installed `mediapipe==1.0.0` imports, exposes Tasks, then dies in the
   native constructor with `compiled with aes enabled ... Illegal instruction`.
   It is not a camera or landmark bug. Earlier `mp.solutions` and
   `_framework_bindings` errors were incompatible/mixed package variants.
4. The loader previously let the native crash/hang take down the orchestrator,
   expire the gesture lease, and let object detection reclaim the camera.

Already landed and pushed: async loading/status phases and cleanup; duplicate
`backend` progress fix; `--probe-model`; flushed probe phases. The current
working fix isolates model construction and inference in a child worker. A
native SIGILL becomes `model_error`, releases camera/state leases, and cannot
kill the orchestrator. The worker also serializes landmarks and supports the
new bounding-box path. Finish, test, commit, and push this worker change.

**Runtime packaging decision:** do not retain `mediapipe==1.0.0` merely for
compatibility. On 64-bit Pi 4 (`aarch64`), use the published
`mediapipe==0.10.18` CPython 3.12 aarch64 wheel. On 32-bit Pi (`armv7l`), use
`mediapipe-rpi4==0.8.8`, which is a CPython 3.7 armhf build. Update
`repair_python_environment.sh` (and the legacy setup script consistently) to
select the architecture-specific package and ABI, recreate an incompatible
venv safely, and ensure systemd starts the selected interpreter. If the robot
cannot provide the required Python ABI, fail with an actionable message rather
than installing a broken namespace. A tested custom wheel may override the
choice with `PICARX_MEDIAPIPE_PACKAGE`.

Acceptance: on the actual Pi, `--probe-model` returns structured success,
gesture status reaches `tracking`, hand target and `bbox` telemetry appear,
head commands stay within pan `[-35,+35]` and tilt `[-30,+30]`, and turning
gesture off/on or killing the worker releases all resources without taking
down `picarx-orchestrator` or `picarx-safety`.

### 2. Automatic repository update/restart — high priority

The repository updater is implemented: configured remote/branch only,
fast-forward-only pull, idle/resource/update lock, event and bus status,
orchestrator re-exec, health check, and rollback marker. It must never stop the
safety daemon. Finish target-Pi validation, failed-start rollback, power-loss
behavior, and service-environment checks. Keep update requests explicitly
configured; never accept an arbitrary ref from speech.

Progress (2026-07-28): the updater now flushes and fsyncs its rollback marker
file and containing directory before the fast-forward merge begins. A sudden
power loss during merge therefore still leaves the previous commit and target
recorded for startup recovery; marker write failure aborts before changing the
checkout. Regression tests verify the marker exists at merge time.

Progress (2026-07-28): post-update startup health now also fails closed when
the orchestrator is not running from the configured venv, the venv launcher is
not selected first on `PATH`, or user-site packages are enabled. This catches
service-environment drift before a new revision is accepted; target-Pi
systemd validation remains outstanding.

Progress (2026-07-28): the legacy `setup_python.sh` entry point now delegates
to `repair_python_environment.sh` instead of installing into the PEP-668
system interpreter with `--break-system-packages`. Both deployment entry
points therefore use the same architecture-specific MediaPipe ABI, venv
compatibility checks, and systemd environment configuration.

### 3. Remote project helper redesign

`remote_assist.py`, companion tools, web controls, SSH host-key validation,
robot-side helper bootstrap, scoped filesystem checks, bounded commands, patch
rollback, and session write authorization exist. Audit the logic end-to-end
against a real provisioned host.

Required redesign: add a transient SSH-password field for hosts that need it;
never persist, echo, log, or send it to an LLM. Replace the preview/apply-
confirmed-patch-only flow with a user-directed coding session: full access to
the explicitly scoped project filesystem, coder-LLM-assisted code edits, and
bounded tests/diagnostics. Retain project-root and symlink/path checks,
cancellation, command/output/time limits, metadata-only audit logs, and
explicit confirmation for destructive operations. Do not turn this into
unauthenticated remote code execution.

Progress (2026-07-28): the web tools console now accepts an optional
password-only-for-connect field. It is cleared from the browser field after
submission and is carried to `sshpass -d` through an anonymous pipe, never in
argv, environment, MQTT results, metadata, or companion/LLM tool input. The
remaining coding-session work is now focused on target-host validation and
workflow polish.

Audit correction (2026-07-28): `RemoteAssist` now forwards that transient
credential only to `RemoteSession.connect`; the typed result and regression
test prove it does not enter connection metadata or published output.

The helper now also exposes bounded, atomic `write_file` and non-recursive
`delete_path` operations for an explicitly authorized coding session. Writes
stay inside the resolved project root, support an expected-content hash to
avoid clobbering concurrent edits, and keep file contents out of audit logs;
edits and deletion require confirmation. Patch operations remain as a
backward-compatible path while the session/LLM orchestration is completed.

Progress (2026-07-28): the Companion coding workflow now tells the thinking model
to inspect before editing, use the approved/resumable plan, preserve expected
hashes, rely only on typed helper results, and run bounded allowlisted diagnostics.
It explicitly forbids claiming edits or passing tests without returned evidence;
the remaining redesign work is end-to-end validation against a provisioned host
and recovery behavior under real network/session failures.

Progress (2026-07-28): a thinking model must now begin an explicit, plan-approved
coding session and carry its returned session ID through destructive remote work.
RemoteAssist enforces that boundary for thinking-originated writes, patches,
rollbacks, and commands, while human web controls remain available separately.
The web tools page can begin/end the scoped session and shows its ID; disconnect
always clears it.

### Thinking-plane tool access (new priority)

The conversational robot should receive the complete typed catalog of
non-movement tools on every thinking turn, rather than a lexical subset that
made multi-step work appear unavailable. It may inspect capabilities and
current activity, query health/perception/session state, manage reminders,
notes, radio, memory, people, Bluetooth, remote coding, and other bounded
services. Movement remains exclusively on the local safety-critical command
path; it must not be exposed as an LLM tool or accepted if a model attempts
to call it anyway.

Progress (2026-07-28): Companion now advertises all non-movement tools,
including `describe_tools`, `get_robot_status`, and typed radio control;
movement tools are filtered at both catalog construction and dispatch. The
tool loop is expanded to eight rounds/16 calls with a hard exhaustion reply,
and latest state mirrors are subscribed for honest “what am I doing?” answers.
Progress (2026-07-28): Companion now has an ephemeral, expiring plan manager.
The thinking model can propose a bounded goal and ordered steps but cannot approve
its own plan. A typed local control path and the web tools page can approve, reject,
or cancel a plan; destructive meeting-note, reminder, note, and remote coding
operations require the approved plan ID before they publish a request. Plan events
and safe outcomes enter the decision journal for later reflection. Voice-native
approval is now restricted to explicit spoken plan phrases while a live plan is
pending; a bare “yes” cannot authorize it. Spoken rejection/cancellation also
uses the typed control path and stops an active thinking loop. Resumable plan
execution now has explicit, model-reported step progress and a read-only
`resume_plan` view so a later turn can continue an approved plan without guessing
which step ran. Completion closes the approval gate. The web tools page shows the
same bounded progress; plan progress does not execute anything itself.

Progress (2026-07-28): Companion now attaches correlation IDs to reminder,
notes, and remote-assist requests, briefly waits for fast daemon results, and
feeds bounded structured results back into the next model round. Slow work is
reported as pending instead of being invented as complete; the status tool
remains the read-only fallback.

Progress (2026-07-28): a typed thinking-control path can cancel active model
loops, and `cancel_current_task` is available as a non-movement tool. Runs
publish bounded lifecycle state, check cancellation between model/tool calls,
and never translate cancellation into a motor command. Long-running remote
operations now accept a typed cancel request: the persistent SSH/helper channel
keeps listening while an allowlisted host process runs, terminates that process
group, returns bounded canceled metadata, and keeps the project session alive.
Actual remote-host validation and a fuller user-directed coder-session workflow
remain next.

Growth architecture progress (2026-07-28): thinking-tool requests and
bounded outcomes now enter the existing `picarx/decision` journal with only
tool names, field names, and safe summaries. Reflection consumes that topic,
so the robot can learn durable patterns about how this individual uses its
capabilities without storing note contents, remote source, passwords, or raw
command arguments. This keeps learning evidence-based rather than adding a
second hard-coded personality memory.

### 4. Follow/perception reliability

Follow already has freshness telemetry, bounded reacquisition, arbitration and
safety reporting, correction `label_memory`, COCO export, deterministic
training-bundle generation, offline training, evaluation gating, and rollback.
Next: reproduce the sit-still case; separate fresh detector-pass time from
cached-payload time; stop/reacquire/give-up deterministically; benchmark real
targets; and never imply that label memory changed detector weights.

## P1 backlog

- **Reminder coherence:** preserve the actual reminder in conversation state.
  A follow-up such as “what are you reminding me to do?” must answer the
  stored text, not claim that no reminder exists or invent a correction.
  Test set, list, due, restart, and follow-up dialogue paths.
- **Notes/meeting lifecycle:** field-validate implemented persistent reminders,
  single notes, consented start/pause/resume/stop meeting transcripts, boot
  briefing, deletion/audit semantics, retention, and voice/web parity.
- **LLM gateway:** shared Claude complexity routing, optional OpenAI fallback,
  redacted telemetry, optional SDKs, and no tool/motion authority in the
  gateway are implemented. Validate live quotas, timeout/fallback behavior,
  privacy classes, and no duplicate side effects.
- **Intent recovery:** high-confidence read-only repair and safety catalog are
  implemented. Finish evidence-aware one-shot retries, redacted learning,
  ambiguity handling, and tests covering remote/destructive/motion refusal.
- **Short local clips:** bounded local audio/video capture/playback/delete with
  RobotState ownership, privacy confirmation, interruptibility, and storage
  limits.
- **Location graph:** completed telemetry/provenance and veto evidence remain
  authoritative. Later add validated fingerprint thresholds, directed edges and
  traversal timestamps, conservative IMU quality policy, audited merge/split,
  then optional embeddings only after an evaluation corpus.

## Completed architecture and capabilities

Lease-based RobotState, central camera ownership, head-intent arbitration,
bounded RC/dead-man behavior, safety limits (global pan `[-75,+75]`, tilt
`[-35,+35]`), follow telemetry/reacquisition, object detection, notes and
reminders, shared LLM routing, web console, radio, speech controls, location
graph/curiosity/navigation recovery, event/decision journaling, coach/self-
training safeguards, and camera overlays are present. Optional dependencies
must remain fail-soft. Camera/head/drive commands must remain vetoable; the
safety daemon is the final authority.

## Known incidents and diagnostic commands

- `picarx-safety` failing with `cannot import name ADC from robot_hat` means
  the wrong Robot HAT distribution is installed; repair the venv with the
  SunFounder git branch and validate `from robot_hat import ADC, Pin, PWM,
  Servo, fileDB, Grayscale_Module, Ultrasonic`.
- Gesture status states `starting`, `camera_wait`, `model_loading`,
  `model_error`, `searching`, and `tracking` are intentional. Always inspect
  `phase`, interpreter, package path/version, backend, model path/size,
  elapsed/frame age, error, and cleanup on `picarx/gesture/status`.
- Reproduce using the service environment and the module's `--probe-model`
  option. The probe must be camera-free and, after the worker fix, must report
  a child exit/trace rather than kill the shell on native failure:
  `timeout --signal=KILL 60s sudo -u picarx env VIRTUAL_ENV=/opt/picarx/venv
  PATH=/opt/picarx/venv/bin:$PATH PYTHONPATH=/opt/picarx/venv/lib/python3.13/site-packages
  PYTHONNOUSERSITE=1 /usr/bin/python3 layer_b/modules/gesture_tracking.py
  --probe-model` (adjust interpreter/site path after packaging fix).
- Inspect `mosquitto_sub -v -t picarx/state/current`,
  `mosquitto_sub -v -t picarx/camera/status`, and
  `sudo fuser -v /tmp/picarx-camera.lock`. Object detection must not own the
  camera while gesture tracking owns it.
- The `externally-managed-environment` error means pip was aimed at system
  Python. Use `repair_python_environment.sh`; do not add `--break-system-
  packages` to the project installer.

## Delivery rules

Use fake camera/MediaPipe/worker/process/servo/thermal/SSH tests off-robot,
then field-test on the Pi. Current local full-suite baseline after the gesture
worker tests is 951 passing tests (`python3 -m unittest discover -s tests -p
'test_*.py'`); warnings from existing resource-cleanup tests are non-fatal.
Keep commits scoped (for example, gesture worker; packaging; roadmap), push
to `master`, and report any target-Pi limitation honestly instead of claiming
that a native package was validated when it was not.
