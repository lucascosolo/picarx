# PiCar-X roadmap / engineering handoff

Updated 2026-07-29. Repository: `lucascosolo/picarx`, branch `master`.
This file is the compact source of truth for future coding sessions. Prefer
small, scoped commits; run the full test suite before pushing. Never weaken the
safety daemon or let an LLM execute arbitrary tools, movement, or shell code.

## Direction (owner, 2026-07-29) — Marco as an individual, not a pipeline

The target architecture is genuine learning and growth, not a wider net of
hand-authored rules. Regex and phrase matching are the old chatbot shape: they
make the robot *appear* to understand a fixed set of sentences and understand
nothing else. Marco should be an individual with memories, abilities, and tools
he reaches for when he needs them, the way a person does. Concretely, every
change from here should move in this direction:

- **Deterministic tables become derived, not authored.** A phrase table is
  acceptable only as a *cache* of decisions Marco already made — populated by
  what worked, aged out when it stops working — never as the definition of what
  he can understand. Adding a hand-written pattern to make one more sentence
  work is the anti-pattern; ask instead why he could not work it out himself.
- **Invert the router.** Today: deterministic match first, model as last-resort
  repair. Target: anything addressed to Marco is his to interpret, with the
  local layer demoted to a fast path for things he has already learned, plus a
  cost/offline gate. This is the real content of the "unify utterance routing"
  item below — the unification is not a tidier keyword list, it is deleting the
  keyword lists' authority. *Started 2026-07-29* with the open conversation
  (`attention.Conversation`): while a conversation is running, the phrase
  tables no longer decide whether it continues — the conversation itself does,
  and everything said inside it goes to Marco rather than to a matcher. The
  deterministic phrases that remain on this path are now only attention
  CONTROLS (wake, sleep), the offline hard edges of the channel, not
  understanding. Still to invert: what happens to an utterance OUTSIDE an open
  conversation, where the tables still hold first refusal.
- **Capabilities are tools he chooses, not phrases he waits for.** A capability
  declares itself once and is reachable both ways (landed: `CapabilityTool` in
  `capability_registry.py`). He should be able to decide to check the clock or
  roll a die because the moment calls for it.
- **Continuity of self.** Memory (`semantic_store`, `label_memory`,
  `person_memory`, reflection, the coach's bandit policies) already exists and
  is the right substrate; what is missing is a persistent identity and history
  that shapes how he answers, so he is the same individual across restarts
  rather than a fresh assistant each turn. *Started 2026-07-29:* `identity.py`
  gives him a configurable name ("Marco"), grounded first-person in the
  personality prompt and folded into the wake phrases so his name addresses
  him. *Also 2026-07-29:* conversations are now episodic events — the closing
  edge of an open channel carries its turn count, length and why it ended,
  event_logger records both edges, and reflection reads them as brackets
  around the day's `heard:` lines. That is the first piece of narrative
  history: the digest can now say "somebody talked with me for six turns"
  instead of showing six unrelated utterances. Still missing: reflection does
  not yet distil those sessions into durable "who I spent time with" facts,
  and the learned self-model still has to be reconciled with a stable core
  identity so growth doesn't erase who he is.

**The boundary that does not move.** Layer A stays hardcoded and Marco has no
authority over it: the safety daemon remains the final motion veto, movement is
never an LLM/thinking-plane capability, and "stop"/"halt" are never filtered or
routed through anything that can think about them. Autonomy grows everywhere
above that line and nowhere below it. Cost, latency, and offline operation are
the other real constraints — the answer to them is a learned cache, not a
hand-written matcher.

## Current state (2026-07-29)

- The repository is on `master`; the local implementation currently passes
  1,119 tests. The full suite is the source-of-truth regression gate, while
  hardware and browser/device validation are still separate release gates.
- The safety architecture is intact: the independent safety daemon remains the
  final motion veto; RobotState leases and the central camera owner coordinate
  RC, speech, gesture, remote assistance, and local capture. Movement is not an
  LLM/thinking-plane capability.
- The thinking plane now has the complete typed non-movement catalog, bounded
  multi-tool loops, status/result correlation, approval plans, cancellation,
  remote coding controls, local notes/reminders/radio, and confirmed local
  audio/video clip capture and management. Decision journaling and reflection
  provide the evidence-based growth path; raw notes, credentials, source, and
  command arguments are not copied into learning records.
- The web console has a fresh dependency-free shell: wider desktop layout,
  phone-sized controls, horizontally scrollable navigation, accessible focus
  states/skip navigation, live connection/battery status, and a dashboard
  two-column layout on larger screens. Existing typed endpoints and RC
  dead-man behavior were not changed. Browser rendering and touch testing on
  the actual Pi remain outstanding.
- Voice input got two owner-reported responsiveness fixes: the phantom leading
  word Vosk invents is now stripped using the decoder's own per-word
  confidence, and plain talk aimed at the robot (second person / question
  shape) reaches the chat path instead of being dropped for not being an
  order. Both are off-robot tested only; see the responsiveness item below for
  what is still open.
- Capability unification has started: `dialog.py` now asks the capability
  router whether anything actually claims an utterance before spending the
  command-repair budget on it, and capabilities declare an optional
  `CapabilityTool` next to their phrase rules so dice and clock are reachable
  from the thinking plane. Both are steps toward the Direction above. The
  conversation is now an open channel rather than a keyword-reset window
  (`attention.Conversation`, `picarx/dialog/conversation`): inside it the
  phrase tables no longer decide whether the conversation continues. Outside
  it they still hold first refusal, and that is the next thing to invert.
- Local clips are bounded and interruptible, but camera/ALSA codec behavior,
  playback devices, and service startup have not yet been validated on the
  target Pi. The gesture native runtime/package decision and provisioned-host
  remote workflow also still need field validation.

## Suggested next steps

1. **Field validation on the Pi:** run the gesture model probe under the
   selected venv, verify systemd startup/recovery, capture and play short audio
   and video clips, and exercise the web console on a phone and desktop while
   observing safety, camera, and RobotState telemetry.
2. **Close remaining P0 gates:** finish the architecture-specific MediaPipe
   packaging/ABI work and target-Pi gesture worker validation; then validate the
   remote coding session against a real provisioned host, including reconnect,
   cancellation, and failed-command recovery.
3. **Finish web parity deliberately:** add clip status/list/play/delete controls
   to the console only through the existing typed clip endpoints, then add
   browser-level smoke coverage for narrow layouts, live updates, consent, and
   offline/degraded responses. Do not duplicate safety logic in JavaScript.
4. **Strengthen individual learning:** collect a small privacy-safe evaluation
   corpus from real interactions, measure whether reflection/self-training
   changes behavior from evidence, and add retention/review controls before
   expanding what the robot stores or adopts.

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
  `OBJECT_DETECTION`, `SPEAKING`, `REMOTE_ASSIST`, `LOCAL_CAPTURE`, `RC`, and
  `SAFETY_STOP`.
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

Progress (2026-07-28): startup recovery now compares `HEAD` with both commits
recorded in the durable marker. A power loss before the merge simply clears
the marker, a failed health check can reset only the intended new revision,
and an unrelated checkout is left untouched with a rollback error for
operator recovery.

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

Progress (2026-07-28): transport exceptions during remote reads, edits, tests,
or cancellation now fail closed: the SSH/helper session is closed, the remote
state claim and coding/write authority are cleared, and the typed result says
the session was disconnected. Typed helper errors still remain recoverable
without throwing away a healthy session.

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
Progress (2026-07-28): a newer detector pass with no person now invalidates the
cached positive target for motion immediately while retaining it in status for
diagnosis; out-of-order producer timestamps cannot resurrect an older target.
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
  Progress (2026-07-28): a shared `ClipStore` now provides generated-ID-only
  paths, atomic media/metadata finalization, incomplete-capture cleanup, and
  bounded per-clip/count/total-storage limits. Capture/playback control now
  uses the central camera subscription, delegates audio to the existing
  microphone stream, owns the `LOCAL_CAPTURE` lease, and interrupts on higher
  priority state. Companion exposes confirmed capture plus bounded clip
  management as non-movement thinking tools. Real-device camera/ALSA codec
  validation remains outstanding.
- **Responsive web UI:** refresh the existing web console with a lightweight,
  attractive responsive design that remains usable on phones and desktops.
  Preserve the current typed safety/control boundaries, keep pages fast on the
  Pi, and validate touch targets, narrow-screen layout, live status updates,
  and degraded/offline behavior.
  Progress (2026-07-28): the shared shell now provides a wider desktop canvas,
  stacked narrow-screen form controls, touch-sized buttons, scrollable nav,
  dashboard reflow, visible connection/battery status, and keyboard skip/focus
  affordances without changing control endpoints. Browser/device validation
  and clip-control parity remain next.
- **Location graph:** completed telemetry/provenance and veto evidence remain
  authoritative. Later add validated fingerprint thresholds, directed edges and
  traversal timestamps, conservative IMU quality policy, audited merge/split,
  then optional embeddings only after an evaluation corpus.
- **Conversational responsiveness (owner-reported, 2026-07-29):** the robot
  "imagines the word 'The' before almost every sentence and refuses to respond
  to almost anything besides pre-programmed command phrases." Two causes, both
  now addressed, plus what is still open.
  - *Phantom lead word (fixed):* Vosk prepends a word nobody said — usually
    "the" — as the cheapest acoustic explanation for a breath, a lip smack, or
    the leading edge of a word the noise gate clipped. It shifts the sentence
    one word right, and `speech_match.looks_directed_command()` keys on the
    FIRST word, so "take me to the kitchen" arriving as "the take me to the
    kitchen" stopped looking like an instruction and was dropped as chatter.
    `speech_match.strip_decoder_artifacts()` now removes it using the decoder's
    OWN per-word confidence (absolute floor, or well below the surrounding
    words) rather than blanket-stripping articles, which would quietly rewrite
    real speech; no word-level evidence means no strip. `audio_nodes` applies
    it before averaging confidence and keeps the original as `raw` on
    `picarx/audio/heard`.
  - *Everything that wasn't an order was dropped (fixed):* `attention.classify`
    recognized only wake phrase / open conversation window / command shape, so
    outside the 45s window plain talk died silently in `dialog.on_directed`.
    New weakest reason `attention.CHAT_SHAPE`
    (`attention.looks_conversational()`): second-person address, or a question
    opener on a sentence-length utterance. It routes to chat but deliberately
    does NOT open the conversation window, and companion's deterministic
    quality gate still decides whether an API call happens.
  - *"Do you like music" hole (fixed 2026-07-29):* a question that merely
    mentions a capability's vocabulary used to classify as COMMAND_SHAPE and go
    to the LLM intent arbiter, where `INTENT_REPAIR_COOLDOWN` (a budget for
    misheard *orders*) dropped it. `dialog._is_talk_about_a_capability()` now
    asks the capability router whether anything actually PARSES the utterance;
    talk that no capability claims, isn't shaped like an instruction, and
    addresses the robot in the second person goes to chat instead. This is the
    first piece of the router inversion in the Direction section.
  - *The conversation ended mid-sentence (fixed 2026-07-29):* the 45s window
    was reset by exactly two events — a wake phrase, or an utterance the local
    phrase tables recognized as a command — so a genuine back-and-forth timed
    out unless the human kept re-addressing the robot or kept saying things the
    matcher already knew. `attention.Conversation` replaces the timestamp
    comparison with a real open channel (voice mode): saying his name or giving
    him a command he acts on OPENS it, and while it is open every turn on
    either side keeps it alive — including plain chat and including Marco's own
    replies (`dialog.on_speak`), so thinking for twenty seconds doesn't hang up
    on you. It ends on `dialog.sleep_phrases` ("stop listening", "goodbye"), on
    `conversation_window_sec` of silence, or at `conversation_max_sec` — the
    backstop, never reset by a turn, that keeps a talkative television from
    holding the channel open forever. State edges are published on
    `picarx/dialog/conversation`. Sleep phrases are read off the RAW heard
    stream because field_agent takes "stop listening" as a motion stop and
    reports it handled first; that order is unchanged and nothing here can
    suppress a safety word.
  - *Still open:* unvalidated on the Pi — whether the artifact repair fires as
    often as the field symptom suggests (watch the `raw` field), whether
    CHAT_SHAPE raises LLM spend near a television, and whether the open channel
    plus the max-duration backstop are the right defaults in a real room.
  - *Landed 2026-07-29:* companion now consumes `picarx/dialog/conversation` —
    an open channel is the honest definition of "what is currently relevant",
    and closing it is what retires stale intent (see the stale-intent item
    below). event_logger records both edges and reflection reads them as
    brackets around the day's `heard:` lines, so a conversation is an episode
    with a length and an ending rather than loose utterances.

- **Stale intent lingers in context — Marco doesn't know when something stops
  mattering (owner-reported, 2026-07-29):** he deletes a note successfully, the
  conversation moves on, and turns later he appends "still waiting on your
  go-ahead to delete that note!" to an unrelated reply. Diagnosed:
  `companion._context_blurb()` recites `thinking plan <id> is
  pending/approved/completed` into EVERY turn's context while the plan is in
  any of those states, and a *completed* plan is never expired at all
  (`ThinkingPlanManager._expire_locked` only expires `pending`/`approved`, and
  only after the 600s TTL). So a finished note-delete plan sits in his context
  and he dutifully narrates it. The same shape affects `pending_reminders` and
  `awaiting_correction`: per-turn state injected with no link between "the task
  is done / we changed subject" and "stop mentioning it."
  *First fix landed 2026-07-29:* a plan's lifetime is now the **conversation's**
  lifetime rather than a wall-clock TTL. `ThinkingPlanManager.retire()` drops a
  plan that is no longer live; `companion.on_conversation()` subscribes to
  `picarx/dialog/conversation` and, when the channel closes, retires terminal
  plans, abandons one still waiting on an approval that will now never come,
  and clears `awaiting_correction`. An *approved* plan is deliberately left
  alone — it may still be executing, and its authority ends on its own TTL.
  `_context_blurb()` no longer recites `completed` plans at all, which is the
  reported symptom's direct cause. Memory (history, learned intents, the
  semantic store) is untouched: this drops pending *intent*, never what he
  knows. *Still open:* `pending_reminders` are still folded in on a wall-clock
  basis and outlive the topic the same way, and relevance is still not
  something Marco judges — inside a single long conversation a resolved intent
  can still linger until the channel closes. The RIGHT fix remains the
  Direction one: stale intent should age out by conversational distance, judged
  from the dialogue, not by a status enum or a timer.
- **Unify utterance routing behind one arbiter (architecture debt):** today,
  "is this utterance for me, and who handles it" logic is split across several
  independently-maintained phrase/keyword lists that have to be hand-kept in
  sync: `field_agent.py`'s `TOOL_KEYWORDS`, `tools_registry.py`'s `_TOOL_WORDS`
  and its `RULES` regex table, `attention.py`'s wake/command-shape model, and
  companion's separate LLM tool-calling catalog. The `field_agent.py` /
  `tools_registry.py` comments already admit this is fragile ("without them
  here both modules would escalate the same text twice"), and every new tool
  (see the dice/clock/weather/web-search entry below) currently means editing
  2-3 of these files by hand with no single source of truth. This is the
  wrong shape and needs a real redesign, not another parallel keyword list.
  Target design: one router/arbiter owns a declarative capability registry
  (each capability — movement, radio, reminders, notes, dice, clock, weather,
  search, chat, ...— registers its own matchers once) and a single dispatch
  order: hardcoded safety words first (never delegable, unchanged), then the
  onboard/local "brain" (deterministic regex/phrase matches against every
  registered capability, replacing today's scattered per-module keyword
  lists), and only when nothing local matches an utterance that *is* addressed
  to the robot, escalate once to the LLM intent arbiter. Critically, the LLM's
  own tool calls should be dispatched back through this same router/arbiter
  instead of companion.py's separate direct-publish tool-calling path, so
  there is exactly one place that decides "who handles this utterance" and
  one audit trail, whether the decision was made locally or by the model.
  This must preserve every existing safety invariant unchanged: motion stays
  off the LLM/thinking-plane path entirely, "stop"/"halt" are never filtered,
  and the dialog broker's turn-taking/open-question semantics
  (`attention.py`, `dialog.py`) keep working. Scope this as its own
  audit-then-refactor pass across `field_agent.py`, `tools_registry.py`,
  `attention.py`, `dialog.py`, `companion.py`'s tool loop, and `arbiter.py`
  (note: `arbiter.py` today is the *motion* priority arbiter to the safety
  daemon, a different concern — decide whether the new command router is a
  new module or an extension of an existing one). Needs careful regression
  coverage given the 1,000-test baseline and the safety-critical invariants
  above; do this deliberately, not as a drive-by rename.
  **Sequencing decision (2026-07-28): do this first.** The dice/clock/
  weather/web-search tools below are the pilot for the new router — build
  them as the first capabilities registered against it, not through another
  round of mirrored `TOOL_KEYWORDS`/`_TOOL_WORDS` lists. If the full router
  isn't ready when that work starts, land at minimum a single shared
  capability-keyword source that both `field_agent.py` and
  `tools_registry.py` import, rather than adding a fourth/fifth hand-copied
  list.
  Progress (2026-07-28), stage 1 of the refactor: `layer_b/capability_
  registry.py` now provides the pure, stdlib-only mechanism (a `Capability`
  declares its phrases, vocabulary, topic, and self-description; a `Router`
  resolves one utterance to match / escalate / unclaimed in registration
  order), and `layer_b/capabilities.py` holds the declarations for reminders,
  notes, remote assist, and radio. `tools_registry.py` is now a thin bus
  adapter over the router, and `field_agent.py`'s `TOOL_KEYWORDS` is derived
  from `capabilities.keywords()` instead of hand-copied, so the two lists can
  no longer drift. Dispatch order and every existing rule are unchanged (the
  old flat table was already grouped in this order); a regression test asserts
  no movement word can become capability vocabulary. Remaining stages:
  fold `attention.py`/`dialog.py` addressing into the same decision, route
  companion's LLM tool calls back through the router, and give the escalation
  path a single audit trail.
- **New tools scaffold — dice, clock, weather, web search:** (dice and clock
  landed 2026-07-28 as the router's first pilot capabilities — see the
  progress note at the end of this entry; weather and web search remain.)
  Add four small
  tools following the existing `tools_registry.py` + `layer_b/modules/tools/`
  pattern (one bus topic and `module_registry.json` entry per tool, always-on
  since none owns exclusive hardware):
  - `dice`: local `random`-only dice rolls and coin flips, no network/LLM —
    instant and free, same category as "party tricks" in the tools_registry
    docstring.
  - `clock`: local "what time is it" / "what's the date" off the Pi's own
    system clock, no network/LLM, same trust boundary reminder_daemon.py
    already relies on for `at: "HH:MM"`.
  - `weather`: fetch from `wttr.in`'s no-key plain-text endpoint (stdlib
    `urllib` only, ~5s timeout, fail-soft to "couldn't reach the weather
    service"); optional `weather.default_location`/units config knobs, empty
    location falls back to wttr.in's IP geolocation.
  - `web_search`: a new `layer_b/web_search_client.py` (stdlib-only, mirrors
    `radio_browser.py`'s style) queries DuckDuckGo's free, keyless Instant
    Answer API for a bounded set of snippets, then one low-complexity
    `LLMGateway.complete()` call turns them into a short spoken summary,
    grounded only in the fetched snippets (never asked to invent facts
    beyond them). No snippets or no LLM available both fail soft — the
    former says nothing useful was found, the latter reads back the best raw
    snippet instead of a summary. This is the one tool here that costs a
    token; log it like other `intent_repair`-class calls, not silently.
  Also route `flip a dice/coin`, `what time/day is it`, `what's the weather`,
  and `search the web for …` / `look up …` in `tools_registry.py`'s RULES,
  add the matching entries to `TOOL_DESCRIPTIONS`, then register each with
  the unified router from the "unify utterance routing" item above instead of
  hand-adding another `_TOOL_WORDS`/`TOOL_KEYWORDS` mirror pair — that item is
  a prerequisite for this one, not a follow-up. Cover each with off-robot
  unit tests (parsing + fail-soft network/LLM paths, per the `harness.py`
  `FakeBus` pattern already used for `reminder_daemon`/`tools_registry`).
  Progress (2026-07-28): `dice` and `clock` are implemented the new way —
  declared once in `layer_b/capabilities.py`, served by
  `layer_b/modules/tools/dice.py` and `tools/clock.py`, registered in
  `module_registry.json`, with no keyword list edited anywhere else. Dice
  bounds (`count`, `sides`) are clamped in the daemon as well as the parser
  because a request can also arrive from a model; the clock speaks
  conversationally ("twenty past four in the afternoon") since the answer goes
  to a speaker. `speech_match.DOMAIN_VOCAB` gained the matching words so "what
  time is it" reads as a command instead of needing a wake phrase. Still open
  for these two: they are not yet in `tool_catalog.py`, so the thinking plane
  cannot call them — that arrives with the stage that routes companion's tool
  calls through the same router.

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
then field-test on the Pi. Current local full-suite baseline is 1,072 passing
tests (`python3 -m unittest discover -s tests -p 'test_*.py'`); warnings from
existing resource-cleanup tests are non-fatal.
Keep commits scoped (for example, gesture worker; packaging; roadmap), push
to `master`, and report any target-Pi limitation honestly instead of claiming
that a native package was validated when it was not.
