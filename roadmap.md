# Validated navigation and memory roadmap

Generated: 2026-07-26  
Repository: <https://github.com/lucascosolo/picarx>

This roadmap was validated against `master` at `f2aeb18`. The existing ownership boundary remains mandatory: `location_graph.py` is the only writer of `spatial.db`; `reflection.py` is the only writer of `semantic.db`; and `event_logger.py` is the only writer of `events.db`. UI and other modules request work over the bus and read databases fail-soft.

## Recommended order and timeline

Implement **7 → 1 → 2 → 3 → 5 (replacement) → 6 → 4**. Resolution telemetry and scan correlation make evidence and later operator actions auditable; configuration and directional history are low-risk foundations. The IMU proposal needs a safe motion-quality replacement rather than pose estimation. Merge/split and embeddings should follow once there is enough retained, measured evidence. Estimated total for the viable work is roughly **100–166 engineering hours**, excluding field calibration and model/data collection.

## 1. Veto evidence and provenance

**Verdict: viable.**

- `locations.veto_count` is only an aggregate today (`layer_b/spatial_store.py:note_veto`); the event payload and the scan context are not retained together.
- A sensor-evidence gate is compatible with the reflection writer model, but an LLM cannot reliably cite supporting evidence unless the reflection input/output contract carries stable evidence IDs.
- Store compact/redacted metadata and an ID, not raw camera frames, in `spatial.db`.

**required_changes**

- `layer_b/spatial_store.py`: extend `_SCHEMA`; add idempotent `SpatialStore._migrate()` after `executescript`; extend `note_veto(location_id, evidence=None, now=None)` to update the count and evidence atomically; add read-only evidence query helpers.
- `layer_b/modules/location_graph.py:on_room_scan`: retain a bounded latest scan context (fingerprint, flattened labels, distance, candidate scores, `scan_id`). `on_action_result` supplies that context plus action/result to `note_veto`. Add an optional fresh world snapshot cache only when a stable `snapshot_id` exists.
- `layer_b/semantic_store.py`: add an additive fact-evidence relation and writer-only helpers. `layer_b/modules/reflection.py:try_reflect` must require/copy cited sensor evidence before an LLM-originated fact exceeds the configured high-confidence ceiling; deterministic analyses may retain their explicit `location_graph`/`self_model` sources.
- `layer_b/modules/event_logger.py` and the room-scan producer must carry a UUID `scan_id`; do not depend on getting SQLite event IDs back across asynchronous broker delivery.

**db_schema_changes**

```sql
CREATE TABLE IF NOT EXISTS veto_evidence (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  location_id INTEGER NOT NULL,
  ts REAL NOT NULL,
  snapshot_id TEXT,
  snapshot_json TEXT,
  labels_json TEXT NOT NULL DEFAULT '[]',
  distance_cm REAL,
  candidate_similarities_json TEXT NOT NULL DEFAULT '[]',
  action_json TEXT NOT NULL DEFAULT '{}',
  result_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_veto_evidence_location_ts
  ON veto_evidence(location_id, ts DESC);

CREATE TABLE IF NOT EXISTS fact_evidence (
  fact_id INTEGER NOT NULL,
  evidence_kind TEXT NOT NULL,
  evidence_db TEXT NOT NULL,
  evidence_id TEXT NOT NULL,
  observed_at REAL,
  PRIMARY KEY (fact_id, evidence_kind, evidence_db, evidence_id)
);
```

Both are additive. `snapshot_id` stays nullable until cross-topic correlation exists. Apply the spatial migration through a new idempotent migration method; no table rebuild is required. Cap/redact `snapshot_json` and never persist image bytes there.

**interactions_with_existing_code**

- `location_graph.py` alone writes `veto_evidence` and `locations`; `event_logger.py` continues to own raw `events.db`; it must not write `spatial.db`.
- `reflection.py` alone writes `fact_evidence` and facts after read-only lookup of spatial/event evidence. For `source='reflection'`, cap confidence (for example 0.70) unless at least the defined number of recent, matching sensor references are attached. Human-confirmed inputs need an explicit separate source policy.
- Use one SQLite transaction for the aggregate counter and evidence row. The location-graph lock protects current-location/last-scan association; no database write belongs on an IMU callback.

**tests_required**

- `tests/test_spatial_store.py`: additive migration, atomic increment/evidence insert, nullable snapshot ID, redaction/size guard, and readonly refusal.
- `tests/test_memory_voice_commands.py`: scan context is attached to a veto at the active location.
- `tests/test_semantic_store.py` and `tests/test_reflection_self_model.py`: legacy facts remain readable; LLM-only high confidence is capped while corroborated facts promote.

**backward_compatibility_risk: medium.** Existing databases remain usable, but changing confidence policy changes which facts companion users see as highly trusted. Deploy in observe-only mode first and record why a proposed promotion was denied.

**approximate_effort: medium, 16–24 hours.**

**prerequisites: 7 preferred.** `scan_id` and durable resolution telemetry make the evidence references reliable; the table itself can land first with nullable IDs.

**suggested_pr:** `spatial: retain veto evidence and gate LLM fact confidence` — Persist compact veto context and require sensor provenance for high-confidence reflection facts.

## 2. Runtime configuration for fingerprint matching

**Verdict: viable.**

- The matching constants are local module globals in `layer_b/spatial_store.py` (`MATCH_THRESHOLD`, `MATCH_MARGIN`, `MIN_DISTINCT_LANDMARKS`, `REVISIT_GAP_SEC`).
- `layer_b/robot_config.py` already provides a single knob registry, atomic config writes, environment overrides, and Config-page rendering.
- Read validated knobs at match time so a saved Config-page change affects the next scan; do not broaden the first PR to uncalibrated feature-weight knobs.

**required_changes**

- `layer_b/robot_config.py:KNOBS`: add `spatial.match_threshold`, `spatial.match_margin`, `spatial.min_distinct_landmarks`, and `spatial.revisit_gap_sec` with defaults and environment names.
- `layer_b/config.json`: materialize the `spatial` defaults and add a concise `_readme` note.
- `layer_b/spatial_store.py`: add `spatial_matching_config()` with numeric validation; use it in `fingerprint_is_distinctive()` and `SpatialStore.match_or_create()`. Keep the current constants as fallback/default names for imports and pure tests.

**db_schema_changes: none.**

**interactions_with_existing_code**

- Matching remains entirely in the `location_graph.py` writer path; no ownership or locking change is needed.
- `robot_config.merge_and_save()` atomically replaces and reloads the config. The helper should call `robot_config.get()` per match, so settings take effect without a location-graph restart; environment values still win.

**tests_required**

- `tests/test_spatial_store.py`: temporary config overrides change threshold, ambiguity margin, landmark minimum, and revisit behavior; malformed/out-of-range settings fall back safely.
- `tests/test_robot_config.py`: registry/default materialization coverage; `tests/test_web_console_pages.py` already verifies that every knob is rendered and should continue to do so.

**backward_compatibility_risk: low.** Defaults exactly reproduce current behavior; invalid user input falls back rather than disabling localization.

**approximate_effort: small, 4–6 hours.**

**prerequisites: none.**

**suggested_pr:** `spatial: expose validated matching thresholds in robot config` — Make recognition and ambiguity thresholds editable without changing their defaults.

## 3. Directed edges and traversal timestamps

**Verdict: viable.**

- `edges` currently canonicalizes endpoints, so it only represents an undirected aggregate and one final timestamp.
- Location changes already identify transitions in `LocationGraph.on_room_scan`; recording directed history there preserves the single-writer rule.
- Preserve the legacy undirected API because `goal_manager.py` and `reflection.py` use it for current behavior and connectivity facts.

**required_changes**

- `layer_b/spatial_store.py`: add spatial migrations; make `note_edge(a, b, now, from_scan_ts=None, to_scan_ts=None)` write the existing edge, a directed aggregate, and an append-only history in one transaction. Add `outgoing_neighbors()`, `directed_edge_list()`, and bounded `traversal_history()` readers.
- `layer_b/modules/location_graph.py:on_room_scan`: pass scan timestamps when the resolved location changes. Do not create transitions for unresolved or same-place scans.
- Keep `SpatialStore.neighbors()` and `edge_list()` unchanged; later routing/analytics can opt into directed readers.

**db_schema_changes**

```sql
CREATE TABLE IF NOT EXISTS directed_edges (
  from_location_id INTEGER NOT NULL,
  to_location_id INTEGER NOT NULL,
  traversals INTEGER NOT NULL DEFAULT 1,
  last_traversed_at REAL NOT NULL,
  PRIMARY KEY (from_location_id, to_location_id)
);
CREATE TABLE IF NOT EXISTS edge_traversals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  from_location_id INTEGER NOT NULL,
  to_location_id INTEGER NOT NULL,
  traversed_at REAL NOT NULL,
  from_scan_ts REAL,
  to_scan_ts REAL
);
CREATE INDEX IF NOT EXISTS idx_edge_traversals_route_time
  ON edge_traversals(from_location_id, to_location_id, traversed_at DESC);
```

This is additive. Do not reinterpret existing `edges` rows as directed history; start directional observations at deployment and keep old aggregates intact.

**interactions_with_existing_code**

- `location_graph.py` is the sole spatial writer and owns all three writes atomically.
- `layer_b/modules/goal_manager.py` keeps using undirected `neighbors()`. `layer_b/modules/reflection.py:try_analyze` must keep receiving the existing three-field `edge_list()` tuple, otherwise its connectivity fact loop breaks.

**tests_required**

- `tests/test_spatial_store.py`: A→B and B→A aggregate separately; every crossing adds a timestamp row; unchanged `neighbors()` and `edge_list()` results remain compatible.
- `tests/test_memory_voice_commands.py` and `tests/test_location_graph_loop.py`: only distinct resolved transitions are recorded and unresolved scans do not create history.

**backward_compatibility_risk: low.** The current undirected schema/read APIs are retained, and empty new tables are valid on upgrade.

**approximate_effort: small/medium, 8–14 hours.**

**prerequisites: none.**

**suggested_pr:** `spatial: record directed traversal history` — Add direction and timestamps while preserving current neighbor and connectivity behavior.

## 4. Embeddings for fingerprints and facts

**Verdict: viable only as a staged, non-authoritative enhancement.**

- `layer_b/embedding_util.py` already offers optional fail-soft MiniLM/ONNX text embeddings, configured in `robot_config.py`, but they are used only by `modules/coach.py` and persisted in its policy JSON.
- MiniLM is appropriate for semantic facts and text recall. It is not a trustworthy replacement for the structured visual fingerprint fields (label/confidence/bin/area/range/motion); using it to auto-merge places would increase false merges.
- At the current expected database size, brute-force cosine over persisted vectors is simpler and safer than an ANN dependency. Introduce ANN only after a measured scale/latency need.

**required_changes**

- First phase: `layer_b/semantic_store.py` gets fact-vector read/write helpers; `layer_b/modules/reflection.py` is the only component that embeds/backfills facts and writes the results. Add a read-only semantic retrieval caller only after an explicit product use case.
- Second, gated phase: `layer_b/spatial_store.py` exposes a versioned handcrafted/camera-validated candidate vector; `layer_b/modules/location_graph.py` alone writes it. It may prefilter/rerank candidates but deterministic `fingerprint_similarity()` remains the resolution authority and ambiguity guard.
- `layer_b/embedding_util.py` and `robot_config.py`: add feature enablement/model-version/max-vector controls and keep absence of model/dependencies a no-op. Document Pi memory/install requirements before enabling MiniLM.

**db_schema_changes**

```sql
-- semantic.db
CREATE TABLE IF NOT EXISTS fact_embeddings (
  fact_id INTEGER NOT NULL,
  model_version TEXT NOT NULL,
  vector_blob BLOB NOT NULL,
  updated_at REAL NOT NULL,
  PRIMARY KEY (fact_id, model_version)
);

-- spatial.db (only after an evaluated visual/vector design exists)
CREATE TABLE IF NOT EXISTS location_embeddings (
  location_id INTEGER NOT NULL,
  model_version TEXT NOT NULL,
  vector_blob BLOB NOT NULL,
  updated_at REAL NOT NULL,
  PRIMARY KEY (location_id, model_version)
);
```

Use L2-normalized float32 blobs and a model version, not JSON vectors. Do not backfill synchronously on the driving path; perform bounded idle/background batches through the respective sole writer. A failed/missing model leaves these tables empty and preserves exact matching.

**interactions_with_existing_code**

- `reflection.py` writes `semantic.db`; `location_graph.py` writes `spatial.db`; no shared embedding worker may write either database directly.
- The current coach policy embedding flow stays independent. Avoid coupling its policy JSON to fact/location vector migrations.
- An ANN index, if ever justified, is a disposable in-memory/read-only index rebuilt from the authoritative table; never make it the source of truth.

**tests_required**

- `tests/test_semantic_store.py`: legacy facts and empty embeddings work; model-version replacement/backfill is idempotent.
- Add mocked-vector pure tests in `tests/test_spatial_store.py`: absent embedding preserves exact decisions, and an embedding candidate cannot override a below-threshold or ambiguous deterministic match.
- Add a focused reflection/recall test when the fact retrieval consumer is defined. Benchmark precision/recall and Pi latency/memory on recorded scans before the location phase.

**backward_compatibility_risk: medium.** Optional tables are safe, but model availability, vector drift, and accidental authority inversion can change recognition quality. Keep it feature-gated and shadow-evaluate first.

**approximate_effort: large, 24–40 hours.**

**prerequisites: 7 recommended.** Use telemetry to build an evaluation corpus; complete 2 and the conservative 5 replacement before considering location reranking.

**suggested_pr:** `memory: add optional versioned fact embeddings` — Persist fail-soft semantic vectors first; defer location-vector matching pending measured validation.

## 5. More aggressive IMU/pose delta integration

**Verdict: not viable as proposed; use a conservative motion-quality and continuity policy instead.**

- The IMU is head-mounted and deliberately publishes robust instantaneous motion/orientation signals, not calibrated body-frame translation or odometry (`layer_b/modules/imu.py`). There are no wheel encoders or SLAM.
- Rich fingerprints can already accept `imu_delta`, and `location_graph.py:on_room_scan` forwards one if present, but `field_agent.py:_handle_scanning_tick` does not attach it. Persisting a dynamic IMU delta in the canonical fingerprint would make a motion value look like a property of a place.
- Integrating raw gyro to choose a place would create false certainty and the exact teleportation problem it is intended to solve.

**Viable replacement**

- `layer_b/modules/location_graph.py`: subscribe to `picarx/sensors/imu`, retain a fresh bounded motion summary, and pass an ephemeral `motion_context` plus previous location into matching. During impact/body movement/stale data, return `motion_unreliable` or apply a small continuity policy rather than selecting a different room.
- `layer_b/modules/field_agent.py:_handle_scanning_tick`: attach fresh scan-time motion metadata from world state only if it has a defined timestamp contract. `layer_b/spatial_store.py` gets a pure policy helper; no coordinates and no pose-derived nearest-place claims.
- Keep the legacy fingerprint shape and absent-context behavior identical. Store motion only as optional scan/audit evidence, not as the saved location fingerprint.

**db_schema_changes: none for the first safe phase.** A later `location_observations` audit table may retain motion context, owned by `location_graph.py`.

**interactions_with_existing_code**

- Only room-scan handling writes spatial data; IMU callbacks cache memory-only state and perform no database writes.
- `location_graph.py` owns the conservative gate. `field_agent.py` remains the scan producer; `imu.py` need not change until a formal interval contract is necessary.

**tests_required**

- `tests/test_spatial_store.py`: pure continuity/motion policy cases and exact legacy behavior with no context.
- `tests/test_memory_voice_commands.py`: a stationary rescan can reconfirm the current candidate; an impact/moving scan remains unresolved and cannot schedule a teleport.
- `tests/test_imu.py`: preserve published motion/freshness fields used by the contract.

**backward_compatibility_risk: medium.** Motion gating deliberately produces more unresolved scans in unstable motion; that is safer than false localization. Ship with counters/telemetry and tune after field observation.

**approximate_effort: medium, 12–20 hours.**

**prerequisites: 2 optional, 7 recommended.** Make policy thresholds configurable only after the initial behavior is measured.

**suggested_pr:** `localization: gate room matches on scan motion quality` — Reject unreliable moving scans and apply temporal continuity without inventing pose estimates.

## 6. Conservative merge/split tooling and web UI flows

**Verdict: viable as an operator-mediated, phased feature.**

- The web console intentionally opens both databases read-only; HTTP handlers must publish edit requests, never write SQLite directly.
- Merge can be transactional with aliases and audit history. A reliable split is not currently safe because only one fingerprint and aggregate counters are retained per location.
- Automated work may propose candidates, but must never mutate locations without operator confirmation and a reversible audit trail.

**required_changes**

- `layer_b/modules/web_console.py`: add read-only place detail/proposal endpoints and a confirmation-protected `POST /places/edit` that validates then publishes `picarx/exploration/place_edit_request`.
- `layer_b/web_ui/places.html` (new) and navigation/assets in `web_console.py`: show location IDs, evidence, aliases, candidate merges, and explicit operator confirmation. The LAN console has no authentication, so destructive controls require a disabled-by-default config flag plus typed/second confirmation.
- `layer_b/modules/location_graph.py`: subscribe to edit requests and own `on_place_edit_request`; remap its current location after a merge.
- `layer_b/spatial_store.py`: add idempotent migrations and transactional `merge_locations()`; initially add only a constrained `split_location()` that requires operator-selected observations. Readers (`get_location`, `all_locations`, `neighbors`, name/object lookup) resolve aliases consistently.
- `layer_b/modules/goal_manager.py`, `field_agent.py`, and `reflection.py`: consume canonical IDs through store APIs. Preserve `reflection.py:try_analyze`'s undirected `edge_list()` contract.

**db_schema_changes**

```sql
ALTER TABLE locations ADD COLUMN status TEXT NOT NULL DEFAULT 'active';
ALTER TABLE locations ADD COLUMN merged_into_id INTEGER;

CREATE TABLE IF NOT EXISTS location_aliases (
  old_location_id INTEGER PRIMARY KEY,
  canonical_location_id INTEGER NOT NULL,
  reason TEXT NOT NULL,
  operator TEXT,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS location_edit_audit (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  operation TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  operator TEXT,
  created_at REAL NOT NULL,
  undone_at REAL
);
CREATE TABLE IF NOT EXISTS location_observations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  location_id INTEGER NOT NULL,
  scan_id TEXT,
  ts REAL NOT NULL,
  fingerprint_json TEXT NOT NULL,
  similarity REAL,
  resolution_reason TEXT
);
```

Check columns with `PRAGMA table_info` before each `ALTER TABLE`. A merge preserves the retired row as an alias/status record; it does not delete historical references. For an initial split, create a child from selected observations and keep historic aggregate visits/vetoes/edges on the parent until an explicit audited reassignment policy exists. Fallback is to disable the edit flag and retain all existing reads.

**interactions_with_existing_code**

- Web console remains read-only and publishes bus commands; `location_graph.py` remains the sole spatial writer and serializes edit transactions with its writer lock.
- Merge must coalesce sightings, counters, legacy undirected edges, directed edges/history, and veto/observation foreign references, remove resulting self-loops, and retain aliases/audit records atomically.
- Optional automatic merge/split logic publishes review-only proposal events. It never calls store mutation methods.

**tests_required**

- `tests/test_spatial_store.py` and `tests/test_spatial_sightings.py`: atomic merge totals/sightings/edges/aliases, invalid/self/unknown operation rejection, rollback, and alias-resolved reads.
- `tests/test_memory_voice_commands.py` and `tests/test_location_graph_loop.py`: request routing, current-ID remap, and no direct writer bypass.
- `tests/test_web_console_pages.py`: new route/assets, payload validation, confirmation gate, and read-only console behavior.
- Split tests must prove selected observations only; do not test or implement inferred historical partitioning without retained evidence.

**backward_compatibility_risk: high.** Location identity is referenced across goals, edges, facts, and speech. Begin with merge-only, manual confirmation, backups, and an append-only audit; defer split automation.

**approximate_effort: large, 28–48 hours.** Merge-only is about 12–18 hours; robust evidence-backed split adds roughly 16–30 hours.

**prerequisites: 7, then 1 recommended.** `location_observations`, correlated scans, and evidence make review/split defensible.

**suggested_pr:** `places: add audited operator merge workflow` — Add read-only place inspection and confirmed, transactional merge requests; keep split as a later evidence-backed phase.

## 7. Rich location-resolution telemetry

**Verdict: viable.**

- `location_change` already reaches `event_logger.py`, whose generic append-only event table needs no schema change; it currently lacks ranked candidates, scan/evidence IDs, and a terminal disambiguation record.
- `SpatialStore.match_or_create()` has only best/second ambiguity information, so it must return a bounded scored candidate list before location graph can publish it.
- The current head saccades in `LocationGraph._schedule_disambiguation` do not themselves produce a room scan: `field_agent.py` is the completed-scan producer and does not consume `disambiguation_needed` today.

**required_changes**

- `layer_b/spatial_store.py:match_or_create`: include top 2–3 `{location_id, similarity}` candidate scores in results, including ambiguous and rejected outcomes; never expose an unbounded location list.
- `layer_b/modules/location_graph.py:on_room_scan` and `_schedule_disambiguation`: generate and propagate `scan_id`, `resolution_id`, `probe_id`, candidate scores, evidence IDs, and terminal disambiguation `{attempt, outcome}` on additive fields of `picarx/exploration/location_change`.
- `layer_b/modules/field_agent.py`: subscribe to `picarx/exploration/disambiguation_needed` and queue exactly one bounded quick-scan FSM carrying the `probe_id`; make it, rather than an uncorrelated future scan, complete the active probe. Avoid duplicate direct sweeps once field agent owns that workflow.
- `layer_b/modules/event_logger.py`: log `picarx/exploration/disambiguation_needed` as an event in addition to its existing room-scan/location-change logging. A dedicated resolution topic is optional; enriched existing events are sufficient initially.

**db_schema_changes: none.** `events.db.events.payload_json` is append-only and accepts additional fields. Optional `location_observations` belongs to #6 and is not needed to ship telemetry.

**interactions_with_existing_code**

- `event_logger.py` remains the events-db writer; location graph must not attempt to obtain its SQLite ID synchronously. UUIDs carried across payloads provide correlation.
- Existing consumers (`field_agent`, `goal_manager`, `reflection`, web console) retain the topic and existing keys. New fields are additive and must be bounded/redacted.
- The quick-scan state machine must deduplicate redelivered requests and report a single terminal outcome, preserving current safety/arbiter ownership of motion.

**tests_required**

- `tests/test_memory_voice_commands.py:LocationGraphDisambiguationTest`: assert ranked candidates, scan/probe correlation, exactly one probe request, and exactly one terminal resolved/unresolved event.
- Add `tests/test_event_logger.py`: resolution and disambiguation payloads persist as valid JSON.
- Add field-agent quick-scan tests (new focused test or `tests/test_location_graph_loop.py`): a request produces one correlated room scan; duplicate/replayed requests do not.

**backward_compatibility_risk: low for telemetry; medium for the active quick-scan handoff.** Existing event consumers ignore unknown keys, but scan-FSM integration must be bounded and safety-vetoable.

**approximate_effort: small/medium, 8–14 hours.**

**prerequisites: none.**

**suggested_pr:** `telemetry: correlate location resolutions and disambiguation probes` — Add bounded candidate evidence to existing events and make active probes produce one identified rescan.

## Suggested patches

### #1: minimal spatial veto-evidence write

```python
# layer_b/spatial_store.py: add to _SCHEMA, then call a new idempotent
# _migrate() from the writable constructor for existing databases.
CREATE TABLE IF NOT EXISTS veto_evidence (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    location_id INTEGER NOT NULL, ts REAL NOT NULL,
    snapshot_id TEXT, snapshot_json TEXT,
    labels_json TEXT NOT NULL DEFAULT '[]', distance_cm REAL,
    candidate_similarities_json TEXT NOT NULL DEFAULT '[]',
    action_json TEXT NOT NULL DEFAULT '{}', result_json TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_veto_evidence_location_ts
    ON veto_evidence(location_id, ts DESC);

# Writer-only: called by location_graph.py, never by event_logger.py.
def note_veto(self, location_id, evidence=None, now=None):
    self._assert_writer()
    now = time.time() if now is None else now
    e = evidence or {}
    with self.conn:
        self.conn.execute("UPDATE locations SET veto_count = veto_count + 1 WHERE id = ?",
                          (location_id,))
        self.conn.execute(
            """INSERT INTO veto_evidence
               (location_id, ts, snapshot_id, labels_json, distance_cm,
                candidate_similarities_json, action_json, result_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (location_id, now, e.get("snapshot_id"),
             json.dumps(e.get("labels", [])), e.get("distance_cm"),
             json.dumps(e.get("candidate_similarities", [])),
             json.dumps(e.get("action", {})), json.dumps(e.get("result", {}))))
```

Keep `snapshot_id` nullable and cap/redact any optional `snapshot_json`. The reflection-side confidence gate belongs in the later semantic-writer part of the same feature, after evidence IDs are available.

### #2: validated matching knobs

```python
# layer_b/robot_config.py: KNOBS (also materialize these values in config.json)
{"section": "spatial", "key": "match_threshold", "type": "float",
 "default": 0.70, "env": "SPATIAL_MATCH_THRESHOLD",
 "desc": "Minimum fingerprint similarity required to recognize a saved place."},
{"section": "spatial", "key": "match_margin", "type": "float",
 "default": 0.15, "env": "SPATIAL_MATCH_MARGIN",
 "desc": "Minimum lead over the next place before a match is unambiguous."},
{"section": "spatial", "key": "min_distinct_landmarks", "type": "int",
 "default": 2, "env": "SPATIAL_MIN_DISTINCT_LANDMARKS",
 "desc": "Minimum distinct landmarks needed to identify an existing place."},
{"section": "spatial", "key": "revisit_gap_sec", "type": "float",
 "default": 120.0, "env": "SPATIAL_REVISIT_GAP_SEC",
 "desc": "Seconds before a matching scan counts as a new visit."},

# layer_b/spatial_store.py: call this from fingerprint_is_distinctive() and
# match_or_create(), so Config-page saves affect the next scan.
def spatial_matching_config():
    def number(key, default, env, low, high=None):
        try:
            value = float(robot_config.get("spatial", key, default, env=env))
        except (TypeError, ValueError):
            return default
        return value if value >= low and (high is None or value <= high) else default
    return {
        "match_threshold": number("match_threshold", MATCH_THRESHOLD,
                                  "SPATIAL_MATCH_THRESHOLD", 0.0, 1.0),
        "match_margin": number("match_margin", MATCH_MARGIN,
                               "SPATIAL_MATCH_MARGIN", 0.0, 1.0),
        "min_distinct_landmarks": max(1, int(number(
            "min_distinct_landmarks", MIN_DISTINCT_LANDMARKS,
            "SPATIAL_MIN_DISTINCT_LANDMARKS", 1.0))),
        "revisit_gap_sec": number("revisit_gap_sec", REVISIT_GAP_SEC,
                                  "SPATIAL_REVISIT_GAP_SEC", 0.0),
    }
```

Use `cfg = spatial_matching_config()` inside the matching path and preserve the existing constants as defaults. The first configuration PR should not expose fingerprint feature weights until they have field calibration data.
