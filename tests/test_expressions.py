"""Ambient personality: the expressions module's context-based and random
reactions, its deference/throttling, and reflection's note-to-self write path.

The decision helpers are pure and tested directly; the dispatcher is driven
with the head-gesture seams made synchronous (no threads, no sleeps)."""
import os
import random
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import expressions  # noqa: E402
import reflection  # noqa: E402
from semantic_store import SemanticStore  # noqa: E402


def _world(**over):
    """A quiet, idle world snapshot; override any top-level field."""
    w = {
        "person": {"name": None, "stale": True},
        "face": {"detected": False, "stale": True},
        "objects": {"items": [], "close_object": False, "stale": False},
        "battery": {"low": False, "critical": False},
        "last_heard": {"text": None, "updated_at": None, "stale": True},
        "last_action": {"action": None, "updated_at": None},
    }
    w.update(over)
    return w


class IsMovingTest(unittest.TestCase):
    def test_fresh_forward_is_moving(self):
        w = _world(last_action={"action": {"direction": "forward"}, "updated_at": 100.0})
        self.assertTrue(expressions._is_moving(w, 100.5))

    def test_stale_forward_is_not_moving(self):
        w = _world(last_action={"action": {"direction": "forward"}, "updated_at": 100.0})
        self.assertFalse(expressions._is_moving(w, 110.0))

    def test_stop_is_not_moving(self):
        w = _world(last_action={"action": {"direction": "stop"}, "updated_at": 100.0})
        self.assertFalse(expressions._is_moving(w, 100.1))

    def test_recentre_turn_is_not_moving(self):
        w = _world(last_action={"action": {"direction": "turn", "angle": 0}, "updated_at": 100.0})
        self.assertFalse(expressions._is_moving(w, 100.1))

    def test_steering_turn_is_moving(self):
        w = _world(last_action={"action": {"direction": "turn", "angle": 20}, "updated_at": 100.0})
        self.assertTrue(expressions._is_moving(w, 100.1))


class IsBusyTest(unittest.TestCase):
    def test_idle_world_is_not_busy(self):
        self.assertFalse(expressions.is_busy(_world(), 1000.0, False, 0.0))

    def test_rc_mode_is_busy(self):
        self.assertTrue(expressions.is_busy(_world(), 1000.0, True, 0.0))

    def test_low_battery_is_busy(self):
        w = _world(battery={"low": True, "critical": False})
        self.assertTrue(expressions.is_busy(w, 1000.0, False, 0.0))

    def test_recent_own_speech_is_busy(self):
        self.assertTrue(expressions.is_busy(_world(), 1000.0, False, 1000.0 - 1))

    def test_recent_human_speech_is_busy(self):
        w = _world(last_heard={"text": "hi", "updated_at": 1000.0 - 2, "stale": False})
        self.assertTrue(expressions.is_busy(w, 1000.0, False, 0.0))

    def test_moving_is_busy(self):
        w = _world(last_action={"action": {"direction": "forward"}, "updated_at": 1000.0})
        self.assertTrue(expressions.is_busy(w, 1000.5, False, 0.0))


class PanDirTest(unittest.TestCase):
    def test_directions(self):
        self.assertEqual(expressions._pan_dir(120), 1)
        self.assertEqual(expressions._pan_dir(-120), -1)
        self.assertEqual(expressions._pan_dir(5), 0)
        self.assertEqual(expressions._pan_dir(None), 0)


class ChooseContextActsTest(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(0)

    def _tools(self, acts):
        return [a["tool"] for a in acts]

    def test_new_person_is_greeted_and_remembered(self):
        w = _world(person={"name": "Sam", "stale": False},
                   face={"detected": True, "frame_center_offset": 120, "stale": False})
        acts, updates = expressions.choose_context_acts(w, 1000.0, {}, {}, self.rng)
        tools = self._tools(acts)
        self.assertIn("speak", tools)
        self.assertIn("curious_tilt", tools)
        self.assertIn("remember", tools)                 # first meeting this session
        self.assertEqual(updates["greeted"], "Sam")
        speak = next(a for a in acts if a["tool"] == "speak")
        self.assertIn("Sam", speak["text"])
        tilt = next(a for a in acts if a["tool"] == "curious_tilt")
        self.assertEqual(tilt["pan_dir"], 1)             # face is to the right

    def test_already_greeted_person_is_left_alone(self):
        w = _world(person={"name": "Sam", "stale": False})
        acts, _ = expressions.choose_context_acts(w, 1000.0, {}, {"Sam": 999.0}, self.rng)
        self.assertEqual(acts, [])

    def test_greeting_after_ttl_does_not_re_remember(self):
        w = _world(person={"name": "Sam", "stale": False})
        greeted = {"Sam": 1000.0 - expressions.PERSON_GREET_TTL - 1}
        acts, _ = expressions.choose_context_acts(w, 1000.0, {}, greeted, self.rng)
        self.assertIn("speak", self._tools(acts))
        self.assertNotIn("remember", self._tools(acts))  # not the first meeting

    def test_stale_person_not_greeted(self):
        w = _world(person={"name": "Sam", "stale": True})
        acts, _ = expressions.choose_context_acts(w, 1000.0, {}, {}, self.rng)
        self.assertEqual(acts, [])

    def test_confident_novel_object_gets_tilt_remark_and_note(self):
        w = _world(objects={"items": [{"label": "guitar", "confidence": 0.9,
                                       "center_offset": -120}],
                            "close_object": False, "stale": False})
        acts, updates = expressions.choose_context_acts(w, 1000.0, {}, {}, self.rng)
        self.assertEqual(set(self._tools(acts)), {"curious_tilt", "speak", "remember"})
        self.assertEqual(updates["reacted_object"], "guitar")
        remark = next(a for a in acts if a["tool"] == "speak")
        self.assertIn("guitar", remark["text"])
        note = next(a for a in acts if a["tool"] == "remember")
        self.assertEqual(note["subject"], "guitar")

    def test_ambiguous_object_left_to_curiosity(self):
        # alt_label present -> curiosity.py's job to ask; expressions stays out.
        w = _world(objects={"items": [{"label": "chair", "alt_label": "speaker",
                                       "confidence": 0.9}],
                            "close_object": False, "stale": False})
        acts, _ = expressions.choose_context_acts(w, 1000.0, {}, {}, self.rng)
        self.assertEqual(acts, [])

    def test_low_confidence_object_skipped(self):
        w = _world(objects={"items": [{"label": "chair", "confidence": 0.6}],
                            "close_object": False, "stale": False})
        acts, _ = expressions.choose_context_acts(w, 1000.0, {}, {}, self.rng)
        self.assertEqual(acts, [])

    def test_recently_reacted_object_skipped(self):
        w = _world(objects={"items": [{"label": "guitar", "confidence": 0.9}],
                            "close_object": False, "stale": False})
        acts, _ = expressions.choose_context_acts(
            w, 1000.0, {"guitar": 999.0}, {}, self.rng)
        self.assertEqual(acts, [])

    def test_stale_objects_are_ignored(self):
        w = _world(objects={"items": [{"label": "guitar", "confidence": 0.9}],
                            "close_object": True, "stale": True})
        acts, _ = expressions.choose_context_acts(w, 1000.0, {}, {}, self.rng)
        self.assertEqual(acts, [])

    def test_close_object_gets_a_startled_remark(self):
        w = _world(objects={"items": [], "close_object": True, "stale": False})
        acts, updates = expressions.choose_context_acts(w, 1000.0, {}, {}, self.rng)
        self.assertEqual(set(self._tools(acts)), {"curious_tilt", "speak"})
        self.assertTrue(updates.get("close_reacted"))

    def test_person_outranks_object(self):
        w = _world(person={"name": "Sam", "stale": False},
                   objects={"items": [{"label": "guitar", "confidence": 0.9}],
                            "close_object": True, "stale": False})
        acts, updates = expressions.choose_context_acts(w, 1000.0, {}, {}, self.rng)
        self.assertEqual(updates.get("greeted"), "Sam")
        self.assertNotIn("reacted_object", updates)


class PickIdleActsTest(unittest.TestCase):
    def test_repertoire_is_covered(self):
        seen = set()
        for seed in range(50):
            acts = expressions.pick_idle_acts(random.Random(seed))
            self.assertTrue(acts)
            seen.add(tuple(a["tool"] for a in acts))
        # All three ambient shapes appear across seeds.
        self.assertIn(("speak",), seen)
        self.assertIn(("look_around",), seen)
        self.assertIn(("curious_tilt", "speak"), seen)


class SubjectOffsetTest(unittest.TestCase):
    def test_person_located_by_fresh_face(self):
        w = _world(face={"detected": True, "frame_center_offset": 90,
                         "frame_width": 640, "stale": False})
        self.assertEqual(expressions._subject_offset(w, ("person", "Sam")), (90, 640))

    def test_person_falls_back_to_person_object(self):
        # No usable face, but a "person"-labelled object box is present.
        w = _world(face={"detected": False, "stale": True},
                   objects={"items": [{"label": "person", "center_offset": -50,
                                       "frame_width": 640}],
                            "close_object": False, "stale": False})
        self.assertEqual(expressions._subject_offset(w, ("person", "Sam")), (-50, 640))

    def test_object_located_by_label(self):
        w = _world(objects={"items": [{"label": "guitar", "center_offset": 120,
                                       "frame_width": 640}],
                            "close_object": False, "stale": False})
        self.assertEqual(expressions._subject_offset(w, ("object", "guitar")), (120, 640))

    def test_lost_when_stale_or_absent(self):
        gone = _world(objects={"items": [], "close_object": False, "stale": False})
        self.assertIsNone(expressions._subject_offset(gone, ("object", "guitar")))
        stale = _world(objects={"items": [{"label": "guitar", "center_offset": 10,
                                           "frame_width": 640}],
                                "close_object": False, "stale": True})
        self.assertIsNone(expressions._subject_offset(stale, ("object", "guitar")))
        self.assertIsNone(expressions._subject_offset(None, ("person", "Sam")))


class AimPanTest(unittest.TestCase):
    def test_pans_toward_the_subject(self):
        # Subject well to the right -> pan increases (turns the head right).
        self.assertGreater(expressions._aim_pan(0, 300, 640), 0)
        # Subject to the left -> pan decreases.
        self.assertLess(expressions._aim_pan(0, -300, 640), 0)

    def test_deadband_holds_a_centred_subject(self):
        self.assertEqual(expressions._aim_pan(25, 5, 640), 25)   # ~1.5% off centre

    def test_step_is_capped(self):
        # A subject at the frame edge still moves the head at most one capped step.
        self.assertEqual(expressions._aim_pan(0, 320, 640), expressions.GAZE_MAX_STEP_DEG)

    def test_pan_is_clamped_to_soft_limit(self):
        self.assertLessEqual(expressions._aim_pan(60, 320, 640), expressions.GAZE_PAN_LIMIT)

    def test_missing_frame_width_holds(self):
        self.assertEqual(expressions._aim_pan(30, 300, None), 30)


class DispatchTest(unittest.TestCase):
    def setUp(self):
        self.e = expressions.Expressions()   # FakeBus via harness
        self.e.rng = random.Random(0)
        # Make head gestures synchronous and instant for deterministic asserts.
        self.e._spawn = lambda fn: fn()
        self.e._sleep = lambda *_a, **_k: None

    def test_speak_and_remember_publish(self):
        self.e._dispatch([
            {"tool": "speak", "text": "hi"},
            {"tool": "remember", "subject": "Sam", "fact": "I met Sam", "confidence": 0.6},
        ], 1000.0, {})
        speak = self.e.bus.last(expressions.SPEAK_TOPIC)
        self.assertEqual(speak["text"], "hi")
        self.assertNotIn("kind", speak)                  # plain, untagged speech
        note = self.e.bus.last(expressions.NOTE_TOPIC)
        self.assertEqual(note["subject"], "Sam")
        self.assertEqual(note["source"], expressions.SOURCE_NAME)

    def test_look_around_sweeps_and_recentres(self):
        self.e._dispatch([{"tool": "look_around"}], 1000.0, {})
        looks = self.e.bus.of(expressions.LOOK_TOPIC)
        pans = [m["action"]["pan"] for m in looks]
        self.assertEqual(pans, list(expressions.LOOK_SWEEP_PANS))
        self.assertEqual(pans[-1], 0)                    # ends centred
        self.assertEqual(looks[0]["source"], expressions.SOURCE_NAME)

    def test_curious_tilt_cocks_then_recentres(self):
        self.e._dispatch([{"tool": "curious_tilt", "pan_dir": 1}], 1000.0, {})
        looks = self.e.bus.of(expressions.LOOK_TOPIC)
        self.assertEqual(looks[0]["action"],
                         {"direction": "look", "pan": expressions.CURIOUS_PAN,
                          "tilt": expressions.CURIOUS_TILT})
        self.assertEqual(looks[-1]["action"]["pan"], 0)  # recentred
        self.assertEqual(looks[-1]["action"]["tilt"], 0)

    def test_curious_tilt_holds_and_follows_identified_subject(self):
        # A tracked object in view: cock, then follow it, then release to centre.
        self.e.latest_world = _world(
            objects={"items": [{"label": "guitar", "center_offset": 260,
                                "frame_width": 640}],
                     "close_object": False, "stale": False})
        self.e._dispatch(
            [{"tool": "curious_tilt", "pan_dir": 1, "track": ("object", "guitar")}],
            1000.0, {})
        actions = [m["action"] for m in self.e.bus.of(expressions.LOOK_TOPIC)]
        self.assertEqual(actions[0], {"direction": "look",
                                      "pan": expressions.CURIOUS_PAN,
                                      "tilt": expressions.CURIOUS_TILT})   # the cock
        # It tracks the subject: at least one aimed, level look toward it (right).
        self.assertTrue(any(a["tilt"] == expressions.GAZE_TILT and a["pan"] > 0
                            for a in actions[1:-1]))
        self.assertEqual(actions[-1], {"direction": "look", "pan": 0, "tilt": 0})
        self.assertGreater(len(actions), 2)                                # not just cock+recentre

    def test_gaze_releases_when_subject_is_lost(self):
        # Track requested but nothing matching is in view -> cock then recentre.
        self.e.latest_world = _world()
        self.e._dispatch(
            [{"tool": "curious_tilt", "pan_dir": 1, "track": ("object", "guitar")}],
            1000.0, {})
        actions = [m["action"] for m in self.e.bus.of(expressions.LOOK_TOPIC)]
        self.assertEqual(actions[0]["tilt"], expressions.CURIOUS_TILT)      # cock
        self.assertEqual(actions[-1], {"direction": "look", "pan": 0, "tilt": 0})
        self.assertEqual(len(actions), 2)                                  # no tracking looks

    def test_gaze_yields_to_a_human_taking_over(self):
        self.e.rc_active = True
        self.e.latest_world = _world(
            objects={"items": [{"label": "guitar", "center_offset": 260,
                                "frame_width": 640}],
                     "close_object": False, "stale": False})
        self.e._hold_gaze(expressions.CURIOUS_PAN, ("object", "guitar"))
        self.assertEqual(self.e.bus.of(expressions.LOOK_TOPIC), [])         # never grabbed the head

    def test_head_gesture_defers_to_another_module(self):
        # Someone else moved the head moments ago: skip head acts, still speak.
        self.e.last_foreign_look_at = 1000.0 - 1
        self.e._dispatch([{"tool": "curious_tilt", "pan_dir": 0},
                          {"tool": "speak", "text": "hi"}], 1000.0, {})
        self.assertEqual(self.e.bus.of(expressions.LOOK_TOPIC), [])
        self.assertIsNotNone(self.e.bus.last(expressions.SPEAK_TOPIC))

    def test_dispatch_records_reaction_state(self):
        self.e._dispatch([{"tool": "speak", "text": "hi"}], 1000.0,
                         {"greeted": "Sam", "reacted_object": "guitar"})
        self.assertEqual(self.e.greeted_people["Sam"], 1000.0)
        self.assertEqual(self.e.reacted_objects["guitar"], 1000.0)
        self.assertEqual(self.e.last_expression_at, 1000.0)


class OnWorldGatingTest(unittest.TestCase):
    def setUp(self):
        self.e = expressions.Expressions()
        self.e.rng = random.Random(0)
        self.e._spawn = lambda fn: fn()
        self.e._sleep = lambda *_a, **_k: None

    def test_context_reaction_fires_and_then_respects_cooldown(self):
        w = _world(person={"name": "Sam", "stale": False})
        self.e.on_world(w)
        self.assertEqual(len(self.e.bus.of(expressions.SPEAK_TOPIC)), 1)
        # A second snapshot inside the cooldown produces no new expression.
        self.e.on_world(w)
        self.assertEqual(len(self.e.bus.of(expressions.SPEAK_TOPIC)), 1)

    def test_busy_world_suppresses_reaction(self):
        w = _world(person={"name": "Sam", "stale": False}, battery={"low": True})
        self.e.on_world(w)
        self.assertEqual(self.e.bus.of(expressions.SPEAK_TOPIC), [])

    def test_rc_mode_suppresses_reaction(self):
        self.e.on_rc_mode({"active": True})
        self.e.on_world(_world(person={"name": "Sam", "stale": False}))
        self.assertEqual(self.e.bus.of(expressions.SPEAK_TOPIC), [])


class OnLookForeignTrackingTest(unittest.TestCase):
    def setUp(self):
        self.e = expressions.Expressions()

    def test_own_look_is_ignored(self):
        self.e.on_look({"source": expressions.SOURCE_NAME})
        self.assertEqual(self.e.last_foreign_look_at, 0.0)

    def test_foreign_look_is_recorded(self):
        self.e.on_look({"source": "field_agent"})
        self.assertGreater(self.e.last_foreign_look_at, 0.0)


class ReflectionNoteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.r = reflection.Reflection.__new__(reflection.Reflection)
        self.r.store = SemanticStore(readonly=False,
                                     db_path=os.path.join(self.tmp, "semantic.db"))

    def test_note_is_written_as_a_fact(self):
        self.r.on_note({"subject": "guitar", "fact": "I have seen a guitar",
                        "confidence": 0.55, "source": "expressions"})
        facts = self.r.store.facts_for("guitar", limit=1)
        self.assertTrue(facts)
        self.assertIn("guitar", facts[0]["fact"])

    def test_confidence_is_clamped(self):
        self.r.on_note({"subject": "x", "fact": "overconfident note", "confidence": 5.0})
        self.assertLessEqual(self.r.store.facts_for("x", limit=1)[0]["confidence"], 0.7)

    def test_empty_note_is_ignored(self):
        self.r.on_note({"subject": "  ", "fact": "nothing"})
        self.r.on_note({"subject": "x", "fact": ""})
        self.assertEqual(self.r.store.fact_count(), 0)

    def test_repeated_note_reinforces_not_duplicates(self):
        payload = {"subject": "guitar", "fact": "I have seen a guitar"}
        self.r.on_note(payload)
        self.r.on_note(payload)
        self.assertEqual(len(self.r.store.facts_for("guitar", limit=5)), 1)


class AffectStateTest(unittest.TestCase):
    """The pure mood model: nudges raise a feeling, time decays it, the
    strongest past threshold is the one to express."""

    def setUp(self):
        self.a = expressions.AffectState(half_life=100.0)

    def test_nudges_accumulate_and_cross_threshold(self):
        t = 1000.0
        self.assertEqual(self.a.dominant(t), (None, 0.0))   # calm at rest
        for _ in range(3):
            self.a.nudge("frustration", 0.22, t)            # a burst of vetoes
        mood, level = self.a.dominant(t)
        self.assertEqual(mood, "frustration")
        self.assertAlmostEqual(level, 0.66, places=3)

    def test_level_is_clamped_to_one(self):
        t = 1000.0
        for _ in range(10):
            self.a.nudge("satisfaction", 0.5, t)
        self.assertLessEqual(self.a.levels["satisfaction"], 1.0)

    def test_decays_over_time(self):
        t = 1000.0
        self.a.nudge("curiosity", 0.8, t)
        self.a.dominant(t + 100.0)                          # one half-life later
        self.assertAlmostEqual(self.a.levels["curiosity"], 0.4, places=3)

    def test_decayed_below_threshold_is_not_dominant(self):
        t = 1000.0
        self.a.nudge("curiosity", 0.5, t)
        self.assertEqual(self.a.dominant(t + 200.0)[0], None)   # faded away

    def test_strongest_mood_wins(self):
        t = 1000.0
        self.a.nudge("curiosity", 0.5, t)
        self.a.nudge("satisfaction", 0.9, t)
        self.assertEqual(self.a.dominant(t)[0], "satisfaction")

    def test_consume_damps_the_expressed_mood(self):
        t = 1000.0
        self.a.nudge("satisfaction", 0.8, t)
        self.a.consume("satisfaction", t)
        self.assertEqual(self.a.dominant(t)[0], None)       # no longer expressible


class PickEmoteActsTest(unittest.TestCase):
    def test_always_has_a_gesture(self):
        for mood in ("curiosity", "frustration", "satisfaction"):
            acts = expressions.pick_emote_acts(mood, random.Random(0))
            self.assertEqual(acts[0], {"tool": "emote_gesture", "mood": mood})

    def test_tag_is_from_the_moods_repertoire_when_present(self):
        # rng seed that yields a tag; the spoken tag must match the mood.
        acts = expressions.pick_emote_acts("frustration", random.Random(2))
        speaks = [a for a in acts if a["tool"] == "speak"]
        for s in speaks:
            self.assertIn(s["text"], expressions.EMOTE_TAGS["frustration"])


class AffectSignalTest(unittest.TestCase):
    """Internal-state bus events nudge the right moods."""

    def setUp(self):
        self.e = expressions.Expressions()

    def _mood(self, name):
        return self.e.affect.levels[name]

    def test_veto_builds_frustration(self):
        for _ in range(3):
            self.e.on_action_result({"result": {"status": "vetoed"}})
        self.assertGreater(self._mood("frustration"), 0.4)

    def test_non_veto_result_is_ignored(self):
        self.e.on_action_result({"result": {"status": "ok"}})
        self.assertEqual(self._mood("frustration"), 0.0)

    def test_cleared_path_is_satisfying(self):
        self.e.on_hypothesis({"resolution": "phantom_reading"})
        self.assertGreater(self._mood("satisfaction"), 0.0)

    def test_still_blocked_is_frustrating(self):
        self.e.on_hypothesis({"resolution": "still_blocked"})
        self.assertGreater(self._mood("frustration"), 0.0)

    def test_goal_reached_is_very_satisfying(self):
        self.e.on_goal_progress({"status": "reached"})
        self.assertGreaterEqual(self._mood("satisfaction"),
                                expressions.AFFECT_EXPRESS_THRESHOLD)

    def test_new_place_sparks_curiosity(self):
        self.e.on_location_change({"is_new": True})
        self.assertGreater(self._mood("curiosity"), 0.0)

    def test_known_place_does_not(self):
        self.e.on_location_change({"is_new": False})
        self.assertEqual(self._mood("curiosity"), 0.0)

    def test_learning_a_label_satisfies_and_stays_curious(self):
        self.e.on_label({"correct_label": "mug"})
        self.assertGreater(self._mood("satisfaction"), 0.0)
        self.assertGreater(self._mood("curiosity"), 0.0)

    def test_training_published_satisfies(self):
        self.e.on_self_trainer_status({"state": "published"})
        self.assertGreater(self._mood("satisfaction"), 0.0)
        self.e.affect.levels["satisfaction"] = 0.0
        self.e.on_self_trainer_status({"state": "training"})   # non-published ignored
        self.assertEqual(self._mood("satisfaction"), 0.0)


class MaybeEmoteTest(unittest.TestCase):
    """The gated dispatch: a strong mood becomes a legible gesture, but only
    when deference allows, and it's consumed so it doesn't loop."""

    def setUp(self):
        self.e = expressions.Expressions()
        self.e.rng = random.Random(0)
        self.e._spawn = lambda fn: fn()
        self.e._sleep = lambda *_a, **_k: None
        self.e.latest_world = _world()
        self.e.last_expression_at = 0.0        # past the cooldown
        self.e.last_foreign_look_at = 0.0      # head free

    def _frustrate(self):
        for _ in range(3):
            self.e.on_action_result({"result": {"status": "vetoed"}})

    def test_strong_mood_emits_a_gesture_and_consumes_it(self):
        self._frustrate()
        self.e._maybe_emote()
        looks = self.e.bus.of(expressions.LOOK_TOPIC)
        self.assertTrue(looks)
        self.assertEqual(looks[-1]["action"]["pan"], 0)     # ends centred
        # Consumed: an immediate second attempt does nothing new.
        self.e.bus.clear()
        self.e.last_expression_at = 0.0
        self.e._maybe_emote()
        self.assertEqual(self.e.bus.of(expressions.LOOK_TOPIC), [])

    def test_defers_while_busy(self):
        self._frustrate()
        self.e.latest_world = _world(battery={"low": True, "critical": False})
        self.e._maybe_emote()
        self.assertEqual(self.e.bus.of(expressions.LOOK_TOPIC), [])

    def test_defers_while_head_in_use(self):
        self._frustrate()
        self.e.last_foreign_look_at = time.time()   # someone else just moved the head
        self.e._maybe_emote()
        self.assertEqual(self.e.bus.of(expressions.LOOK_TOPIC), [])

    def test_calm_robot_does_not_emote(self):
        self.e._maybe_emote()
        self.assertEqual(self.e.bus.of(expressions.LOOK_TOPIC), [])


class QuietRequiredTest(unittest.TestCase):
    """_quiet_required is is_busy minus the 'we're moving' half - so a moving
    (but conversation-free) robot is is_busy yet NOT quiet-required."""

    def test_moving_alone_is_busy_but_not_quiet_required(self):
        w = _world(last_action={"action": {"direction": "forward", "speed": 20},
                                "updated_at": 1000.0})
        self.assertTrue(expressions.is_busy(w, 1000.5, False, 0.0))
        self.assertFalse(expressions._quiet_required(w, 1000.5, False, 0.0))

    def test_conversation_and_power_still_require_quiet(self):
        self.assertTrue(expressions._quiet_required(_world(), 1000.0, True, 0.0))  # rc
        self.assertTrue(expressions._quiet_required(
            _world(battery={"low": True, "critical": False}), 1000.0, False, 0.0))
        self.assertTrue(expressions._quiet_required(  # we just spoke
            _world(), 1000.0, False, 1000.0 - 1))
        w = _world(last_heard={"text": "hi", "updated_at": 1000.0 - 2, "stale": False})
        self.assertTrue(expressions._quiet_required(w, 1000.0, False, 0.0))  # human spoke


class VoiceFrustrationTest(unittest.TestCase):
    """The real-time frustration cue: audible on the speaker WHILE driving (when
    the head-gesture emote would defer), but still deferential to conversation /
    power / a human driver, and on its own cooldown."""

    def setUp(self):
        self.e = expressions.Expressions()
        self.e.rng = random.Random(0)
        # A world where the robot is actively driving (head-emote would defer).
        self.driving = _world(last_action={
            "action": {"direction": "forward", "speed": 20}, "updated_at": time.time()})
        self.e.latest_world = self.driving

    def _frustrate(self):
        for _ in range(3):        # ~0.62, past FRUSTRATION_VOICE_THRESHOLD
            self.e.on_action_result({"result": {"status": "vetoed"}})

    def _tags(self):
        return [p["text"] for p in self.e.bus.of(expressions.SPEAK_TOPIC)]

    def test_frustration_is_voiced_while_driving(self):
        self._frustrate()
        self.assertTrue(expressions.is_busy(self.driving, time.time(), False, 0.0))
        self.e._maybe_voice_frustration()
        self.assertIn(self._tags()[-1], expressions.EMOTE_TAGS["frustration"])

    def test_consumed_so_it_does_not_repeat_next_tick(self):
        self._frustrate()
        self.e._maybe_voice_frustration()
        self.e.bus.clear()
        self.e.last_frustration_voice_at = 0.0     # ignore the cooldown for this probe
        self.e._maybe_voice_frustration()          # mood was consumed below threshold
        self.assertEqual(self._tags(), [])

    def test_own_cooldown_blocks_a_rapid_second_cue(self):
        self._frustrate()
        self.e._maybe_voice_frustration()
        self.e.bus.clear()
        self._frustrate()                          # frustration spikes again immediately
        self.e._maybe_voice_frustration()          # ...but the cooldown holds it
        self.assertEqual(self._tags(), [])

    def test_defers_to_a_live_conversation(self):
        self._frustrate()
        self.e.latest_world = _world(
            last_action=self.driving["last_action"],
            last_heard={"text": "hello", "updated_at": time.time(), "stale": False})
        self.e._maybe_voice_frustration()
        self.assertEqual(self._tags(), [])

    def test_calm_robot_stays_silent(self):
        self.e._maybe_voice_frustration()
        self.assertEqual(self._tags(), [])

    def test_satisfaction_spike_is_not_voiced(self):
        for _ in range(2):
            self.e.on_goal_progress({"status": "reached"})   # strong satisfaction
        self.e._maybe_voice_frustration()
        self.assertEqual(self._tags(), [])


if __name__ == "__main__":
    unittest.main()
