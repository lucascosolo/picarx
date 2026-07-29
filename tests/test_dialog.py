"""The dialog broker (layer_b/modules/dialog.py): the single owner of the open
question and the turn-taking decision. These lock down the behaviour the old
per-module "capture the next utterance" races could not guarantee - most
importantly that a command (spoken or a web-console button on the same heard
topic) is never swallowed as an answer."""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import dialog  # noqa: E402

ANSWER = "picarx/dialog/answer"
CLEARED = "picarx/dialog/cleared"
UNHANDLED = "picarx/audio/unhandled"
UNCERTAIN = "picarx/audio/uncertain"


class DialogBrokerTest(unittest.TestCase):
    def setUp(self):
        self.d = dialog.DialogBroker()   # __init__ only builds a FakeBus + lock

    def _ask(self, asker="curiosity", qid="q1", kind="label",
             options=("chair", "speaker"), ttl=12.0):
        self.d.on_ask({"asker": asker, "question_id": qid, "kind": kind,
                       "options": list(options), "ttl": ttl})

    def _heard(self, text, **extra):
        self.d.on_heard({"text": text, **extra})

    # ---- answer routing ----

    def test_answer_routes_to_asker_and_clears(self):
        self._ask()
        self._heard("it's a speaker", confidence=0.9)
        ans = self.d.bus.last(ANSWER)
        self.assertIsNotNone(ans)
        self.assertEqual(ans["asker"], "curiosity")
        self.assertEqual(ans["question_id"], "q1")
        self.assertEqual(ans["text"], "it's a speaker")
        self.assertEqual(ans["confidence"], 0.9)
        self.assertIsNone(self.d.question)   # one-shot: cleared after answering

    def test_no_open_question_means_no_routing(self):
        self._heard("it's a speaker")
        self.assertIsNone(self.d.bus.last(ANSWER))

    def test_meeting_recording_owns_heard_speech(self):
        self._ask()
        self.d.on_notes_state({"active_meeting": {"state": "recording"}})
        self._heard("it's a speaker")
        self.d.on_directed({"text": "what is the agenda"})
        self.assertIsNone(self.d.bus.last(ANSWER))
        self.assertIsNone(self.d.bus.last(UNHANDLED))
        self.assertIsNone(self.d.bus.last(UNCERTAIN))

    # ---- the regression that motivated looks_like_label_answer ----

    def test_command_is_not_swallowed_as_a_label(self):
        self._ask()   # curiosity label question open
        self._heard("who am i")                      # a command/question, not a label
        self.assertIsNone(self.d.bus.last(ANSWER))   # not routed as the answer
        self.assertIsNotNone(self.d.question)        # question stays open
        self._heard("it's a speaker")                # a real answer still lands
        self.assertEqual(self.d.bus.last(ANSWER)["text"], "it's a speaker")
        self.assertIsNone(self.d.question)

    def test_wake_addressed_command_is_not_an_answer(self):
        self._ask()
        self._heard("robot stop")                    # a fresh command to the robot
        self.assertIsNone(self.d.bus.last(ANSWER))
        self.assertIsNotNone(self.d.question)        # still waiting for a real answer

    def test_repair_and_correction_echoes_ignored(self):
        self._ask()
        self._heard("speaker", source="intent_repair")
        self._heard("speaker", source="user_correction")
        self.assertIsNone(self.d.bus.last(ANSWER))
        self.assertIsNotNone(self.d.question)

    # ---- correction-kind questions accept command words ----

    def test_correction_question_accepts_a_command_word(self):
        self._ask(asker="companion", qid="c1", kind="correction", options=())
        self._heard("battery")
        ans = self.d.bus.last(ANSWER)
        self.assertEqual(ans["asker"], "companion")
        self.assertEqual(ans["text"], "battery")

    def test_correction_question_ignores_a_feedback_verdict(self):
        self._ask(asker="companion", qid="c1", kind="correction", options=())
        self._heard("that's wrong")
        self.assertIsNone(self.d.bus.last(ANSWER))
        self.assertIsNotNone(self.d.question)

    # ---- one question at a time ----

    def test_new_question_replaces_and_clears_old(self):
        self._ask(asker="curiosity", qid="q1")
        self._ask(asker="companion", qid="c1", kind="correction", options=())
        cleared = self.d.bus.last(CLEARED)
        self.assertEqual(cleared["question_id"], "q1")
        self.assertEqual(cleared["asker"], "curiosity")
        self.assertEqual(cleared["reason"], "replaced")
        self.assertEqual(self.d.question.id, "c1")

    # ---- expiry ----

    def test_expiry_via_sweeper_emits_cleared(self):
        self._ask(ttl=12.0)
        self.d.question.deadline = time.time() - 1     # force it past due
        self.d._sweep_once(time.time())
        cleared = self.d.bus.last(CLEARED)
        self.assertEqual(cleared["question_id"], "q1")
        self.assertEqual(cleared["reason"], "expired")
        self.assertIsNone(self.d.question)

    def test_expired_question_does_not_capture_an_answer(self):
        self._ask()
        self.d.question.deadline = time.time() - 1
        self._heard("it's a speaker")
        self.assertIsNone(self.d.bus.last(ANSWER))     # too late - not routed
        self.assertEqual(self.d.bus.last(CLEARED)["reason"], "expired")

    def test_no_ttl_means_never_expires(self):
        self._ask(ttl=None)
        self.d._sweep_once(time.time() + 10_000)
        self.assertIsNone(self.d.bus.last(CLEARED))
        self.assertIsNotNone(self.d.question)


class DirectedRoutingTest(unittest.TestCase):
    """on_directed: the addressing half of turn-taking, moved here from
    field_agent. field_agent forwards its command-misses; the broker decides
    chat vs the intent arbiter vs drop, and owns the conversation window."""

    def setUp(self):
        self.d = dialog.DialogBroker()   # fresh: conversation window closed

    def _directed(self, **payload):
        self.d.on_directed(payload)

    def test_wake_addressed_miss_goes_to_chat_stripped(self):
        self._directed(text="robot what do you think", confidence=0.8)
        msg = self.d.bus.last(UNHANDLED)
        self.assertEqual(msg["text"], "what do you think")   # wake phrase stripped
        self.assertEqual(msg["confidence"], 0.8)

    def test_command_shaped_miss_goes_to_the_intent_arbiter(self):
        self._directed(text="take me to the kitchen")
        msg = self.d.bus.last(UNCERTAIN)
        self.assertEqual(msg["text"], "take me to the kitchen")
        self.assertEqual(msg["from"], "field_agent")
        self.assertIsNone(self.d.bus.last(UNHANDLED))

    def test_repaired_command_shaped_miss_is_dropped(self):
        # Loop guard: a repair that still matched nothing must not re-escalate.
        self._directed(text="take me to the kitchen", from_repair=True)
        self.assertIsNone(self.d.bus.last(UNCERTAIN))

    def test_plain_chatter_is_dropped_when_the_window_is_closed(self):
        self._directed(text="the weather is nice today")
        self.assertIsNone(self.d.bus.last(UNHANDLED))
        self.assertIsNone(self.d.bus.last(UNCERTAIN))

    def test_handled_command_opens_the_conversation_window(self):
        # A matched command (handled ping) keeps a wake-less follow-up directed:
        # plain chatter that would be dropped is now forwarded as conversation.
        self._directed(handled=True)
        self._directed(text="the weather is nice today")
        self.assertEqual(self.d.bus.last(UNHANDLED)["text"], "the weather is nice today")
        self.assertIsNone(self.d.bus.last(UNCERTAIN))   # chat, not a command

    def test_wake_opens_the_window_for_a_following_plain_utterance(self):
        self._directed(text="robot hello there")               # wake -> opens window
        self.d.bus.clear()
        self._directed(text="the weather is nice today")       # now in-conversation
        self.assertEqual(self.d.bus.last(UNHANDLED)["text"], "the weather is nice today")

    def test_command_shaped_miss_does_not_open_the_window(self):
        # A bare command-shaped miss escalates but must NOT hold the window open
        # (matching the old field_agent rule), so a later plain utterance drops.
        self._directed(text="take me to the kitchen")
        self.d.bus.clear()
        self._directed(text="the weather is nice today")
        self.assertIsNone(self.d.bus.last(UNHANDLED))

    def test_talk_aimed_at_the_robot_reaches_chat(self):
        # No wake word, no open window, no command vocabulary - but plainly
        # spoken TO the robot. This used to be dropped in silence.
        self._directed(text="how are you feeling today", confidence=0.8)
        self.assertEqual(self.d.bus.last(UNHANDLED)["text"],
                         "how are you feeling today")
        self.assertIsNone(self.d.bus.last(UNCERTAIN))

    def test_chat_shaped_talk_does_not_open_the_window(self):
        # Overhearing one question must not hand the next 45 seconds of room
        # noise to the chat path.
        self._directed(text="how are you feeling today")
        self.d.bus.clear()
        self._directed(text="the weather is nice today")
        self.assertIsNone(self.d.bus.last(UNHANDLED))

    def test_talk_about_a_capability_goes_to_chat_not_the_arbiter(self):
        # Command-SHAPED only because it names a capability's vocabulary. No
        # capability parses it and it's addressed to the robot in the second
        # person, so it's conversation - and must not be spent from the
        # command-repair budget.
        for text in ("do you like music", "what do you think of the radio",
                     "are you any good at dice"):
            self.d.bus.clear()
            self._directed(text=text)
            self.assertEqual(self.d.bus.last(UNHANDLED)["text"], text, text)
            self.assertIsNone(self.d.bus.last(UNCERTAIN), text)

    def test_real_commands_still_reach_the_command_path(self):
        # Instruction shape wins over second person ("can you play some music"),
        # and so does a capability that genuinely parses the utterance.
        for text in ("can you play some music", "turn off the radio",
                     "do you know what time it is"):
            self.d.bus.clear()
            self._directed(text=text)
            self.assertIsNotNone(self.d.bus.last(UNCERTAIN), text)
            self.assertIsNone(self.d.bus.last(UNHANDLED), text)

    def test_room_questions_still_escalate(self):
        # A question opener alone is not second-person address: "is the radio
        # on" is a status question the arbiter can repair into a real command.
        self._directed(text="is the radio on")
        self.assertIsNotNone(self.d.bus.last(UNCERTAIN))

    def test_capability_talk_does_not_open_the_window(self):
        self._directed(text="do you like music")
        self.d.bus.clear()
        self._directed(text="the weather is nice today")
        self.assertIsNone(self.d.bus.last(UNHANDLED))

    def test_empty_or_missing_text_is_a_noop(self):
        self._directed(text="   ")
        self._directed(confidence=0.5)
        self.assertEqual(self.d.bus.of(UNHANDLED), [])
        self.assertEqual(self.d.bus.of(UNCERTAIN), [])

    # ---- dedup: an answer is not ALSO re-forwarded as chat ----

    def _open_label_question(self):
        self.d.on_ask({"asker": "curiosity", "question_id": "q1", "kind": "label",
                       "options": ["chair", "speaker"], "ttl": 12.0})
        self._directed(handled=True)      # open the window so a follow-up would forward

    def test_a_routed_answer_is_not_re_forwarded_to_chat(self):
        self._open_label_question()
        self.d.on_heard({"text": "it's a speaker"})       # routed to curiosity
        self.assertIsNotNone(self.d.bus.last(ANSWER))
        self._directed(text="it's a speaker")             # field_agent's echo of the same
        self.assertIsNone(self.d.bus.last(UNHANDLED))     # ...suppressed
        self.assertIsNone(self.d.bus.last(UNCERTAIN))

    def test_dedup_is_case_insensitive(self):
        self._open_label_question()
        self.d.on_heard({"text": "It's A Speaker"})        # on_heard preserves case
        self._directed(text="it's a speaker")              # field_agent lowercases
        self.assertIsNone(self.d.bus.last(UNHANDLED))

    def test_a_different_utterance_after_an_answer_still_forwards(self):
        self._open_label_question()
        self.d.on_heard({"text": "it's a speaker"})
        self._directed(text="the weather is nice today")   # a real follow-up, not the answer
        self.assertEqual(self.d.bus.last(UNHANDLED)["text"], "the weather is nice today")

    def test_dedup_expires_so_a_later_repeat_is_not_swallowed(self):
        self._open_label_question()
        self.d.on_heard({"text": "it's a speaker"})
        self.d._answered = ("it's a speaker",
                            time.time() - dialog.ANSWER_SUPPRESS_SEC - 1)  # age it out
        self._directed(text="it's a speaker")
        self.assertEqual(self.d.bus.last(UNHANDLED)["text"], "it's a speaker")


class OpenConversationTest(unittest.TestCase):
    """Voice mode: the channel opens when someone starts a conversation, stays
    open while it is actually being had, and ends when it is over."""

    def setUp(self):
        self.d = dialog.DialogBroker()

    def _directed(self, **payload):
        self.d.on_directed(payload)

    def _age(self, seconds):
        """Pretend SECONDS passed since the last turn (idle clock only)."""
        self.d.conversation.last_turn_at -= seconds

    # ---- staying open ----

    def test_plain_conversation_keeps_the_channel_open(self):
        # The point of the change: a back-and-forth sustains itself. Before,
        # only a wake phrase or a matched command reset the window, so an
        # ordinary conversation timed out mid-sentence.
        self._directed(text="robot hello there")               # opens
        for _ in range(3):
            self._age(dialog.CONVERSATION_WINDOW_SEC - 5)      # nearly idle out
            self.d.bus.clear()
            self._directed(text="that is interesting tell me more")
            self.assertIsNotNone(self.d.bus.last(UNHANDLED))   # still conversation
        self.assertTrue(self.d.conversation.is_open(time.time()))

    def test_the_robot_speaking_holds_the_channel_open(self):
        # A reply that took a while to think about must not be followed by the
        # channel having quietly closed under it.
        self._directed(text="robot hello there")
        self._age(dialog.CONVERSATION_WINDOW_SEC - 1)
        self.d.on_speak({"text": "Sorry, I was thinking about that."})
        self._age(dialog.CONVERSATION_WINDOW_SEC - 5)
        self.d.bus.clear()
        self._directed(text="the weather is nice today")
        self.assertIsNotNone(self.d.bus.last(UNHANDLED))

    def test_speaking_alone_never_opens_a_channel(self):
        # An announcement into an empty room must not start the robot listening.
        self.d.on_speak({"text": "My battery is getting low."})
        self.assertFalse(self.d.conversation.is_open(time.time()))
        self._directed(text="the weather is nice today")
        self.assertIsNone(self.d.bus.last(UNHANDLED))

    # ---- ending ----

    def test_silence_closes_the_channel(self):
        self._directed(text="robot hello there")
        self.d.bus.clear()
        self._age(dialog.CONVERSATION_WINDOW_SEC + 1)
        self.d._sweep_once(time.time())
        state = self.d.bus.last(dialog.CONVERSATION_TOPIC)
        self.assertFalse(state["open"])
        self.assertEqual(state["reason"], "idle")
        # ...and it closes silently: no goodbye muttered into an empty room.
        self.assertEqual(self.d.bus.of("picarx/audio/speak"), [])

    def test_a_long_conversation_ends_even_while_still_being_talked_at(self):
        # The backstop against a television holding the channel open forever:
        # max_sec is never reset by a turn.
        self._directed(text="robot hello there")
        self.d.conversation.opened_at -= dialog.CONVERSATION_MAX_SEC + 1
        self.d._sweep_once(time.time())
        self.assertFalse(self.d.conversation.is_open(time.time()))
        self.assertEqual(self.d.bus.last(dialog.CONVERSATION_TOPIC)["reason"],
                         "max_duration")

    def test_stop_listening_hangs_up_and_says_so(self):
        self._directed(text="robot hello there")
        self.d.on_heard({"text": "okay stop listening"})
        self.assertFalse(self.d.conversation.is_open(time.time()))
        self.assertEqual(self.d.bus.last("picarx/audio/speak")["text"],
                         dialog.SLEEP_ACK)
        self.assertEqual(self.d.bus.last(dialog.CONVERSATION_TOPIC)["reason"],
                         "asked")

    def test_the_handled_echo_of_stop_listening_does_not_reopen_it(self):
        # field_agent acts on "stop listening" as a motion stop and reports it
        # handled a beat later - which must not undo the hang-up.
        self._directed(text="robot hello there")
        self.d.on_heard({"text": "stop listening"})
        self._directed(handled=True)
        self.assertFalse(self.d.conversation.is_open(time.time()))

    def test_stop_listening_when_nobody_was_talking_stays_silent(self):
        self.d.on_heard({"text": "goodbye"})
        self.assertEqual(self.d.bus.of("picarx/audio/speak"), [])
        self.assertIsNone(self.d.bus.last(dialog.CONVERSATION_TOPIC))

    def test_a_sleep_word_inside_a_longer_word_does_not_hang_up(self):
        self._directed(text="robot hello there")
        self.d.on_heard({"text": "i am feeling sleepy"})
        self.assertTrue(self.d.conversation.is_open(time.time()))

    # ---- the state topic ----

    def test_opening_is_announced_once_not_per_turn(self):
        self._directed(text="robot hello there")
        self._directed(text="and what do you think about that")
        self._directed(text="tell me more about it")
        states = self.d.bus.of(dialog.CONVERSATION_TOPIC)
        self.assertEqual(len(states), 1)
        self.assertTrue(states[0]["open"])
        self.assertEqual(states[0]["reason"], "wake")

    def test_the_closing_edge_carries_the_shape_of_the_conversation(self):
        # The close is the only place that knows how long it lasted and how
        # many turns it took; downstream (event_logger -> reflection) that is
        # what makes it an episode instead of loose utterances.
        self._directed(text="robot hello there")
        self._directed(text="and what do you think about that")
        self.d.conversation.opened_at -= 30.0
        self.d.on_heard({"text": "goodbye"})
        state = self.d.bus.last(dialog.CONVERSATION_TOPIC)
        self.assertFalse(state["open"])
        self.assertEqual(state["turns"], 2)
        self.assertGreaterEqual(state["duration"], 30.0)
        self.assertEqual(state["since"], self.d.bus.of(
            dialog.CONVERSATION_TOPIC)[0]["since"] - 30.0)


if __name__ == "__main__":
    unittest.main()
