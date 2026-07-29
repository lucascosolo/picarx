#!/usr/bin/env python3
# layer_b/attention.py
"""
Attention & turn-taking model (Layer B) - the ONE place that answers two
questions about a heard utterance:

  1. "Is this addressed to me?"  -> classify() / is_addressed()
  2. "Is this an answer to my open question?" -> answers_question()

Both used to be decided implicitly and in several places at once: field_agent
matched a wake phrase and kept a 45s conversation window; curiosity and
companion each grabbed "the next utterance" off the raw picarx/audio/heard
stream and guarded it after the fact (curiosity's looks_like_label_answer,
companion's parse_feedback skip). Nothing coordinated them, so a command, a
web-console button, or the other module's answer could be swallowed by whoever
happened to be waiting.

This module is the shared, PURE (no I/O, no bus, no clock) model those callers
now share. dialog.py (the central broker) uses it to route every utterance
exactly once; field_agent uses it for its forwarding decision. Deterministic
and stdlib-only, same philosophy as speech_match, so the whole thing is
unit-testable off-robot.
"""
import re
from collections import namedtuple

import speech_match

# Question kinds - what SHAPE of reply resolves an open question. The kind
# changes what counts as an answer (a label is a noun; a correction is
# whatever the user says next; a yes/no wants an affirmation), so each asker
# declares its kind when it registers the question with the dialog broker.
LABEL = "label"          # "is that a chair or a speaker?" -> a noun/option
CORRECTION = "correction"  # "what did you want me to do?" -> anything but feedback
YES_NO = "yes_no"        # a plain confirmation
FREEFORM = "freeform"    # any spoken content resolves it


class Question:
    """A single open question the robot is waiting on an answer to. Plain
    data (the dialog broker owns the live one); `deadline` is an absolute
    epoch time, 0 meaning 'never expires'."""

    __slots__ = ("id", "asker", "kind", "options", "deadline", "prompt")

    def __init__(self, id, asker, kind=FREEFORM, options=None, deadline=0.0,
                 prompt=None):
        self.id = id
        self.asker = asker
        self.kind = kind
        self.options = list(options or [])
        self.deadline = deadline
        self.prompt = prompt

    def expired(self, now):
        return bool(self.deadline) and now > self.deadline


# ---------------------------------------------------------------------
# "Is this addressed to me?"
# ---------------------------------------------------------------------
# classify() returns all three signals the routers need: whether the
# utterance is addressed at all, WHY (so callers can treat a wake-word chat,
# a mid-conversation follow-up, and a bare command-shaped phrase differently),
# and the wake-phrase-stripped remainder to actually act on.
Addressing = namedtuple("Addressing", "addressed reason remainder")

# reason values, most-explicit first:
WAKE = "wake"                  # opened with a wake phrase ("robot, ...")
CONVERSATION = "conversation"  # within the no-wake-word follow-up window
COMMAND_SHAPE = "command_shape"  # robot vocabulary / imperative shape, no wake word
CHAT_SHAPE = "chat_shape"      # spoken TO the robot, but as talk rather than an order

# ---------------------------------------------------------------------
# "Talk aimed at me" - the difference between a mute robot and a companion
# ---------------------------------------------------------------------
# The three reasons above all want an ORDER: a wake phrase, an open window, or
# robot vocabulary / an imperative. Everything else was dropped in silence, so
# outside the 45-second window the robot answered pre-programmed phrasings and
# nothing else - "do you like music", "how are you", "what do you think that
# is" all died before reaching the chat path.
#
# Second-person address is the signal that costs nothing and means everything:
# people say "you" to the thing they're talking to. A question opener is the
# weaker sibling (people also ask the room), so it takes a longer utterance to
# count. Both are only a routing hint - companion.py's quality gate is still
# what decides whether an LLM call happens, so a chatty television reaches a
# deterministic word-list screen, not the API.
_SECOND_PERSON = {"you", "your", "youre", "you're", "yours", "yourself", "u"}
_QUESTION_OPENERS = {
    "what", "whats", "what's", "where", "when", "why", "how", "who", "whos",
    "who's", "which", "is", "are", "am", "do", "does", "did", "can", "could",
    "would", "should", "will", "have", "has", "tell", "any",
}
# A lone "why" or "you" is as likely to be noise as speech; talk aimed at
# someone is a sentence.
CHAT_MIN_TOKENS = 3


def addresses_the_robot(text):
    """True when an utterance speaks to the robot in the second person ("do
    YOU like music", "what do YOU think").

    This is the stronger half of looks_conversational, exposed on its own
    because routers use it to distinguish "asking the robot something" from
    "asking the room something": a question opener is ambiguous, but "you" is
    a person addressing the thing in front of them."""
    toks = speech_match.tokens(text)
    if len(toks) < CHAT_MIN_TOKENS:
        return False
    return any(t in _SECOND_PERSON for t in toks)


def looks_conversational(text):
    """True when an utterance is aimed at the robot as SPEECH rather than as a
    command - it addresses it in the second person, or opens like a question.

    Deliberately shallow: this only decides whether the utterance gets to be
    HEARD by the chat path, never whether anything is spent answering it."""
    if addresses_the_robot(text):
        return True
    toks = speech_match.tokens(text)
    if len(toks) < CHAT_MIN_TOKENS:
        return False
    return toks[0] in _QUESTION_OPENERS


def contains_phrase(text, phrases):
    """The first phrase in PHRASES that appears in TEXT as a contiguous run of
    whole words, or None.

    Whole-word runs only, no regex and no substring test: "go to sleep" must
    not fire on "sleepy", and "goodbye" must not fire on "goodbyes". Used for
    the small set of attention CONTROLS (see Conversation) - never for
    understanding what someone meant."""
    toks = speech_match.tokens(text)
    for phrase in phrases:
        want = speech_match.tokens(phrase)
        if not want or len(want) > len(toks):
            continue
        for i in range(len(toks) - len(want) + 1):
            if toks[i:i + len(want)] == want:
                return phrase
    return None


def normalize_wake_phrases(value):
    """Coerce a wake-phrase config value to a lowercase tuple. config.json
    stores a JSON list; the env override arrives as a comma-separated string.
    Shared by dialog.py and field_agent so both read the one dialog.wake_phrases
    knob identically."""
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value or [])
    return tuple(p.strip().lower() for p in (str(x) for x in parts) if p.strip())


def strip_wake_phrase(text, wake_phrases):
    """The utterance with its leading wake phrase removed, or None if it
    doesn't start with one. Whole-word match only ("robotics class" starts
    with "robot" but was never addressed to the robot); a bare wake word on
    its own becomes "hello" so it still greets. Moved verbatim from
    field_agent so the wake-word rule lives with the rest of the model."""
    text = (text or "").lower().strip()
    for phrase in wake_phrases:
        if not text.startswith(phrase):
            continue
        rest = text[len(phrase):]
        if rest and rest[0].isalnum():
            continue
        remainder = rest.strip(" ,.:;-!?")
        return remainder if remainder else "hello"
    return None


def classify(text, canon=None, *, wake_phrases=(), in_conversation=False):
    """Decide whether TEXT is addressed to the robot and why. `canon` is the
    speech_match.canonicalize() form (pass it in so we don't recompute it);
    `in_conversation` is the caller's 'we spoke recently' window state.

    Most-explicit first: an explicit wake phrase wins, then an open
    conversation window, then a bare command-shaped utterance, and finally
    plain talk aimed at the robot (see looks_conversational). Pure - the
    caller owns the clock and the window."""
    remainder = strip_wake_phrase(text, wake_phrases)
    if remainder is not None:
        return Addressing(True, WAKE, remainder)
    if in_conversation:
        return Addressing(True, CONVERSATION, (text or "").lower().strip())
    if canon is None:
        canon = speech_match.canonicalize(text)
    if speech_match.looks_command_like(canon) or speech_match.looks_directed_command(text):
        return Addressing(True, COMMAND_SHAPE, (text or "").lower().strip())
    if looks_conversational(text):
        return Addressing(True, CHAT_SHAPE, (text or "").lower().strip())
    return Addressing(False, None, None)


def is_addressed(text, canon=None, *, wake_phrases=(), in_conversation=False):
    """(bool, reason) convenience over classify() for callers that don't need
    the stripped remainder."""
    a = classify(text, canon, wake_phrases=wake_phrases,
                 in_conversation=in_conversation)
    return a.addressed, a.reason


# ---------------------------------------------------------------------
# The open conversation ("voice mode")
# ---------------------------------------------------------------------
# The window used to be a bare timestamp comparison ("did something address
# the robot in the last 45 seconds?"), and only two events reset it: a wake
# phrase, or a command the local matcher recognized. That is a request/response
# shape wearing a conversation's clothes - the human had to keep re-addressing
# the robot, or keep saying things the phrase tables happened to know, or the
# conversation quietly ended mid-sentence.
#
# A real conversation SUSTAINS ITSELF: it opens when someone starts it, stays
# open while the two of you are actually going back and forth (either side's
# turn counts), and ends when it is over - either because someone said so, or
# because nobody said anything for a while. That is what this models, as
# explicit state instead of arithmetic on one timestamp, so the routers can ask
# "are we talking?" rather than "was there a keyword recently?".
#
# Two independent clocks, because they answer different questions:
#   idle_sec - how long a silence ends the conversation. Reset by every turn.
#   max_sec  - the longest one conversation can run before the robot needs to
#              be addressed again. NOT reset by anything: it is the backstop
#              that stops a talkative television (or the robot answering its
#              own replies) from holding the channel open indefinitely.
# Pure: the caller owns the clock and the bus, same as the rest of this module.
DEFAULT_IDLE_SEC = 45.0
DEFAULT_MAX_SEC = 600.0

# why a conversation ended
IDLE = "idle"                  # nobody said anything for idle_sec
MAX_DURATION = "max_duration"  # ran past max_sec without being re-addressed
ASKED = "asked"                # someone told the robot to stop listening


class Conversation:
    """One open conversation. Closed until something opens it."""

    __slots__ = ("idle_sec", "max_sec", "opened_at", "last_turn_at", "turns",
                 "reason")

    def __init__(self, idle_sec=DEFAULT_IDLE_SEC, max_sec=DEFAULT_MAX_SEC):
        self.idle_sec = float(idle_sec)
        self.max_sec = float(max_sec)
        self.opened_at = 0.0
        self.last_turn_at = 0.0
        self.turns = 0
        self.reason = None

    def open(self, now, reason=WAKE):
        """Start a conversation, or count a turn if one is already running.
        Returns True only when it was NEWLY opened, so a caller can announce
        the transition exactly once."""
        if self.is_open(now):
            self.touch(now)
            return False
        self.opened_at = now
        self.last_turn_at = now
        self.turns = 1
        self.reason = reason
        return True

    def touch(self, now):
        """Count one more turn of an ALREADY-open conversation, resetting the
        idle clock. Never opens one: the robot speaking, or someone chatting
        near it, extends a conversation that exists but must not start one -
        otherwise an announcement to an empty room would open the channel."""
        if not self.is_open(now):
            return False
        self.last_turn_at = now
        self.turns += 1
        return True

    def closing_reason(self, now):
        """Why this conversation is over, or None while it is still running."""
        if not self.opened_at:
            return None
        if self.idle_sec and (now - self.last_turn_at) >= self.idle_sec:
            return IDLE
        if self.max_sec and (now - self.opened_at) >= self.max_sec:
            return MAX_DURATION
        return None

    def is_open(self, now):
        return bool(self.opened_at) and self.closing_reason(now) is None

    def close(self):
        """End it. Returns True if it had been open (so the caller only
        announces a real ending)."""
        was_open = bool(self.opened_at)
        self.opened_at = 0.0
        self.last_turn_at = 0.0
        self.turns = 0
        self.reason = None
        return was_open

    def remaining(self, now):
        """Seconds until this conversation closes on its own (0 when closed)."""
        if not self.opened_at:
            return 0.0
        by_idle = (self.idle_sec - (now - self.last_turn_at)
                   if self.idle_sec else float("inf"))
        by_max = (self.max_sec - (now - self.opened_at)
                  if self.max_sec else float("inf"))
        return max(0.0, min(by_idle, by_max))


# ---------------------------------------------------------------------
# "Is this an answer to my open question?"  (turn-taking)
# ---------------------------------------------------------------------
# Words that aren't the noun in a spoken label answer ("it's a speaker", "I
# think that's a chair", "no, a mug") - stripped so what's left is the label.
_ANSWER_FILLER = {
    "it's", "its", "it", "is", "a", "an", "the", "that's", "thats", "that",
    "this", "i", "think", "maybe", "looks", "look", "like", "actually", "no",
    "not", "but", "sorry", "um", "uh", "well", "you", "mean", "meant", "of",
    "course", "yeah", "yes", "nope", "just", "some", "kind", "sort",
}

# Words that mark an utterance as a COMMAND or QUESTION addressed to the robot
# rather than a label answer. If any appears, a "what is that?" question must
# not swallow it as a label - object labels are nouns and never contain these.
# Guards the real bug: a command issued while a question is open (spoken, or a
# web-console button on the same heard topic - e.g. "who am I") was being
# mis-stored as a label ("I'll remember that's a 'who am'").
_NON_LABEL_WORDS = {
    # interrogatives
    "who", "whom", "whose", "whos", "what", "whats", "where", "when", "why",
    "how", "which",
    # question-leading auxiliaries ("are you...", "can you...", "do you...")
    "are", "am", "do", "does", "did", "can", "could", "would", "will", "should",
    # command verbs / motion the robot acts on
    "stop", "halt", "go", "come", "follow", "wander", "explore", "turn", "move",
    "drive", "spin", "find", "search", "play", "pause", "resume", "forward",
    "backward", "left", "right",
}


def parse_label_answer(text, options):
    """Extract the intended label from a spoken answer, or None.

    An explicitly offered option named anywhere in the answer wins ("it's
    the speaker" -> "speaker"); otherwise the answer's remaining non-filler
    words become a fresh label ("that's a coffee mug" -> "coffee mug"). Pure
    and hardware-free so it's unit-testable off the robot."""
    low = (text or "").lower()
    for opt in options:
        if opt and re.search(rf"\b{re.escape(opt)}\b", low):
            return opt
    words = [w for w in re.findall(r"[a-z']+", low) if w not in _ANSWER_FILLER]
    return " ".join(words[:3]) if words else None


def answers_question(text, question):
    """True if TEXT could plausibly be the answer to QUESTION - and not a
    fresh command or question aimed at the robot. What counts depends on the
    question's kind:

      LABEL      - an affirmation/negation, an offered option, or a plain noun
                   label, but NOT a command/interrogative (see _NON_LABEL_WORDS).
      CORRECTION - anything the user says that ISN'T itself feedback ("that's
                   wrong" is graded elsewhere); the clarification may well be a
                   command word ("battery"), which is the whole point.
      YES_NO     - an explicit yes/no style short reply, or a feedback verdict.
      FREEFORM   - any non-empty spoken content resolves it.

    Pure; the broker owns which Question is live and whether it has expired."""
    low = (text or "").lower().strip()
    if not low:
        return False
    is_feedback = speech_match.parse_feedback(low) is not None
    kind = getattr(question, "kind", LABEL)

    if kind == CORRECTION:
        # A graded verdict ("that's not what I meant") is intent feedback, not
        # the clarification we're waiting for - leave it for the feedback path.
        return not is_feedback
    if is_feedback:
        return True                        # "yes, that's right" / "no, a mug"
    if kind == FREEFORM:
        return True
    if kind == YES_NO:
        toks = re.findall(r"[a-z']+", low)
        return any(t in speech_match.STRONG_SHORT_REPLIES for t in toks)
    # LABEL (default): a noun label, unless it's really a command/question.
    toks = re.findall(r"[a-z']+", low)
    return not any(t in _NON_LABEL_WORDS for t in toks)


def looks_like_label_answer(text, options):
    """Back-compat thin wrapper: does TEXT look like the answer to a
    "what is that?" label question? Kept as a named entry point (and for its
    existing unit tests) over the general answers_question()."""
    return answers_question(text, Question("_", "_", kind=LABEL, options=options))
