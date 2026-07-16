#!/usr/bin/env python3
# /home/picarx/layer_b/speech_match.py
"""
Shared voice-command normalization - the tolerance layer between what
Vosk heard and what the rule tables expect.

The command routers (tools_registry.py, field_agent.py) match on
specific words: "play radio" works, "play the radio for me" may not,
and an STT near-miss ("play the radial") never does - so people end up
repeating themselves in ever more robotic phrasing. This module makes
matching tolerant WITHOUT loosening the rules themselves:

  canonicalize(text) does two cheap, deterministic things:
    1. drops filler tokens ("play the radio for me please" ->
       "play radio"), so word ORDER and CONTENT matter but politeness
       and articles don't;
    2. snaps near-miss tokens onto the robot's known command vocabulary
       using stdlib difflib ("radial" -> "radio", "explorer" ->
       "explore"), so one mangled phoneme doesn't force a re-take.

  looks_command_like(canonical) answers "did this utterance PROBABLY
  try to be a command?" - it's how routers decide that an unmatched
  utterance is worth escalating to the LLM intent arbiter
  (picarx/audio/uncertain) instead of dropping it on the floor.

Deliberate limits: stdlib only, no models, deterministic, and the
snapping is conservative (>=4-char tokens, high cutoff) - a wrong snap
that FIRES a command is worse than a miss that gets escalated. Number
words are never treated as filler ("one oh two point five" must
survive canonicalization for dial parsing to work downstream).
"""
import difflib
import re

# Tokens that carry no command meaning. NOTE: number words ("oh", "one",
# "point") are deliberately absent - they're load-bearing for spoken
# radio dials. "a" is dropped even though parse_dial knows "a hundred";
# that reading is rare enough to lose, unlike "oh".
FILLER_WORDS = {
    "the", "a", "an", "please", "some", "for", "me", "my", "now", "just",
    "can", "could", "would", "will", "you", "kindly", "hey", "um", "uh",
    "er", "it", "that", "this", "again", "go", "ahead", "and",
}

# The robot's command vocabulary: both the snap targets for near-miss
# repair and the "was this probably meant for the robot?" test set.
# Keep this mirroring what tools_registry RULES and field_agent's hard
# commands actually key on - a word listed here but matched nowhere just
# makes looks_command_like() eager for no benefit.
DOMAIN_VOCAB = {
    # radio / tools
    "radio", "station", "stations", "tune", "tuning", "frequency", "dial",
    "play", "playing", "music", "volume", "find", "search", "next", "skip",
    "tools",
    # field_agent hard commands
    "explore", "stop", "halt", "battery", "charge", "status", "report",
    "history", "objects", "object", "notice", "map", "places", "where",
    "hello",
}

# Only snap tokens of this length or more (short words collide too
# easily: "to"/"do", "on"/"oh"), onto targets of similar length.
_SNAP_MIN_LEN = 4
_SNAP_CUTOFF = 0.8
_SNAP_TARGETS = sorted(w for w in DOMAIN_VOCAB if len(w) >= _SNAP_MIN_LEN)


def tokens(text):
    """Lowercase word tokens; keeps digits and in-word apostrophes so
    "98.7" -> "98", "7" and "what's" survives as one token."""
    return re.findall(r"[a-z0-9']+", (text or "").lower())


def _snap(token):
    """Return the vocabulary word this token was probably meant to be,
    or the token unchanged. Conservative on purpose - see module doc.
    Besides the similarity cutoff, the candidate must be within one
    character of the token's length: an STT near-miss garbles sounds,
    it doesn't grow words ("raydio"->"radio" is a mishearing;
    "nice"->"notice" is just two different words scoring 0.8)."""
    if len(token) < _SNAP_MIN_LEN or token in DOMAIN_VOCAB or token.isdigit():
        return token
    close = difflib.get_close_matches(token, _SNAP_TARGETS, n=1, cutoff=_SNAP_CUTOFF)
    if close and abs(len(close[0]) - len(token)) <= 1:
        return close[0]
    return token


def canonicalize(text):
    """Filler-free, near-miss-repaired version of an utterance, for
    MATCHING only - never speak or store this form, it's lossy."""
    return " ".join(_snap(t) for t in tokens(text) if t not in FILLER_WORDS)


def looks_command_like(canonical_text):
    """True if a canonicalized utterance contains any robot vocabulary -
    the cheap signal that an unmatched utterance deserves escalation to
    the LLM intent arbiter rather than being dropped."""
    return any(t in DOMAIN_VOCAB for t in canonical_text.split())


# ---------------------------------------------------------------------
# Utterance quality scoring (noise rejection)
# ---------------------------------------------------------------------
# Background noise regularly decodes to SOMETHING - a lone "the", a limp
# two-word fragment - and each one that reaches the chat path is a paid
# LLM call answering the television. These scores are the cheap,
# deterministic screen in front of that: no models (a POS tagger on the
# Pi would cost more CPU than the LLM calls it saves), just word lists
# and arithmetic, same philosophy as canonicalize() above.

# A lone one of these is noise, period - background chatter's most common
# decode products. Only ever applied to SINGLE-word utterances: all of
# them are perfectly meaningful inside a longer sentence.
WEAK_SINGLE_WORDS = {"the", "a", "an", "and", "or", "um", "uh", "hmm",
                     "huh", "er", "oh", "ah"}

# Meaningful one-worders: legitimate mid-conversation replies and
# attention-getters that must NOT be scored like noise just for being
# short ("yes" answering a question is real speech).
STRONG_SHORT_REPLIES = {"yes", "no", "yeah", "nope", "sure", "okay", "ok",
                        "thanks", "bye", "hi", "hello", "why", "how", "what",
                        "stop", "halt"}

# "This asks for something": command verbs the robot acts on plus
# question openers. Broader than DOMAIN_VOCAB on purpose - chat requests
# ("tell me a story") are legitimate LLM work even with zero robot
# vocabulary in them.
ACTION_WORDS = {
    "play", "stop", "halt", "find", "search", "tune", "turn", "go", "come",
    "tell", "show", "explore", "remember", "remind", "report", "look",
    "follow", "sing", "say", "set", "list", "skip", "pause", "start",
    "open", "close", "switch", "change", "describe", "what", "where",
    "when", "who", "how", "why", "is", "are", "do", "does", "can", "could",
    "would",
}


def intent_score(text):
    """Rule-based "does this ask for something" score in 0..1:
    action/question word x0.5 + content word x0.3 + length bonus x0.2."""
    toks = tokens(text)
    if not toks:
        return 0.0
    has_action = any(t in ACTION_WORDS or t in DOMAIN_VOCAB for t in toks)
    has_content = any(len(t) >= 3 and t not in FILLER_WORDS
                      and t not in ACTION_WORDS for t in toks)
    length_bonus = min(1.0, (len(toks) - 1) / 4.0)
    return 0.5 * has_action + 0.3 * has_content + 0.2 * length_bonus


def quality_score(text, confidence=None):
    """0..1 "was this real, directed speech?" - what the LLM paths check
    before spending a call. A single weak word scores 0 outright; strong
    short replies keep a floor so a mid-conversation "yes" survives;
    decoder confidence (when the STT provides one) scales the rest."""
    toks = tokens(text)
    if not toks:
        return 0.0
    if len(toks) == 1 and toks[0] in WEAK_SINGLE_WORDS:
        return 0.0
    score = intent_score(text)
    if len(toks) <= 2 and any(t in STRONG_SHORT_REPLIES for t in toks):
        score = max(score, 0.6)
    if confidence is not None:
        score *= 0.5 + 0.5 * max(0.0, min(1.0, confidence))
    return round(score, 3)
