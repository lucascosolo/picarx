#!/usr/bin/env python3
# layer_b/capability_registry.py
"""
Declarative capability registry + the local half of the utterance router.

Background: "is this utterance for me, and who handles it" used to live in
several hand-synchronized lists - field_agent's TOOL_KEYWORDS, tools_registry's
_TOOL_WORDS and its flat RULES table - so every new skill meant editing two or
three files and hoping they stayed in agreement. This module is the single
place a capability declares itself:

    Capability(
        name="radio",
        topic="picarx/tools/radio",
        keywords=("radio", "station", "tune", ...),
        rules=[(pattern, builder), ...],
        say="play radio / stop radio / ...",
        description="streams internet radio through my speaker",
    )

and the single place that decides which one owns an utterance.

Deliberate limits, mirroring speech_match.py: pure and stdlib-only. No bus, no
clock, no I/O, no LLM. Modules own the side effects; this owns the decision, so
it is fully testable off-robot and cheap enough to run on every transcript.

Dispatch contract (Router.route):
  1. rules are tried capability by capability in registration order, and rule
     by rule inside each - first match wins, exactly like the flat table it
     replaces, so ordering stays the visible, reviewable thing it was;
  2. a builder returning None means "matched the shape but not the params" and
     routing falls through to the next rule;
  3. if nothing matched but the utterance carries a registered capability's
     keyword, the decision is ESCALATE - the LLM intent arbiter gets a chance
     at it instead of the utterance vanishing;
  4. otherwise UNCLAIMED, and the caller (dialog/companion/chat) decides.

Safety invariants this must never break: movement is NOT a capability here -
safety-relevant commands stay on field_agent's fast local path, and no model
output can route into one. Keyword ownership only ever decides who *may*
handle an utterance; it never suppresses "stop"/"halt".
"""

MATCH = "match"
ESCALATE = "escalate"
UNCLAIMED = "unclaimed"


class Capability:
    """One declared skill: what it answers to, and how to build its request.

    rules is a sequence of (compiled_pattern, builder) pairs, or
    (compiled_pattern, builder, topic) when a capability publishes to more
    than one topic (reminders set vs. control, for example). Builders are
    called as builder(match, canonical_text) and return the payload dict, or
    None to decline and let the next rule try.
    """

    def __init__(self, name, topic=None, rules=(), keywords=(), say="",
                 description="", escalate=True):
        self.name = name
        self.topic = topic
        self.rules = tuple(rules)
        self.keywords = tuple(dict.fromkeys(keywords))
        self.say = say
        self.description = description
        # Capabilities whose vocabulary is too generic to justify sending
        # unparsed utterances to the arbiter can opt out.
        self.escalate = escalate
        for rule in self.rules:
            if len(rule) not in (2, 3):
                raise ValueError(f"{name}: rule must be (pattern, builder[, topic])")
            if len(rule) == 2 and topic is None:
                raise ValueError(f"{name}: rule needs a topic (capability has none)")

    def describe(self):
        """The public catalog entry, as published on picarx/tools/available."""
        return {"name": self.name, "topic": self.topic,
                "say": self.say, "description": self.description}

    def match(self, canon):
        """(topic, payload) for the first rule that fully matches, else None."""
        for rule in self.rules:
            pattern, build = rule[0], rule[1]
            topic = rule[2] if len(rule) == 3 else self.topic
            m = pattern.search(canon)
            if not m:
                continue
            payload = build(m, canon)
            if payload is None:
                continue  # shape matched, params didn't - try the next rule
            return topic, payload
        return None


class Decision:
    """What the router concluded about one utterance."""

    def __init__(self, kind, capability=None, topic=None, payload=None,
                 keyword=None):
        self.kind = kind
        self.capability = capability
        self.topic = topic
        self.payload = payload
        self.keyword = keyword

    @property
    def matched(self):
        return self.kind == MATCH

    @property
    def claimed(self):
        """True when some capability owns this utterance - either it parsed
        cleanly or its vocabulary appeared. Callers outside the tool path
        (field_agent) use this to stay out of the way."""
        return self.kind in (MATCH, ESCALATE)

    def __repr__(self):
        name = self.capability.name if self.capability else None
        return f"<Decision {self.kind} {name} {self.topic}>"


class Router:
    """Local, deterministic dispatch across the declared capabilities."""

    def __init__(self, capabilities=()):
        self.capabilities = []
        for capability in capabilities:
            self.register(capability)

    def register(self, capability):
        if any(c.name == capability.name for c in self.capabilities):
            raise ValueError(f"duplicate capability: {capability.name}")
        self.capabilities.append(capability)
        return capability

    def keywords(self):
        """Every registered capability's vocabulary, deduplicated. This is the
        single source that used to be copied into TOOL_KEYWORDS/_TOOL_WORDS."""
        words = []
        for capability in self.capabilities:
            words.extend(capability.keywords)
        return tuple(dict.fromkeys(words))

    def describe(self):
        return [c.describe() for c in self.capabilities]

    def owner_of(self, text):
        """The first capability whose vocabulary appears in `text`, else None.
        Whole-token comparison would be wrong here: multi-word keywords such
        as "remote assist" are deliberately supported, so this is a substring
        test on already-canonicalized text."""
        for capability in self.capabilities:
            if not capability.escalate:
                continue
            for word in capability.keywords:
                if word in text:
                    return capability, word
        return None, None

    def route(self, canon, extra_text=""):
        """Route one canonicalized utterance. `extra_text` is optionally the
        raw transcript, searched for keywords alongside the canonical form so
        a word that canonicalization dropped can still claim ownership."""
        canon = (canon or "").lower().strip()
        for capability in self.capabilities:
            hit = capability.match(canon)
            if hit is not None:
                topic, payload = hit
                return Decision(MATCH, capability, topic, payload)
        haystack = f"{canon} {(extra_text or '').lower()}".strip()
        capability, word = self.owner_of(haystack)
        if capability is not None:
            return Decision(ESCALATE, capability, keyword=word)
        return Decision(UNCLAIMED)
