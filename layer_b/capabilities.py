#!/usr/bin/env python3
# layer_b/capabilities.py
"""
The robot's declared capabilities - one place, one source of truth.

Every skill that can be reached by voice declares itself here: the phrases it
answers to, the vocabulary that marks an utterance as *its* even when parsing
fails, the bus topic it publishes on, and how it describes itself to a human or
to the thinking plane. `capability_registry.py` holds the mechanism; this holds
the content.

Modules import from here rather than keeping their own copy:
  - tools_registry.py routes heard speech through ROUTER and publishes;
  - field_agent.py asks ROUTER whether an utterance already belongs to a tool
    before treating it as a drive/report command;
  - the catalog published on picarx/tools/available is ROUTER.describe().

Order matters: capabilities are tried top to bottom, and rules within a
capability likewise, so more specific declarations come first. This is the same
first-match-wins ordering the old flat RULES table had, kept visible instead of
implied.

Movement is deliberately absent. Drive/steer/stop stay on the local
safety-critical path in field_agent and the safety daemon, are never routed by
vocabulary, and must never become a registered capability.
"""
import re

from capability_registry import Capability, CapabilityTool, Router

REMOTE_TOPIC = "picarx/tools/remote_assist"
REMINDER_SET_TOPIC = "picarx/tools/reminder/set"
REMINDER_CONTROL_TOPIC = "picarx/tools/reminder/control"
NOTES_TOPIC = "picarx/tools/notes"
RADIO_TOPIC = "picarx/tools/radio"
DICE_TOPIC = "picarx/tools/dice"
CLOCK_TOPIC = "picarx/tools/clock"

# ---------- spoken-number → dial string ----------
# Vosk transcribes a frequency as WORDS ("ninety eight point seven",
# "one oh two point five"), sometimes as digits ("98.7"). This turns
# either into a canonical dial string like "98.7" so the radio can
# match it. Frequencies use two spoken conventions - grouped tens
# ("ninety eight" = 98) and digit-by-digit ("one oh two" = 102) - and
# this handles both, keyed off whether every token is a single digit.
_ONES = {"zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4,
         "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
          "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_NUMWORD = {**_ONES, **_TEENS, **_TENS, "hundred": 100, "a": 1}


def _spoken_amount(value):
    """Parse the small numeric vocabulary used by reminder phrases."""
    value = str(value or "").strip().lower()
    try:
        return float(value)
    except ValueError:
        pass
    if value in _ONES:
        return float(_ONES[value])
    if value in _TEENS:
        return float(_TEENS[value])
    if value in _TENS:
        return float(_TENS[value])
    return None


def _side_to_digits(tokens):
    """Convert one side of the dial (before or after the point) to a
    digit string. Returns None if a token isn't a number word."""
    if not tokens:
        return ""
    if any(t not in _NUMWORD for t in tokens):
        return None
    # All single-digit words (ones/oh) -> digit-by-digit ("one oh two").
    if all(t in _ONES for t in tokens):
        return "".join(str(_ONES[t]) for t in tokens)
    # Otherwise grouped arithmetic ("ninety eight", "one hundred eight").
    current = 0
    for t in tokens:
        if t == "hundred":
            current = (current or 1) * 100
        else:
            current += _NUMWORD[t]
    return str(current)


def parse_dial(text):
    """Return a canonical dial string ('98.7') from a spoken/typed
    frequency, or None if the text doesn't clearly contain one."""
    text = text.lower()
    # Digits already present: "98.7", "98 7", "1025", "987".
    m = re.search(r"\b(\d{2,4})(?:[.\s](\d))?\b", text)
    if m and m.group(2):
        return f"{m.group(1)}.{m.group(2)}"
    if m and len(m.group(1)) >= 4:                    # e.g. "1025" -> 102.5
        return f"{m.group(1)[:-1]}.{m.group(1)[-1]}"
    if m and len(m.group(1)) == 3:                    # e.g. "987" -> 98.7
        return f"{m.group(1)[:2]}.{m.group(1)[2]}"
    if m:
        return m.group(1)                             # bare "98"
    # Word form. Split on point/dot, keep only number words either side.
    tokens = re.findall(r"[a-z]+", text)
    tokens = [t for t in tokens if t in _NUMWORD or t in ("point", "dot")]
    if not tokens:
        return None
    if "point" in tokens or "dot" in tokens:
        sep = "point" if "point" in tokens else "dot"
        i = tokens.index(sep)
        whole = _side_to_digits(tokens[:i])
        frac = _side_to_digits(tokens[i + 1:])
        if whole and frac:
            return f"{whole}.{frac}"
        return whole or None
    return _side_to_digits(tokens)


# ---------- payload builders ----------
# A builder receives (regex match, canonicalized text) and returns the payload
# to publish, or None for "matched the shape but couldn't extract params" so
# routing falls through to the next rule.

def _tune_payload(m, text):
    dial = parse_dial(text)
    return {"command": "play", "dial": dial} if dial else None


# Words that aren't part of the genre/name in a search utterance:
# "radio find some soft rock for me please" -> keywords "soft rock".
_FIND_FILLER = {"radio", "station", "stations", "find", "search", "look",
                "for", "a", "an", "some", "me", "please", "the", "up",
                "on", "of", "to", "my", "play"}


def _find_payload(m, text):
    words = [w for w in re.findall(r"[a-z0-9]+", text) if w not in _FIND_FILLER]
    if not words or all(w.isdigit() for w in words):
        return None  # nothing searchable / it's a dial - let later rules tune it
    return {"command": "find", "keywords": " ".join(words)}


def _remote_payload(m, text):
    # Keep extraction deliberately narrow. Hostnames/IPs are validated again
    # by remote_assist.py; this only prevents ordinary navigation phrases such
    # as "connect to the kitchen" from becoming network requests.
    host = None
    if m:
        for i in range(1, (len(m.groups()) if m.lastindex else 0) + 1):
            if m.group(i):
                host = m.group(i).strip().rstrip(".,")
                break
    if not host:
        return None
    # speech_match.canonicalize() separates punctuation in IPv4 addresses
    # ("192.168.1.20" -> "192 168 1 20"). Put the four octets back together
    # before the remote module performs its own strict validation.
    if re.fullmatch(r"\d{1,3}(?:\s+\d{1,3}){3}", host):
        host = ".".join(host.split())
    return {"command": "connect", "host": host}


def _reminder_payload(m, text):
    amount = _spoken_amount(m.group("amount")) if m and m.group("amount") else None
    if amount is not None:
        unit = m.group("unit").lower()
        minutes = amount / 60.0 if unit.startswith("second") else amount
        if unit.startswith("hour"):
            minutes = amount * 60.0
        message = m.group("message").strip(" .,!?")
        if message and minutes > 0:
            return {"message": message, "delay_minutes": minutes}
    if m and m.group("clock"):
        hour, minute = m.group("clock").split()
        message = m.group("clock_message").strip(" .,!?")
        if message:
            return {"message": message, "at": f"{int(hour):02d}:{int(minute):02d}"}
    return None


def _note_payload(m, text):
    body = (m.group("body") if m else "").strip(" .,!?")
    return {"command": "create", "text": body, "source": "voice"} if body else None


_DICE_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4,
                     "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                     "ten": 10, "pair": 2, "couple": 2}
MAX_DICE = 10
MIN_DICE_SIDES = 2
MAX_DICE_SIDES = 1000


def _dice_payload(m, text):
    """How many dice, and how many sides. Everything is optional: a bare
    "roll the dice" is one ordinary six-sided die."""
    sides = 6
    shorthand = re.search(r"\bd\s?(\d{1,4})\b", text)      # "roll a d20"
    spelled = re.search(r"\b(\d{1,4})[\s-]*sided\b", text)  # "twenty sided die"
    if spelled:
        sides = int(spelled.group(1))
    elif shorthand:
        sides = int(shorthand.group(1))
    if not MIN_DICE_SIDES <= sides <= MAX_DICE_SIDES:
        return None

    # "two dice", "roll 3 dice", "a pair of dice" - the count word sits a
    # word or two ahead of the noun, never after it.
    count = 1
    digits = None if spelled else re.search(
        r"\b(\d{1,2})(?:\s+\w+){0,2}\s+(?:dice|die)\b", text)
    if digits:
        count = int(digits.group(1))
    else:
        for word, value in _DICE_COUNT_WORDS.items():
            if re.search(rf"\b{word}\b(?:\s+\w+){{0,2}}\s+(?:dice|die)\b", text):
                count = value
                break
    count = max(1, min(MAX_DICE, count))
    return {"command": "roll", "count": count, "sides": sides}


def _query_payload(m, text):
    query = (m.group("query") if m and m.groupdict().get("query") else "").strip(" .,!?")
    query = re.sub(r"^(?:about|for|named|called)\s+", "", query)
    return {"command": "delete", "query": query, "confirmed": True,
            "source": "voice"} if query else None


# ---------- declarations ----------
# Patterns run against speech_match.canonicalize()d text ("play the radio for
# me please" arrives here as "play radio"), so they only need to cover
# meaningful word variants, not filler permutations.

REMINDERS = Capability(
    name="reminders",
    topic=REMINDER_CONTROL_TOPIC,
    keywords=("remind", "reminder", "reminders"),
    say="remind me in <time> to <text> / list or cancel reminders",
    description="persists bounded local reminders and speaks them at the "
                "requested time",
    rules=[
        # Explicit, deterministic reminder phrases.  More ambiguous requests
        # such as "remind me to call mom" remain available to the companion
        # LLM, which can ask for a time instead of inventing one.
        (re.compile(r"\bremind(?:\s+me)?\s+in\s+(?P<amount>\d+(?:\.\d+)?|"
                    r"one|two|three|four|five|six|seven|eight|nine|ten|eleven|"
                    r"twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|"
                    r"fifty|sixty)\s+(?P<unit>seconds?|minutes?|hours?)\s+"
                    r"(?:to\s+)?(?P<message>.+)$|"
                    r"\bremind(?:\s+me)?\s+at\s+(?P<clock>\d{1,2}\s+\d{2})\s+"
                    r"(?:to\s+)?(?P<clock_message>.+)$"),
         _reminder_payload, REMINDER_SET_TOPIC),
        (re.compile(r"\b(?:list|show|what are)\b.*\breminders?\b|"
                    r"\bwhat reminders\b"),
         lambda m, t: {"command": "list", "source": "voice"}),
        (re.compile(r"\b(?:cancel|delete|remove)\b\s+reminders?\s+(?P<query>.+)$"),
         _query_payload),
    ],
)

NOTES = Capability(
    name="notes",
    topic=NOTES_TOPIC,
    keywords=("note", "notes", "meeting"),
    say="take a note <text> / start, pause, resume, stop meeting notes / "
        "list or delete notes",
    description="stores local user notes and consented meeting transcripts",
    rules=[
        # A note with content is a single memory entry; a bare "take notes" is
        # a consented continuous meeting session.
        (re.compile(r"\b(?:take|make|write|add)\s+(?:a\s+)?note\b"
                    r"(?:\s+(?:that|saying|about))?\s*(?P<body>.+)$"),
         _note_payload),
        (re.compile(r"\b(?:start|begin)\s+(?:taking\s+)?(?:meeting\s+)?notes?\b|"
                    r"\btake\s+(?:meeting\s+)?notes?\b"),
         lambda m, t: {"command": "start", "confirmed": True, "source": "voice"}),
        (re.compile(r"\b(?:pause|resume|stop|end|finish)\b.*\bmeeting\s+notes?\b"),
         lambda m, t: {"command": ("pause" if "pause" in m.group(0) else
                                   "resume" if "resume" in m.group(0) else "stop"),
                       "source": "voice"}),
        (re.compile(r"\b(?:list|show|search)\b.*\bnotes?\b"),
         lambda m, t: {"command": "list", "source": "voice"}),
        (re.compile(r"\b(?:delete|remove)\s+notes?\s+(?P<query>.+)$"),
         _query_payload),
    ],
)

REMOTE_ASSIST = Capability(
    name="remote_assist",
    topic=REMOTE_TOPIC,
    keywords=("ssh", "remote assist", "remote", "host"),
    say="ssh into <host> / give or revoke remote write access / "
        "disconnect remote assist",
    description="connects to a provisioned host helper over verified SSH so I "
                "can inspect a project, preview approved patches, and run "
                "bounded debugging commands",
    rules=[
        # Remote project assistance. Require ssh/remote/host/computer wording
        # or an unmistakable "ssh into" phrase so this never captures a place
        # goal.
        (re.compile(r"\b(?:revoke|remove|disable)\b.*\b(?:remote|ssh|host|project)\b"
                    r".*\bwrite(?: access| permission)?\b"),
         lambda m, t: {"command": "revoke_write"}),
        (re.compile(r"\b(?:grant|give|allow|enable)\b.*\b(?:remote|ssh|host|project)\b"
                    r".*\bwrite(?: access| permission)?\b|"
                    r"\b(?:remote|ssh|host|project)\b.*\bwrite access\b"),
         lambda m, t: {"command": "authorize_write", "confirmed": True}),
        (re.compile(r"\b(?:ssh\s+(?:into\s+)?|(?:remote|host|computer)\s+)"
                    r"((?:\d{1,3}(?:[ .]\d{1,3}){3})|[a-z][a-z0-9.-]{1,62})\b|"
                    r"\bconnect\s+to\s+((?:\d{1,3}(?:[ .]\d{1,3}){3})|"
                    r"[a-z0-9-]+\.[a-z0-9.-]+)\b"),
         _remote_payload),
        (re.compile(r"\b(?:disconnect|close)\b.*\b(?:ssh|remote|host|computer)\b|"
                    r"\bstop\s+remote\s+assist\b"),
         lambda m, t: {"command": "disconnect"}),
    ],
)

RADIO = Capability(
    name="radio",
    topic=RADIO_TOPIC,
    keywords=("radio", "station", "stations", "tune", "dial", "frequency",
              "fm", "music", "song"),
    say="play radio / stop radio / next station / station <name> / "
        "tune to <number> / radio find <genre or name> / "
        "what's playing / list stations",
    description="streams internet radio through my speaker; tune saved dials, "
                "or search the live radio-browser.info directory by keyword "
                "and cycle results with next station",
    rules=[
        (re.compile(r"\b(?:stop|pause|turn off|shut off|kill)\b.*\b(?:radio|music)\b|"
                    r"\b(?:radio|music) off\b"),
         lambda m, t: {"command": "stop"}),
        # Live directory search: needs a find/search word AND radio/station
        # in the utterance ("radio find soft rock", "find me a jazz station").
        (re.compile(r"\b(?:find|search)\b(?=.*\b(?:radio|station)\b)|"
                    r"\b(?:radio|station)\b(?=.*\b(?:find|search)\b)"),
         _find_payload),
        (re.compile(r"\bwhat(?:'s| is)?\s+playing\b|\bradio status\b"),
         lambda m, t: {"command": "status"}),
        (re.compile(r"\blist\b.*\bstations?\b|\bwhat stations\b"),
         lambda m, t: {"command": "list"}),
        (re.compile(r"\b(?:next|change|switch|skip|another|different)\b.*"
                    r"\b(?:station|radio|song|music)\b|"
                    r"\b(?:station|song)\b.*\b(?:next|skip)\b"),
         lambda m, t: {"command": "next"}),
        # Tune to a frequency/dial: needs a tuning word AND a number.
        # NOTE: "to" is deliberately NOT a tuning word - "\bto\b.*\d" matched
        # any utterance shaped like "... to <number> ..." ("set a timer to 20
        # minutes", "count to 10") and hijacked it into a radio tune. Real
        # tune requests always carry one of the actual radio words below.
        (re.compile(r"\b(?:tune|station|frequency|dial|fm)\b.*\d|"
                    r"\b(?:tune|station|frequency|dial|fm)\b.*\b(?:one|two|three|four|five|six|"
                    r"seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|"
                    r"seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|"
                    r"eighty|ninety|hundred|oh|zero)\b"),
         _tune_payload),
        (re.compile(r"\b(?:play|start|put on)\b.*\b(?:radio|music|tunes)\b|"
                    r"\b(?:radio|music) on\b"),
         lambda m, t: {"command": "play"}),
        # Named station: "station <name>" (only when it's not a number).
        (re.compile(r"\bstation\s+([a-z][a-z\s]*)"),
         lambda m, t: {"command": "play", "station": m.group(1).strip()}),
    ],
)

def _dice_tool(tool_input):
    """The model's typed arguments, bounded the same way the spoken parser
    bounds them. The capability keeps the last word on its own limits: a
    request outside them comes back as a sentence, not a publish."""
    command = str((tool_input or {}).get("command") or "roll").lower()
    if command == "flip":
        return {"command": "flip"}
    if command != "roll":
        return "I can roll dice or flip a coin, nothing else."
    count = int((tool_input or {}).get("count") or 1)
    sides = int((tool_input or {}).get("sides") or 6)
    if not 1 <= count <= MAX_DICE:
        return f"I can roll between 1 and {MAX_DICE} dice at once."
    if not MIN_DICE_SIDES <= sides <= MAX_DICE_SIDES:
        return (f"A die needs between {MIN_DICE_SIDES} and "
                f"{MAX_DICE_SIDES} sides.")
    return {"command": "roll", "count": count, "sides": sides}


def _clock_tool(tool_input):
    command = str((tool_input or {}).get("command") or "time").lower()
    if command not in ("time", "date", "datetime"):
        return "I can read the time, the date, or both."
    return {"command": command}


DICE = Capability(
    name="dice",
    topic=DICE_TOPIC,
    tool=CapabilityTool(
        name="roll_dice",
        description="Roll dice or flip a coin, and hear the outcome. Use it "
                    "whenever chance should decide something - the person asks "
                    "you to, or you are choosing between options and would "
                    "rather leave it to luck than pick. The result is real "
                    "randomness from the robot, not a number you made up.",
        input_schema={"type": "object", "properties": {
            "command": {"type": "string", "enum": ["roll", "flip"]},
            "count": {"type": "integer", "minimum": 1, "maximum": MAX_DICE,
                      "description": "how many dice (roll only)"},
            "sides": {"type": "integer", "minimum": MIN_DICE_SIDES,
                      "maximum": MAX_DICE_SIDES,
                      "description": "sides per die (roll only), 6 by default"}},
            "required": ["command"]},
        build=_dice_tool,
        result_topic=DICE_TOPIC + "/result"),
    keywords=("dice", "coin flip", "flip a coin"),
    say="roll a dice / roll two dice / roll a d20 / flip a coin",
    description="rolls dice and flips coins locally - no network, no thinking, "
                "just an honest random number",
    rules=[
        (re.compile(r"\b(?:flip|toss)\b.*\bcoin\b|\bcoin\s+(?:flip|toss)\b|"
                    r"\bheads\s+or\s+tails\b"),
         lambda m, t: {"command": "flip"}),
        (re.compile(r"\b(?:roll|throw|shake)\b.*\b(?:dice|die|d\d{1,4})\b|"
                    r"\bdice\s+roll\b"),
         _dice_payload),
    ],
)

CLOCK = Capability(
    name="clock",
    topic=CLOCK_TOPIC,
    tool=CapabilityTool(
        name="read_clock",
        description="Read the current local time and date off the robot's own "
                    "clock. Use it whenever the actual time matters - working "
                    "out how long ago something happened, whether it is late, "
                    "or what to call today - rather than guessing from context.",
        input_schema={"type": "object", "properties": {
            "command": {"type": "string", "enum": ["time", "date", "datetime"],
                        "description": "which to read; 'datetime' for both"}},
            "required": ["command"]},
        build=_clock_tool,
        result_topic=CLOCK_TOPIC + "/result"),
    keywords=("what time", "the time", "what day", "what date", "the date"),
    say="what time is it / what's the date / what day is it",
    description="reads the current local time and date off my own clock - no "
                "network and no thinking needed",
    rules=[
        # Matched against canonicalized text, where trailing filler is already
        # gone ("what time is it" arrives as "what time is").
        (re.compile(r"\bwhat(?:'s)?(?:\s+\w+){0,2}\s+time\b|"
                    r"\btime\s+is\s+it\b|\btell\s+(?:\w+\s+)?time\b|"
                    r"\bcurrent\s+time\b"),
         lambda m, t: {"command": "time"}),
        (re.compile(r"\bwhat(?:'s)?(?:\s+\w+){0,2}\s+(?:date|day)\b|"
                    r"\btoday'?s?\s+date\b|\bwhat\s+day\b"),
         lambda m, t: {"command": "date"}),
    ],
)

# Registration order is dispatch order. The four original tools keep their
# existing precedence; the newer local answers are appended rather than
# inserted, so no previously-routed phrase changes owner.
ROUTER = Router([REMINDERS, NOTES, REMOTE_ASSIST, RADIO, DICE, CLOCK])

# Asking the robot what it can do is answered by the registry itself rather
# than by any one capability, but the word still belongs to the tool path so
# field_agent leaves "what tools do you have" alone.
CATALOG_KEYWORDS = ("tools",)


def keywords():
    """Every word that marks an utterance as tool-owned. The single source
    that TOOL_KEYWORDS/_TOOL_WORDS used to duplicate by hand."""
    return ROUTER.keywords() + CATALOG_KEYWORDS


def describe():
    """Catalog entries for picarx/tools/available and the thinking plane."""
    return ROUTER.describe()


def thinking_tools():
    """(capability, CapabilityTool) for every capability that has declared
    itself to the thinking plane. companion.py merges these into the typed
    tool list it hands the model, so a capability declared here is reachable
    BOTH by a spoken phrase and by the robot's own decision to use it."""
    return ROUTER.tools()


def tool_named(name):
    return ROUTER.tool_named(name)
