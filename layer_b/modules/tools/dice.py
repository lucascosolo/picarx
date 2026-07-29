#!/usr/bin/env python3
# layer_b/modules/tools/dice.py
"""
Dice daemon (Layer B tool) - rolls dice and flips coins.

The smallest possible capability, and deliberately so: it is the reference
example of a tool that answers entirely on-board. No network, no LLM, no
hardware, no stored state - `random` and the speaker. Requests arrive on
picarx/tools/dice from the capability router (voice) or the thinking plane,
and the spoken result is published on the existing TTS topic.

Request payload:
  {"command": "roll", "count": 2, "sides": 6}   # count/sides optional
  {"command": "flip"}                            # a coin

Bounds are enforced here as well as in the router's parser, since a request
can also arrive from a model: at most MAX_DICE dice of at most MAX_SIDES
sides, so a mistyped "roll 10000 dice" is clamped rather than read aloud for
a minute. Every roll is published on picarx/tools/dice/result and journaled,
so an outcome the user reacts to is available as evidence later.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from broker_client import Bus

import random
import time

DICE_TOPIC = "picarx/tools/dice"
RESULT_TOPIC = "picarx/tools/dice/result"
SPEAK_TOPIC = "picarx/audio/speak"
DECISION_TOPIC = "picarx/decision"

MAX_DICE = 10
MIN_SIDES = 2
MAX_SIDES = 1000
DEFAULT_SIDES = 6


def _clamp_int(value, default, low, high):
    """Fail-soft integer coercion: anything unparseable becomes the default
    rather than raising inside a bus callback."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _correlated(result, request):
    """The result, plus the correlation keys a thinking-plane caller waits on.

    A request that arrived from a spoken phrase carries no request_id and the
    payload is unchanged - the extra keys only appear when someone is actually
    waiting for the answer, which is what lets the robot use its own dice roll
    in a sentence instead of only speaking it aloud."""
    request_id = str((request or {}).get("request_id") or "").strip()
    if not request_id:
        return result
    return dict(result, request_id=request_id, ok=True, result=result)


class Dice:
    def __init__(self, rng=None):
        self.bus = Bus()
        self.rng = rng or random.Random()

    def _say(self, text):
        self.bus.publish(SPEAK_TOPIC, {"text": text, "ts": time.time()})

    def roll(self, count=1, sides=DEFAULT_SIDES):
        count = _clamp_int(count, 1, 1, MAX_DICE)
        sides = _clamp_int(sides, DEFAULT_SIDES, MIN_SIDES, MAX_SIDES)
        values = [self.rng.randint(1, sides) for _ in range(count)]
        if count == 1:
            spoken = f"{values[0]}."
        else:
            spoken = (", ".join(str(v) for v in values[:-1]) +
                      f" and {values[-1]}, for a total of {sum(values)}.")
        return {"command": "roll", "count": count, "sides": sides,
                "values": values, "total": sum(values), "spoken": spoken}

    def flip(self):
        face = self.rng.choice(["heads", "tails"])
        return {"command": "flip", "face": face, "spoken": f"{face.capitalize()}."}

    def on_request(self, payload):
        try:
            if not isinstance(payload, dict):
                return
            command = str(payload.get("command") or "roll").lower()
            if command == "flip":
                result = self.flip()
            elif command == "roll":
                result = self.roll(payload.get("count", 1),
                                   payload.get("sides", DEFAULT_SIDES))
            else:
                return
            result["ts"] = time.time()
            print(f"Dice: {result}")
            self._say(result["spoken"])
            self.bus.publish(RESULT_TOPIC, _correlated(result, payload))
            self.bus.publish(DECISION_TOPIC, {
                "source": "dice", "kind": "tool_result",
                "choice": {k: result[k] for k in ("command", "count", "sides",
                                                  "total", "face")
                           if k in result},
                "reason": "random outcome produced locally", "ts": result["ts"]})
        except Exception as e:                       # never kill the callback
            print(f"Dice: request failed: {e}")

    def run(self):
        self.bus.subscribe(DICE_TOPIC, self.on_request)
        print("Dice tool active.")
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    Dice().run()
