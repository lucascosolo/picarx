#!/usr/bin/env python3
# layer_b/modules/tools/clock.py
"""
Clock daemon (Layer B tool) - what time is it, what day is it.

Reads the Pi's own system clock and speaks the answer. No network, no LLM,
no stored state: the same trust boundary reminder_daemon.py already relies on
for `at: "HH:MM"`, so if one is right the other is too.

Request payload, from the capability router (voice) or the thinking plane:
  {"command": "time"}      # "It's twenty past four in the afternoon."
  {"command": "date"}      # "It's Tuesday, July 28th."
  {"command": "datetime"}  # both

Times are spoken the way a person would say them rather than as digits, since
the answer goes to a speaker and "sixteen twenty" is not how anyone asks. The
clock injection point exists for tests; production always uses local time.
"""
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from broker_client import Bus

import time
from datetime import datetime

CLOCK_TOPIC = "picarx/tools/clock"
RESULT_TOPIC = "picarx/tools/clock/result"
SPEAK_TOPIC = "picarx/audio/speak"

_HOURS = {0: "twelve", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
          6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
          11: "eleven", 12: "twelve"}
_MINUTES = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six",
            7: "seven", 8: "eight", 9: "nine", 10: "ten", 11: "eleven",
            12: "twelve", 13: "thirteen", 14: "fourteen", 15: "quarter",
            16: "sixteen", 17: "seventeen", 18: "eighteen", 19: "nineteen",
            20: "twenty", 21: "twenty one", 22: "twenty two",
            23: "twenty three", 24: "twenty four", 25: "twenty five",
            26: "twenty six", 27: "twenty seven", 28: "twenty eight",
            29: "twenty nine", 30: "half"}


def _part_of_day(hour):
    if hour < 12:
        return "in the morning"
    if hour < 17:
        return "in the afternoon"
    if hour < 21:
        return "in the evening"
    return "at night"


def spoken_time(now):
    """Conversational clock reading: 'twenty past four in the afternoon'."""
    hour, minute = now.hour, now.minute
    if minute == 0:
        return f"{_HOURS[hour % 12]} o'clock {_part_of_day(hour)}"
    if minute <= 30:
        return (f"{_MINUTES[minute]} past {_HOURS[hour % 12]} "
                f"{_part_of_day(hour)}")
    next_hour = (hour + 1) % 24
    return (f"{_MINUTES[60 - minute]} to {_HOURS[next_hour % 12]} "
            f"{_part_of_day(next_hour)}")


_ORDINAL_SUFFIX = {1: "st", 2: "nd", 3: "rd"}


def _ordinal(day):
    if 11 <= day <= 13:
        return f"{day}th"
    return f"{day}{_ORDINAL_SUFFIX.get(day % 10, 'th')}"


def spoken_date(now):
    """'Tuesday, July 28th' - the way the answer is actually asked for."""
    return f"{now.strftime('%A')}, {now.strftime('%B')} {_ordinal(now.day)}"


class Clock:
    def __init__(self, clock=None):
        self.bus = Bus()
        # Injection point for tests; production reads the Pi's local clock.
        self.clock = clock or datetime.now

    def answer(self, command="time"):
        now = self.clock()
        command = str(command or "time").lower()
        if command == "date":
            spoken = f"It's {spoken_date(now)}."
        elif command == "datetime":
            spoken = f"It's {spoken_time(now)} on {spoken_date(now)}."
        else:
            command = "time"
            spoken = f"It's {spoken_time(now)}."
        return {"command": command, "spoken": spoken,
                "iso": now.replace(microsecond=0).isoformat()}

    def on_request(self, payload):
        try:
            if not isinstance(payload, dict):
                return
            result = self.answer(payload.get("command", "time"))
            result["ts"] = time.time()
            print(f"Clock: {result['spoken']}")
            self.bus.publish(SPEAK_TOPIC, {"text": result["spoken"],
                                           "ts": result["ts"]})
            self.bus.publish(RESULT_TOPIC, result)
        except Exception as e:                       # never kill the callback
            print(f"Clock: request failed: {e}")

    def run(self):
        self.bus.subscribe(CLOCK_TOPIC, self.on_request)
        print("Clock tool active.")
        while True:
            time.sleep(3600)


if __name__ == "__main__":
    Clock().run()
