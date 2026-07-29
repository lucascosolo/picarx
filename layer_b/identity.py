#!/usr/bin/env python3
# layer_b/identity.py
"""
Who the robot is - the one place its name and self-reference live.

This is deliberately tiny, and deliberately its own module rather than a
literal inside companion's system prompt. Continuity of self starts with the
most basic fact an individual has: a name it knows and answers to. The robot
had none - every module that needed to speak as it called itself "a PiCar-X",
so it was a generic device wearing a personality, not someone.

Pure and stdlib-only, same philosophy as speech_match/attention: it reads the
`identity` config knobs (env > config.json > default) once and hands back
plain strings. No bus, no clock, no I/O beyond the config read. The name is a
knob, not a constant, so a second robot on the same codebase is a different
individual by configuration, not by a code fork.

The identity here grounds PERSONALITY only - how the robot refers to itself
when it talks. It is never a safety or authority signal: it does not gate
movement, it does not filter "stop"/"halt", and knowing its own name grants it
nothing it could not already do.
"""
import robot_config

DEFAULT_NAME = "Marco"
DEFAULT_PRONOUNS = "he/him"


def name():
    """The robot's own name - what it calls itself and answers to."""
    value = str(robot_config.get("identity", "name", DEFAULT_NAME,
                                 env="ROBOT_NAME") or "").strip()
    return value or DEFAULT_NAME


def pronouns():
    """How the robot refers to itself, for the personality prompt only."""
    value = str(robot_config.get("identity", "pronouns", DEFAULT_PRONOUNS,
                                 env="ROBOT_PRONOUNS") or "").strip()
    return value or DEFAULT_PRONOUNS


def self_intro():
    """A short first-person identity statement to seed the personality prompt.
    Kept deliberately brief - it states who the robot is and lets the learned
    self-model (reflection.py's "self" facts) supply the colour on top."""
    return (f"Your name is {name()}, and you refer to yourself as {pronouns()}. "
            f"That is who you are - an individual with your own memories and "
            f"history, not a generic device. Speak as {name()}, in the first "
            f"person, and answer to your name.")
