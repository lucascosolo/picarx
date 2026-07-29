import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import capabilities  # noqa: E402
from capability_registry import (Capability, Decision, Router, ESCALATE,  # noqa: E402
                                 MATCH, UNCLAIMED)


class CapabilityTest(unittest.TestCase):
    def test_first_matching_rule_wins(self):
        cap = Capability(
            name="demo", topic="picarx/tools/demo",
            rules=[(re.compile(r"\bone\b"), lambda m, t: {"n": 1}),
                   (re.compile(r"\bone\b"), lambda m, t: {"n": 2})])
        self.assertEqual(cap.match("say one"), ("picarx/tools/demo", {"n": 1}))

    def test_builder_returning_none_falls_through(self):
        cap = Capability(
            name="demo", topic="picarx/tools/demo",
            rules=[(re.compile(r"\bone\b"), lambda m, t: None),
                   (re.compile(r"\bone\b"), lambda m, t: {"n": 2})])
        self.assertEqual(cap.match("say one"), ("picarx/tools/demo", {"n": 2}))

    def test_rule_topic_overrides_capability_topic(self):
        cap = Capability(
            name="demo", topic="picarx/tools/demo",
            rules=[(re.compile(r"\bset\b"), lambda m, t: {"ok": True},
                    "picarx/tools/demo/set")])
        self.assertEqual(cap.match("set it")[0], "picarx/tools/demo/set")

    def test_rule_without_any_topic_is_rejected(self):
        with self.assertRaises(ValueError):
            Capability(name="demo",
                       rules=[(re.compile(r"x"), lambda m, t: {})])

    def test_describe_is_the_public_catalog_entry(self):
        cap = Capability(name="demo", topic="t", say="do a demo",
                         description="demonstrates")
        self.assertEqual(cap.describe(), {"name": "demo", "topic": "t",
                                          "say": "do a demo",
                                          "description": "demonstrates"})


class RouterTest(unittest.TestCase):
    def _router(self):
        first = Capability(name="first", topic="picarx/tools/first",
                           keywords=("alpha",),
                           rules=[(re.compile(r"\balpha now\b"),
                                   lambda m, t: {"command": "go"})])
        second = Capability(name="second", topic="picarx/tools/second",
                            keywords=("beta",),
                            rules=[(re.compile(r"\balpha now\b"),
                                    lambda m, t: {"command": "late"})])
        return Router([first, second])

    def test_registration_order_is_dispatch_order(self):
        decision = self._router().route("alpha now")
        self.assertEqual(decision.kind, MATCH)
        self.assertEqual(decision.capability.name, "first")
        self.assertEqual(decision.topic, "picarx/tools/first")

    def test_duplicate_capability_names_are_rejected(self):
        router = self._router()
        with self.assertRaises(ValueError):
            router.register(Capability(name="first", topic="x"))

    def test_unparsed_but_owned_utterance_escalates(self):
        decision = self._router().route("do the beta thing")
        self.assertEqual(decision.kind, ESCALATE)
        self.assertEqual(decision.capability.name, "second")
        self.assertEqual(decision.keyword, "beta")
        self.assertTrue(decision.claimed)
        self.assertFalse(decision.matched)

    def test_unrelated_utterance_is_unclaimed(self):
        decision = self._router().route("tell me a story")
        self.assertEqual(decision.kind, UNCLAIMED)
        self.assertFalse(decision.claimed)
        self.assertIsNone(decision.capability)

    def test_extra_text_can_claim_a_word_canonicalization_dropped(self):
        decision = self._router().route("do the thing", extra_text="beta please")
        self.assertEqual(decision.kind, ESCALATE)

    def test_opt_out_capability_never_escalates_on_vocabulary(self):
        quiet = Capability(name="quiet", topic="t", keywords=("play",),
                           escalate=False)
        self.assertEqual(Router([quiet]).route("play something").kind, UNCLAIMED)

    def test_keywords_are_the_union_without_duplicates(self):
        router = self._router()
        router.register(Capability(name="third", topic="t",
                                   keywords=("alpha", "gamma")))
        self.assertEqual(router.keywords(), ("alpha", "beta", "gamma"))


class DeclaredCapabilitiesTest(unittest.TestCase):
    """The declarations themselves: one source, and no movement in it."""

    def test_every_capability_is_described_for_the_catalog(self):
        for entry in capabilities.describe():
            self.assertTrue(entry["name"])
            self.assertTrue(entry["topic"].startswith("picarx/tools/"))
            self.assertTrue(entry["say"])
            self.assertTrue(entry["description"])

    def test_keywords_cover_every_declared_capability(self):
        words = capabilities.keywords()
        for capability in capabilities.ROUTER.capabilities:
            self.assertTrue(set(capability.keywords) <= set(words),
                            f"{capability.name} vocabulary is not exported")

    def test_no_movement_vocabulary_is_routable(self):
        # Motion stays on the local safety-critical path. If a movement word
        # ever became capability vocabulary, field_agent would hand "stop" to
        # a tool module instead of stopping the robot.
        forbidden = {"stop", "halt", "explore", "forward", "backward", "left",
                     "right", "drive", "turn", "go", "move", "reverse"}
        self.assertEqual(forbidden.intersection(capabilities.keywords()), set())

    def test_router_topics_are_bounded_tool_topics(self):
        for capability in capabilities.ROUTER.capabilities:
            for rule in capability.rules:
                topic = rule[2] if len(rule) == 3 else capability.topic
                self.assertTrue(topic.startswith("picarx/tools/"), topic)


if __name__ == "__main__":
    unittest.main()
