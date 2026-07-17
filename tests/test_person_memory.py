import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import person_memory as pm  # noqa: E402


class EnrollParseTest(unittest.TestCase):
    def test_remember_me_i_am(self):
        self.assertEqual(pm.parse_enroll_command("remember me i am lucas"), "lucas")
        self.assertEqual(pm.parse_enroll_command("remember me, i'm alice"), "alice")

    def test_remember_my_face(self):
        self.assertEqual(
            pm.parse_enroll_command("remember my face, my name is bob"), "bob")

    def test_my_name_is(self):
        self.assertEqual(pm.parse_enroll_command("my name is lucas"), "lucas")

    def test_reversed_order(self):
        self.assertEqual(pm.parse_enroll_command("i am alice remember me"), "alice")

    def test_no_enroll_in_ordinary_speech(self):
        self.assertIsNone(pm.parse_enroll_command("remember to buy milk"))
        self.assertIsNone(pm.parse_enroll_command("what's my name"))
        self.assertIsNone(pm.parse_enroll_command("play the radio"))

    def test_filler_word_is_not_a_name(self):
        self.assertIsNone(pm.parse_enroll_command("my name is not"))


class ForgetParseTest(unittest.TestCase):
    def test_forget_name(self):
        self.assertEqual(pm.parse_forget_command("forget lucas"), "lucas")
        self.assertEqual(pm.parse_forget_command("forget about alice"), "alice")

    def test_forget_me(self):
        self.assertEqual(pm.parse_forget_command("forget me"), "me")

    def test_dont_forget_is_not_a_forget(self):
        self.assertIsNone(pm.parse_forget_command("don't forget to charge"))
        self.assertIsNone(pm.parse_forget_command("do not forget me"))

    def test_forget_to_do_something_ignored(self):
        self.assertIsNone(pm.parse_forget_command("you can forget that idea"))
        self.assertIsNone(pm.parse_forget_command("forget it"))


class _FakeRecognizer:
    """Deterministic stand-in for the LBPH wrapper: 'decodes' any payload
    to a token and predicts from a scripted queue."""
    def __init__(self, predictions=(), available=True):
        self.available = available
        self.predictions = list(predictions)
        self.samples = []       # (name, crop) added during enrollment
        self.retrained = 0
        self.forgotten = []

    def decode_crop(self, jpeg_b64):
        return jpeg_b64  # tests pass simple tokens through

    def predict(self, gray):
        return self.predictions.pop(0) if self.predictions else None

    def add_sample(self, name, gray):
        self.samples.append((name, gray))

    def retrain(self):
        self.retrained += 1

    def forget(self, name):
        self.forgotten.append(name)
        return name in ("lucas", "alice")

    def known_names(self):
        return []


class IdentityDebounceTest(unittest.TestCase):
    def _module(self, predictions):
        return pm.PersonMemory(recognizer=_FakeRecognizer(predictions))

    def test_single_prediction_not_published(self):
        m = self._module([("lucas", 40.0)])
        m.on_face_crop({"jpeg": "crop"})
        self.assertEqual(m.bus.of(pm.PERSON_TOPIC), [])

    def test_stable_prediction_published_once(self):
        m = self._module([("lucas", 40.0)] * 3)
        for _ in range(3):
            m.on_face_crop({"jpeg": "crop"})
        published = m.bus.of(pm.PERSON_TOPIC)
        self.assertEqual(len(published), 1)  # re-publish is rate-limited
        self.assertEqual(published[0]["name"], "lucas")

    def test_flicker_between_people_not_published(self):
        m = self._module([("lucas", 40.0), ("alice", 40.0)] * 2)
        for _ in range(4):
            m.on_face_crop({"jpeg": "crop"})
        self.assertEqual(m.bus.of(pm.PERSON_TOPIC), [])

    def test_person_change_publishes_new_name(self):
        m = self._module([("lucas", 40.0)] * 2 + [("alice", 40.0)] * 2)
        for _ in range(4):
            m.on_face_crop({"jpeg": "crop"})
        names = [p["name"] for p in m.bus.of(pm.PERSON_TOPIC)]
        self.assertEqual(names, ["lucas", "alice"])

    def test_unknown_face_publishes_nothing(self):
        m = self._module([None] * 4)
        for _ in range(4):
            m.on_face_crop({"jpeg": "crop"})
        self.assertEqual(m.bus.of(pm.PERSON_TOPIC), [])


class EnrollmentFlowTest(unittest.TestCase):
    def test_enrollment_collects_then_trains_and_confirms(self):
        rec = _FakeRecognizer()
        m = pm.PersonMemory(recognizer=rec)
        m.on_heard({"text": "remember me i am lucas"})
        self.assertIsNotNone(m.enrolling)
        for _ in range(pm.ENROLL_SAMPLES):
            m.on_face_crop({"jpeg": "crop"})
        self.assertIsNone(m.enrolling)
        self.assertEqual(len(rec.samples), pm.ENROLL_SAMPLES)
        self.assertEqual(rec.retrained, 1)
        speech = " ".join(p["text"] for p in m.bus.of(pm.SPEAK_TOPIC))
        self.assertIn("lucas", speech)

    def test_enrollment_times_out(self):
        rec = _FakeRecognizer()
        m = pm.PersonMemory(recognizer=rec)
        m.on_heard({"text": "my name is lucas"})
        m.enrolling["deadline"] = time.time() - 1.0
        m.on_face_crop({"jpeg": "crop"})
        self.assertIsNone(m.enrolling)
        self.assertEqual(rec.retrained, 0)

    def test_unavailable_recognizer_apologizes_once(self):
        m = pm.PersonMemory(recognizer=_FakeRecognizer(available=False))
        m.on_heard({"text": "my name is lucas"})
        m.on_heard({"text": "my name is lucas"})
        self.assertIsNone(m.enrolling)
        apologies = [p for p in m.bus.of(pm.SPEAK_TOPIC)
                     if "face memory" in p["text"]]
        self.assertEqual(len(apologies), 1)

    def test_forget_by_name(self):
        rec = _FakeRecognizer()
        m = pm.PersonMemory(recognizer=rec)
        m.on_heard({"text": "forget lucas"})
        self.assertEqual(rec.forgotten, ["lucas"])
        speech = " ".join(p["text"] for p in m.bus.of(pm.SPEAK_TOPIC))
        self.assertIn("forgotten lucas", speech)

    def test_forget_unknown_person(self):
        rec = _FakeRecognizer()
        m = pm.PersonMemory(recognizer=rec)
        m.on_heard({"text": "forget zebra"})
        speech = " ".join(p["text"] for p in m.bus.of(pm.SPEAK_TOPIC))
        self.assertIn("don't know anyone called zebra", speech)


if __name__ == "__main__":
    unittest.main()
