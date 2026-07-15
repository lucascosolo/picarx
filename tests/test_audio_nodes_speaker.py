import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import audio_nodes  # noqa: E402


class SpeakerEnableTest(unittest.TestCase):
    def setUp(self):
        # Build an AudioNode without its heavy __init__ (audio/model setup).
        self.node = audio_nodes.AudioNode.__new__(audio_nodes.AudioNode)
        self._orig = (audio_nodes.SPEAKER_ENABLE_RETRIES,
                      audio_nodes.SPEAKER_ENABLE_INTERVAL)

    def tearDown(self):
        (audio_nodes.SPEAKER_ENABLE_RETRIES,
         audio_nodes.SPEAKER_ENABLE_INTERVAL) = self._orig

    def test_enable_once_runs_configured_command(self):
        recorded = {}

        def fake_run(argv, **kwargs):
            recorded["argv"] = argv
            return None
        orig = audio_nodes.subprocess.run
        audio_nodes.subprocess.run = fake_run
        try:
            self.assertTrue(self.node._enable_speakers_once())
        finally:
            audio_nodes.subprocess.run = orig
        self.assertEqual(recorded["argv"], audio_nodes.SPEAKER_ENABLE_CMD.split())

    def test_enable_once_failsoft_on_missing_binary(self):
        def boom(argv, **kwargs):
            raise FileNotFoundError("no robot_hat")
        orig = audio_nodes.subprocess.run
        audio_nodes.subprocess.run = boom
        try:
            self.assertFalse(self.node._enable_speakers_once())  # no raise
        finally:
            audio_nodes.subprocess.run = orig

    def test_enable_retries_across_boot_window(self):
        # This is the fix: a single enable races with the HAT init, so it
        # must re-assert several times.
        audio_nodes.SPEAKER_ENABLE_RETRIES = 4
        audio_nodes.SPEAKER_ENABLE_INTERVAL = 0.0
        calls = {"n": 0}
        self.node._enable_speakers_once = lambda: (calls.__setitem__("n", calls["n"] + 1)
                                                   or True)
        thread = self.node._enable_speakers()
        thread.join(timeout=5)
        self.assertEqual(calls["n"], 4)

    def test_stops_retrying_when_binary_absent(self):
        # If the command isn't even available, retrying can't help - stop.
        audio_nodes.SPEAKER_ENABLE_RETRIES = 8
        audio_nodes.SPEAKER_ENABLE_INTERVAL = 0.0
        calls = {"n": 0}
        self.node._enable_speakers_once = lambda: (calls.__setitem__("n", calls["n"] + 1)
                                                   and False)
        thread = self.node._enable_speakers()
        thread.join(timeout=5)
        self.assertEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
