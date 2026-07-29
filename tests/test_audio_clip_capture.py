import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import audio_nodes  # noqa: E402
from clip_store import ClipStore  # noqa: E402


class AudioClipCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.node = audio_nodes.AudioNode.__new__(audio_nodes.AudioNode)
        self.node.bus = harness.FakeBus()
        self.node._clip_lock = threading.RLock()
        self.node._audio_clip = None
        self.node._clip_store = ClipStore(self.tmp.name, max_duration=5,
                                          max_clip_bytes=10000)
        self.node._audio_stream_ready = True
        self.node.mic_enabled = True

    def tearDown(self):
        self.tmp.cleanup()

    def test_capture_uses_existing_pcm_stream_and_finalizes_wav(self):
        pending = self.node.on_clip_control({
            "command": "capture", "confirmed": True,
            "duration_sec": 1, "request_id": "a1"})
        self.assertTrue(pending["ok"])
        self.assertTrue(pending["result"]["pending"])
        self.node._write_audio_clip(b"\x00\x01" * 100, now=time.time() + 2)
        result = self.node.bus.last(audio_nodes.AUDIO_CLIP_RESULT_TOPIC)
        self.assertTrue(result["ok"])
        row = result["result"]
        self.assertEqual(row["kind"], "audio")
        with open(self.node._clip_store.path(row["id"]), "rb") as stream:
            self.assertEqual(stream.read(4), b"RIFF")

    def test_capture_requires_consent_and_stop_is_interruptible(self):
        denied = self.node.on_clip_control({"command": "capture",
                                             "duration_sec": 1})
        self.assertFalse(denied["ok"])
        self.node.on_clip_control({"command": "capture", "confirmed": True,
                                   "duration_sec": 5, "request_id": "a2"})
        stopped = self.node.on_clip_control({"command": "stop",
                                             "request_id": "a3"})
        self.assertFalse(stopped["ok"])
        self.assertIsNone(self.node._audio_clip)
        self.assertEqual(self.node._clip_store.list(), [])

    def test_unavailable_mic_does_not_create_clip(self):
        self.node._audio_stream_ready = False
        result = self.node.on_clip_control({"command": "capture",
                                            "confirmed": True})
        self.assertFalse(result["ok"])
        self.assertEqual(self.node._clip_store.list(), [])


if __name__ == "__main__":
    unittest.main()
