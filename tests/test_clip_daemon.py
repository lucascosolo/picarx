import base64
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402
sys.path.insert(0, os.path.join(harness.MODULES, "tools"))

from clip_store import ClipStore  # noqa: E402
from clip_daemon import ClipDaemon  # noqa: E402


class FakeCamera:
    def __init__(self):
        self.ensures = 0
        self.releases = 0

    def ensure(self):
        self.ensures += 1

    def release(self):
        self.releases += 1


class ClipDaemonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.camera = FakeCamera()
        self.daemon = ClipDaemon(store=ClipStore(self.tmp.name),
                                 camera=self.camera)
        self.daemon.on_robot_state({"state": "IDLE", "owner": "robot_state"})

    def tearDown(self):
        self.tmp.cleanup()

    def test_video_capture_requires_consent_and_uses_camera_subscription(self):
        denied = self.daemon.on_control({"command": "capture", "kind": "video"})
        self.assertFalse(denied["ok"])
        pending = self.daemon.on_control({"command": "capture", "kind": "video",
                                          "duration_sec": 1,
                                          "confirmed": True, "request_id": "v1"})
        self.assertTrue(pending["ok"])
        self.assertTrue(pending["result"]["pending"])
        self.assertGreater(self.camera.ensures, 0)
        jpeg = base64.b64encode(b"\xff\xd8fake\xff\xd9").decode("ascii")
        self.daemon.on_frame({"jpeg": jpeg})
        self.daemon._finish_video()
        result = self.daemon.bus.last("picarx/tools/clip/result")
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["kind"], "video")
        self.assertTrue(self.daemon.store.list())

    def test_capture_is_interrupted_by_safety_state(self):
        self.daemon.on_control({"command": "capture", "kind": "video",
                                "confirmed": True, "request_id": "v2"})
        self.daemon.on_robot_state({"state": "SAFETY_STOP", "owner": "safety"})
        self.assertIsNone(self.daemon._video)
        self.assertEqual(self.daemon.store.list(), [])

    def test_list_delete_and_invalid_playback_are_bounded(self):
        row = self.daemon.store.begin("video", 1)
        with open(row["temporary_path"], "wb") as stream:
            stream.write(b"jpeg")
        self.daemon.store.finalize(row, 1)
        listed = self.daemon.on_control({"command": "list", "request_id": "l1"})
        self.assertEqual(len(listed["result"]["clips"]), 1)
        missing = self.daemon.on_control({"command": "play", "id": "bad"})
        self.assertFalse(missing["ok"])
        denied = self.daemon.on_control({"command": "delete", "id": row["id"]})
        self.assertFalse(denied["ok"])
        deleted = self.daemon.on_control({"command": "delete", "id": row["id"],
                                          "confirmed": True})
        self.assertTrue(deleted["ok"])

    def test_audio_capture_is_forwarded_without_opening_a_second_mic(self):
        pending = self.daemon.on_control({"command": "capture", "kind": "audio",
                                          "duration_sec": 1, "confirmed": True,
                                          "request_id": "a1"})
        self.assertTrue(pending["ok"])
        request = self.daemon.bus.last("picarx/tools/clip/audio")
        self.assertEqual(request["command"], "capture")
        self.assertEqual(request["request_id"], "a1")
        self.daemon.on_audio_result({"ok": True, "command": "capture",
                                     "request_id": "a1", "result": {
                                         "id": "abc", "kind": "audio"}})
        self.assertIsNone(self.daemon._audio)


if __name__ == "__main__":
    unittest.main()
