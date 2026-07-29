import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402,F401

from clip_store import ClipError, ClipStore  # noqa: E402


class ClipStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.now = [100.0]
        self.store = ClipStore(self.tmp.name, max_duration=5,
                               max_clip_bytes=10, max_clips=2,
                               max_total_bytes=15, clock=lambda: self.now[0])

    def tearDown(self):
        self.tmp.cleanup()

    def _finish(self, kind="audio", data=b"clip"):
        reservation = self.store.begin(kind, 1.25)
        with open(reservation["temporary_path"], "ab") as stream:
            stream.write(data)
        self.now[0] += 1
        return self.store.finalize(reservation)

    def test_atomic_finalize_and_restart_listing(self):
        row = self._finish("audio", b"hello")
        self.assertEqual(row["kind"], "audio")
        self.assertEqual(row["bytes"], 5)
        self.assertTrue(self.store.path(row["id"]).endswith(".wav"))
        self.assertEqual(self.store.list()[0]["id"], row["id"])
        restarted = ClipStore(self.tmp.name, clock=lambda: self.now[0])
        self.assertEqual(restarted.list()[0]["id"], row["id"])
        self.assertFalse(any(name.endswith(".part")
                             for name in os.listdir(self.tmp.name)))

    def test_video_uses_mjpeg_extension_and_metadata_is_bounded(self):
        row = self._finish("video", b"jpeg")
        self.assertTrue(self.store.path(row["id"]).endswith(".mjpeg"))
        with open(os.path.join(self.tmp.name, row["id"] + ".json"),
                  encoding="utf-8") as stream:
            self.assertNotIn("jpeg", stream.read())

    def test_limits_reject_duration_size_count_and_total(self):
        with self.assertRaises(ClipError):
            self.store.begin("audio", 6)
        with self.assertRaises(ClipError):
            self._finish(data=b"01234567890")
        self._finish(data=b"12345678")
        with self.assertRaises(ClipError):
            self._finish(data=b"12345678")  # total limit, not silent eviction
        # Count limit is independently enforced when enough room is available.
        other = ClipStore(self.tmp.name + "-count", max_clip_bytes=20,
                          max_clips=1, max_total_bytes=40)
        try:
            reservation = other.begin("audio", 1)
            with open(reservation["temporary_path"], "wb") as stream:
                stream.write(b"x")
            other.finalize(reservation)
            with self.assertRaises(ClipError):
                other.begin("audio", 1)
        finally:
            other_root = other.root
            other = None
            import shutil
            shutil.rmtree(other_root, ignore_errors=True)

    def test_abort_and_restart_remove_incomplete_capture(self):
        reservation = self.store.begin("video", 1)
        self.assertTrue(os.path.exists(reservation["temporary_path"]))
        self.store.abort(reservation)
        self.assertFalse(os.path.exists(reservation["temporary_path"]))
        reservation = self.store.begin("audio", 1)
        restarted = ClipStore(self.tmp.name)
        self.assertFalse(os.path.exists(reservation["temporary_path"]))
        restarted.cleanup_incomplete()

    def test_delete_requires_generated_id_and_removes_media_and_metadata(self):
        row = self._finish()
        deleted = self.store.delete(row["id"])
        self.assertEqual(deleted["id"], row["id"])
        self.assertEqual(self.store.list(), [])
        with self.assertRaises(ClipError):
            self.store.delete("../escape")


if __name__ == "__main__":
    unittest.main()
