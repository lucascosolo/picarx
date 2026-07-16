import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import audio_nodes  # noqa: E402


class _FakePopen:
    """Stand-in for the espeak subprocess.Popen in _speak_espeak."""
    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.stdout = types.SimpleNamespace(close=lambda: None)
        self.waited = False

    def wait(self):
        self.waited = True


def _bare_node():
    """AudioNode without its heavy __init__; wire only the TTS fields."""
    import queue
    node = audio_nodes.AudioNode.__new__(audio_nodes.AudioNode)
    node.mute_until = 0.0
    node.kokoro = None
    node._tts_queue = queue.Queue()
    node._tts_worker_started = False
    node._last_amp_assert_at = 0.0
    return node


class TtsFallbackLadderTest(unittest.TestCase):
    def setUp(self):
        self.node = _bare_node()
        self._popen_calls = []
        self._run_calls = []
        self._orig_popen = audio_nodes.subprocess.Popen
        self._orig_run = audio_nodes.subprocess.run

        def fake_popen(argv, **kw):
            p = _FakePopen(argv, **kw)
            self._popen_calls.append(p)
            return p

        def fake_run(argv, **kw):
            self._run_calls.append(argv)
            return None

        audio_nodes.subprocess.Popen = fake_popen
        audio_nodes.subprocess.run = fake_run

    def tearDown(self):
        audio_nodes.subprocess.Popen = self._orig_popen
        audio_nodes.subprocess.run = self._orig_run
        sys.modules.pop("sounddevice", None)

    def _espeak_used(self):
        return any(p.argv[:1] == ["espeak"] for p in self._popen_calls)

    # ---- fallback: no Kokoro -> espeak ----

    def test_falls_back_to_espeak_when_kokoro_absent(self):
        self.node.kokoro = None
        self.node._render_and_play("hello there")
        self.assertTrue(self._espeak_used())
        self.assertEqual(self._popen_calls[0].argv, ["espeak", "--stdout", "hello there"])
        # mic-mute is released (finite) once playback is done.
        self.assertNotEqual(self.node.mute_until, float("inf"))

    # ---- fallback: Kokoro runtime error mid-synthesis -> espeak ----

    def test_kokoro_runtime_error_falls_back(self):
        class BoomKokoro:
            def create(self, *a, **k):
                raise RuntimeError("onnx blew up")
        self.node.kokoro = BoomKokoro()
        self.node._render_and_play("still talking")
        self.assertTrue(self._espeak_used())  # seamlessly used espeak

    # ---- happy path: Kokoro synthesizes, sounddevice plays, no espeak ----

    def test_kokoro_success_uses_sounddevice_not_espeak(self):
        played = {}
        fake_sd = types.ModuleType("sounddevice")
        fake_sd.play = lambda samples, rate, **kw: played.update(samples=samples, rate=rate)
        fake_sd.wait = lambda: played.update(waited=True)
        sys.modules["sounddevice"] = fake_sd

        class GoodKokoro:
            def create(self, text, voice=None, speed=None, lang=None):
                return ([0.0, 0.1, -0.1], 24000)
        self.node.kokoro = GoodKokoro()

        self.node._render_and_play("neural voice")
        self.assertEqual(played.get("rate"), 24000)
        self.assertTrue(played.get("waited"))
        self.assertFalse(self._espeak_used())   # Kokoro path, espeak untouched

    # ---- both engines down -> no crash, mute still released ----

    def test_both_engines_fail_does_not_crash(self):
        self.node.kokoro = None

        def boom(argv, **kw):
            raise FileNotFoundError("no espeak")
        audio_nodes.subprocess.Popen = boom
        self.node._render_and_play("silence")   # must not raise
        self.assertNotEqual(self.node.mute_until, float("inf"))


class AmpReassertTest(unittest.TestCase):
    """The amp is re-asserted right before playback (throttled), so a HAT
    init resetting the GPIO after the boot burst can't leave the robot mute."""

    def setUp(self):
        self.node = _bare_node()
        self._orig_popen = audio_nodes.subprocess.Popen
        self._orig_run = audio_nodes.subprocess.run
        self.run_calls = []
        audio_nodes.subprocess.Popen = lambda argv, **kw: _FakePopen(argv, **kw)
        audio_nodes.subprocess.run = lambda argv, **kw: self.run_calls.append(argv)

    def tearDown(self):
        audio_nodes.subprocess.Popen = self._orig_popen
        audio_nodes.subprocess.run = self._orig_run

    def _enable_calls(self):
        enable_argv = audio_nodes.SPEAKER_ENABLE_CMD.split()
        return [c for c in self.run_calls if c == enable_argv]

    def test_amp_reasserted_before_utterance(self):
        self.node._render_and_play("hello")
        self.assertEqual(len(self._enable_calls()), 1)

    def test_reassert_throttled_within_interval(self):
        # Two lines back to back: the amp was just asserted, don't pay the
        # robot_hat CLI spawn again.
        self.node._render_and_play("line one")
        self.node._render_and_play("line two")
        self.assertEqual(len(self._enable_calls()), 1)

    def test_reassert_fires_again_after_interval(self):
        self.node._render_and_play("early line")
        # Simulate the throttle window passing (e.g. a reset happened since).
        self.node._last_amp_assert_at -= (audio_nodes.SPEAKER_REASSERT_INTERVAL + 1)
        self.node._render_and_play("later line")
        self.assertEqual(len(self._enable_calls()), 2)


class TtsQueueTest(unittest.TestCase):
    def setUp(self):
        self.node = _bare_node()
        # Don't spin the real playback thread - we assert on queue contents,
        # so a background worker draining it would race. Mark it "started"
        # without launching a thread.
        self.node._ensure_tts_worker = lambda: setattr(
            self.node, "_tts_worker_started", True)

    def test_speak_is_nonblocking_enqueue(self):
        self.node.speak("hi")
        self.assertEqual(self.node._tts_queue.qsize(), 1)
        self.assertTrue(self.node._tts_worker_started)

    def test_handle_speak_request_drops_stale_at_enqueue(self):
        self.node.handle_speak_request(
            {"text": "old news", "ts": 0.0})   # ancient timestamp
        self.assertEqual(self.node._tts_queue.qsize(), 0)

    def test_handle_speak_request_enqueues_fresh(self):
        import time
        self.node.handle_speak_request({"text": "fresh", "ts": time.time()})
        self.assertEqual(self.node._tts_queue.qsize(), 1)

    def test_worker_drops_stale_at_dequeue(self):
        rendered = []
        self.node._render_and_play = lambda t: rendered.append(t)
        # Put one stale, one fresh; run one worker pass over each manually.
        self.node._tts_queue.put(("stale", 0.0))
        self.node._tts_queue.put(("fresh", None))

        # Drain exactly two items through the worker's body without the
        # infinite loop: replicate its dequeue+staleness+render steps.
        import time
        for _ in range(2):
            text, ts = self.node._tts_queue.get()
            if ts is not None and (time.time() - ts) > self.node.SPEAK_MAX_AGE_SEC:
                continue
            self.node._render_and_play(text)
        self.assertEqual(rendered, ["fresh"])


if __name__ == "__main__":
    unittest.main()
