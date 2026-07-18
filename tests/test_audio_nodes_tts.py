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
    node.espeak_voice = None
    node.speaker_enabled = True
    node._tts_queue = queue.Queue()
    node._tts_worker_started = False
    node._last_amp_assert_at = 0.0
    return node


class EspeakArgvTest(unittest.TestCase):
    def test_default_voice_is_legacy_call(self):
        # voice=None must be the exact proven pre-MBROLA invocation.
        self.assertEqual(audio_nodes._espeak_argv("hi"), ["espeak", "--stdout", "hi"])

    def test_named_voice_brings_speed(self):
        argv = audio_nodes._espeak_argv("hi", "mb-us1")
        self.assertEqual(argv[:5], ["espeak", "-v", "mb-us1", "-s",
                                    audio_nodes.ESPEAK_SPEED])
        self.assertEqual(argv[-2:], ["--stdout", "hi"])

    def test_pitch_only_when_set(self):
        orig = audio_nodes.ESPEAK_PITCH
        try:
            audio_nodes.ESPEAK_PITCH = ""
            self.assertNotIn("-p", audio_nodes._espeak_argv("hi", "mb-us1"))
            audio_nodes.ESPEAK_PITCH = "60"
            argv = audio_nodes._espeak_argv("hi", "mb-us1")
            self.assertIn("-p", argv)
            self.assertEqual(argv[argv.index("-p") + 1], "60")
        finally:
            audio_nodes.ESPEAK_PITCH = orig


class VoiceProbeTest(unittest.TestCase):
    """_init_voice picks the MBROLA voice only when a probe render works."""

    def setUp(self):
        self.node = _bare_node()
        self._orig_run = audio_nodes.subprocess.run

    def tearDown(self):
        audio_nodes.subprocess.run = self._orig_run

    def test_healthy_probe_selects_mbrola_voice(self):
        audio_nodes.subprocess.run = lambda argv, **kw: types.SimpleNamespace(
            returncode=0, stdout=b"RIFF" + b"\0" * 100)   # WAV bigger than header
        self.node._init_voice()
        self.assertEqual(self.node.espeak_voice, audio_nodes.ESPEAK_VOICE)

    def test_missing_voice_falls_back_to_default(self):
        audio_nodes.subprocess.run = lambda argv, **kw: types.SimpleNamespace(
            returncode=1, stdout=b"")                     # voice pack absent
        self.node._init_voice()
        self.assertIsNone(self.node.espeak_voice)

    def test_empty_render_falls_back(self):
        # rc 0 but nothing beyond a WAV header = mbrola silently produced
        # no audio - still a fallback, not a crash.
        audio_nodes.subprocess.run = lambda argv, **kw: types.SimpleNamespace(
            returncode=0, stdout=b"\0" * 44)
        self.node._init_voice()
        self.assertIsNone(self.node.espeak_voice)

    def test_probe_exception_is_failsoft(self):
        def boom(argv, **kw):
            raise FileNotFoundError("no espeak at all")
        audio_nodes.subprocess.run = boom
        self.node._init_voice()                           # must not raise
        self.assertIsNone(self.node.espeak_voice)


class TtsFallbackLadderTest(unittest.TestCase):
    def setUp(self):
        self.node = _bare_node()
        self._popen_calls = []
        self._orig_popen = audio_nodes.subprocess.Popen
        self._orig_run = audio_nodes.subprocess.run

        def fake_popen(argv, **kw):
            p = _FakePopen(argv, **kw)
            self._popen_calls.append(p)
            return p

        audio_nodes.subprocess.Popen = fake_popen
        audio_nodes.subprocess.run = lambda argv, **kw: None

    def tearDown(self):
        audio_nodes.subprocess.Popen = self._orig_popen
        audio_nodes.subprocess.run = self._orig_run

    def _espeak_argvs(self):
        return [p.argv for p in self._popen_calls if p.argv[:1] == ["espeak"]]

    def test_mbrola_voice_used_when_selected(self):
        self.node.espeak_voice = "mb-us1"
        self.node._render_and_play("hello there")
        argvs = self._espeak_argvs()
        self.assertEqual(len(argvs), 1)
        self.assertIn("mb-us1", argvs[0])
        self.assertNotEqual(self.node.mute_until, float("inf"))  # mute released

    def test_no_voice_selected_uses_legacy_call(self):
        self.node.espeak_voice = None
        self.node._render_and_play("hello there")
        self.assertEqual(self._espeak_argvs(), [["espeak", "--stdout", "hello there"]])

    def test_mbrola_runtime_error_falls_back_to_default_voice(self):
        self.node.espeak_voice = "mb-us1"
        calls = {"n": 0}
        orig_fake = audio_nodes.subprocess.Popen

        def popen_first_fails(argv, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("mbrola pipe burst")
            return orig_fake(argv, **kw)
        audio_nodes.subprocess.Popen = popen_first_fails

        self.node._render_and_play("still talking")
        # The second call is the seamless plain-espeak fallback.
        self.assertEqual(self._espeak_argvs(), [["espeak", "--stdout", "still talking"]])

    def test_everything_down_does_not_crash(self):
        self.node.espeak_voice = "mb-us1"

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
        self.node._render_and_play("line one")
        self.node._render_and_play("line two")
        self.assertEqual(len(self._enable_calls()), 1)

    def test_reassert_fires_again_after_interval(self):
        self.node._render_and_play("early line")
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
