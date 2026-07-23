"""The audio_nodes speech gate: VadGate (webrtcvad-backed) behaviour and the
fail-soft selection that falls back to the adaptive EnergyGate when webrtcvad
isn't installed (which is exactly the off-robot / un-pip'd case)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness  # noqa: E402

import audio_nodes  # noqa: E402

FRAME = b"\x10\x00" * 480   # one 30ms/960-byte frame at 16kHz/16-bit


class _FakeVad:
    """webrtcvad stand-in: is_speech returns whatever we tell it to."""
    def __init__(self, verdict=True):
        self.verdict = verdict

    def is_speech(self, frame, rate):
        return self.verdict


class MakeGateFallbackTest(unittest.TestCase):
    def test_falls_back_to_energy_gate_without_webrtcvad(self):
        # webrtcvad isn't installed off-robot, so even with VAD enabled the
        # module must come up on the energy gate rather than failing.
        self.assertTrue(audio_nodes.VAD_ENABLED)
        gate = audio_nodes._make_gate()
        self.assertIsInstance(gate, audio_nodes.EnergyGate)

    def test_disabled_uses_energy_gate(self):
        orig = audio_nodes.VAD_ENABLED
        audio_nodes.VAD_ENABLED = False
        try:
            self.assertIsInstance(audio_nodes._make_gate(), audio_nodes.EnergyGate)
        finally:
            audio_nodes.VAD_ENABLED = orig


class VadGateTest(unittest.TestCase):
    def test_voiced_frame_opens_gate(self):
        g = audio_nodes.VadGate(_FakeVad(True))
        self.assertTrue(g.process(FRAME, rms=500, now=100.0))

    def test_gate_holds_through_hangover_then_closes(self):
        g = audio_nodes.VadGate(_FakeVad(True))
        g.process(FRAME, rms=500, now=100.0)               # opens
        g.vad.verdict = False
        # still within the trailing-silence hangover -> stays open
        self.assertTrue(g.process(FRAME, rms=50,
                                  now=100.0 + audio_nodes.TRAILING_SILENCE_SEC - 0.1))
        # past the hangover -> closes
        self.assertFalse(g.process(FRAME, rms=50,
                                   now=100.0 + audio_nodes.TRAILING_SILENCE_SEC + 0.5))

    def test_silence_keeps_gate_closed(self):
        g = audio_nodes.VadGate(_FakeVad(False))
        self.assertFalse(g.process(FRAME, rms=40, now=100.0))

    def test_partial_frames_are_buffered_across_chunks(self):
        g = audio_nodes.VadGate(_FakeVad(True))
        half = b"\x10\x00" * 240   # 480 bytes - not a whole 960-byte frame yet
        self.assertFalse(g.process(half, rms=500, now=100.0))   # nothing to judge
        self.assertTrue(g.process(half, rms=500, now=100.1))    # completes the frame

    def test_tracks_a_noise_floor_for_the_snr_check(self):
        # The SNR reject downstream reads gate.noise_floor regardless of gate
        # type, so VadGate must keep one populated.
        g = audio_nodes.VadGate(_FakeVad(False))
        g.process(FRAME, rms=120, now=100.0)
        self.assertIsNotNone(g.noise_floor)


if __name__ == "__main__":
    unittest.main()
