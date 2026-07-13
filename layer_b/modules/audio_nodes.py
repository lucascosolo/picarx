#!/usr/bin/env python3
# /home/picarx/layer_b/modules/audio_nodes.py
"""
CPU footprint (Pi 4, running alongside vision_basic.py's SSD detector)
-----------------------------------------------------------------------
The previous version fed every single captured chunk into
rec.AcceptWaveform() unconditionally, all day, whether or not anyone
was talking - Vosk's acoustic model + decoder runs on every chunk it's
given, silence included, so that's continuous ~100% CPU on one core
even in a silent room. That's the actual reason voice recognition was
sluggish and inaccurate: it wasn't behind because the model was slow,
it was behind because it was constantly decoding silence and had no
CPU headroom left for real speech.

This version adds a cheap energy gate before Kaldi ever sees a chunk:
compute a chunk's RMS amplitude (a handful of int multiplications -
negligible cost) and only call AcceptWaveform() while that's above the
ambient noise floor by NOISE_MULTIPLIER, plus a trailing window
(TRAILING_SILENCE_SEC) afterward so an in-progress utterance's natural
word-to-word pauses don't get cut off, and so Kaldi still gets the bit
of trailing silence its own endpoint rules expect before finalizing a
result. During confirmed extended silence, no decoding happens at all.

The trigger is ADAPTIVE, not a fixed number: a slow-moving estimate of
the ambient noise floor is tracked continuously (only updated during
confirmed silence, so speech itself can never drag the floor upward),
and the actual trigger threshold is recomputed from that floor every
chunk. A fixed absolute threshold turned out to be a losing game -
every change to mic gain, room noise, or hardware meant re-guessing a
magic number from scratch, and if the true noise floor sits close to
speech level (a low-SNR mic - see AUDIO_GAIN below), no single fixed
number cleanly separates the two anyway. This won't fix a mic that's
fundamentally too insensitive to hear you clearly, but it does mean
the gate keeps tracking real conditions instead of a stale guess.

A short PREBUFFER is also kept during silence (a few hundred ms) and
flushed into the recognizer the instant the gate triggers, so the very
start of an utterance - which is often quieter than the rest and might
not immediately clear the threshold - doesn't get chopped off. That
chopped-onset effect is what "reacts slowly, doesn't hear the start of
what I said" usually is.

Run with AUDIO_DEBUG_LEVELS=1 in the environment to print live
rms/floor/threshold readings if you need to sanity check it's tracking
sensibly for your room.

Some USB mics (especially cheap ones with no ALSA-exposed mixer
control at all - `alsamixer` reporting "This device does not have any
controls" is exactly that) capture at a fixed, very low hardware gain
with no way to raise it at the ALSA level. AUDIO_GAIN below digitally
amplifies every captured chunk before it reaches either the energy
gate or Kaldi, to compensate. It's not as good as a real hardware
preamp (it amplifies the noise floor right along with the signal), but
it's the only lever available when there's no mixer control to turn.
"""
import os
import getpass
os.getlogin = getpass.getuser

# Cap BLAS/OpenMP-level threading before vosk (and whatever numeric
# libraries it's built on) is imported, so this process leaves real
# headroom for vision_basic.py's SSD detector instead of grabbing
# every core it can.
THREAD_LIMIT = 2
os.environ.setdefault("OMP_NUM_THREADS", str(THREAD_LIMIT))
os.environ.setdefault("OPENBLAS_NUM_THREADS", str(THREAD_LIMIT))

import sys
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus

import array
import time
import json
import subprocess
from collections import deque
from vosk import Model, KaldiRecognizer

try:
    import audioop  # stdlib; removed in 3.13+, hence the fallback below
except ImportError:
    audioop = None

# Upgraded from the original small model ("model-en", still present on
# disk for rollback) to Vosk's lgraph model: same engine/pipeline, a
# meaningfully more accurate decoding graph. Its conf/model.conf has
# also been narrowed (lower max-active/beam, much lower lattice-beam -
# we only ever take the 1-best result, so a wide lattice-beam is pure
# wasted computation for this pipeline) to cut per-chunk decode cost.
MODEL_PATH = "/home/picarx/layer_b/modules/models/model-en-lgraph"

CHUNK_BYTES = 4000              # ~125ms per chunk at 16kHz/16-bit/mono

# The local "Stopping" reflex below used to fire on every single
# recognized utterance containing "stop"/"halt", with no rate limit at
# all - unlike every other announcement in this system. TTS playback
# is also blocking (aplay has to finish before the next one can play),
# so repeating "stop" a few times in frustration queued up that many
# blocking announcements, which could only play back-to-back - that's
# the "Stopping! Stopping! Stopping!" pileup and the perceived delay.
# The actual stop command (cancel the intent) goes through a separate,
# unthrottled path in field_agent.py and isn't affected by this at all -
# this cooldown only throttles the audible confirmation.
STOP_REFLEX_COOLDOWN = 4.0

# --- adaptive energy gate tuning ---
NOISE_FLOOR_ALPHA = 0.05        # how fast the ambient-floor estimate adapts (per chunk)
NOISE_MULTIPLIER = 2.2          # trigger this many times above the tracked floor
MIN_THRESHOLD = 60              # absolute floor, in case ambient is near-silent
TRAILING_SILENCE_SEC = 1.2      # keep decoding this long after the last loud chunk
PREBUFFER_CHUNKS = 3            # ~375ms kept during silence so speech onset isn't chopped
DEBUG_LEVELS = bool(os.environ.get("AUDIO_DEBUG_LEVELS"))

# Digital gain applied to every captured chunk before anything else
# sees it - see the module docstring for why this exists. 1.0 = no
# change. Tune with AUDIO_DEBUG_LEVELS=1 alongside SILENCE_RMS_THRESHOLD.
AUDIO_GAIN = float(os.environ.get("AUDIO_GAIN", "12.0"))


def _apply_gain(data, gain):
    if gain == 1.0 or not data:
        return data
    if audioop is not None:
        try:
            return audioop.mul(data, 2, gain)  # width=2 bytes (16-bit), clips automatically
        except audioop.error:
            pass
    # Fallback (or if audioop rejected the input for some reason):
    # same idea in pure Python, manually clipped to int16 range.
    samples = array.array("h", data)
    boosted = array.array("h", (max(-32768, min(32767, int(s * gain))) for s in samples))
    return boosted.tobytes()


def _chunk_rms(data):
    """Cheap energy estimate for a raw 16-bit PCM chunk - a handful of
    int multiply-adds, nowhere near the cost of even one decode step."""
    if not data:
        return 0.0
    samples = array.array("h", data)
    return (sum(s * s for s in samples) / len(samples)) ** 0.5


class AudioNode:
    def __init__(self):
        self.bus = Bus()

        # TTS: shell out to espeak directly per call, rather than using
        # pyttsx3. pyttsx3's espeak driver has a known issue on Linux
        # where the engine's internal event loop only reliably runs
        # once per process - a second engine.say()/runAndWait() call
        # silently does nothing (no exception), which is why speech
        # worked once at startup and then went silent for every
        # subsequent picarx/audio/speak message. Calling espeak fresh
        # each time avoids that class of bug entirely.

        # Initialize local STT Model
        if not os.path.exists(MODEL_PATH):
            print(f"Vosk model not found at {MODEL_PATH}. Speech recognition disabled.")
            self.model = None
        else:
            self.model = Model(MODEL_PATH)
            self.rec = KaldiRecognizer(self.model, 16000)

    # Bypass ALSA's "default" device resolution (which normally hands
    # off to PulseAudio) and target the physical sound card directly.
    # A systemd service has no PulseAudio session to connect to, so
    # anything routed through "default" silently fails there even
    # though it works fine from an interactive login shell. Card index
    # from `aplay -l`: card 0 is the HifiBerry DAC HAT (the real
    # speaker output); cards 2/3 are just the Pi's HDMI outputs.
    AUDIO_OUT_DEVICE = "plughw:0,0"

    def speak(self, text):
        print(f"PiCar Speaking: {text}")
        espeak_proc = subprocess.Popen(["espeak", "--stdout", text], stdout=subprocess.PIPE)
        subprocess.run(
            ["aplay", "-D", self.AUDIO_OUT_DEVICE, "-q"],
            stdin=espeak_proc.stdout,
            check=False,
        )
        espeak_proc.stdout.close()
        espeak_proc.wait()

    def handle_speak_request(self, payload):
        text = payload.get("text", "")
        if text:
            self.speak(text)

    def run(self):
        self.bus.subscribe("picarx/audio/speak", self.handle_speak_request)

        if not self.model:
            print("Audio node running in output-only mode.")
            while True:
                time.sleep(1)

        print("Audio node running: starting ALSA audio stream...")
        self.speak("Systems initialized. Voice control active.")

        # Use arecord with plughw to automatically resample the USB mic to 16000Hz.
        # Card index from `arecord -l`: card 1 is the USB PnP Sound Device (the mic).
        cmd = [
            "arecord",
            "-D", "plughw:1,0",
            "-f", "S16_LE",
            "-c", "1",
            "-r", "16000",
            "-t", "raw",
            "-q" # Quiet mode, no standard output text
        ]

        # Start the microphone stream directly from the OS
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        speaking_until = 0.0    # keep decoding until this timestamp
        noise_floor = None      # bootstrapped from the first chunk we see
        prebuffer = deque(maxlen=PREBUFFER_CHUNKS)
        last_stop_reflex_at = 0.0

        while True:
            # Read a chunk at a time from the arecord output
            data = process.stdout.read(CHUNK_BYTES)
            if len(data) == 0:
                print("Audio node: arecord produced no data (device open failure or process exit) - STT is now dead until this module restarts")
                break

            data = _apply_gain(data, AUDIO_GAIN)

            now = time.time()
            rms = _chunk_rms(data)
            if noise_floor is None:
                noise_floor = rms
            threshold = max(MIN_THRESHOLD, noise_floor * NOISE_MULTIPLIER)

            if rms > threshold:
                speaking_until = now + TRAILING_SILENCE_SEC
            elif now > speaking_until:
                # Only drift the floor estimate during confirmed
                # silence, so speech itself can never drag it upward.
                noise_floor = noise_floor * (1 - NOISE_FLOOR_ALPHA) + rms * NOISE_FLOOR_ALPHA

            if DEBUG_LEVELS:
                print(f"Audio node: chunk rms={rms:.0f} floor={noise_floor:.0f} threshold={threshold:.0f}")

            if now > speaking_until:
                # Confirmed silence - skip Kaldi entirely for this chunk
                # (the actual CPU win: no acoustic model, no decoder
                # search, nothing, until sound picks up again), but
                # keep a short rolling buffer so if speech starts on
                # the very next chunk, we don't lose the onset.
                prebuffer.append(data)
                continue

            # Just triggered (or still within the trailing window) -
            # flush any pre-buffered quiet-adjacent audio first so the
            # start of the utterance isn't chopped off, then feed this
            # chunk normally.
            if prebuffer:
                for buffered in prebuffer:
                    self.rec.AcceptWaveform(buffered)
                prebuffer.clear()

            if self.rec.AcceptWaveform(data):
                result = json.loads(self.rec.Result())
                text = result.get("text", "").lower()

                if text:
                    print(f"Heard locally: '{text}'")
                    # Publish what was heard to the MQTT bus
                    self.bus.publish("picarx/audio/heard", {"text": text})

                    # Core local reflex - throttled (see STOP_REFLEX_COOLDOWN
                    # above) so repeating "stop" doesn't queue a pile of
                    # blocking TTS announcements that delay everything else.
                    if ("halt" in text or "stop" in text) and (now - last_stop_reflex_at) > STOP_REFLEX_COOLDOWN:
                        last_stop_reflex_at = now
                        self.bus.publish("picarx/audio/speak", {"text": "Stopping"})

if __name__ == "__main__":
    node = AudioNode()
    node.run()
