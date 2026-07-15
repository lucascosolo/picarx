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
import math
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
#
# To switch models without editing code, point VOSK_MODEL_PATH at any
# unpacked Vosk model directory (one containing am/, conf/, graph/ ...).
# Grab one from https://alphacephei.com/vosk/models - e.g. the larger
# vosk-model-en-us-0.22 for better accuracy (needs much more RAM/CPU, so
# watch this Pi's budget), or a small model for lower latency. The engine
# and the whole pipeline here are model-agnostic; only this path changes.
MODEL_PATH = os.environ.get(
    "VOSK_MODEL_PATH", "/home/picarx/layer_b/modules/models/model-en-lgraph")

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

# --- stuck-open gate failsafe ---
# Field data (debug_monitor, 7.5min sample): this process averaged 74%
# CPU with peaks at 100% - the gate was effectively never closing. The
# floor estimate only adapts during CONFIRMED SILENCE, so any sustained
# sound source (driving-motor noise through a gain-12 mic, a TV, the
# robot's own constant TTS during a fail-state storm) keeps rms above
# threshold forever, and the floor never gets a silent moment to catch
# up: decode runs continuously from then on. Real utterances last a few
# seconds; anything keeping the gate open for FLOOR_RESEED_AFTER_SEC
# straight is ambient by definition. Two-layer fix:
#   - while the gate is open, the floor also drifts (much more slowly)
#     TOWARD the observed rms, so continuous noise gradually raises the
#     threshold above itself and the gate re-closes on its own;
#   - if it's still open after FLOOR_RESEED_AFTER_SEC, the floor is
#     hard-reseeded to the recent rms average and the gate forced shut.
OPEN_DRIFT_ALPHA = 0.005        # per-chunk (~125ms): ~25s time constant while gate is open
FLOOR_RESEED_AFTER_SEC = 20.0   # continuously open longer than this -> reseed floor, close gate
RESEED_RMS_WINDOW = 40          # chunks (~5s) of recent rms kept for the reseed value

# Tail kept muted after our own TTS playback finishes, so the speaker's
# reverb doesn't re-open the gate. See mute_until in AudioNode.speak() -
# without this the robot hears and dutifully decodes every one of its
# own announcements (which, during a busy exploration session, is a
# near-continuous stream) - burning decode CPU and keeping the gate open.
SELF_SPEECH_MUTE_TAIL_SEC = 0.4


class EnergyGate:
    """Decides, chunk by chunk, whether Kaldi should see any audio.

    Pure bookkeeping (no I/O), factored out of the capture loop so the
    stuck-open failsafe logic is testable off-robot.
    """

    def __init__(self):
        self.noise_floor = None
        self.speaking_until = 0.0
        self.open_since = None          # when the gate last transitioned closed -> open
        self.recent_rms = deque(maxlen=RESEED_RMS_WINDOW)

    def process(self, rms, now):
        """Returns True if this chunk should be decoded."""
        self.recent_rms.append(rms)
        if self.noise_floor is None:
            self.noise_floor = rms
        threshold = max(MIN_THRESHOLD, self.noise_floor * NOISE_MULTIPLIER)

        if rms > threshold:
            self.speaking_until = now + TRAILING_SILENCE_SEC
            if self.open_since is None:
                self.open_since = now

        gate_open = now <= self.speaking_until

        if gate_open:
            # Slow drift toward sustained sound so continuous ambient
            # noise eventually raises the threshold above itself.
            self.noise_floor += OPEN_DRIFT_ALPHA * (rms - self.noise_floor)
            if now - self.open_since > FLOOR_RESEED_AFTER_SEC:
                avg = sum(self.recent_rms) / len(self.recent_rms)
                print(f"Audio gate: open {FLOOR_RESEED_AFTER_SEC:.0f}s straight - treating as "
                      f"ambient noise, reseeding floor {self.noise_floor:.0f} -> {avg:.0f}")
                self.noise_floor = avg
                self.speaking_until = 0.0
                self.open_since = None
                return False
        else:
            self.open_since = None
            # Confirmed silence - normal (faster) floor adaptation.
            self.noise_floor += NOISE_FLOOR_ALPHA * (rms - self.noise_floor)

        if DEBUG_LEVELS:
            print(f"Audio node: chunk rms={rms:.0f} floor={self.noise_floor:.0f} threshold={threshold:.0f}")
        return gate_open

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


# --- voice-band noise filter (resource-light, streaming) ---
# Steady background noise in a room is dominated by energy OUTSIDE the
# band human speech actually lives in: HVAC/fan/traffic rumble below
# ~150 Hz and hiss/clatter above ~4 kHz. Two cascaded second-order
# biquads (a high-pass then a low-pass, RBJ cookbook, Butterworth Q)
# strip both, so what reaches the energy gate and Kaldi is mostly
# voice. This directly fixes "the gate won't open in a noisy room":
# the tracked noise floor stops being inflated by rumble the mic can't
# even use. Cost is a few multiply-adds per sample - ~160k mult/sec at
# our chunk rate, negligible next to a single decode step. Entirely
# fail-soft and env-toggleable; if disabled or it errors, audio passes
# through untouched and behaviour is exactly as before.
BANDPASS_ENABLED = os.environ.get("AUDIO_BANDPASS", "1") not in ("0", "", "false", "no")
BANDPASS_HP_HZ = float(os.environ.get("AUDIO_BANDPASS_HP", "150"))   # kill rumble below this
BANDPASS_LP_HZ = float(os.environ.get("AUDIO_BANDPASS_LP", "4000"))  # kill hiss above this
SAMPLE_RATE = 16000


class _Biquad:
    """One second-order section, Direct Form II transposed, float state."""
    def __init__(self, b0, b1, b2, a1, a2):
        self.b0, self.b1, self.b2, self.a1, self.a2 = b0, b1, b2, a1, a2
        self.z1 = 0.0
        self.z2 = 0.0

    @classmethod
    def highpass(cls, fc, fs, q=0.7071):
        w0 = 2.0 * math.pi * fc / fs
        cw, sw = math.cos(w0), math.sin(w0)
        alpha = sw / (2.0 * q)
        a0 = 1.0 + alpha
        return cls((1.0 + cw) / 2.0 / a0, -(1.0 + cw) / a0, (1.0 + cw) / 2.0 / a0,
                   (-2.0 * cw) / a0, (1.0 - alpha) / a0)

    @classmethod
    def lowpass(cls, fc, fs, q=0.7071):
        w0 = 2.0 * math.pi * fc / fs
        cw, sw = math.cos(w0), math.sin(w0)
        alpha = sw / (2.0 * q)
        a0 = 1.0 + alpha
        return cls((1.0 - cw) / 2.0 / a0, (1.0 - cw) / a0, (1.0 - cw) / 2.0 / a0,
                   (-2.0 * cw) / a0, (1.0 - alpha) / a0)


class _PassThrough:
    """Null filter used when AUDIO_BANDPASS is disabled."""
    def process(self, data):
        return data


class VoiceBandFilter:
    """Cascaded high-pass + low-pass over a raw 16-bit PCM chunk stream.
    State persists across chunks so there's no per-chunk edge click."""
    def __init__(self, fs=SAMPLE_RATE, hp=BANDPASS_HP_HZ, lp=BANDPASS_LP_HZ):
        # Only build sections that make sense for the rate (lp must be
        # below Nyquist); clamp defensively so bad env values fail soft.
        self.sections = []
        if 20.0 < hp < fs / 2.0:
            self.sections.append(_Biquad.highpass(hp, fs))
        if 20.0 < lp < fs / 2.0 and lp > hp:
            self.sections.append(_Biquad.lowpass(lp, fs))

    def process(self, data):
        if not self.sections or not data:
            return data
        try:
            samples = array.array("h", data)
        except ValueError:
            return data
        for bq in self.sections:
            b0, b1, b2, a1, a2 = bq.b0, bq.b1, bq.b2, bq.a1, bq.a2
            z1, z2 = bq.z1, bq.z2
            for i, x in enumerate(samples):
                y = b0 * x + z1
                z1 = b1 * x - a1 * y + z2
                z2 = b2 * x - a2 * y
                # clamp back to int16
                samples[i] = 32767 if y > 32767 else (-32768 if y < -32768 else int(y))
            bq.z1, bq.z2 = z1, z2
        return samples.tobytes()


class AudioNode:
    def __init__(self):
        self.bus = Bus()
        self.mute_until = 0.0   # mic ignored while our own TTS is playing (see speak())
        # Remote mic kill-switch (picarx/audio/mic_control) - lets the
        # web console silence STT entirely in a loud room (or while the
        # radio plays) so background noise can't fire false commands.
        # Speech OUT is unaffected; only recognition is gated.
        self.mic_enabled = True

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
    AUDIO_OUT_DEVICE = "plug:robot_speaker"

    def speak(self, text):
        print(f"PiCar Speaking: {text}")
        # Mute our own mic for the duration of playback (+ a short
        # reverb tail): the mic sits right next to the speaker, so
        # without this every announcement gets captured, holds the
        # energy gate open, and burns a full STT decode on our own
        # voice. Set BEFORE playback starts and refreshed after, since
        # playback is blocking and this handler runs on the MQTT
        # callback thread while the mic loop keeps consuming chunks.
        self.mute_until = float("inf")
        try:
            espeak_proc = subprocess.Popen(["espeak", "--stdout", text], stdout=subprocess.PIPE)
            subprocess.run(
                ["aplay", "-D", self.AUDIO_OUT_DEVICE, "-q"],
                stdin=espeak_proc.stdout,
                check=False,
            )
            espeak_proc.stdout.close()
            espeak_proc.wait()
        finally:
            self.mute_until = time.time() + SELF_SPEECH_MUTE_TAIL_SEC

    # Announcements older than this on arrival get dropped instead of
    # spoken. aplay playback is blocking and serial, so during a busy
    # stretch (fail-state loop) speak requests stack up in the MQTT
    # callback queue and play long after the moment they described -
    # the robot ends up narrating decisions from half a minute ago while
    # visibly doing something else. Stale narration is worse than none.
    SPEAK_MAX_AGE_SEC = 6.0

    def handle_speak_request(self, payload):
        text = payload.get("text", "")
        if not text:
            return
        ts = payload.get("ts")
        if ts is not None and (time.time() - float(ts)) > self.SPEAK_MAX_AGE_SEC:
            print(f"(dropping stale queued announcement: {text})")
            return
        self.speak(text)

    def _enable_speakers(self):
        # The robot_hat speaker amp is gated by a GPIO switch that comes
        # up DISABLED on every boot - without flipping it on first, aplay
        # plays to a dead output and no speech is ever heard. Must run
        # after the hat is initialized (safety_daemon's Picarx() does
        # that) and before our first speak(). Override the command via
        # SPEAKER_ENABLE_CMD if your robot_hat build names it differently.
        cmd = os.environ.get("SPEAKER_ENABLE_CMD", "robot_hat enable_speaker")
        try:
            subprocess.run(cmd.split(), check=False, timeout=5,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"Audio node: ran '{cmd}' to enable the speaker amp.")
        except (FileNotFoundError, subprocess.SubprocessError) as e:
            print(f"Audio node: could not enable speakers via '{cmd}': {e}")

    def on_mic_control(self, payload):
        enabled = bool(payload.get("enabled", True))
        if enabled != self.mic_enabled:
            self.mic_enabled = enabled
            print(f"Audio node: microphone {'enabled' if enabled else 'disabled'} remotely")
        # Always answer with current state so UIs can sync on connect.
        self.bus.publish("picarx/audio/mic_state",
                         {"enabled": self.mic_enabled, "ts": time.time()})

    def run(self):
        self.bus.subscribe("picarx/audio/speak", self.handle_speak_request)
        self.bus.subscribe("picarx/audio/mic_control", self.on_mic_control)
        self._enable_speakers()

        if not self.model:
            print("Audio node running in output-only mode.")
            while True:
                time.sleep(1)

        print("Audio node running: starting ALSA audio stream...")
        self.speak("Systems initialized. Standing by for instructions.")

        # Use arecord with plughw to automatically resample the USB mic to 16000Hz.
        # Card index from `arecord -l`: card 1 is the USB PnP Sound Device (the mic).
        cmd = [
            "arecord",
            "-D", "plug:robot_mic",
            "-f", "S16_LE",
            "-c", "1",
            "-r", "16000",
            "-t", "raw",
            "-q" # Quiet mode, no standard output text
        ]

        # Start the microphone stream directly from the OS
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE)

        gate = EnergyGate()
        voice_filter = VoiceBandFilter() if BANDPASS_ENABLED else _PassThrough()
        print(f"Audio node: voice-band filter {'on' if BANDPASS_ENABLED else 'off'} "
              f"({BANDPASS_HP_HZ:.0f}-{BANDPASS_LP_HZ:.0f} Hz)")
        prebuffer = deque(maxlen=PREBUFFER_CHUNKS)
        self._last_stop_reflex_at = 0.0
        gate_was_open = False

        while True:
            # Read a chunk at a time from the arecord output
            data = process.stdout.read(CHUNK_BYTES)
            if len(data) == 0:
                print("Audio node: arecord produced no data (device open failure or process exit) - STT is now dead until this module restarts")
                break

            now = time.time()

            # Remotely disabled (web console kill-switch): keep reading
            # so the arecord pipe never backs up, but decode nothing.
            if not self.mic_enabled:
                prebuffer.clear()
                continue

            # Our own TTS is playing (or just finished) - drop the
            # chunk entirely: no decode, no gate/floor update (the
            # speaker blast would poison the ambient estimate), no
            # prebuffer (we don't want our own voice flushed into the
            # recognizer when the gate next opens).
            if now < self.mute_until:
                prebuffer.clear()
                continue

            # Strip out-of-band noise FIRST (on the raw chunk), then
            # apply gain - so the gain amplifies mostly voice instead of
            # rumble, and the gate below keys on voice-band energy.
            data = voice_filter.process(data)
            data = _apply_gain(data, AUDIO_GAIN)
            rms = _chunk_rms(data)

            if not gate.process(rms, now):
                # Confirmed silence. Because we stop feeding Kaldi during
                # silence, its own endpointer never sees the trailing
                # silence it needs to finalize - so on the FIRST silent
                # chunk after speech, finalize the utterance ourselves
                # (FinalResult). Without this, successive phrases pile
                # into one un-ended utterance ("explore explore explore")
                # until Kaldi flushes on its own many seconds later.
                if gate_was_open:
                    gate_was_open = False
                    self._emit_result(self.rec.FinalResult(), now)
                # Keep a short rolling buffer so if speech starts on the
                # very next chunk, we don't lose the onset.
                prebuffer.append(data)
                continue

            gate_was_open = True

            # Just triggered (or still within the trailing window) -
            # flush any pre-buffered quiet-adjacent audio first so the
            # start of the utterance isn't chopped off, then feed this
            # chunk normally.
            if prebuffer:
                for buffered in prebuffer:
                    self.rec.AcceptWaveform(buffered)
                prebuffer.clear()

            # A natural in-speech endpoint (e.g. a real pause) still
            # finalizes immediately; otherwise we wait for gate close.
            if self.rec.AcceptWaveform(data):
                self._emit_result(self.rec.Result(), now)

    def _emit_result(self, result_json, now):
        try:
            text = json.loads(result_json).get("text", "").lower().strip()
        except (json.JSONDecodeError, AttributeError):
            return
        if not text:
            return
        print(f"Heard locally: '{text}'")
        self.bus.publish("picarx/audio/heard", {"text": text})
        # Core local reflex - throttled (see STOP_REFLEX_COOLDOWN) so
        # repeating "stop" doesn't queue a pile of blocking TTS.
        if ("halt" in text or "stop" in text) and (now - self._last_stop_reflex_at) > STOP_REFLEX_COOLDOWN:
            self._last_stop_reflex_at = now
            self.bus.publish("picarx/audio/speak", {"text": "Stopping"})

if __name__ == "__main__":
    node = AudioNode()
    node.run()
