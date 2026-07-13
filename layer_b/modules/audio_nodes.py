#!/usr/bin/env python3
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus

import time
import json
import subprocess
from vosk import Model, KaldiRecognizer

# Upgraded from the original small model ("model-en", still present on
# disk for rollback) to Vosk's lgraph model: same engine/pipeline, a
# meaningfully more accurate decoding graph, and still light enough to
# run in real time on a Pi (unlike Vosk's full ~1.8GB server model).
MODEL_PATH = "/home/picarx/layer_b/modules/models/model-en-lgraph"

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
        
        while True:
            # Read 4000 bytes at a time from the arecord output
            data = process.stdout.read(4000)
            if len(data) == 0:
                print("Audio node: arecord produced no data (device open failure or process exit) - STT is now dead until this module restarts")
                break
                
            if self.rec.AcceptWaveform(data):
                result = json.loads(self.rec.Result())
                text = result.get("text", "").lower()
                
                if text:
                    print(f"Heard locally: '{text}'")
                    # Publish what was heard to the MQTT bus
                    self.bus.publish("picarx/audio/heard", {"text": text})
                    
                    # Core local reflexes
                    if "halt" in text or "stop" in text:
                        self.bus.publish("picarx/audio/speak", {"text": "Stopping"})

if __name__ == "__main__":
    node = AudioNode()
    node.run()