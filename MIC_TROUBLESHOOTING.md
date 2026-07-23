# Microphone troubleshooting

The mic pipeline is `audio_nodes.py` (Layer B): a USB mic → ALSA capture → gain
+ band-pass → a **speech gate** → Vosk (Kaldi) speech-to-text → published on
**`picarx/audio/heard`**. Command routers (companion, field_agent, tools) read
that topic. TTS ("speech out") is a *separate* path in the same module, so the
robot can talk while the mic is dead, or vice-versa.

The speech gate is now a **voice-activity detector** (`webrtcvad`) rather than
the old amplitude threshold. Install it once on the robot:

```
pip install webrtcvad
```

If it isn't installed the module logs one line and **falls back to the adaptive
energy/SNR gate automatically** — nothing breaks, so this is optional but
recommended. Tune with `audio.vad` (on/off) and `audio.vad_aggressiveness`
(`0` hears more / `3` rejects more) in `layer_b/config.json`. The SNR/confidence
reject in step 7 still runs on top of either gate.

Nothing in the recent IMU / self-trainer work touches this pipeline — the mic is
USB, the IMU is I²C — so start by assuming a hardware / ALSA / config cause.

Work top to bottom; each step tells you which layer is at fault.

---

## 0. First: is it "no input" or "garbage / rejected"?

On the robot (or over SSH), watch what the mic actually emits:

```bash
mosquitto_sub -t 'picarx/audio/heard' -v
```

Now speak a clear command.

- **Nothing at all, ever** → capture/mute/model problem → steps 1–5.
- **Messages appear but your words are wrong / low quality** → the mic hears,
  but decoding or the gate is off → steps 6–7.
- **A steady stream of messages even in silence** → the mic is picking up noise
  (bad ground, motor/electrical hum, mic next to speaker). This *also* explains
  a self-trainer stuck on `busy` (see step 8).

Also check the self-trainer's own diagnosis — it now reports what keeps it awake:

```bash
mosquitto_sub -t 'picarx/self_trainer/status' -v
# look at:  "last_activity": "audio/heard"   and   "idle_for_sec"
```
If `last_activity` is `audio/heard` and `idle_for_sec` keeps resetting to ~0,
the mic is spraying messages — a mic problem, not a self-training one.

---

## 1. Is the audio node running, and did it start cleanly?

```bash
pgrep -af audio_nodes.py                 # expect one line
sudo journalctl -u picarx-orchestrator --since "20 min ago" | grep -iE "audio|vosk|mic|speak|alsa|espeak"
```

Look for, near startup:
- `Vosk model not found at <path>. Speech recognition disabled.` → step 5.
- ALSA / device-open errors (`cannot open`, `Device or resource busy`,
  `No such file or directory`) → step 4.
- a Python traceback → the module crashed; the last lines name the cause.

## 2. Is the mic muted by the kill-switch?

There is a remote mic kill-switch (`picarx/audio/mic_control`, driven by the web
console Audio card). Check the current state and re-enable it:

```bash
mosquitto_sub -t 'picarx/audio/mic_state' -v -C 1     # {"enabled": true|false}
# if false, turn it back on:
mosquitto_pub -t 'picarx/audio/mic_control' -m '{"enabled": true}'
```

## 3. Is the mic stuck muted behind TTS? (self-speech mute)

The mic is muted while the robot speaks (it sits next to the speaker). That mute
is released in a `finally`, so a TTS *error* recovers — **but a hung `aplay`
(stuck audio device) leaves the mic muted indefinitely.** Symptom: the mic died
right after the robot last spoke.

```bash
pgrep -af "aplay|espeak"                 # a lingering aplay = a stuck playback
# clear a stuck playback + restart the audio node:
pkill -f aplay ; pkill -f espeak
sudo systemctl restart picarx-orchestrator.service
```
If this is the recurring cause, tell me and I'll add a hard timeout around the
`aplay` call so a stuck speaker can never wedge the mic.

## 4. ALSA / USB hardware — can Linux capture at all?

```bash
arecord -l                               # is the USB mic listed as a capture device?
arecord -d 3 -f cd /tmp/mic.wav          # record 3s (speak now)
aplay /tmp/mic.wav                       # play it back
```
- Mic **not listed** → USB/power/cabling. Reseat it; `dmesg | tail -30` for
  USB errors; try another port.
- Records **silence** → wrong default capture device or a hardware mixer muting
  it: `alsamixer` → F4 (capture) → unmute (`M`) and raise the level, and/or set
  the right card as default in `~/.asoundrc` / `/etc/asound.conf`.
- Records **fine** here but the robot still hears nothing → the capture works;
  the problem is above ALSA (steps 2, 3, 5, 6).

## 5. Vosk model present?

```bash
python3 -c "import sys; sys.path.insert(0,'layer_b'); import robot_config as c; \
print(c.get('audio','vosk_model_path', c.base_path('modules','models','model-en-lgraph')))"
ls -la layer_b/modules/models/model-en-lgraph    # or whatever the path prints
```
A missing/incomplete model dir disables STT entirely (you'll see the log line in
step 1). Re-download the model to that path.

## 6. Confirm the raw levels (is sound reaching the gate?)

Turn on live level printing, then restart and watch the log while you speak:

```bash
# in layer_b/config.json set  "audio": { "debug_levels": true, ... }
# (or:  export AUDIO_DEBUG_LEVELS=1  in the service env)
sudo systemctl restart picarx-orchestrator.service
sudo journalctl -u picarx-orchestrator -f | grep -i level
```
- Levels stay near the noise floor while you speak → capture/gain problem
  (step 4, or raise `audio.gain`).
- Levels jump when you speak but no `picarx/audio/heard` → the SNR/confidence
  gate is rejecting it → step 7.

## 7. The gate may be too strict (or too loose)

`audio_nodes` drops decodes that don't clear these (all in `config.json` /
env, live on the Config page):

| Knob | Default | Effect |
|---|---|---|
| `audio.gain` | 12.0 | digital amplification for low-gain USB mics — **raise** if quiet |
| `audio.heard_min_snr` | 2.5 | utterance peak must beat the noise floor by this ×  — **lower** to accept quieter speech |
| `audio.heard_min_confidence` | 0.3 | Kaldi mean per-word confidence floor |
| `audio.bandpass` / `_hp_hz` / `_lp_hz` | on / 150 / 4000 | band-pass filter; a mis-set band can kill speech |

A quiet mic that "stopped working" often just needs `gain` up or `heard_min_snr`
down. A noisy mic firing constantly (step 0, third case) needs `heard_min_snr`
**up**.

**Did your tuned values get reset?** The orchestrator now materializes defaults
into `config.json` at startup (`robot_config.sync_defaults`), but it only *adds
missing* knobs and preserves existing ones. Confirm your audio section is intact:

```bash
python3 -c "import json; print(json.load(open('layer_b/config.json'))['audio'])"
```
If it shows defaults where you had custom values, restore them (and let me know —
that would be a real bug in the materializer).

## 8. If it correlates with self-training being stuck `busy`

A mic spraying false `picarx/audio/heard` keeps the self-trainer's idle clock
pinned (each message counts as activity), so it never trains. Fixing the mic
(step 7 — raise `heard_min_snr`) fixes both. Confirm with the status topic in
step 0: once the mic is quiet, `idle_for_sec` should climb toward
`idle_needed_sec` and the state should move `busy → idle → training`.

---

## What to send me if you're still stuck

```bash
sudo journalctl -u picarx-orchestrator --since "20 min ago" | grep -iE "audio|vosk|mic|speak|alsa|espeak" > /tmp/mic_log.txt
arecord -l >> /tmp/mic_log.txt
python3 -c "import json; print(json.load(open('layer_b/config.json'))['audio'])" >> /tmp/mic_log.txt
mosquitto_sub -t 'picarx/audio/mic_state' -v -C 1 >> /tmp/mic_log.txt
```
Paste `/tmp/mic_log.txt` plus which of the step-0 cases you see.
