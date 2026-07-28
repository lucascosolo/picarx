# Gesture tracking model troubleshooting

The tracker now initializes MediaPipe asynchronously. `model_loading` no longer
blocks the camera/state loop, and every loading status includes `phase`,
`elapsed_sec`, `frame_age_sec`, `timeout_sec`, `interpreter`, MediaPipe package
versions/files, backend, and hand-model path/size. A timeout or exception ends
in `model_error`, releases the camera and RobotState lease, and leaves object
detection free to recover. This separates a model/runtime problem from camera
ownership and hand-detection problems.

1. Check the live status and capture the complete error, rather than only the
   console label:

   ```bash
   mosquitto_sub -t picarx/gesture/status -v
   sudo journalctl -u picarx-orchestrator.service -f -o cat
   ```

   In `model_loading`, note the last `phase` (`import`, `asset_downloading`,
   or `constructing`) and the reported `interpreter`, `mediapipe_file`,
   `package_versions`, `model_path`, and `model_size_bytes`. If the status ends
   at `model_error`, capture `error`, `exception`, and `cleanup` as well.

2. Run the import check with the same Python executable used to launch Layer B:

   ```bash
   /opt/picarx/venv/bin/python -c 'import sys; print(sys.executable); import mediapipe as mp; print(mp.__file__); print(getattr(mp, "__version__", "unknown"))'
   /opt/picarx/venv/bin/python -m pip show mediapipe mediapipe-rpi4
   ```

   Use the exact interpreter and environment reported by the status payload;
   on the production services this is normally `/opt/picarx/venv/bin/python`
   with `PYTHONPATH` set by the `picarx-orchestrator.service` drop-in. To
   reproduce the service import without the web UI:

   ```bash
   sudo -u picarx env PYTHONNOUSERSITE=1 \
     /opt/picarx/venv/bin/python -c 'import mediapipe as mp; print(mp.__file__)'
   ```

   If this reports `No module named 'mediapipe'` (the current development
   environment does), the cause is a missing runtime dependency, not a bad
   camera frame. Install MediaPipe in the Layer B environment using that same
   interpreter, then restart the module. If the import names another missing
   package, install or repair that dependency instead.

3. If the import succeeds, check the camera controller and then enable gesture
   tracking:

   ```bash
   mosquitto_sub -t picarx/camera/status -v
   mosquitto_sub -t picarx/gesture/status -v
   # then enable gesture tracking from the Tools page
   ```

   The controller should report the active subscriber and requested FPS. It is
   the only process allowed to open Picamera2; `camera_error` is a
   Picamera2/configuration failure, while `process_error` means MediaPipe
   rejected a decoded frame. Only after the gesture status reaches `tracking`
   should lighting, framing, and hand-loss behavior be investigated.

4. If the model timed out, check network access to the Tasks asset and the
   native binding before retrying. The retry is explicit and does not retain a
   failed camera/state lease:

   ```bash
   curl -I --max-time 15 \
     https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
   mosquitto_pub -t picarx/gesture/control \
     -m '{"enabled":true,"retry":true}'
   ```

   The tracker includes the exact `sys.executable` and native package path in
   the status payload, avoiding the common situation where MediaPipe is
   installed into one Python environment while the orchestrator runs another.

5. Isolate native model construction from MQTT and camera ownership. Run this
   with the same interpreter and environment shown in the status record:

   ```bash
   timeout --signal=KILL 60s \
     sudo -u picarx env VIRTUAL_ENV=/opt/picarx/venv \
     PATH=/opt/picarx/venv/bin:$PATH \
     PYTHONPATH=/opt/picarx/venv/lib/python3.13/site-packages \
     PYTHONNOUSERSITE=1 \
     /usr/bin/python3 layer_b/modules/gesture_tracking.py --probe-model
   ```

   A JSON ok=true result proves the native constructor returns; a timeout
   proves the installed ARM binding is hanging before inference. The current
   production mediapipe version 1.0.0 is not an official PyPI release
   matching the current upstream release line. Upstream documents that its
   Python package does not provide Linux aarch64 wheels, so the robot must
   use a tested ARM build or a locally built wheel rather than an arbitrary
   generic package. Do not mix it with mediapipe-rpi4; both distributions
   install the same mediapipe namespace.

   The probe prints each phase as a flushed JSON line. If it reaches
   `phase=constructing` and then emits the AES/SIGILL fatal error, the crash
   is in the native MediaPipe constructor. If it dies before `phase=imported`,
   isolate the import itself:

   ```bash
   /usr/bin/python3 -c 'import mediapipe; print("mediapipe import ok")'
   /usr/bin/python3 -c 'from mediapipe.tasks.python import vision; print("tasks import ok")'
   ```

   Also identify the exact crashing object instead of assuming it is
   MediaPipe:

   ```bash
   grep -R -a -l 'go/sigill-fail-fast' \
     /opt/picarx/venv/lib/python3.13/site-packages 2>/dev/null
   ```
