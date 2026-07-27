# Gesture tracking model troubleshooting

The web console's `model_error` state is emitted by `_load_hands()`, before
any frame is sent to the model. This separates a model/runtime problem from
camera ownership and hand-detection problems.

1. Check the live status and capture the complete error, rather than only the
   console label:

   ```bash
   mosquitto_sub -t picarx/gesture/status -v
   ```

2. Run the import check with the same Python executable used to launch Layer B:

   ```bash
   python3 -c 'import sys; print(sys.executable); import mediapipe as mp; print(mp.__version__)'
   python3 -m pip show mediapipe
   ```

   If this reports `No module named 'mediapipe'` (the current development
   environment does), the cause is a missing runtime dependency, not a bad
   camera frame. Install MediaPipe in the Layer B environment using that same
   interpreter, then restart the module. If the import names another missing
   package, install or repair that dependency instead.

3. If the import succeeds, check whether the camera is available:

   ```bash
   mosquitto_sub -t picarx/gesture/status -v
   # then enable gesture tracking from the Tools page
   ```

   `camera_wait` means another process owns the camera; `camera_error` is a
   Picamera2/configuration failure; `process_error` means MediaPipe rejected a
   captured frame. Only after the status reaches `tracking` should lighting,
   framing, and hand-loss behavior be investigated.

The tracker now includes the missing package and the exact `sys.executable` in
the `model_error` payload. This avoids the common situation where MediaPipe is
installed into one Python environment while the orchestrator runs another.
