#!/usr/bin/env bash
# Install the Python dependencies used by Layer B, the safety daemon, and the
# offline tools.  The import check makes this safe to run repeatedly: packages
# that are already available to the selected interpreter are not reinstalled.
set -euo pipefail

PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python interpreter not found: $PYTHON" >&2
    exit 1
fi

if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
    echo "pip is not available for $PYTHON; install python3-pip first." >&2
    exit 1
fi

# Entries are "import name|pip distribution".  Some distributions expose a
# different import name (paho-mqtt -> paho, opencv -> cv2).
dependencies=(
    "anthropic|anthropic"
    "cv2|opencv-contrib-python-headless"
    "mediapipe|mediapipe"
    "numpy|numpy"
    "onnxruntime|onnxruntime"
    "paho.mqtt|paho-mqtt"
    "robot_hat|git+https://github.com/sunfounder/robot-hat.git@2.5.x"
    "tokenizers|tokenizers"
    "vosk|vosk"
    "webrtcvad|webrtcvad"
)

for dependency in "${dependencies[@]}"; do
    import_name="${dependency%%|*}"
    package_name="${dependency##*|}"
    if [[ "$import_name" == "cv2" ]]; then
        # Importing cv2 alone is not enough: the vision module needs the DNN
        # model loaders, which are absent from some minimal distro builds.
        cv2_ok="$($PYTHON -c 'import cv2; d=getattr(cv2, "dnn", None); raise SystemExit(0 if d and (hasattr(d, "readNetFromCaffe") or hasattr(d, "readNet")) and (hasattr(d, "readNetFromDarknet") or hasattr(d, "readNet")) else 1' 2>/dev/null && echo yes || echo no)"
    else
        cv2_ok=yes
    fi
    if [[ "$import_name" == "robot_hat" ]]; then
        cv2_ok=no
    elif [[ "$import_name" != "cv2" ]] && ! "$PYTHON" -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('$import_name') else 1)"; then
        cv2_ok=no
    fi
    if [[ "$cv2_ok" == yes ]]; then
        echo "skip: $package_name (import $import_name already works)"
    else
        echo "install: $package_name"
        if [[ "$import_name" == "robot_hat" ]]; then
            "$PYTHON" -m pip uninstall -y robot-hat robot_hat >/dev/null 2>&1 || true
        fi
        "$PYTHON" -m pip install --upgrade --break-system-packages "$package_name"
    fi
done

"$PYTHON" -c 'from robot_hat import ADC, Pin, PWM, Servo, fileDB, Grayscale_Module, Ultrasonic'

# Picamera2 is distributed by Raspberry Pi OS rather than PyPI.  Install it
# through apt when available, while still making the script useful on a host
# machine where camera hardware is intentionally absent.
if "$PYTHON" -c "import importlib.util; raise SystemExit(0 if importlib.util.find_spec('picamera2') else 1)"; then
    echo "skip: python3-picamera2 (import picamera2 already works)"
elif command -v apt-get >/dev/null 2>&1; then
    echo "install: python3-picamera2 (Raspberry Pi OS package)"
    if [[ "$(id -u)" -eq 0 ]]; then
        apt-get install -y python3-picamera2
    elif command -v sudo >/dev/null 2>&1; then
        sudo apt-get install -y python3-picamera2
    else
        echo "picamera2 is missing and sudo is unavailable; install python3-picamera2 manually." >&2
        exit 1
    fi
else
    echo "warning: picamera2 is missing; install Raspberry Pi OS package python3-picamera2 on the robot." >&2
fi

echo "Python dependency setup complete for $PYTHON."
