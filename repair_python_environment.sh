#!/usr/bin/env bash
# Install this project's Python dependencies in a dedicated virtual
# environment.  Debian and Raspberry Pi OS mark their system interpreter as
# externally managed (PEP 668), so installing these distributions into it is
# both unnecessary and unsafe.
set -euo pipefail

# The virtual environment and its packages are root-owned by default because
# the target processes are system services.  Both locations can be changed by
# the caller when a service uses a different layout.
if [[ "$(id -u)" -ne 0 ]]; then
    # sudo commonly removes arbitrary environment variables.  Preserve the
    # two settings that affect where and how the environment is built.
    exec sudo env \
        "PYTHON=${PYTHON-}" \
        "PICARX_VENV=${PICARX_VENV-}" \
        "$0" "$@"
fi

if [[ -n "${PYTHON:-}" ]]; then
    if [[ "$PYTHON" == */* ]]; then
        BASE_PYTHON="$PYTHON"
    else
        BASE_PYTHON="$(command -v "$PYTHON" || true)"
    fi
elif [[ -x /usr/bin/python3 ]]; then
    BASE_PYTHON=/usr/bin/python3
else
    BASE_PYTHON="$(command -v python3 || true)"
fi
if [[ -z "$BASE_PYTHON" || ! -x "$BASE_PYTHON" ]]; then
    echo "Python interpreter not found: ${PYTHON:-python3}" >&2
    exit 1
fi

VENV="${PICARX_VENV:-/opt/picarx/venv}"
if [[ "$VENV" != /* ]]; then
    echo "PICARX_VENV must be an absolute path: $VENV" >&2
    exit 1
fi
VENV_PYTHON="$VENV/bin/python"

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Creating Python virtual environment: $VENV"
    mkdir -p "$(dirname "$VENV")"
    if ! "$BASE_PYTHON" -m venv "$VENV"; then
        cat >&2 <<EOF
Could not create $VENV.
Install the venv support for the selected interpreter and run this script
again (on Debian/Raspberry Pi OS this is usually: apt install python3-venv).
EOF
        exit 1
    fi
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "Virtual environment is missing its Python executable: $VENV_PYTHON" >&2
    exit 1
fi
if ! "$VENV_PYTHON" -m pip --version >/dev/null 2>&1; then
    echo "pip is not available in $VENV; recreate it with python3-venv." >&2
    exit 1
fi

# Keep this list aligned with setup_python.sh. Import and distribution names
# differ for paho-mqtt and OpenCV. The official MediaPipe PyPI build does not
# cover Raspberry Pi's ARM wheel/runtime combinations reliably, so use the
# Pi 4 build there. Override this when deploying a separately built official
# wheel (for example: PICARX_MEDIAPIPE_PACKAGE=mediapipe).
case "$(uname -m)" in
    aarch64|armv7l|armv8l)
        MEDIAPIPE_PACKAGE="${PICARX_MEDIAPIPE_PACKAGE:-mediapipe-rpi4}"
        ;;
    *)
        MEDIAPIPE_PACKAGE="${PICARX_MEDIAPIPE_PACKAGE:-mediapipe}"
        ;;
esac
dependencies=(
    "anthropic|anthropic"
    "cv2|opencv-contrib-python-headless"
    "mediapipe|$MEDIAPIPE_PACKAGE"
    "numpy|numpy"
    "onnxruntime|onnxruntime"
    "paho.mqtt|paho-mqtt"
    # Do not use the unqualified PyPI name here: it is a different, newer
    # project that also claims the robot_hat import and lacks SunFounder's
    # ADC/Pin/PWM API expected by picarx. PiCar-X's supported library is the
    # SunFounder 2.5.x branch.
    "robot_hat|git+https://github.com/sunfounder/robot-hat.git@2.5.x"
    "tokenizers|tokenizers"
    "vosk|vosk"
    "webrtcvad|webrtcvad"
)

users=(root)
add_user() {
    local candidate="$1"
    [[ -n "$candidate" ]] || return 0
    id "$candidate" >/dev/null 2>&1 || return 0
    local existing
    for existing in "${users[@]}"; do
        [[ "$existing" == "$candidate" ]] && return 0
    done
    users+=("$candidate")
}

# Discover the accounts used by the services so that the final import check
# also verifies that those accounts can read the root-owned environment.
for unit in picarx-safety.service picarx-orchestrator.service; do
    if command -v systemctl >/dev/null 2>&1 && systemctl cat "$unit" >/dev/null 2>&1; then
        service_user="$(systemctl show "$unit" -p User --value 2>/dev/null || true)"
        add_user "${service_user:-root}"
        echo "$unit user: ${service_user:-root}"
    fi
done
add_user picarx

as_user() {
    local user="$1"
    shift
    if [[ "$user" == root ]]; then
        "$@"
    else
        runuser -u "$user" -- "$@"
    fi
}

venv_site="$("$VENV_PYTHON" -c 'import site; print(site.getsitepackages()[0])')"
echo "Base interpreter: $BASE_PYTHON"
echo "Installing allow-listed dependencies in: $VENV"
for dependency in "${dependencies[@]}"; do
    import_name="${dependency%%|*}"
    package_name="${dependency##*|}"
    if [[ "$import_name" == "robot_hat" ]]; then
        # Remove a previously installed, same-named incompatible distribution
        # before installing the official package. Otherwise pip can leave
        # module files from both projects in the shared robot_hat namespace.
        "$VENV_PYTHON" -m pip uninstall -y robot-hat robot_hat >/dev/null 2>&1 || true
        "$VENV_PYTHON" -m pip install --force-reinstall "$package_name"
        continue
    fi
    if [[ "$import_name" == "mediapipe" && "$package_name" == "mediapipe-rpi4" ]]; then
        # Both distributions install the mediapipe import package. Remove a
        # newer generic install first, otherwise mixed Python/native files can
        # make model construction abort before the module can report an error.
        "$VENV_PYTHON" -m pip uninstall -y mediapipe mediapipe-rpi4 \
            >/dev/null 2>&1 || true
        "$VENV_PYTHON" -m pip install --force-reinstall "$package_name"
        continue
    fi
    "$VENV_PYTHON" -m pip install --upgrade "$package_name"
done

# Keep the venv readable by service accounts without changing anything in a
# user's home directory or in the system Python installation.
chown -R root:root "$VENV"
chmod -R a+rX "$VENV"

# Existing units may invoke /usr/bin/python3 with an absolute ExecStart.  A
# systemd drop-in cannot replace that command through PATH alone, so expose
# the venv's site-packages explicitly as well as putting its bin directory
# first for subprocesses and bare `python`/`pip` commands.
if command -v systemctl >/dev/null 2>&1; then
    changed_units=0
    for unit in picarx-safety.service picarx-orchestrator.service; do
        if systemctl cat "$unit" >/dev/null 2>&1; then
            dropin="/etc/systemd/system/${unit}.d"
            mkdir -p "$dropin"
            {
                printf '%s\n' '[Service]'
                printf 'Environment=VIRTUAL_ENV=%s\n' "$VENV"
                printf 'Environment=PATH=%s/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin\n' "$VENV"
                printf 'Environment=PYTHONPATH=%s\n' "$venv_site"
                printf '%s\n' 'Environment=PYTHONNOUSERSITE=1'
            } > "$dropin/picarx-python-environment.conf"
            changed_units=1
        fi
    done
    if [[ "$changed_units" -eq 1 ]]; then
        if ! systemctl daemon-reload; then
            echo "warning: systemd daemon-reload failed; reload it before restarting the services." >&2
        fi
    fi
fi

echo "Validating imports with the service environment..."
for user in "${users[@]}"; do
    for dependency in "${dependencies[@]}"; do
        import_name="${dependency%%|*}"
        as_user "$user" env \
            VIRTUAL_ENV="$VENV" \
            PATH="$VENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
            PYTHONPATH="$venv_site" \
            PYTHONNOUSERSITE=1 \
            "$BASE_PYTHON" -c "import $import_name" >/dev/null
    done
    as_user "$user" env \
        VIRTUAL_ENV="$VENV" \
        PATH="$VENV/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
        PYTHONPATH="$venv_site" \
        PYTHONNOUSERSITE=1 \
        "$BASE_PYTHON" -c \
        'from robot_hat import ADC, Pin, PWM, Servo, fileDB, Grayscale_Module, Ultrasonic' \
        >/dev/null
    echo "validated: $user"
done

echo "Done. Restart picarx-safety and picarx-orchestrator to apply the Python environment."
