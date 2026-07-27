#!/usr/bin/env bash
# Consolidate the project's Python dependencies into the system interpreter
# used by systemd. Only this project's allow-listed distributions are touched;
# unrelated system/user packages are left alone.
set -euo pipefail

if [[ "$(id -u)" -ne 0 ]]; then
    exec sudo "$0" "$@"
fi

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON="$PYTHON"
elif [[ -x /usr/bin/python3 ]]; then
    PYTHON=/usr/bin/python3
else
    PYTHON="$(command -v python3)"
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "Python interpreter not found: $PYTHON" >&2
    exit 1
fi
if ! "$PYTHON" -m pip --version >/dev/null 2>&1; then
    echo "pip is not available for $PYTHON" >&2
    exit 1
fi

# Keep this list aligned with setup_python.sh. Import and distribution names
# differ for paho-mqtt and OpenCV.
dependencies=(
    "anthropic|anthropic"
    "cv2|opencv-contrib-python-headless"
    "mediapipe|mediapipe"
    "numpy|numpy"
    "onnxruntime|onnxruntime"
    "paho.mqtt|paho-mqtt"
    "robot_hat|robot-hat"
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

# A systemd service runs as root only when its unit has no User= setting.
# Inspect both units instead of assuming that is true.
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

echo "Canonical interpreter: $PYTHON"
echo "Removing only project dependency copies from user site-packages..."
for user in "${users[@]}"; do
    user_site="$(as_user "$user" "$PYTHON" -c 'import site; print(site.getusersitepackages())' 2>/dev/null || true)"
    [[ -n "$user_site" ]] || continue
    for dependency in "${dependencies[@]}"; do
        import_name="${dependency%%|*}"
        package_name="${dependency##*|}"
        origin="$(as_user "$user" env -u PYTHONNOUSERSITE "$PYTHON" -c \
            "import importlib.util; s=importlib.util.find_spec('$import_name'); print(getattr(s, 'origin', '') if s else '')" \
            2>/dev/null || true)"
        if [[ "$origin" == "$user_site"/* ]]; then
            echo "remove: $package_name from $user user site ($origin)"
            as_user "$user" "$PYTHON" -m pip uninstall -y "$package_name" >/dev/null
        fi
    done
done

echo "Installing/upgrading the allow-listed dependencies for $PYTHON..."
for dependency in "${dependencies[@]}"; do
    package_name="${dependency##*|}"
    "$PYTHON" -m pip install --upgrade --break-system-packages "$package_name"
done

# Normalize ownership only for the distributions just installed, and only
# under standard system Python locations. Never recursively chown arbitrary
# paths returned by a package or anything in a user's home directory.
for dependency in "${dependencies[@]}"; do
    package_name="${dependency##*|}"
    package_root="$("$PYTHON" -c \
        "import importlib.metadata as m; print(m.distribution('$package_name').locate_file(''))" \
        2>/dev/null || true)"
    package_root="$(realpath -m "$package_root" 2>/dev/null || true)"
    case "$package_root" in
        /usr/lib/python*/*|/usr/local/lib/python*/*)
            chown -R root:root "$package_root"
            chmod -R a+rX "$package_root"
            ;;
        "")
            echo "warning: could not locate installed files for $package_name" >&2
            ;;
        *)
            echo "warning: refusing to change ownership outside system Python: $package_root" >&2
            ;;
    esac
done

# Prevent future ~/.local package shadowing for the two services. This does
# not force either service to run as root; it makes both use the canonical
# system installation selected above.
if command -v systemctl >/dev/null 2>&1; then
    for unit in picarx-safety.service picarx-orchestrator.service; do
        if systemctl cat "$unit" >/dev/null 2>&1; then
            dropin="/etc/systemd/system/${unit}.d"
            mkdir -p "$dropin"
            printf '%s\n' '[Service]' 'Environment=PYTHONNOUSERSITE=1' \
                > "$dropin/python-environment.conf"
        fi
    done
    systemctl daemon-reload
fi

echo "Validating imports with user-site packages disabled..."
for user in "${users[@]}"; do
    for dependency in "${dependencies[@]}"; do
        import_name="${dependency%%|*}"
        as_user "$user" env PYTHONNOUSERSITE=1 "$PYTHON" -c \
            "import $import_name" >/dev/null
    done
    echo "validated: $user"
done

echo "Done. Restart picarx-safety and picarx-orchestrator to apply the systemd environment guard."
