#!/usr/bin/env bash
# Compatibility entry point for older deployments.
#
# The original script installed into whichever interpreter happened to be
# called ``python3``. That is unsafe on current Debian/Raspberry Pi OS, where
# the system interpreter is PEP-668 externally managed. Keep the old filename
# working, but use the same architecture-aware venv and systemd environment
# repair path as production deployments.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/repair_python_environment.sh" "$@"
