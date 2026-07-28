#!/usr/bin/env python3
# layer_b/tools/grayscale_calibration.py
"""
Standalone grayscale sensor diagnostic.

Keep safety_daemon.py running while using this diagnostic. It is the sole
owner of the grayscale hardware; this script asks it for readings over the
same Unix socket used by the runtime sensor consumer.

Usage:
    python3 grayscale_calibration.py

Then physically carry/drive the robot slowly over:
  - plain carpet
  - plain tile
  - the carpet/tile transition seam
  - an actual edge/drop-off (a step, table edge, etc - do this one
    carefully, at low height, to see what a REAL cliff reads as)

Watch the three printed values (left, middle, right sensor - order
depends on your wiring/library version, verify against SunFounder's
docs for your HAT if unsure) and note the LOWEST value you see over
normal floor, and the value you see over an actual edge. The gap
between those two numbers is what CLIFF_THRESHOLD needs to sit
between. If normal-floor readings ever dip close to or below the
current threshold (200), that confirms it's a false-positive/
calibration issue rather than an actual sensor fault.
"""
import os
import getpass
os.getlogin = getpass.getuser

import json
import socket
import time

SOCKET_PATH = "/tmp/picarx_safety.sock"


def read_grayscale():
    """Read the safety daemon's HAT-owned grayscale sensor snapshot."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(1.0)
        client.connect(SOCKET_PATH)
        client.sendall(json.dumps({"query": "grayscale"}).encode())
        result = json.loads(client.recv(1024).decode())
    if "error" in result:
        raise RuntimeError(result["error"])
    return result["grayscale"]

print("Reading grayscale sensors. Ctrl+C to stop.")
print("Drive/carry the robot over carpet, tile, the seam, and a real edge.\n")

try:
    while True:
        values = read_grayscale()
        print(f"grayscale: {values}")
        time.sleep(0.2)
except KeyboardInterrupt:
    print("\nStopped.")
