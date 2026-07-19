#!/usr/bin/env python3
# layer_b/tools/grayscale_calibration.py
"""
Standalone grayscale sensor diagnostic.

STOP safety_daemon.py before running this - both scripts try to own
the same hardware (px.get_grayscale_data()), and running them
simultaneously risks conflicting GPIO/ADC access, same class of
problem as the earlier "device busy" mic issue.

Usage:
    sudo systemctl stop picarx-orchestrator   # or however yours is stopped
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

import time
from picarx import Picarx

px = Picarx()

print("Reading grayscale sensors. Ctrl+C to stop.")
print("Drive/carry the robot over carpet, tile, the seam, and a real edge.\n")

try:
    while True:
        values = px.get_grayscale_data()
        print(f"grayscale: {values}")
        time.sleep(0.2)
except KeyboardInterrupt:
    print("\nStopped.")