#!/usr/bin/env python3
import os
import getpass
os.getlogin = getpass.getuser

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from broker_client import Bus
import socket
import json
import time

SOCKET_PATH = "/tmp/picarx_safety.sock"

def query_safety_distance():
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(2.0)
            s.connect(SOCKET_PATH)
            s.sendall(json.dumps({"query": "distance"}).encode())
            resp = s.recv(1024)
        return json.loads(resp.decode())
    except Exception as e:
        return {"error": str(e)}

def run():
    bus = Bus()
    print("distance_sensor module running, querying safety daemon socket...")
    
    while True:
        data = query_safety_distance()
        if "distance_cm" in data:
            reading = data["distance_cm"]
            # The ultrasonic sensor returns negative values (commonly
            # -1 or -2) to signal "no echo received" - out of range,
            # angled away from any surface, or a bad read. That is
            # NOT a valid distance and must not be forwarded as one;
            # downstream code (world_state.py, field_agent.py) has no
            # way to tell "sensor says -2cm" apart from "sensor says
            # 2cm" unless we filter it out here at the source. We
            # simply skip publishing this cycle, so consumers see the
            # last good reading age out and go stale (an honest
            # signal) rather than seeing a false near-obstacle.
            if reading is not None and reading >= 0:
                bus.publish("picarx/sensors/distance", {"distance_cm": reading})
            else:
                print(f"Distance sensor returned invalid reading ({reading}), not publishing")
        else:
            print(f"Failed to read distance via socket: {data.get('error')}")
        time.sleep(0.5)

if __name__ == "__main__":
    run()