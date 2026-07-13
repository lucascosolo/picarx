#!/usr/bin/env python3
# /home/picarx/layer_b/modules/vision_basic.py
"""
Basic vision module - publishes face / motion detection data to MQTT
for Layer C skills to consume (e.g. follow_me.py). Uses OpenCV's 
built-in Haar cascade.
"""
import sys
sys.path.insert(0, "/home/picarx/layer_b")
from broker_client import Bus
from picamera2 import Picamera2
import cv2
import time

DETECT_INTERVAL = 0.2

def run():
  bus = Bus()
  picam2 = Picamera2()
  config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
  picam2.configure(config)
  picam2.start()
  time.sleep(1)

  face_cascade = cv2.CascadeClassifier(
    "/home/picarx/layer_b/modules/cascades/cascades.xml"
  )

  print("vision basic module running, publishing to picarx/vision/faces")

  while True:
    frame = picam2.capture_array()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) > 0:
      #publish the largest detected face
      x, y, w, h = max(faces.tolist(), key=lambda f: f[2])
      frame_width = frame.shape[1]
      bus.publish("picarx/vision/faces", {
        "detected": True,
        "x": x, "y": y, "w": w, "h": h,
        "frame_width": frame_width,
        "frame_center_offset": (x + w // 2) - (frame_width // 2)
      })
    else:
      bus.publish("picarx/vision/faces", {"detected": False})

    time.sleep(DETECT_INTERVAL)

if __name__ == "__main__":
  run()
