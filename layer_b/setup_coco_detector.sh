#!/bin/sh
# One-time download of the YOLOv4-tiny COCO detector files that upgrade
# vision_basic.py from 20 VOC classes to 80 COCO classes. Run on the
# robot; ~24MB total. modules/models/ is gitignored on purpose - model
# weights live on the robot, not in git. vision_basic.py picks the new
# model up automatically on its next restart (and falls back to the old
# VOC model if any of these files are missing or corrupt).
set -e
DIR="$(dirname "$0")/modules/models/yolov4-tiny"
mkdir -p "$DIR"

echo "Downloading YOLOv4-tiny (COCO) into $DIR ..."
curl -L --fail -o "$DIR/yolov4-tiny.cfg" \
  "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg"
curl -L --fail -o "$DIR/coco.names" \
  "https://raw.githubusercontent.com/AlexeyAB/darknet/master/data/coco.names"
curl -L --fail -o "$DIR/yolov4-tiny.weights" \
  "https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights"

echo "Done. Restart the vision module (or the orchestrator) to switch to 80-class detection."
