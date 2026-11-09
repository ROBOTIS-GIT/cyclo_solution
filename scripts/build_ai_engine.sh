#!/usr/bin/env bash
# Run inside the GPU-enabled demo container, on the Jetson that will execute it.
set -euo pipefail
model="${1:-/root/ros2_ws/src/demo_1f/models/yolov8n-seg.onnx}"
engine="${2:-/root/ros2_ws/install/demo_1f/share/demo_1f/models/yolov8n-seg.engine}"
trtexec="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
[[ -r "$model" && -x "$trtexec" ]] || { echo "Missing ONNX model or TensorRT trtexec" >&2; exit 1; }
mkdir -p "$(dirname "$engine")"
# Internal FP16, static FP32 IO. Engines must be rebuilt for another GPU/TRT version.
"$trtexec" --onnx="$model" --saveEngine="$engine.tmp" --fp16 --skipInference
mv "$engine.tmp" "$engine"
sha256sum "$model" > "$engine.model.sha256"
echo "Built $engine. Launch perception manually; no ROS commands were sent."
