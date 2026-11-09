# Head-camera AI model

`yolov8n-seg.onnx` is the official Ultralytics YOLOv8n-seg COCO checkpoint exported with Ultralytics 8.3.221, ONNX opset 12, batch 1, 640x640, static shapes, simplification enabled, no embedded NMS. The original checkpoint is https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-seg.pt . Model licensing: https://www.ultralytics.com/license (AGPL-3.0 / Enterprise). Runtime inference is C++; Python Ultralytics is only used to export weights, not installed in the demo runtime.

Outputs must be FP32 `[1,116,8400]` and `[1,32,160,160]`. Do not substitute a detection-only, end-to-end, custom-class-count or different-input-size model. COCO class 41 (`cup`) is enabled by default. Bottle class 39 is deliberately excluded. This generic model is not trained specifically on the user's tumbler.

Optional `yolov8n-seg.engine` is built **on the target Jetson**, with TensorRT 10 and internal FP16/FP32 IO. It is not portable across GPUs or TensorRT versions. `ai.backend=auto` selects this engine if installed and supported by the build; otherwise it reports `opencv_cpu`. A present but invalid engine fails startup instead of silently changing backends. `ai.backend=tensorrt` requires TensorRT. Model loading never downloads files.

Inside a GPU-enabled demo container:

```bash
bash /root/ros2_ws/src/demo_1f/scripts/build_ai_engine.sh
```

For persistence across image builds, copy the generated engine back to the robot host's `demo_1f/models/` before rebuilding. Never copy the Jetson engine to an unrelated GPU's model directory. An absent model produces an explicit startup error.

Reproduce the ONNX export in a separate export environment with PyTorch/torchvision and Ultralytics 8.3.221:

```python
from ultralytics import YOLO
YOLO('yolov8n-seg.pt').export(format='onnx', imgsz=640, batch=1,
                           opset=12, simplify=True, dynamic=False, device='cpu')
```

After export, run `python3 scripts/prepare_ai_onnx.py models/yolov8n-seg.onnx` in that environment. It materializes the final stride constant's broadcast from `[1,8400]` to `[1,4,8400]` for OpenCV 4.6. This is mathematically identical; inference output shape and weights are unchanged.
