#!/usr/bin/env bash
# Separate Python environment; preserve ROS numpy/OpenCV and Jetson's vendor torch.
set -euo pipefail
task_env="${PICK_PYTHON_ENV:-/opt/demo-pick-venv}"
# A new venv does not inherit the packages of a parent venv. Keep Jetson's
# existing torch/torchvision visible without copying or replacing its CUDA wheels.
task_vendor_site="$(python3 -c 'import importlib.util, pathlib; spec = importlib.util.find_spec("torch"); print(pathlib.Path(spec.origin).parent.parent if spec and spec.origin else "")')"
if python3 -c 'import ensurepip' >/dev/null 2>&1; then
  python3 -m venv --system-site-packages "$task_env"
else
  task_bootstrap="$(mktemp -d)"
  trap 'rm -rf "$task_bootstrap"' EXIT
  python3 -m pip install --target "$task_bootstrap" virtualenv
  PYTHONPATH="$task_bootstrap${PYTHONPATH:+:$PYTHONPATH}" \
    python3 -m virtualenv --system-site-packages "$task_env"
fi
task_python="$task_env/bin/python3"
if [[ -n "$task_vendor_site" ]]; then
  task_site="$("$task_python" -c 'import site; print(site.getsitepackages()[0])')"
  printf '%s\n' "$task_vendor_site" > "$task_site/jetson-vendor.pth"
fi
if ! "$task_python" -c 'import torch, torchvision' >/dev/null 2>&1; then
  if [[ "$(uname -m)" != x86_64 ]]; then
    echo 'This ARM environment needs a working torch/torchvision installation matched to its JetPack.' >&2
    echo 'Run this script from the existing inference environment after installing the vendor builds.' >&2
    exit 1
  fi
  "$task_python" -m pip install torch==2.7.1 torchvision==0.22.1 \
    --index-url https://download.pytorch.org/whl/cpu
fi
# Install without automatic torch/OpenCV replacements, especially on Jetson.
"$task_python" -m pip install --index-url "${PICK_PIP_INDEX_URL:-https://pypi.org/simple}" --no-deps ultralytics==8.3.228 ultralytics-thop
"$task_python" -m pip install --index-url "${PICK_PIP_INDEX_URL:-https://pypi.org/simple}" 'numpy<2' 'matplotlib<3.10' pillow pyyaml requests scipy psutil polars
"$task_python" -c 'from ultralytics import YOLO; import torch; print("Pick inference ready; CUDA:", torch.cuda.is_available())'
printf '\nLaunch with: pick_python:=%s\n' "$task_python"
