#!/usr/bin/env python3
"""Read-only dependency check. Does not connect to a robot or start perception."""
import importlib
import sys


def main():
    missing = []
    for name in ('numpy', 'cv2', 'torch', 'torchvision', 'ultralytics', 'rclpy', 'ffw_pick', 'ffw_grasp'):
        try:
            module = importlib.import_module(name)
            print(f'{name}: {getattr(module, "__version__", "OK")}')
        except Exception as error:
            missing.append(name)
            print(f'{name}: {error}')
    if missing:
        print('Missing or incompatible runtime dependencies: ' + ', '.join(missing))
        return 1
    import torch
    print(f'PyTorch CUDA available: {torch.cuda.is_available()}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
