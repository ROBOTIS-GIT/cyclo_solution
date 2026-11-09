#!/usr/bin/env python3
"""Materialize YOLOv8's final stride broadcast for OpenCV 4.6 compatibility.

Run after the documented static ONNX export. This changes no model arithmetic.
Requires onnx + numpy in the export environment, not the ROS runtime.
"""

import argparse

import numpy as np
import onnx
from onnx import numpy_helper

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model")
args = parser.parse_args()
model = onnx.load(args.model)
matched = 0
for node in model.graph.node:
    if node.name != "/model.22/Mul_2" or node.op_type != "Mul":
        continue
    for tensor in model.graph.initializer:
        if tensor.name == node.input[1]:
            data = numpy_helper.to_array(tensor)
            if data.shape not in {(1, 8400), (1, 4, 8400)}:
                raise ValueError(f"Unexpected stride shape: {data.shape}")
            expanded = np.broadcast_to(data, (1, 4, 8400)).copy()
            tensor.CopyFrom(numpy_helper.from_array(expanded, tensor.name))
            matched += 1
if matched != 1:
    raise ValueError("Expected exactly one YOLOv8 segmentation stride multiply")
onnx.checker.check_model(model)
onnx.save(model, args.model)
print("Materialized stride broadcast; model validated")
