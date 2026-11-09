"""Numeric depth conversion and filtering; ROS-independent."""

import numpy as np


def depth_meters(raw, encoding):
    if encoding.lower() not in ("16uc1", "mono16", "32fc1"):
        raise ValueError("원본 16UC1/32FC1 depth가 필요합니다")
    return raw.astype(np.float32) * (0.001 if encoding.lower() != "32fc1" else 1.0)


def filter_depth(raw, encoding, near, far):
    meters = depth_meters(raw, encoding)
    keep = np.isfinite(meters) & (meters >= near) & (meters <= far)
    filtered = raw.copy()
    filtered[~keep] = 0
    return filtered, meters, keep


def gray_depth(meters, valid, near, far):
    # Keep 0 exclusively for invalid/excluded pixels; preserve distances separately.
    scale = np.nan_to_num((far - meters) / (far - near), nan=0, posinf=0, neginf=0)
    gray = (1 + np.clip(scale, 0, 1) * 254).astype(np.uint8)
    gray[~valid] = 0
    return gray
