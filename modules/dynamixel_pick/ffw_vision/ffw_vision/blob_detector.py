from dataclasses import dataclass
import math

import cv2
from ffw_vision.depth_stats import clip_roi
import numpy as np

OPEN_KERNEL = np.ones((3, 3), np.uint8)
CLOSE_KERNEL = np.ones((5, 5), np.uint8)
RING_GAP_KERNEL = np.ones((7, 7), np.uint8)  # skip 3 px of mixed depth at object edges
RING_OUTER_KERNEL = np.ones((27, 27), np.uint8)  # local floor ring out to 13 px
MIN_RING_PX = 50
TOP_SHARE = 20  # percent of nearest pixels that sets the top level


@dataclass(frozen=True)
class DetectorParams:
    roi: tuple = (0, 0, 0, 0)  # u_min, v_min, u_max, v_max; all zeros = full image
    dark_v_max: int = 110  # HSV value below this is object
    gray_s_max: int = 120  # low saturation ...
    gray_v_max: int = 180  # ... and not bright is object
    min_area: int = 300
    max_single_area: int = 1800
    max_single_length: float = 60.0
    min_height: float = 0.012  # m above floor; lower blobs are dropped
    standing_min_height: float = 0.039
    min_depth_ratio: float = 0.3  # valid depth share needed for a height
    top_tolerance: float = 0.008  # m below the top level still counted as top surface
    min_top_ratio: float = 0.3  # top surface share of the blob needed to use it
    square_ratio: float = 1.25  # length / width below this = near square
    min_depth: float = 0.05
    max_depth: float = 2.0


@dataclass(frozen=True)
class Blob:
    u: float
    v: float
    length: float
    width: float
    angle: float
    angle_period: float
    area: int
    height: float  # NaN when unknown
    pose_label: str
    ambiguous: bool
    top_height: float = float('nan')  # level of the nearest TOP_SHARE, diagnostics only
    floor: float = float('nan')  # depth used as floor for this blob
    color_box: tuple = None  # (u, v, length, width, angle) of the whole color blob


def normalize_angle(angle, period=math.pi):
    """Wrap into [-period/2, period/2)."""
    return (angle + period / 2) % period - period / 2


def object_mask(color_bgr, params):
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    mask = (v < params.dark_v_max) | ((s < params.gray_s_max) & (v < params.gray_v_max))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, OPEN_KERNEL)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, CLOSE_KERNEL)


def rotated_box(mask):
    """Box of nonzero pixels: u, v, length, width, angle of the long side."""
    rows, cols = np.nonzero(mask)
    rect = cv2.minAreaRect(np.column_stack((cols, rows)).astype(np.float32))
    (u, v), corners = rect[0], cv2.boxPoints(rect)
    side_a, side_b = corners[1] - corners[0], corners[2] - corners[1]
    if np.hypot(*side_a) < np.hypot(*side_b):
        side_a, side_b = side_b, side_a
    length, width = np.hypot(*side_a), np.hypot(*side_b)
    angle = normalize_angle(math.atan2(side_a[1], side_a[0]))
    return float(u), float(v), float(length), float(width), angle


def valid_depth(depth, params):
    with np.errstate(invalid='ignore'):
        return np.isfinite(depth) & (depth >= params.min_depth) & (depth <= params.max_depth)


def local_floor(depth_ok, depth, blob_mask, all_mask, roi, fallback):
    """Median depth in a ring around the blob, away from every object; fallback if too few."""
    near = cv2.dilate(blob_mask, RING_OUTER_KERNEL).astype(bool)
    objects = cv2.dilate(all_mask, RING_GAP_KERNEL).astype(bool)
    ring = near & ~objects & roi & depth_ok
    return float(np.median(depth[ring])) if ring.sum() >= MIN_RING_PX else fallback


def largest_component(mask):
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count < 2:
        return mask
    return (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)


def top_surface(depth, depth_ok, blob_mask, floor, params):
    """(top mask or None, median height of it, top level); heights in meters above floor."""
    nan = float('nan')
    blob = blob_mask.astype(bool)
    good = blob & depth_ok
    if not math.isfinite(floor) or good.sum() < params.min_depth_ratio * blob.sum():
        return None, nan, nan
    heights = floor - depth
    top_level = float(np.percentile(heights[good], 100 - TOP_SHARE))
    top = (good & (heights >= top_level - params.top_tolerance)).astype(np.uint8)
    top = largest_component(cv2.morphologyEx(top, cv2.MORPH_OPEN, OPEN_KERNEL))
    if top.sum() < params.min_top_ratio * blob.sum():
        return None, nan, top_level
    return top, float(np.median(heights[top.astype(bool)])), top_level


def detect(color_bgr, depth, params):
    """Dark blobs in roi; depth in meters (NaN = invalid) or None. Returns (blobs, floor)."""
    height_px, width_px = color_bgr.shape[:2]
    u0, v0, u1, v1 = clip_roi(params.roi, width_px, height_px)
    roi = np.zeros((height_px, width_px), bool)
    roi[v0:v1, u0:u1] = True
    mask = object_mask(color_bgr, params) & roi.astype(np.uint8)

    floor = float('nan')
    depth_ok = None
    if depth is not None:
        depth_ok = valid_depth(depth, params)
        floor_px = roi & ~mask.astype(bool) & depth_ok
        if floor_px.any():
            floor = float(np.median(depth[floor_px]))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    blobs = []
    for i in range(1, count):
        left, top, box_w, box_h, area = stats[i]
        if area < params.min_area:
            continue
        blob_mask = (labels == i).astype(np.uint8)
        color_box = rotated_box(blob_mask)
        on_edge = left <= u0 or top <= v0 or left + box_w >= u1 or top + box_h >= v1
        ambiguous = (area > params.max_single_area or color_box[2] > params.max_single_length
                     or on_edge)

        blob_floor, top_mask, height, top_level = floor, None, float('nan'), float('nan')
        if depth is not None:
            blob_floor = local_floor(depth_ok, depth, blob_mask, mask, roi, floor)
            top_mask, height, top_level = top_surface(
                depth, depth_ok, blob_mask, blob_floor, params)
        if math.isfinite(top_level) and top_level < params.min_height:
            continue  # flat: shadow or print

        # top surface only for single objects; side faces shift the color blob
        box = rotated_box(top_mask) if top_mask is not None and not ambiguous else color_box
        u, v, length, width, angle = box
        if ambiguous or not math.isfinite(height):
            pose_label = 'unknown'
        else:
            pose_label = 'standing' if height >= params.standing_min_height else 'lying'
        if ambiguous:
            period = 0.0
        else:
            period = math.pi / 2 if length / width < params.square_ratio else math.pi
        blobs.append(Blob(u, v, length, width, angle, period, int(area), height,
                          pose_label, ambiguous, top_level, blob_floor, color_box))
    return blobs, floor
